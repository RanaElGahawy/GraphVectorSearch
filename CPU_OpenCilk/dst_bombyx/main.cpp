// Mapping of Falcon onto Cilk tasks:
//   * dst_search is the control logic of one query processing pipeline. It runs
//     serially: pops candidate groups (S1), filters visited neighbors with a
//     Bloom filter (S2), and merges results into the queues (S5, S6).
//   * eval_range is the Bloom-fetch-compute work of one group: fetch the
//     neighbor vectors and compute their distances to the query (S3, S4). It
//     splits the neighbors in halves down to grain distances per task, so
//     grain = 1 is one task per neighbor.
//   * search_batch spawns one dst_search per query (across-query parallelism).
//
// DST in fork-join form: groups of mc candidates are launched in epochs of mg
// groups, and there is one cilk_sync per epoch. Step g of an epoch merges group
// g of the previous epoch (the delayed synchronization), selects a new group
// and spawns its evaluation without waiting for it. So up to mg groups are
// evaluated at the same time, overlapped with the serial merges and selections,
// and every group is selected from queues that do not yet contain the results
// of the mg-1 most recently launched groups, as in Algorithm 2.
// mg = 1, mc = 1 is best-first search (Algorithm 1), mg = 1, mc > 1 is
// multi-candidate search (MCS).
//
// The spawned tasks only write the distances of their own group, which the
// serial control logic reads only after the next cilk_sync, so the output is
// deterministic for any number of workers.

#include <cfloat>  // For FLT_MAX
#include <cstdint> // For uint32_t

//  The set of libraries that can only be used in OpenCilk software
#include <cilk/cilk.h>

//  Driver Code specific
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdio.h>
#include <string>
#include <utility>
#include <vector>

#define BLOOM_BITS 262144 // 256 Kbit, same as Falcon
#define BLOOM_WORDS 8192  // BLOOM_BITS / 32
#define BLOOM_HASHES 3

// Global variables for the driver
int n;         // number of database vectors
int dim;       // vector dimensionality
int maxDegree; // row stride of the adjacency array
int entry;     // entry node of the graph
float *vecs;   // n * dim
int *degree;   // n
int *links;    // n * maxDegree

/**************** Distance computation (S3, S4)
 * L2 distance squared
 * squared Euclidean distance between two vectors
 * S4: Computer distances****************/

inline float l2sq(const float *a, const float *b, int d)
{
  float acc = 0;
  for (int i = 0; i < d; i = i + 1)
  {
    float t = a[i] - b[i];
    acc += t * t;
  }
  return acc;
}

// Bloom-fetch-compute jobs: distances between the query and vectors ids[lo..hi)
// The only parallel work inside the query
// Same as the BFC unit in the falcon
// grain: how many distances one PE computes per task it receives.
void eval_range(const float *q, const float *vecs, int d, const int *ids,
                float *dists, int lo, int hi, int grain)
{
  if (hi - lo <= grain)
  {
    for (int i = lo; i < hi; i = i + 1)
    {
      dists[i] = l2sq(q, &vecs[ids[i] * d], d);
    }
  }
  else
  {
    int mid = lo + (hi - lo) / 2;
    cilk_spawn eval_range(q, vecs, d, ids, dists, lo, mid, grain);
    cilk_spawn eval_range(q, vecs, d, ids, dists, mid, hi, grain);
    cilk_sync;
  }
}

/**************** Bloom filter (S2) ****************/

// MurmurHash2 of a 4-byte key
inline uint32_t murmur2(uint32_t k, uint32_t seed)
{
  uint32_t m = 0x5bd1e995;
  k = k * m;
  k = k ^ (k >> 24);
  k = k * m;
  uint32_t h = seed ^ 4;
  h = h * m;
  h = h ^ k;
  h = h ^ (h >> 13);
  h = h * m;
  h = h ^ (h >> 15);
  return h;
}

// Marks node as visited. Returns 1 if it was (definitely) not visited before.
inline int bloom_test_and_set(uint32_t *bloom, uint32_t node)
{
  int seen = 1;
  for (uint32_t h = 0; h < BLOOM_HASHES; h = h + 1)
  {
    uint32_t pos = murmur2(node, h + 1) & (BLOOM_BITS - 1);
    uint32_t bit = 1u << (pos & 31);
    if ((bloom[pos >> 5] & bit) == 0)
    {
      seen = 0;
    }
    bloom[pos >> 5] = bloom[pos >> 5] | bit;
  }
  return 1 - seen;
}

/**************** Priority queues (S5, S6) ****************/

// Both queues are arrays sorted by ascending distance.
// Inserts (dist, id) keeping the closest cap elements. Returns the new size.
inline int sorted_insert(float *qd, int *qi, int size, int cap, float dist,
                         int id)
{
  int new_size = size;
  if (size < cap || dist < qd[cap - 1])
  {
    if (size < cap)
    {
      new_size = size + 1;
    }
    int j = new_size - 1;
    while (j > 0 && qd[j - 1] > dist)
    {
      qd[j] = qd[j - 1];
      qi[j] = qi[j - 1];
      j = j - 1;
    }
    qd[j] = dist;
    qi[j] = id;
  }
  return new_size;
}

// Max(R.dist): the distance a candidate must not exceed to be evaluated
inline float threshold(const float *rd, int r_size, int ef)
{
  float t = FLT_MAX;
  if (r_size == ef)
  {
    t = rd[ef - 1];
  }
  return t;
}

// Merges the distances of one evaluated group into R and C.
// meta: [0] = R size, [1] = C size, [2] = evaluated candidates, [3] = distance
// computations
// mainly the synchronization step: insert then sort based on threshhold -> S5-S6
inline void merge_group(float *rd, int *ri, float *cd, int *ci, int *meta,
                        int ef, int cand_cap, const int *ids,
                        const float *dists, int cnt)
{
  for (int i = 0; i < cnt; i = i + 1)
  {
    // only nodes that enter R become candidates (same as Falcon)
    if (dists[i] < threshold(rd, meta[0], ef))
    {
      meta[0] = sorted_insert(rd, ri, meta[0], ef, dists[i], ids[i]);
      meta[1] = sorted_insert(cd, ci, meta[1], cand_cap, dists[i], ids[i]);
    }
  }
  meta[3] = meta[3] + cnt;
}

/**************** Candidate group selection (S1, S2) ****************/

// Extract-Min(C, mc, threshold): pops up to mc qualified candidates and writes
// their not yet visited neighbors to ids. Returns the number of popped
// candidates; *cnt is set to the number of neighbors to evaluate.
inline int select_group(float *rd, float *cd, int *ci, int *meta, int ef,
                        int mc, const int *degree, const int *links,
                        int maxDegree, uint32_t *bloom, int *ids, int *cnt)
{
  int popped = 0;
  int c = 0;
  float t = threshold(rd, meta[0], ef);
  while (popped < mc && meta[1] > 0 && cd[0] <= t)
  {
    int node = ci[0];
    // pop the front of C
    for (int j = 1; j < meta[1]; j = j + 1)
    {
      cd[j - 1] = cd[j];
      ci[j - 1] = ci[j];
    }
    meta[1] = meta[1] - 1;
    popped = popped + 1;
    // fetch neighbor ids and filter visited ones
    for (int j = 0; j < degree[node]; j = j + 1)
    {
      int nb = links[node * maxDegree + j];
      if (bloom_test_and_set(bloom, nb) == 1)
      {
        ids[c] = nb;
        c = c + 1;
      }
    }
  }
  *cnt = c;
  meta[2] = meta[2] + popped;
  return popped;
}

/**************** DST search of one query ****************/

// Per-query buffers (all owned by this query):
//   rd, ri:          result queue R, ef entries
//   cd, ci:          candidate queue C, cand_cap entries
//   bloom:           BLOOM_WORDS words
//   slot_ids/dists:  2 * mg group slots of (mc * maxDegree) entries each, the
//                    groups of the current and of the previous epoch
//   slot_cnt:        2 * mg entries
//   meta:            4 entries (see merge_group)
//   out:             k result ids
void dst_search(const float *q, const float *vecs, const int *degree,
                const int *links, int d, int maxDegree, int entry, int ef,
                int cand_cap, int mc, int mg, int grain, int k, float *rd,
                int *ri, float *cd, int *ci, uint32_t *bloom, int *slot_ids,
                float *slot_dists, int *slot_cnt, int *meta, int *out)
{
  int slot_cap = mc * maxDegree;
  for (int i = 0; i < BLOOM_WORDS; i = i + 1)
  {
    bloom[i] = 0;
  }
  for (int i = 0; i < 2 * mg; i = i + 1)
  {
    slot_cnt[i] = 0;
  }
  // C = R = Visited = {entry}
  float d0 = l2sq(q, &vecs[entry * d], d);
  bloom_test_and_set(bloom, entry);
  rd[0] = d0;
  ri[0] = entry;
  cd[0] = d0;
  ci[0] = entry;
  meta[0] = 1;
  meta[1] = 1;
  meta[2] = 0;
  meta[3] = 1;

  int cur = 0; // the current epoch uses slots cur * mg .. cur * mg + mg - 1
  int active = 1;
  while (active == 1)
  {
    int launched = 0;
    for (int g = 0; g < mg; g = g + 1)
    {
      int ps = (1 - cur) * mg + g; // group g of the previous epoch
      int cs = cur * mg + g;       // group g of this epoch
      // delayed synchronization: merge the earliest group in the pipeline,
      // evaluated before the last cilk_sync
      merge_group(rd, ri, cd, ci, meta, ef, cand_cap, &slot_ids[ps * slot_cap],
                  &slot_dists[ps * slot_cap], slot_cnt[ps]);
      slot_cnt[ps] = 0;
      // launch a new group, without waiting for the groups in flight
      int popped = select_group(rd, cd, ci, meta, ef, mc, degree, links,
                                maxDegree, bloom, &slot_ids[cs * slot_cap],
                                &slot_cnt[cs]);
      launched = launched + popped;
      cilk_spawn eval_range(q, vecs, d, &slot_ids[cs * slot_cap],
                            &slot_dists[cs * slot_cap], 0, slot_cnt[cs], grain);
    }
    cilk_sync;
    cur = 1 - cur;

    // all groups of the previous epoch are merged, so if nothing was launched
    // the pipeline is empty, and the last selection found no qualified
    // candidates after the last merge
    if (launched == 0)
    {
      active = 0;
    }
  }

  for (int i = 0; i < k; i = i + 1)
  {
    out[i] = -1;
    if (i < meta[0])
    {
      out[i] = ri[i];
    }
  }
}

/**************** Batch of queries (across-query parallelism) ****************/

// The per-query buffers are laid out back to back, one block per query.
void search_batch(int nq, const float *queries, const float *vecs,
                  const int *degree, const int *links, int d, int maxDegree,
                  int entry, int ef, int cand_cap, int mc, int mg, int grain,
                  int k, float *rd, int *ri, float *cd, int *ci,
                  uint32_t *bloom, int *slot_ids, float *slot_dists,
                  int *slot_cnt, int *meta, int *out)
{
  int slot_size = 2 * mg * mc * maxDegree;
  for (int qi = 0; qi < nq; qi = qi + 1)
  {
    cilk_spawn dst_search(&queries[qi * d], vecs, degree, links, d, maxDegree,
                          entry, ef, cand_cap, mc, mg, grain, k, &rd[qi * ef],
                          &ri[qi * ef], &cd[qi * cand_cap], &ci[qi * cand_cap],
                          &bloom[qi * BLOOM_WORDS], &slot_ids[qi * slot_size],
                          &slot_dists[qi * slot_size], &slot_cnt[qi * 2 * mg],
                          &meta[qi * 4], &out[qi * k]);
  }
  cilk_sync;
}

/**************** Driver ****************/

#pragma BOMBYX IGNORE buildSynthetic
/***
 * Random vectors and a navigable kNN graph: the maxDegree/2 nearest neighbors
 * of each node plus reverse edges, topped up with the rest of its kNN list.
 * The entry node is the medoid. Also computes the exact top-10 ground truth.
 */
void buildSynthetic(int nq, unsigned seed, std::vector<float> &queries,
                    std::vector<int> &gt)
{
  std::mt19937 rng(seed);
  std::normal_distribution<float> N01(0.f, 1.f);
  vecs = new float[(size_t)n * dim];
  for (size_t i = 0; i < (size_t)n * dim; i++)
    vecs[i] = N01(rng);
  queries.resize((size_t)nq * dim);
  for (size_t i = 0; i < queries.size(); i++)
    queries[i] = N01(rng);

  const int K = maxDegree;
  std::vector<int> knn((size_t)n * K);
  std::vector<std::pair<float, int>> all(n);
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
      all[j] = {j == i ? FLT_MAX : l2sq(&vecs[(size_t)i * dim], &vecs[(size_t)j * dim], dim), j};
    std::partial_sort(all.begin(), all.begin() + K, all.end());
    for (int j = 0; j < K; j++)
      knn[(size_t)i * K + j] = all[j].second;
  }
  std::vector<std::vector<int>> adj(n);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < K / 2; j++)
      adj[i].push_back(knn[(size_t)i * K + j]);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < K / 2; j++)
    {
      int v = knn[(size_t)i * K + j];
      if ((int)adj[v].size() < K && std::find(adj[v].begin(), adj[v].end(), i) == adj[v].end())
        adj[v].push_back(i);
    }
  for (int i = 0; i < n; i++)
    for (int j = K / 2; j < K && (int)adj[i].size() < K; j++)
    {
      int v = knn[(size_t)i * K + j];
      if (std::find(adj[i].begin(), adj[i].end(), v) == adj[i].end())
        adj[i].push_back(v);
    }
  degree = new int[n];
  links = new int[(size_t)n * K];
  std::fill_n(links, (size_t)n * K, 0);
  for (int i = 0; i < n; i++)
  {
    degree[i] = adj[i].size();
    std::copy(adj[i].begin(), adj[i].end(), links + (size_t)i * K);
  }

  std::vector<float> mean(dim, 0.f);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < dim; j++)
      mean[j] += vecs[(size_t)i * dim + j] / n;
  float best = FLT_MAX;
  for (int i = 0; i < n; i++)
  {
    float dd = l2sq(mean.data(), &vecs[(size_t)i * dim], dim);
    if (dd < best)
    {
      best = dd;
      entry = i;
    }
  }

  gt.resize((size_t)nq * 10);
  for (int qi = 0; qi < nq; qi++)
  {
    for (int j = 0; j < n; j++)
      all[j] = {l2sq(&queries[(size_t)qi * dim], &vecs[(size_t)j * dim], dim), j};
    std::partial_sort(all.begin(), all.begin() + 10, all.end());
    for (int j = 0; j < 10; j++)
      gt[(size_t)qi * 10 + j] = all[j].second;
  }
}

#pragma BOMBYX IGNORE main
int main(int argc, char *argv[])
{
  if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help"))
  {
    std::cout << "Usage: " << argv[0] << " [mc=1] [mg=1] [n=5000] [dim=32] [nq=200] [maxDegree=32] [ef=64] [grain=1]\n"
              << "  mc = candidates per group, mg = groups in the pipeline\n"
              << "  grain = distance computations per task\n"
              << "  mc=1 mg=1 is best-first search, mg=1 mc>1 is multi-candidate search\n";
    return 0;
  }
  int mc = argc > 1 ? atoi(argv[1]) : 1;
  int mg = argc > 2 ? atoi(argv[2]) : 1;
  n = argc > 3 ? atoi(argv[3]) : 5000;
  dim = argc > 4 ? atoi(argv[4]) : 32;
  int nq = argc > 5 ? atoi(argv[5]) : 200;
  maxDegree = argc > 6 ? atoi(argv[6]) : 32;
  int ef = argc > 7 ? atoi(argv[7]) : 64;
  int grain = argc > 8 ? atoi(argv[8]) : 1;
  int cand_cap = 64; // candidate queue size
  int k = 10;
  if (mc < 1 || mg < 1 || ef < k || grain < 1)
  {
    std::cerr << "need mc >= 1, mg >= 1, ef >= 10, grain >= 1" << std::endl;
    return 1;
  }

  std::vector<float> queries;
  std::vector<int> gt;
  buildSynthetic(nq, 42, queries, gt);

  // per-query buffers
  size_t slot_size = (size_t)2 * mg * mc * maxDegree;
  std::vector<float> rd((size_t)nq * ef), cd((size_t)nq * cand_cap), slot_dists(nq * slot_size);
  std::vector<int> ri((size_t)nq * ef), ci((size_t)nq * cand_cap), slot_ids(nq * slot_size);
  std::vector<int> slot_cnt((size_t)nq * 2 * mg), meta((size_t)nq * 4), out((size_t)nq * k);
  std::vector<uint32_t> bloom((size_t)nq * BLOOM_WORDS);

  auto t0 = std::chrono::steady_clock::now();
  search_batch(nq, queries.data(), vecs, degree, links, dim, maxDegree, entry, ef,
               cand_cap, mc, mg, grain, k, rd.data(), ri.data(), cd.data(), ci.data(),
               bloom.data(), slot_ids.data(), slot_dists.data(), slot_cnt.data(),
               meta.data(), out.data());
  double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

  long hits = 0, cands = 0, dists = 0;
  for (int qi = 0; qi < nq; qi++)
  {
    for (int a = 0; a < 10; a++)
      for (int b = 0; b < k; b++)
        if (gt[(size_t)qi * 10 + a] == out[(size_t)qi * k + b])
          hits++;
    cands += meta[(size_t)qi * 4 + 2];
    dists += meta[(size_t)qi * 4 + 3];
  }

  // deterministic part of the output (diff this against the Bombyx output)
  printf("n=%d dim=%d maxDegree=%d entry=%d nq=%d ef=%d mc=%d mg=%d\n", n, dim,
         maxDegree, entry, nq, ef, mc, mg);
  printf("R@10=%.4f avg evaluated candidates=%.1f avg distance computations=%.1f\n",
         hits / (nq * 10.0), (double)cands / nq, (double)dists / nq);
  for (int qi = 0; qi < std::min(nq, 3); qi++)
  {
    printf("query %d:", qi);
    for (int b = 0; b < k; b++)
      printf(" %d", out[(size_t)qi * k + b]);
    printf("\n");
  }
  fprintf(stderr, "time %.2f ms\n", ms);

  delete[] vecs;
  delete[] degree;
  delete[] links;
  return 0;
}
