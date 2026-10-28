#include "cilk_explicit.hh"
// A single file implementation of Delayed-Synchronization Traversal (DST) for
// graph vector search (Falcon, Jiang et al. VLDB'25, Algorithm 2) using OpenCilk.
// Written in the Bombyx subset so it can later be converted for HLS / HardCilk:
//   * no classes / structs, only functions and plain arrays
//   * no break / continue / early return in task functions (flags instead)
//   * loops use i = i + 1
//   * driver code (data generation, main) is marked BOMBYX IGNORE
//
// Mapping of Falcon onto Cilk tasks:
//   * dst_search is the control logic of one query processing pipeline. It runs
//     serially: pops candidate groups (S1), filters visited neighbors with a
//     Bloom filter (S2), and merges results into the queues (S5, S6).
//   * dist_task is one Bloom-fetch-compute unit job: fetch one vector and compute
//     its distance to the query (S3, S4). One task is spawned per neighbor.
//   * search_batch spawns one dst_search per query (across-query parallelism).
//
// DST in fork-join form: up to mg candidate groups (mc candidates each) are in
// the pipeline. Every round launches one new group and, once the pipeline is
// full, merges only the earliest group (the delayed synchronization). New groups
// are therefore selected from queues that do not yet contain the results of the
// mg-1 most recent groups. mg = 1, mc = 1 is best-first search (Algorithm 1),
// mg = 1, mc > 1 is multi-candidate search (MCS).
//
// The spawned tasks write disjoint array entries and all queue updates are
// serial, so the output is deterministic for any number of workers.

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

/**************** Distance computation (S3, S4) ****************/

float l2sq(const float *a, const float *b, int d);
THREAD(dist_task);
uint32_t murmur2(uint32_t k0, uint32_t seed);
int bloom_test_and_set(uint32_t *bloom, uint32_t node);
int sorted_insert(float *qd, int *qi, int size, int cap, float dist, int id);
float threshold(const float *rd, int r_size, int ef);
void merge_group(float *rd, int *ri, float *cd, int *ci, int *meta, int ef, int cand_cap, const int *ids, const float *dists, int cnt);
int select_group(float *rd, float *cd, int *ci, int *meta, int ef, int mc, const int *degree, const int *links, int maxDegree, uint32_t *bloom, int *ids, int *cnt);
THREAD(dst_search);
void search_batch(int nq, const float *queries, const float *vecs, const int *degree, const int *links, int d, int maxDegree, int entry, int ef, int cand_cap, int mc, int mg, int k0, float *rd, int *ri, float *cd, int *ci, uint32_t *bloom, int *slot_ids, float *slot_dists, int *slot_cnt, int *meta, int *out);
THREAD(dst_search_exit0);
THREAD(dst_search_reentry0);
THREAD(search_batch_cont0);
THREAD(dst_search_reentry0_cont0);


CLOSURE_DEF(dist_task,
    const float *q;
    const float *vecs;
    int d;
    const int *ids;
    float *dists;
    int i;
);
CLOSURE_DEF(dst_search,
    const float *q;
    const float *vecs;
    const int *degree;
    const int *links;
    int d;
    int maxDegree;
    int entry;
    int ef;
    int cand_cap;
    int mc;
    int mg;
    int k0;
    float *rd;
    int *ri;
    float *cd;
    int *ci;
    uint32_t *bloom;
    int *slot_ids;
    float *slot_dists;
    int *slot_cnt;
    int *meta;
    int *out;
);
CLOSURE_DEF(dst_search_exit0,
    int k0;
    int *ri;
    int *meta;
    int *out;
    int i1;
);
CLOSURE_DEF(dst_search_reentry0,
    const float *q;
    const float *vecs;
    const int *degree;
    const int *links;
    int d;
    int maxDegree;
    int ef;
    int cand_cap;
    int mc;
    int mg;
    int k0;
    float *rd;
    int *ri;
    float *cd;
    int *ci;
    uint32_t *bloom;
    int *slot_ids;
    float *slot_dists;
    int *slot_cnt;
    int *meta;
    int *out;
    int slot_cap;
    int head;
    int tail;
    int pending;
    int active;
    int s;
    int popped;
    int i0;
    int i1;
);
CLOSURE_DEF(search_batch_cont0,
);
CLOSURE_DEF(dst_search_reentry0_cont0,
    const float *q;
    const int *links;
    int d;
    int maxDegree;
    const float *vecs;
    const int *degree;
    int ef;
    int pending;
    int active;
    int cand_cap;
    int mc;
    int mg;
    int k0;
    float *rd;
    int *ri;
    float *cd;
    int *ci;
    uint32_t *bloom;
    int *slot_ids;
    float *slot_dists;
    int *slot_cnt;
    int *meta;
    int *out;
    int slot_cap;
    int head;
    int tail;
    int s;
    int popped;
    int i0;
    int i1;
);


// One Bloom-fetch-compute job: distance between the query and vector ids[i].


/**************** Bloom filter (S2) ****************/

// MurmurHash2 of a 4-byte key


// Marks node as visited. Returns 1 if it was (definitely) not visited before.


/**************** Priority queues (S5, S6) ****************/

// Both queues are arrays sorted by ascending distance.
// Inserts (dist, id) keeping the closest cap elements. Returns the new size.


// Max(R.dist): the distance a candidate must not exceed to be evaluated


// Merges the distances of one evaluated group into R and C.
// meta: [0] = R size, [1] = C size, [2] = evaluated candidates, [3] = distance
// computations


/**************** Candidate group selection (S1, S2) ****************/

// Extract-Min(C, mc, threshold): pops up to mc qualified candidates and writes
// their not yet visited neighbors to ids. Returns the number of popped
// candidates; *cnt is set to the number of neighbors to evaluate.


/**************** DST search of one query ****************/

// Per-query buffers (all owned by this query):
//   rd, ri:          result queue R, ef entries
//   cd, ci:          candidate queue C, cand_cap entries
//   bloom:           BLOOM_WORDS words
//   slot_ids/dists:  mg pipeline slots of (mc * maxDegree) entries each
//   slot_cnt:        mg entries
//   meta:            4 entries (see merge_group)
//   out:             k result ids


/**************** Batch of queries (across-query parallelism) ****************/

// The per-query buffers are laid out back to back, one block per query.


/**************** Driver ****************/

#pragma BOMBYX IGNORE buildSynthetic
/***
 * Random vectors and a navigable kNN graph: the maxDegree/2 nearest neighbors
 * of each node plus reverse edges, topped up with the rest of its kNN list.
 * The entry node is the medoid. Also computes the exact top-10 ground truth.
 */
void buildSynthetic(int nq, unsigned seed, std::vector<float> &queries,
                    std::vector<int> &gt) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> N01(0.f, 1.f);
  vecs = new float[(size_t)n * dim];
  for (size_t i = 0; i < (size_t)n * dim; i++) vecs[i] = N01(rng);
  queries.resize((size_t)nq * dim);
  for (size_t i = 0; i < queries.size(); i++) queries[i] = N01(rng);

  const int K = maxDegree;
  std::vector<int> knn((size_t)n * K);
  std::vector<std::pair<float, int>> all(n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++)
      all[j] = {j == i ? FLT_MAX : l2sq(&vecs[(size_t)i * dim], &vecs[(size_t)j * dim], dim), j};
    std::partial_sort(all.begin(), all.begin() + K, all.end());
    for (int j = 0; j < K; j++) knn[(size_t)i * K + j] = all[j].second;
  }
  std::vector<std::vector<int>> adj(n);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < K / 2; j++) adj[i].push_back(knn[(size_t)i * K + j]);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < K / 2; j++) {
      int v = knn[(size_t)i * K + j];
      if ((int)adj[v].size() < K && std::find(adj[v].begin(), adj[v].end(), i) == adj[v].end())
        adj[v].push_back(i);
    }
  for (int i = 0; i < n; i++)
    for (int j = K / 2; j < K && (int)adj[i].size() < K; j++) {
      int v = knn[(size_t)i * K + j];
      if (std::find(adj[i].begin(), adj[i].end(), v) == adj[i].end()) adj[i].push_back(v);
    }
  degree = new int[n];
  links = new int[(size_t)n * K];
  std::fill_n(links, (size_t)n * K, 0);
  for (int i = 0; i < n; i++) {
    degree[i] = adj[i].size();
    std::copy(adj[i].begin(), adj[i].end(), links + (size_t)i * K);
  }

  std::vector<float> mean(dim, 0.f);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < dim; j++) mean[j] += vecs[(size_t)i * dim + j] / n;
  float best = FLT_MAX;
  for (int i = 0; i < n; i++) {
    float dd = l2sq(mean.data(), &vecs[(size_t)i * dim], dim);
    if (dd < best) { best = dd; entry = i; }
  }

  gt.resize((size_t)nq * 10);
  for (int qi = 0; qi < nq; qi++) {
    for (int j = 0; j < n; j++) all[j] = {l2sq(&queries[(size_t)qi * dim], &vecs[(size_t)j * dim], dim), j};
    std::partial_sort(all.begin(), all.begin() + 10, all.end());
    for (int j = 0; j < 10; j++) gt[(size_t)qi * 10 + j] = all[j].second;
  }
}

#pragma BOMBYX IGNORE main
int main(int argc, char *argv[]) {
  if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
    std::cout << "Usage: " << argv[0] << " [mc=1] [mg=1] [n=5000] [dim=32] [nq=200] [maxDegree=32] [ef=64]\n"
              << "  mc = candidates per group, mg = groups in the pipeline\n"
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
  int cand_cap = 64; // candidate queue size
  int k = 10;
  if (mc < 1 || mg < 1 || ef < k) {
    std::cerr << "need mc >= 1, mg >= 1, ef >= 10" << std::endl;
    return 1;
  }

  std::vector<float> queries;
  std::vector<int> gt;
  buildSynthetic(nq, 42, queries, gt);

  // per-query buffers
  size_t slot_size = (size_t)mg * mc * maxDegree;
  std::vector<float> rd((size_t)nq * ef), cd((size_t)nq * cand_cap), slot_dists(nq * slot_size);
  std::vector<int> ri((size_t)nq * ef), ci((size_t)nq * cand_cap), slot_ids(nq * slot_size);
  std::vector<int> slot_cnt((size_t)nq * mg), meta((size_t)nq * 4), out((size_t)nq * k);
  std::vector<uint32_t> bloom((size_t)nq * BLOOM_WORDS);

  auto t0 = std::chrono::steady_clock::now();
  search_batch(nq, queries.data(), vecs, degree, links, dim, maxDegree, entry, ef,
               cand_cap, mc, mg, k, rd.data(), ri.data(), cd.data(), ci.data(),
               bloom.data(), slot_ids.data(), slot_dists.data(), slot_cnt.data(),
               meta.data(), out.data());
  double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

  long hits = 0, cands = 0, dists = 0;
  for (int qi = 0; qi < nq; qi++) {
    for (int a = 0; a < 10; a++)
      for (int b = 0; b < k; b++)
        if (gt[(size_t)qi * 10 + a] == out[(size_t)qi * k + b]) hits++;
    cands += meta[(size_t)qi * 4 + 2];
    dists += meta[(size_t)qi * 4 + 3];
  }

  // deterministic part of the output (diff this against the Bombyx output)
  printf("n=%d dim=%d maxDegree=%d entry=%d nq=%d ef=%d mc=%d mg=%d\n", n, dim,
         maxDegree, entry, nq, ef, mc, mg);
  printf("R@10=%.4f avg evaluated candidates=%.1f avg distance computations=%.1f\n",
         hits / (nq * 10.0), (double)cands / nq, (double)dists / nq);
  for (int qi = 0; qi < std::min(nq, 3); qi++) {
    printf("query %d:", qi);
    for (int b = 0; b < k; b++) printf(" %d", out[(size_t)qi * k + b]);
    printf("\n");
  }
  fprintf(stderr, "time %.2f ms\n", ms);

  delete[] vecs;
  delete[] degree;
  delete[] links;
  return 0;
}

float l2sq(const float *a, const float *b, int d) {
    float acc;
    int i;
    float t;
    acc = 0;
    for (i = 0;(i < d);i = (i + 1)) {
        t = (a[i] - b[i]);
        acc = (acc + (t * t));
    }
    return acc;
}
THREAD(dist_task) {
    dist_task_closure *largs = (dist_task_closure*)(args.get());
    largs->dists[largs->i] = l2sq(largs->q,&(largs->vecs[(largs->ids[largs->i] * largs->d)]),largs->d);
    return;
}
uint32_t murmur2(uint32_t k0, uint32_t seed) {
    uint32_t m;
    uint32_t h;
    m = 1540483477;
    k0 = (k0 * m);
    k0 = (k0 ^ (k0 >> 24));
    k0 = (k0 * m);
    h = (seed ^ 4);
    h = (h * m);
    h = (h ^ k0);
    h = (h ^ (h >> 13));
    h = (h * m);
    h = (h ^ (h >> 15));
    return h;
}
int bloom_test_and_set(uint32_t *bloom, uint32_t node) {
    int seen;
    uint32_t h;
    uint32_t pos;
    uint32_t bit;
    seen = 1;
    for (h = 0;(h < 3);h = (h + 1)) {
        pos = (murmur2(node,(h + 1)) & (262144 - 1));
        bit = (1U << (pos & 31));
        if (((bloom[(pos >> 5)] & bit) == 0)) {
            seen = 0;
        }
        bloom[(pos >> 5)] = (bloom[(pos >> 5)] | bit);
    }
    return (1 - seen);
}
int sorted_insert(float *qd, int *qi, int size, int cap, float dist, int id) {
    int new_size;
    int j;
    new_size = size;
    if (((size < cap) || (dist < qd[(cap - 1)]))) {
        if ((size < cap)) {
            new_size = (size + 1);
        }
        j = (new_size - 1);
        while (((j > 0) && (qd[(j - 1)] > dist))) {
            qd[j] = qd[(j - 1)];
            qi[j] = qi[(j - 1)];
            j = (j - 1);
        }
        qd[j] = dist;
        qi[j] = id;
    }
    return new_size;
}
float threshold(const float *rd, int r_size, int ef) {
    float t;
    t = 3.40282347E+38F;
    if ((r_size == ef)) {
        t = rd[(ef - 1)];
    }
    return t;
}
void merge_group(float *rd, int *ri, float *cd, int *ci, int *meta, int ef, int cand_cap, const int *ids, const float *dists, int cnt) {
    int i;
    for (i = 0;(i < cnt);i = (i + 1)) {
        if ((dists[i] < threshold(rd,meta[0],ef))) {
            meta[0] = sorted_insert(rd,ri,meta[0],ef,dists[i],ids[i]);
            meta[1] = sorted_insert(cd,ci,meta[1],cand_cap,dists[i],ids[i]);
        }
    }
    meta[3] = (meta[3] + cnt);
}
int select_group(float *rd, float *cd, int *ci, int *meta, int ef, int mc, const int *degree, const int *links, int maxDegree, uint32_t *bloom, int *ids, int *cnt) {
    int popped;
    int c;
    float t;
    int node;
    int j;
    int j0;
    int nb;
    popped = 0;
    c = 0;
    t = threshold(rd,meta[0],ef);
    while ((((popped < mc) && (meta[1] > 0)) && (cd[0] <= t))) {
        node = ci[0];
        for (j = 1;(j < meta[1]);j = (j + 1)) {
            cd[(j - 1)] = cd[j];
            ci[(j - 1)] = ci[j];
        }
        meta[1] = (meta[1] - 1);
        popped = (popped + 1);
        for (j0 = 0;(j0 < degree[node]);j0 = (j0 + 1)) {
            nb = links[((node * maxDegree) + j0)];
            if ((bloom_test_and_set(bloom,nb) == 1)) {
                ids[c] = nb;
                c = (c + 1);
            }
        }
    }
    *(cnt) = c;
    meta[2] = (meta[2] + popped);
    return popped;
}
THREAD(dst_search) {
    int slot_cap;
    int i;
    float d0;
    int head;
    int tail;
    int pending;
    int active;
    int s;
    int popped;
    int i0;
    int i1;
    dst_search_closure *largs = (dst_search_closure*)(args.get());
    slot_cap = (largs->mc * largs->maxDegree);
    for (i = 0;(i < 8192);i = (i + 1)) {
        largs->bloom[i] = 0;
    }
    d0 = l2sq(largs->q,&(largs->vecs[(largs->entry * largs->d)]),largs->d);
    bloom_test_and_set(largs->bloom,largs->entry);
    largs->rd[0] = d0;
    largs->ri[0] = largs->entry;
    largs->cd[0] = d0;
    largs->ci[0] = largs->entry;
    largs->meta[0] = 1;
    largs->meta[1] = 1;
    largs->meta[2] = 0;
    largs->meta[3] = 1;
    head = 0;
    tail = 0;
    pending = 0;
    active = 1;
    auto sp0c = std::make_shared<dst_search_reentry0_closure>(largs->k);
    sp0c->q = largs->q;
    sp0c->vecs = largs->vecs;
    sp0c->degree = largs->degree;
    sp0c->links = largs->links;
    sp0c->d = largs->d;
    sp0c->maxDegree = largs->maxDegree;
    sp0c->ef = largs->ef;
    sp0c->cand_cap = largs->cand_cap;
    sp0c->mc = largs->mc;
    sp0c->mg = largs->mg;
    sp0c->k0 = largs->k0;
    sp0c->rd = largs->rd;
    sp0c->ri = largs->ri;
    sp0c->cd = largs->cd;
    sp0c->ci = largs->ci;
    sp0c->bloom = largs->bloom;
    sp0c->slot_ids = largs->slot_ids;
    sp0c->slot_dists = largs->slot_dists;
    sp0c->slot_cnt = largs->slot_cnt;
    sp0c->meta = largs->meta;
    sp0c->out = largs->out;
    sp0c->slot_cap = slot_cap;
    sp0c->head = head;
    sp0c->tail = tail;
    sp0c->pending = pending;
    sp0c->active = active;
    sp0c->s = s;
    sp0c->popped = popped;
    sp0c->i0 = i0;
    sp0c->i1 = i1;
    cilk_spawn taskSpawn(sp0c->getTask(), sp0c);
    return;
}
void search_batch(int nq, const float *queries, const float *vecs, const int *degree, const int *links, int d, int maxDegree, int entry, int ef, int cand_cap, int mc, int mg, int k0, float *rd, int *ri, float *cd, int *ci, uint32_t *bloom, int *slot_ids, float *slot_dists, int *slot_cnt, int *meta, int *out) {
    int slot_size;
    int qi;
    slot_size = ((mg * mc) * maxDegree);
    search_batch_cont0_closure SN_search_batch_cont0c(CONT_DUMMY);
    spawn_next<search_batch_cont0_closure> SN_search_batch_cont0(SN_search_batch_cont0c);
    for (qi = 0;(qi < nq);qi = (qi + 1)) {
        cont sp0k;
        SN_BIND_VOID(SN_search_batch_cont0, &sp0k);
        dst_search_closure sp0c(sp0k);
        sp0c.q = &(queries[(qi * d)]);
        sp0c.vecs = vecs;
        sp0c.degree = degree;
        sp0c.links = links;
        sp0c.d = d;
        sp0c.maxDegree = maxDegree;
        sp0c.entry = entry;
        sp0c.ef = ef;
        sp0c.cand_cap = cand_cap;
        sp0c.mc = mc;
        sp0c.mg = mg;
        sp0c.k0 = k0;
        sp0c.rd = &(rd[(qi * ef)]);
        sp0c.ri = &(ri[(qi * ef)]);
        sp0c.cd = &(cd[(qi * cand_cap)]);
        sp0c.ci = &(ci[(qi * cand_cap)]);
        sp0c.bloom = &(bloom[(qi * 8192)]);
        sp0c.slot_ids = &(slot_ids[(qi * slot_size)]);
        sp0c.slot_dists = &(slot_dists[(qi * slot_size)]);
        sp0c.slot_cnt = &(slot_cnt[(qi * mg)]);
        sp0c.meta = &(meta[(qi * 4)]);
        sp0c.out = &(out[(qi * k0)]);
        spawn<dst_search_closure> sp0(sp0c);

    }
    // Original sync was here
}
THREAD(dst_search_exit0) {
    dst_search_exit0_closure *largs = (dst_search_exit0_closure*)(args.get());
    for (largs->i1 = 0;(largs->i1 < largs->k0);largs->i1 = (largs->i1 + 1)) {
        largs->out[largs->i1] = (-1);
        if ((largs->i1 < largs->meta[0])) {
            largs->out[largs->i1] = largs->ri[largs->i1];
        }
    }
    return;
}
THREAD(dst_search_reentry0) {
    dst_search_reentry0_closure *largs = (dst_search_reentry0_closure*)(args.get());
    if ((largs->active == 1)) {
        largs->s = largs->head;
        largs->popped = select_group(largs->rd,largs->cd,largs->ci,largs->meta,largs->ef,largs->mc,largs->degree,largs->links,largs->maxDegree,largs->bloom,&(largs->slot_ids[(largs->s * largs->slot_cap)]),&(largs->slot_cnt[largs->s]));
        dst_search_reentry0_cont0_closure SN_dst_search_reentry0_cont0c(largs->k);
        spawn_next<dst_search_reentry0_cont0_closure> SN_dst_search_reentry0_cont0(SN_dst_search_reentry0_cont0c);
        if ((largs->popped > 0)) {
            largs->pending = (largs->pending + 1);
            largs->head = ((largs->head + 1) % largs->mg);
        }
        for (largs->i0 = 0;(largs->i0 < largs->slot_cnt[largs->s]);largs->i0 = (largs->i0 + 1)) {
            cont sp0k;
            SN_BIND_VOID(SN_dst_search_reentry0_cont0, &sp0k);
            dist_task_closure sp0c(sp0k);
            sp0c.q = largs->q;
            sp0c.vecs = largs->vecs;
            sp0c.d = largs->d;
            sp0c.ids = &(largs->slot_ids[(largs->s * largs->slot_cap)]);
            sp0c.dists = &(largs->slot_dists[(largs->s * largs->slot_cap)]);
            sp0c.i = largs->i0;
            spawn<dist_task_closure> sp0(sp0c);

        }
        ((dst_search_reentry0_cont0_closure*)SN_dst_search_reentry0_cont0.cls.get())->popped = largs->popped;
        ((dst_search_reentry0_cont0_closure*)SN_dst_search_reentry0_cont0.cls.get())->s = largs->s;
        ((dst_search_reentry0_cont0_closure*)SN_dst_search_reentry0_cont0.cls.get())->tail = largs->tail;
        ((dst_search_reentry0_cont0_closure*)SN_dst_search_reentry0_cont0.cls.get())->slot_cap = largs->slot_cap;
        ((dst_search_reentry0_cont0_closure*)SN_dst_search_reentry0_cont0.cls.get())->meta = largs->meta;
        ((dst_search_reentry0_cont0_closure*)SN_dst_search_reentry0_cont0.cls.get())->out = largs->out;
        ((dst_search_reentry0_cont0_closure*)SN_dst_search_reentry0_cont0.cls.get())->cd = largs->cd;
        ((dst_search_reentry0_cont0_closure*)SN_dst_search_reentry0_cont0.cls.get())->slot_cnt = largs->slot_cnt;
        ((dst_search_reentry0_cont0_closure*)SN_dst_search_reentry0_cont0.cls.get())->slot_dists = largs->slot_dists;
        ((dst_search_reentry0_cont0_closure*)SN_dst_search_reentry0_cont0.cls.get())->mc = largs->mc;
        ((dst_search_reentry0_cont0_closure*)SN_dst_search_reentry0_cont0.cls.get())->bloom = largs->bloom;
        ((dst_search_reentry0_cont0_closure*)SN_dst_search_reentry0_cont0.cls.get())->slot_ids = largs->slot_ids;
        ((dst_search_reentry0_cont0_closure*)SN_dst_search_reentry0_cont0.cls.get())->ci = largs->ci;
        ((dst_search_reentry0_cont0_closure*)SN_dst_search_reentry0_cont0.cls.get())->mg = largs->mg;
        ((dst_search_reentry0_cont0_closure*)SN_dst_search_reentry0_cont0.cls.get())->links = largs->links;
        ((dst_search_reentry0_cont0_closure*)SN_dst_search_reentry0_cont0.cls.get())->q = largs->q;
        ((dst_search_reentry0_cont0_closure*)SN_dst_search_reentry0_cont0.cls.get())->maxDegree = largs->maxDegree;
        ((dst_search_reentry0_cont0_closure*)SN_dst_search_reentry0_cont0.cls.get())->k0 = largs->k0;
        ((dst_search_reentry0_cont0_closure*)SN_dst_search_reentry0_cont0.cls.get())->i0 = largs->i0;
        ((dst_search_reentry0_cont0_closure*)SN_dst_search_reentry0_cont0.cls.get())->cand_cap = largs->cand_cap;
        ((dst_search_reentry0_cont0_closure*)SN_dst_search_reentry0_cont0.cls.get())->rd = largs->rd;
        ((dst_search_reentry0_cont0_closure*)SN_dst_search_reentry0_cont0.cls.get())->ri = largs->ri;
        ((dst_search_reentry0_cont0_closure*)SN_dst_search_reentry0_cont0.cls.get())->vecs = largs->vecs;
        ((dst_search_reentry0_cont0_closure*)SN_dst_search_reentry0_cont0.cls.get())->active = largs->active;
        ((dst_search_reentry0_cont0_closure*)SN_dst_search_reentry0_cont0.cls.get())->head = largs->head;
        ((dst_search_reentry0_cont0_closure*)SN_dst_search_reentry0_cont0.cls.get())->pending = largs->pending;
        ((dst_search_reentry0_cont0_closure*)SN_dst_search_reentry0_cont0.cls.get())->i1 = largs->i1;
        ((dst_search_reentry0_cont0_closure*)SN_dst_search_reentry0_cont0.cls.get())->degree = largs->degree;
        ((dst_search_reentry0_cont0_closure*)SN_dst_search_reentry0_cont0.cls.get())->d = largs->d;
        ((dst_search_reentry0_cont0_closure*)SN_dst_search_reentry0_cont0.cls.get())->ef = largs->ef;
        // Original sync was here
    } else {
        auto sp1c = std::make_shared<dst_search_exit0_closure>(largs->k);
        sp1c->k0 = largs->k0;
        sp1c->ri = largs->ri;
        sp1c->meta = largs->meta;
        sp1c->out = largs->out;
        sp1c->i1 = largs->i1;
        cilk_spawn taskSpawn(sp1c->getTask(), sp1c);
        return;
    }
}
THREAD(search_batch_cont0) {
    search_batch_cont0_closure *largs = (search_batch_cont0_closure*)(args.get());
    return;
}
THREAD(dst_search_reentry0_cont0) {
    dst_search_reentry0_cont0_closure *largs = (dst_search_reentry0_cont0_closure*)(args.get());
    if (((largs->pending > 0) && ((largs->pending == largs->mg) || (largs->popped == 0)))) {
        merge_group(largs->rd,largs->ri,largs->cd,largs->ci,largs->meta,largs->ef,largs->cand_cap,&(largs->slot_ids[(largs->tail * largs->slot_cap)]),&(largs->slot_dists[(largs->tail * largs->slot_cap)]),largs->slot_cnt[largs->tail]);
        largs->tail = ((largs->tail + 1) % largs->mg);
        largs->pending = (largs->pending - 1);
    }
    if (((largs->pending == 0) && ((largs->meta[1] == 0) || (largs->cd[0] > threshold(largs->rd,largs->meta[0],largs->ef))))) {
        largs->active = 0;
    }
    auto sp0c = std::make_shared<dst_search_reentry0_closure>(largs->k);
    sp0c->q = largs->q;
    sp0c->vecs = largs->vecs;
    sp0c->degree = largs->degree;
    sp0c->links = largs->links;
    sp0c->d = largs->d;
    sp0c->maxDegree = largs->maxDegree;
    sp0c->ef = largs->ef;
    sp0c->cand_cap = largs->cand_cap;
    sp0c->mc = largs->mc;
    sp0c->mg = largs->mg;
    sp0c->k0 = largs->k0;
    sp0c->rd = largs->rd;
    sp0c->ri = largs->ri;
    sp0c->cd = largs->cd;
    sp0c->ci = largs->ci;
    sp0c->bloom = largs->bloom;
    sp0c->slot_ids = largs->slot_ids;
    sp0c->slot_dists = largs->slot_dists;
    sp0c->slot_cnt = largs->slot_cnt;
    sp0c->meta = largs->meta;
    sp0c->out = largs->out;
    sp0c->slot_cap = largs->slot_cap;
    sp0c->head = largs->head;
    sp0c->tail = largs->tail;
    sp0c->pending = largs->pending;
    sp0c->active = largs->active;
    sp0c->s = largs->s;
    sp0c->popped = largs->popped;
    sp0c->i0 = largs->i0;
    sp0c->i1 = largs->i1;
    cilk_spawn taskSpawn(sp0c->getTask(), sp0c);
    return;
}
