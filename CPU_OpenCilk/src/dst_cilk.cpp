// Delayed-Synchronization Traversal (DST) for graph vector search on CPUs, using OpenCilk.
//
// CPU port of the Falcon/DST algorithm (Jiang et al., VLDB'25, Algorithm 2). It reads the same
// graph files as the FPGA kernels (the 1-channel export of hnsw_to_FPGA.py / nsg_to_FPGA.py).
//
// How the accelerator maps onto Cilk:
//   * Candidate groups in the pipeline (mg) -> "lanes". When a lane finishes evaluating its group it
//     takes the query lock, merges its distances into the result/candidate queues (delayed sync of
//     just that group), refills the pipeline up to mg groups, cilk_spawns the extra groups and keeps
//     evaluating the first one itself. There is no busy waiting: whoever completes the last in-flight
//     group with nothing left to launch ends the query.
//   * Bloom-fetch-compute units -> the distance computations of a group run in a cilk_for
//     (intra-query mode). In inter-query mode the cilk_for is over the queries of a batch instead.
//   * On-chip Bloom filter -> a register-blocked Bloom filter (all hash bits of a key live in one
//     64-bit word), so one atomic fetch_or is an exact test-and-set across concurrent lanes.
//
// mg = 1, mc = 1 is classic best-first search (Algorithm 1); mg = 1, mc > 1 is MCS.

#if defined(__cilk)
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#define HAVE_CILK 1
#else
// Serial elision, so the code also builds (and runs single-threaded) with a plain C++ compiler.
#define cilk_spawn
#define cilk_sync
#define cilk_for for
#define HAVE_CILK 0
#endif

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <string>
#include <vector>

static const float kInf = std::numeric_limits<float>::infinity();
static const int kMaxMC = 16; // max candidates per group
static const int kMaxMG = 64; // max groups in the pipeline

// ---------------------------------------------------------------------------------------------
// Index
// ---------------------------------------------------------------------------------------------

struct Index {
	int n = 0;          // number of database vectors
	int d = 0;          // dimensionality
	int d_pad = 0;      // row stride in floats (multiple of 16 -> rows are 64-byte aligned)
	int max_degree = 0; // row stride of the adjacency array
	int entry = 0;      // entry node (HNSW: top-level entry point, used on the base layer)
	float* vecs = nullptr;  // n * d_pad
	std::vector<int> degree;    // n
	std::vector<int> links;     // n * max_degree
	std::vector<int> labels;    // internal id -> label (HNSW); empty means identity

	const float* vec(int i) const { return vecs + (size_t)i * d_pad; }
	const int* nbrs(int i) const { return links.data() + (size_t)i * max_degree; }
	int label(int i) const { return labels.empty() ? i : labels[i]; }
	~Index() { free(vecs); }
};

static float* alloc_floats(size_t count) {
	size_t bytes = (count * sizeof(float) + 63) / 64 * 64;
	void* p = aligned_alloc(64, bytes);
	if (!p) { fprintf(stderr, "out of memory (%zu bytes)\n", bytes); exit(1); }
	memset(p, 0, bytes);
	return (float*)p;
}

static FILE* open_or_die(const std::string& path) {
	FILE* f = fopen(path.c_str(), "rb");
	if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
	return f;
}

static long file_size(const std::string& path) {
	FILE* f = open_or_die(path);
	fseek(f, 0, SEEK_END);
	long s = ftell(f);
	fclose(f);
	return s;
}

static std::string join(const std::string& dir, const std::string& name) {
	return (!dir.empty() && dir.back() == '/') ? dir + name : dir + "/" + name;
}

// Loads the 1-channel FPGA export:
//   meta.bin                   HNSW: n, max_level, entry, maxM, maxM0    NSG: n, entry, width
//   ground_links_1_chan_0.bin  per node: [64 B header: int num_links] [max_degree ints, padded to 64 B]
//   ground_vectors_1_chan_0.bin per node: [d floats padded to 64 B] [64 B visited tag (unused)]
//   ground_labels.bin          HNSW only: int label per node
static void load_falcon_index(Index& idx, const std::string& dir, const std::string& graph, int d) {
	FILE* f = open_or_die(join(dir, "meta.bin"));
	int meta[5] = {0};
	if (graph == "HNSW") {
		if (fread(meta, sizeof(int), 5, f) != 5) { fprintf(stderr, "bad meta.bin\n"); exit(1); }
		idx.n = meta[0]; idx.entry = meta[2]; idx.max_degree = meta[4];
	} else {
		if (fread(meta, sizeof(int), 3, f) != 3) { fprintf(stderr, "bad meta.bin\n"); exit(1); }
		idx.n = meta[0]; idx.entry = meta[1]; idx.max_degree = meta[2];
	}
	fclose(f);
	idx.d = d;
	idx.d_pad = (d + 15) / 16 * 16;
	printf("index: n=%d d=%d entry=%d max_degree=%d\n", idx.n, idx.d, idx.entry, idx.max_degree);

	const size_t n = idx.n;
	const size_t link_row_ints = 16 + (idx.max_degree + 15) / 16 * 16;
	const size_t vec_row_floats = idx.d_pad + 16;

	std::string links_path = join(dir, "ground_links_1_chan_0.bin");
	std::string vecs_path = join(dir, "ground_vectors_1_chan_0.bin");
	if ((size_t)file_size(links_path) != n * link_row_ints * 4 ||
		(size_t)file_size(vecs_path) != n * vec_row_floats * 4) {
		fprintf(stderr, "file sizes do not match meta.bin / dimensionality %d\n", d);
		exit(1);
	}

	// stream the files in chunks and repack them densely (drops the FPGA paddings)
	const size_t chunk = 1 << 16;
	idx.degree.resize(n);
	idx.links.resize(n * idx.max_degree);
	std::vector<int> ibuf(chunk * link_row_ints);
	f = open_or_die(links_path);
	for (size_t s = 0; s < n; s += chunk) {
		size_t m = std::min(chunk, n - s);
		if (fread(ibuf.data(), 4, m * link_row_ints, f) != m * link_row_ints) { fprintf(stderr, "short read\n"); exit(1); }
		for (size_t i = 0; i < m; i++) {
			const int* row = ibuf.data() + i * link_row_ints;
			idx.degree[s + i] = std::min(row[0], idx.max_degree);
			memcpy(&idx.links[(s + i) * idx.max_degree], row + 16, idx.max_degree * sizeof(int));
		}
	}
	fclose(f);

	idx.vecs = alloc_floats(n * idx.d_pad);
	std::vector<float> fbuf(chunk * vec_row_floats);
	f = open_or_die(vecs_path);
	for (size_t s = 0; s < n; s += chunk) {
		size_t m = std::min(chunk, n - s);
		if (fread(fbuf.data(), 4, m * vec_row_floats, f) != m * vec_row_floats) { fprintf(stderr, "short read\n"); exit(1); }
		for (size_t i = 0; i < m; i++)
			memcpy(idx.vecs + (s + i) * idx.d_pad, fbuf.data() + i * vec_row_floats, idx.d_pad * sizeof(float));
	}
	fclose(f);

	if (graph == "HNSW") {
		idx.labels.resize(n);
		f = open_or_die(join(dir, "ground_labels.bin"));
		if (fread(idx.labels.data(), 4, n, f) != n) { fprintf(stderr, "short read labels\n"); exit(1); }
		fclose(f);
	}
}

// ---------------------------------------------------------------------------------------------
// Queries and ground truth (same file formats as the FPGA host programs)
// ---------------------------------------------------------------------------------------------

struct Queries {
	int nq = 0, d_pad = 0;
	float* data = nullptr; // nq * d_pad
	const float* q(int i) const { return data + (size_t)i * d_pad; }
	~Queries() { free(data); }
};

// formats: bvecs (SIFT, uint8), fbin (Deep/GLOVE, 8 B header), i8bin (SPACEV, 8 B header), f32raw (SBERT)
static void load_queries(Queries& Q, const std::string& path, const std::string& fmt, int d, int nq) {
	FILE* f = open_or_die(path);
	Q.nq = nq;
	Q.d_pad = (d + 15) / 16 * 16;
	Q.data = alloc_floats((size_t)nq * Q.d_pad);
	if (fmt == "bvecs") {
		std::vector<uint8_t> row(4 + d);
		for (int i = 0; i < nq; i++) {
			if (fread(row.data(), 1, row.size(), f) != row.size()) { fprintf(stderr, "short read queries\n"); exit(1); }
			for (int j = 0; j < d; j++) Q.data[(size_t)i * Q.d_pad + j] = (float)row[4 + j];
		}
	} else if (fmt == "fbin" || fmt == "i8bin" || fmt == "f32raw") {
		if (fmt != "f32raw") fseek(f, 8, SEEK_SET);
		size_t esz = fmt == "i8bin" ? 1 : 4;
		std::vector<char> row(esz * d);
		for (int i = 0; i < nq; i++) {
			if (fread(row.data(), 1, row.size(), f) != row.size()) { fprintf(stderr, "short read queries\n"); exit(1); }
			for (int j = 0; j < d; j++) {
				float v;
				if (esz == 1) v = (float)(int8_t)row[j];
				else memcpy(&v, &row[4 * j], 4);
				Q.data[(size_t)i * Q.d_pad + j] = v;
			}
		}
	} else {
		fprintf(stderr, "unknown query format %s\n", fmt.c_str());
		exit(1);
	}
	fclose(f);
}

// formats: ivecs (SIFT: per row int k + k ints), ibin (8 B header n, k + n*k ints). Keeps top-10.
static std::vector<int> load_gt(const std::string& path, const std::string& fmt, int nq, int keep) {
	FILE* f = open_or_die(path);
	std::vector<int> gt((size_t)nq * keep);
	int k = 0;
	if (fmt == "ibin") {
		int hdr[2];
		if (fread(hdr, 4, 2, f) != 2) { fprintf(stderr, "bad gt\n"); exit(1); }
		k = hdr[1];
	}
	std::vector<int> row;
	for (int i = 0; i < nq; i++) {
		if (fmt == "ivecs" && fread(&k, 4, 1, f) != 1) { fprintf(stderr, "short read gt\n"); exit(1); }
		row.resize(k);
		if (fread(row.data(), 4, k, f) != (size_t)k) { fprintf(stderr, "short read gt\n"); exit(1); }
		for (int j = 0; j < keep; j++) gt[(size_t)i * keep + j] = row[j];
	}
	fclose(f);
	return gt;
}

// ---------------------------------------------------------------------------------------------
// Synthetic data (for testing without the datasets): Gaussian vectors + a symmetrized kNN graph
// ---------------------------------------------------------------------------------------------

static inline float l2sq(const float* __restrict a, const float* __restrict b, int d_pad);

static void build_synthetic(Index& idx, Queries& Q, std::vector<int>& gt, int n, int d, int nq,
							int max_degree, uint64_t seed) {
	idx.n = n; idx.d = d; idx.d_pad = (d + 15) / 16 * 16; idx.max_degree = max_degree;
	idx.vecs = alloc_floats((size_t)n * idx.d_pad);
	Q.nq = nq; Q.d_pad = idx.d_pad;
	Q.data = alloc_floats((size_t)nq * Q.d_pad);
	std::mt19937_64 rng(seed);
	std::normal_distribution<float> N01(0.f, 1.f);
	for (int i = 0; i < n; i++) for (int j = 0; j < d; j++) idx.vecs[(size_t)i * idx.d_pad + j] = N01(rng);
	for (int i = 0; i < nq; i++) for (int j = 0; j < d; j++) Q.data[(size_t)i * Q.d_pad + j] = N01(rng);

	// exact kNN lists (K = max_degree)
	const int K = max_degree;
	std::vector<int> knn((size_t)n * K);
	cilk_for (int i = 0; i < n; i++) {
		std::vector<std::pair<float, int>> all(n);
		for (int j = 0; j < n; j++) all[j] = {j == i ? kInf : l2sq(idx.vec(i), idx.vec(j), idx.d_pad), j};
		std::partial_sort(all.begin(), all.begin() + K, all.end());
		for (int j = 0; j < K; j++) knn[(size_t)i * K + j] = all[j].second;
	}
	// graph: K/2 nearest + reverse edges (makes the graph navigable), topped up with the rest of kNN
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
	idx.degree.resize(n);
	idx.links.assign((size_t)n * K, 0);
	for (int i = 0; i < n; i++) {
		idx.degree[i] = adj[i].size();
		std::copy(adj[i].begin(), adj[i].end(), idx.links.begin() + (size_t)i * K);
	}
	// entry = medoid (node closest to the mean)
	std::vector<float> mean(idx.d_pad, 0.f);
	for (int i = 0; i < n; i++) for (int j = 0; j < d; j++) mean[j] += idx.vecs[(size_t)i * idx.d_pad + j] / n;
	float best = kInf;
	for (int i = 0; i < n; i++) {
		float dist = l2sq(mean.data(), idx.vec(i), idx.d_pad);
		if (dist < best) { best = dist; idx.entry = i; }
	}
	// ground truth top-10
	gt.assign((size_t)nq * 10, 0);
	cilk_for (int qi = 0; qi < nq; qi++) {
		std::vector<std::pair<float, int>> all(n);
		for (int j = 0; j < n; j++) all[j] = {l2sq(Q.q(qi), idx.vec(j), idx.d_pad), j};
		std::partial_sort(all.begin(), all.begin() + 10, all.end());
		for (int j = 0; j < 10; j++) gt[(size_t)qi * 10 + j] = all[j].second;
	}
	printf("synthetic: n=%d d=%d nq=%d max_degree=%d entry=%d\n", n, d, nq, max_degree, idx.entry);
}

// ---------------------------------------------------------------------------------------------
// Search building blocks
// ---------------------------------------------------------------------------------------------

// squared L2 over the padded row (paddings are zero in both operands)
static inline float l2sq(const float* __restrict a, const float* __restrict b, int d_pad) {
	float acc = 0.f;
#pragma clang loop vectorize(enable) interleave(enable)
	for (int i = 0; i < d_pad; i++) {
		float t = a[i] - b[i];
		acc += t * t;
	}
	return acc;
}

static inline uint32_t murmur2_key4(uint32_t k, uint32_t seed) {
	const uint32_t m = 0x5bd1e995;
	k *= m; k ^= k >> 24; k *= m;
	uint32_t h = seed ^ 4u;
	h *= m; h ^= k;
	h ^= h >> 13; h *= m; h ^= h >> 15;
	return h;
}

// Register-blocked Bloom filter: num_hash bits, all inside one 64-bit word chosen by the key.
// test_and_set is a single atomic fetch_or, so concurrent lanes never both claim the same node.
class VisitedBloom {
public:
	VisitedBloom(int log2_bits, int num_hash) : num_hash_(num_hash) {
		int log2_words = std::max(0, log2_bits - 6);
		mask_ = (1u << log2_words) - 1;
		words_.reset(new std::atomic<uint64_t>[(size_t)mask_ + 1]);
		for (size_t i = 0; i <= mask_; i++) words_[i].store(0, std::memory_order_relaxed);
	}
	// returns true if the node was (probably) not visited before, and marks it visited
	bool test_and_set(uint32_t node) {
		uint32_t h1 = murmur2_key4(node, 1);
		uint32_t h2 = murmur2_key4(node, 2);
		uint64_t bits = 0;
		for (int i = 0; i < num_hash_; i++) bits |= 1ull << ((h2 >> (6 * i)) & 63);
		uint64_t old = words_[h1 & mask_].fetch_or(bits, std::memory_order_relaxed);
		return (old & bits) != bits;
	}
private:
	int num_hash_;
	uint32_t mask_;
	std::unique_ptr<std::atomic<uint64_t>[]> words_;
};

struct Neighbor {
	float dist;
	int id;
};

// result queue R: ascending by distance, keeps the closest `cap`
struct ResultQueue {
	std::vector<Neighbor> v;
	int cap;
	explicit ResultQueue(int c) : cap(c) { v.reserve(c + 1); }
	float threshold() const { return (int)v.size() < cap ? kInf : v.back().dist; }
	bool insert(Neighbor x) {
		if (x.dist >= threshold()) return false;
		auto it = std::upper_bound(v.begin(), v.end(), x.dist, [](float d, const Neighbor& e) { return d < e.dist; });
		v.insert(it, x);
		if ((int)v.size() > cap) v.pop_back();
		return true;
	}
};

// candidate queue C: stored descending, so the best candidate is at the back
struct CandidateQueue {
	std::vector<Neighbor> v;
	int cap;
	explicit CandidateQueue(int c) : cap(c) { v.reserve(c + 1); }
	void insert(Neighbor x) {
		if ((int)v.size() == cap && x.dist >= v.front().dist) return;
		auto it = std::upper_bound(v.begin(), v.end(), x.dist, [](float d, const Neighbor& e) { return d > e.dist; });
		v.insert(it, x);
		if ((int)v.size() > cap) v.erase(v.begin());
	}
	bool top_qualifies(float threshold) const { return !v.empty() && v.back().dist <= threshold; }
	int pop() { int id = v.back().id; v.pop_back(); return id; }
};

struct Params {
	int ef = 64;          // result queue size l
	int cand_queue = 32;  // candidate queue size (Falcon hardware uses 32)
	int mc = 1;           // candidates per group
	int mg = 1;           // groups in the pipeline
	int k = 10;
	int bloom_log2_bits = 18; // 256 Kbit, same as Falcon with one BFC unit
	int bloom_hash = 3;
	int grain = 8;        // distance computations per cilk_for leaf (intra-query mode)
	bool intra = true;    // parallelize the distance computations of a group
};

struct Group {
	int n = 0;
	int ids[kMaxMC];
};

struct QueryStats {
	long hops = 0;      // candidates evaluated (incl. entry)
	long dist_comps = 0;
};

// Per-query search state. Everything except the Bloom filter is guarded by `lock`.
struct QueryCtx {
	const Index& idx;
	const float* q;
	const Params& p;
	VisitedBloom visited;
	std::mutex lock;
	ResultQueue R;
	CandidateQueue C;
	int in_flight = 0;
	QueryStats stats;

	QueryCtx(const Index& i, const float* qv, const Params& pp)
		: idx(i), q(qv), p(pp), visited(pp.bloom_log2_bits, pp.bloom_hash), R(pp.ef), C(pp.cand_queue) {}
};

// S1-S4: fetch neighbor ids, Bloom filter, fetch vectors, compute distances
static void evaluate_group(QueryCtx& S, const Group& g, std::vector<Neighbor>& out) {
	const Index& idx = S.idx;
	out.clear();
	for (int c = 0; c < g.n; c++) {
		const int* nb = idx.nbrs(g.ids[c]);
		int deg = idx.degree[g.ids[c]];
		for (int j = 0; j < deg; j++) {
			if (S.visited.test_and_set((uint32_t)nb[j])) {
				out.push_back({0.f, nb[j]});
				__builtin_prefetch(idx.vec(nb[j]));
			}
		}
	}
	const int m = out.size();
	if (S.p.intra && m > S.p.grain) {
		const int grain = S.p.grain;
		const int chunks = (m + grain - 1) / grain;
		cilk_for (int ch = 0; ch < chunks; ch++) {
			int end = std::min(m, (ch + 1) * grain);
			for (int i = ch * grain; i < end; i++) out[i].dist = l2sq(S.q, idx.vec(out[i].id), idx.d_pad);
		}
	} else {
		for (int i = 0; i < m; i++) out[i].dist = l2sq(S.q, idx.vec(out[i].id), idx.d_pad);
	}
}

// One lane of the DST pipeline (Algorithm 2). Evaluates `g`; on completion merges the results
// (S5/S6 + delayed sync of this group only) and refills the pipeline up to mg groups.
static void run_lane(QueryCtx& S, Group g) {
	std::vector<Neighbor> buf;
	buf.reserve((size_t)S.p.mc * S.idx.max_degree);
	Group next[kMaxMG];
	while (true) {
		evaluate_group(S, g, buf);

		int n_next = 0;
		{
			std::lock_guard<std::mutex> guard(S.lock);
			S.stats.dist_comps += buf.size();
			for (const Neighbor& x : buf)
				if (S.R.insert(x)) S.C.insert(x); // FPGA: only results that enter R become candidates
			S.in_flight--;
			// fill the pipeline
			while (S.in_flight < S.p.mg) {
				float threshold = S.R.threshold();
				Group& ng = next[n_next];
				ng.n = 0;
				while (ng.n < S.p.mc && S.C.top_qualifies(threshold)) ng.ids[ng.n++] = S.C.pop();
				if (ng.n == 0) break;
				S.stats.hops += ng.n;
				S.in_flight++;
				n_next++;
			}
		}
		if (n_next == 0) return; // other lanes (if any) still own in-flight groups
		for (int i = 1; i < n_next; i++) cilk_spawn run_lane(S, next[i]);
		g = next[0];
	}
	// implicit cilk_sync on return
}

// Returns the top-k labels for one query.
static void search_one(const Index& idx, const float* q, const Params& p, int* out_labels, QueryStats* st) {
	QueryCtx S(idx, q, p);
	S.visited.test_and_set((uint32_t)idx.entry);
	S.R.insert({l2sq(q, idx.vec(idx.entry), idx.d_pad), idx.entry});
	S.stats.dist_comps = 1;
	S.stats.hops = 1;
	S.in_flight = 1;
	Group g0;
	g0.n = 1;
	g0.ids[0] = idx.entry;
	run_lane(S, g0);
	// run_lane returns only after all lanes it spawned (transitively) have returned
	for (int i = 0; i < p.k; i++) out_labels[i] = i < (int)S.R.v.size() ? idx.label(S.R.v[i].id) : -1;
	*st = S.stats;
}

// ---------------------------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------------------------

struct RunResult {
	double total_s = 0, qps = 0, recall1 = 0, recall10 = 0, avg_hops = 0, avg_dist = 0;
	double lat_median_ms = 0, lat_p95_ms = 0; // per batch
};

static double now_s() {
	return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static RunResult run(const Index& idx, const Queries& Q, const std::vector<int>& gt, int gt_k,
					 const Params& p, int batch) {
	const int nq = Q.nq;
	std::vector<int> out((size_t)nq * p.k);
	std::vector<QueryStats> st(nq);
	std::vector<double> lat;
	double t_all = now_s();
	for (int b = 0; b < nq; b += batch) {
		int e = std::min(nq, b + batch);
		double t0 = now_s();
		if (p.intra) {
			for (int i = b; i < e; i++) search_one(idx, Q.q(i), p, &out[(size_t)i * p.k], &st[i]);
		} else {
			cilk_for (int i = b; i < e; i++) search_one(idx, Q.q(i), p, &out[(size_t)i * p.k], &st[i]);
		}
		lat.push_back((now_s() - t0) * 1e3);
	}
	RunResult r;
	r.total_s = now_s() - t_all;
	r.qps = nq / r.total_s;
	std::sort(lat.begin(), lat.end());
	r.lat_median_ms = lat[lat.size() / 2];
	r.lat_p95_ms = lat[std::min(lat.size() - 1, (size_t)(lat.size() * 0.95))];
	long c1 = 0, c10 = 0, hops = 0, dists = 0;
	for (int i = 0; i < nq; i++) {
		const int* o = &out[(size_t)i * p.k];
		const int* g = &gt[(size_t)i * gt_k];
		if (o[0] == g[0]) c1++;
		for (int a = 0; a < 10 && a < p.k; a++)
			for (int b2 = 0; b2 < 10 && b2 < p.k; b2++)
				if (g[a] == o[b2]) { c10++; break; }
		hops += st[i].hops;
		dists += st[i].dist_comps;
	}
	r.recall1 = (double)c1 / nq;
	r.recall10 = (double)c10 / (nq * 10.0);
	r.avg_hops = (double)hops / nq;
	r.avg_dist = (double)dists / nq;
	return r;
}

static void usage() {
	printf(
		"Usage:\n"
		"  dst_cilk --index-dir DIR --graph HNSW|NSG --dataset SIFT|Deep|SPACEV|SBERT|GLOVE\n"
		"           --query FILE --gt FILE [options]\n"
		"  dst_cilk --synthetic N [--dim 32] [--nq 1000] [--md 32] [options]\n"
		"Options:\n"
		"  --mc N          candidates per group (default 1)\n"
		"  --mg N          groups in the pipeline (default 1); mc=mg=1 is best-first search\n"
		"  --ef N          result queue size (default 64)\n"
		"  --cand-queue N  candidate queue size (default 32)\n"
		"  --mode intra|inter  intra: parallel within a query, inter: cilk_for over a batch\n"
		"  --batch N       queries per batch (default 1)\n"
		"  --nq N          number of queries (default 10000, or 1000 for synthetic)\n"
		"  --bloom-bits N  log2 of the Bloom filter size in bits (default 18)\n"
		"  --grain N       distance computations per cilk_for leaf (default 8)\n"
		"  --sweep         run mc=1..4 x mg=1..7 and print a table (like Fig. 7)\n"
		"  --query-format bvecs|fbin|i8bin|f32raw, --gt-format ivecs|ibin, --dim D\n"
		"                  override what --dataset implies\n"
		"Workers: set CILK_NWORKERS.\n");
}

int main(int argc, char** argv) {
	Params p;
	std::string index_dir, graph = "HNSW", dataset, qpath, gtpath, qfmt, gtfmt, mode = "intra";
	int d = 0, nq = -1, batch = 1, synth_n = 0, md = 32;
	bool sweep = false;
	for (int i = 1; i < argc; i++) {
		std::string a = argv[i];
		auto next = [&]() -> std::string {
			if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(1); }
			return argv[++i];
		};
		if (a == "--index-dir") index_dir = next();
		else if (a == "--graph") graph = next();
		else if (a == "--dataset") dataset = next();
		else if (a == "--query") qpath = next();
		else if (a == "--gt") gtpath = next();
		else if (a == "--query-format") qfmt = next();
		else if (a == "--gt-format") gtfmt = next();
		else if (a == "--dim") d = atoi(next().c_str());
		else if (a == "--nq") nq = atoi(next().c_str());
		else if (a == "--mc") p.mc = atoi(next().c_str());
		else if (a == "--mg") p.mg = atoi(next().c_str());
		else if (a == "--ef") p.ef = atoi(next().c_str());
		else if (a == "--cand-queue") p.cand_queue = atoi(next().c_str());
		else if (a == "--mode") mode = next();
		else if (a == "--batch") batch = atoi(next().c_str());
		else if (a == "--bloom-bits") p.bloom_log2_bits = atoi(next().c_str());
		else if (a == "--grain") p.grain = atoi(next().c_str());
		else if (a == "--synthetic") synth_n = atoi(next().c_str());
		else if (a == "--md") md = atoi(next().c_str());
		else if (a == "--sweep") sweep = true;
		else if (a == "-h" || a == "--help") { usage(); return 0; }
		else { fprintf(stderr, "unknown argument %s\n", a.c_str()); usage(); return 1; }
	}
	if (mode != "intra" && mode != "inter") { fprintf(stderr, "--mode must be intra or inter\n"); return 1; }
	p.intra = mode == "intra";
	if (p.mc < 1 || p.mc > kMaxMC || p.mg < 1 || p.mg > kMaxMG) {
		fprintf(stderr, "need 1 <= mc <= %d and 1 <= mg <= %d\n", kMaxMC, kMaxMG);
		return 1;
	}
	if (p.k > p.ef) p.k = p.ef;

	Index idx;
	Queries Q;
	std::vector<int> gt;
	int gt_k = 10;
	if (synth_n > 0) {
		build_synthetic(idx, Q, gt, synth_n, d > 0 ? d : 32, nq > 0 ? nq : 1000, md, 42);
	} else {
		if (index_dir.empty() || qpath.empty() || gtpath.empty()) { usage(); return 1; }
		if (dataset == "SIFT") { if (!d) d = 128; if (qfmt.empty()) qfmt = "bvecs"; if (gtfmt.empty()) gtfmt = "ivecs"; }
		else if (dataset == "Deep") { if (!d) d = 96; if (qfmt.empty()) qfmt = "fbin"; }
		else if (dataset == "SPACEV") { if (!d) d = 100; if (qfmt.empty()) qfmt = "i8bin"; }
		else if (dataset == "SBERT") { if (!d) d = 384; if (qfmt.empty()) qfmt = "f32raw"; }
		else if (dataset == "GLOVE") { if (!d) d = 300; if (qfmt.empty()) qfmt = "fbin"; }
		if (gtfmt.empty()) gtfmt = "ibin";
		if (!d || qfmt.empty()) { fprintf(stderr, "set --dataset or --dim/--query-format\n"); return 1; }
		if (graph != "HNSW" && graph != "NSG") { fprintf(stderr, "--graph must be HNSW or NSG\n"); return 1; }
		if (nq <= 0) nq = 10000;
		load_falcon_index(idx, index_dir, graph, d);
		load_queries(Q, qpath, qfmt, d, nq);
		gt = load_gt(gtpath, gtfmt, nq, gt_k);
	}

#if HAVE_CILK
	printf("OpenCilk workers: %d\n", __cilkrts_get_nworkers());
#else
	printf("built without OpenCilk: running serially\n");
#endif
	printf("mode=%s batch=%d ef=%d cand_queue=%d bloom=2^%d bits\n",
		   mode.c_str(), batch, p.ef, p.cand_queue, p.bloom_log2_bits);

	if (!sweep) {
		run(idx, Q, gt, gt_k, p, batch); // warm-up
		RunResult r = run(idx, Q, gt, gt_k, p, batch);
		printf("mc=%d mg=%d  QPS=%.1f  batch latency median=%.3f ms p95=%.3f ms  "
			   "R@1=%.4f R@10=%.4f  avg hops=%.1f  avg dist comps=%.1f\n",
			   p.mc, p.mg, r.qps, r.lat_median_ms, r.lat_p95_ms, r.recall1, r.recall10, r.avg_hops, r.avg_dist);
		return 0;
	}

	// Fig. 7-style sweep; speedup is relative to BFS (mc = mg = 1)
	printf("mc,mg,qps,speedup_over_bfs,lat_median_ms,lat_p95_ms,recall1,recall10,avg_hops,norm_hops,avg_dist_comps\n");
	double bfs_time = 0, bfs_hops = 0;
	for (int mc = 1; mc <= 4; mc++) {
		for (int mg = 1; mg <= 7; mg++) {
			Params pp = p;
			pp.mc = mc;
			pp.mg = mg;
			run(idx, Q, gt, gt_k, pp, batch); // warm-up
			RunResult r = run(idx, Q, gt, gt_k, pp, batch);
			if (mc == 1 && mg == 1) { bfs_time = r.total_s; bfs_hops = r.avg_hops; }
			printf("%d,%d,%.1f,%.3f,%.4f,%.4f,%.4f,%.4f,%.1f,%.3f,%.1f\n", mc, mg, r.qps, bfs_time / r.total_s,
				   r.lat_median_ms, r.lat_p95_ms, r.recall1, r.recall10, r.avg_hops, r.avg_hops / bfs_hops, r.avg_dist);
			fflush(stdout);
		}
	}
	return 0;
}
