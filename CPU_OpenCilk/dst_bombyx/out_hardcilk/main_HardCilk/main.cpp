#include "hls_stream.h"
#include "main_defs.h"

float inline l2sq(void *mem, addr_t a, addr_t b, uint32_t d) {
  float acc;
  uint32_t i;
  float t;
  acc = 0;
  for (i = 0;(i < d);i = (i + 1)) {
    t = (MEM_ARR_IN(mem, a, i, const float) - MEM_ARR_IN(mem, b, i, const float));
    acc = (acc + (t * t));
  }
  return acc;
}

uint32_t inline murmur2(uint32_t k0, uint32_t seed) {
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

uint32_t inline bloom_test_and_set(void *mem, addr_t bloom, uint32_t node) {
  uint32_t seen;
  uint32_t h;
  uint32_t pos;
  uint32_t bit;
  seen = 1;
  for (h = 0;(h < 3);h = (h + 1)) {
    pos = (murmur2(node, (h + 1)) & (262144 - 1));
    bit = (1U << (pos & 31));
    if (((MEM_ARR_IN(mem, bloom, (pos >> 5), uint32_t) & bit) == 0)) {
      seen = 0;
    }
    MEM_ARR_OUT(mem, bloom, (pos >> 5), uint32_t, (MEM_ARR_IN(mem, bloom, (pos >> 5), uint32_t) | bit));
  }
  return (1 - seen);
}

uint32_t inline sorted_insert(void *mem, addr_t qd, addr_t qi, uint32_t size, uint32_t cap, float dist, uint32_t id) {
  uint32_t new_size;
  uint32_t j;
  new_size = size;
  if (((size < cap) || (dist < MEM_ARR_IN(mem, qd, (cap - 1), float)))) {
    if ((size < cap)) {
      new_size = (size + 1);
    }
    j = (new_size - 1);
    while (((j > 0) && (MEM_ARR_IN(mem, qd, (j - 1), float) > dist))) {
      MEM_ARR_OUT(mem, qd, j, float, MEM_ARR_IN(mem, qd, (j - 1), float));
      MEM_ARR_OUT(mem, qi, j, int, MEM_ARR_IN(mem, qi, (j - 1), int));
      j = (j - 1);
    }
    MEM_ARR_OUT(mem, qd, j, float, dist);
    MEM_ARR_OUT(mem, qi, j, int, id);
  }
  return new_size;
}

float inline threshold(void *mem, addr_t rd, uint32_t r_size, uint32_t ef) {
  float t;
  t = 3.40282347E+38F;
  if ((r_size == ef)) {
    t = MEM_ARR_IN(mem, rd, (ef - 1), const float);
  }
  return t;
}

void inline merge_group(void *mem, addr_t rd, addr_t ri, addr_t cd, addr_t ci, addr_t meta, uint32_t ef, uint32_t cand_cap, addr_t ids, addr_t dists, uint32_t cnt) {
  uint32_t i;
  for (i = 0;(i < cnt);i = (i + 1)) {
    if ((MEM_ARR_IN(mem, dists, i, const float) < threshold(rd, MEM_ARR_IN(mem, meta, 0, int), ef))) {
      MEM_ARR_OUT(mem, meta, 0, int, sorted_insert(mem, rd, ri, MEM_ARR_IN(mem, meta, 0, int), ef, MEM_ARR_IN(mem, dists, i, const float), MEM_ARR_IN(mem, ids, i, const int)));
      MEM_ARR_OUT(mem, meta, 1, int, sorted_insert(mem, cd, ci, MEM_ARR_IN(mem, meta, 1, int), cand_cap, MEM_ARR_IN(mem, dists, i, const float), MEM_ARR_IN(mem, ids, i, const int)));
    }
  }
  MEM_ARR_OUT(mem, meta, 3, int, (MEM_ARR_IN(mem, meta, 3, int) + cnt));
}

uint32_t inline select_group(void *mem, addr_t rd, addr_t cd, addr_t ci, addr_t meta, uint32_t ef, uint32_t mc, addr_t degree, addr_t links, uint32_t maxDegree, addr_t bloom, addr_t ids, addr_t cnt) {
  uint32_t popped;
  uint32_t c;
  float t;
  uint32_t node;
  uint32_t j;
  uint32_t j0;
  uint32_t nb;
  popped = 0;
  c = 0;
  t = threshold(mem, rd, MEM_ARR_IN(mem, meta, 0, int), ef);
  while ((((popped < mc) && (MEM_ARR_IN(mem, meta, 1, int) > 0)) && (MEM_ARR_IN(mem, cd, 0, float) <= t))) {
    node = MEM_ARR_IN(mem, ci, 0, int);
    for (j = 1;(j < MEM_ARR_IN(mem, meta, 1, int));j = (j + 1)) {
      MEM_ARR_OUT(mem, cd, (j - 1), float, MEM_ARR_IN(mem, cd, j, float));
      MEM_ARR_OUT(mem, ci, (j - 1), int, MEM_ARR_IN(mem, ci, j, int));
    }
    MEM_ARR_OUT(mem, meta, 1, int, (MEM_ARR_IN(mem, meta, 1, int) - 1));
    popped = (popped + 1);
    for (j0 = 0;(j0 < MEM_ARR_IN(mem, degree, node, const int));j0 = (j0 + 1)) {
      nb = MEM_ARR_IN(mem, links, ((node * maxDegree) + j0), const int);
      if ((bloom_test_and_set(bloom, nb) == 1)) {
        MEM_ARR_OUT(mem, ids, c, int, nb);
        c = (c + 1);
      }
    }
  }
  MEM_OUT(mem, cnt, int, c);
  MEM_ARR_OUT(mem, meta, 2, int, (MEM_ARR_IN(mem, meta, 2, int) + popped));
  return popped;
}

void dst_search (
  void *mem,
  hls::stream<dst_search_task> &taskIn,
  hls::stream<dst_search_reentry0_task> &taskGlobalOut_dst_search_reentry0
) {

#pragma HLS INTERFACE mode = axis port = taskIn
#pragma HLS INTERFACE mode = axis port = taskGlobalOut_dst_search_reentry0
#pragma HLS INTERFACE mode = m_axi port = mem latency = 16
#pragma HLS INTERFACE ap_ctrl_none port = return
#pragma HLS PIPELINE II = 1 style = flp

  uint32_t slot_cap;
  uint32_t i;
  uint32_t i0;
  float d0;
  uint32_t cur;
  uint32_t active;
  uint32_t launched;
  uint32_t g;
  uint32_t ps;
  uint32_t cs;
  uint32_t popped;
  uint32_t i1;
  dst_search_task args = taskIn.read();

  slot_cap = (args.mc * args.maxDegree);
  for (i = 0;(i < 8192);i = (i + 1)) {
    MEM_ARR_OUT(mem, args.bloom, i, uint32_t, 0);
  }
  for (i0 = 0;(i0 < (2 * args.mg));i0 = (i0 + 1)) {
    MEM_ARR_OUT(mem, args.slot_cnt, i0, int, 0);
  }
  d0 = l2sq(mem, args.q, (args.vecs + ((args.entry * args.d)) * sizeof(const float)), args.d);
  bloom_test_and_set(mem, args.bloom, args.entry);
  MEM_ARR_OUT(mem, args.rd, 0, float, d0);
  MEM_ARR_OUT(mem, args.ri, 0, int, args.entry);
  MEM_ARR_OUT(mem, args.cd, 0, float, d0);
  MEM_ARR_OUT(mem, args.ci, 0, int, args.entry);
  MEM_ARR_OUT(mem, args.meta, 0, int, 1);
  MEM_ARR_OUT(mem, args.meta, 1, int, 1);
  MEM_ARR_OUT(mem, args.meta, 2, int, 0);
  MEM_ARR_OUT(mem, args.meta, 3, int, 1);
  cur = 0;
  active = 1;
  dst_search_reentry0_task dst_search_reentry0_args0;
  dst_search_reentry0_args0._cont = args._cont;
  dst_search_reentry0_args0.q = args.q;
  dst_search_reentry0_args0.vecs = args.vecs;
  dst_search_reentry0_args0.degree = args.degree;
  dst_search_reentry0_args0.links = args.links;
  dst_search_reentry0_args0.d = args.d;
  dst_search_reentry0_args0.maxDegree = args.maxDegree;
  dst_search_reentry0_args0.entry = args.entry;
  dst_search_reentry0_args0.ef = args.ef;
  dst_search_reentry0_args0.cand_cap = args.cand_cap;
  dst_search_reentry0_args0.mc = args.mc;
  dst_search_reentry0_args0.mg = args.mg;
  dst_search_reentry0_args0.grain = args.grain;
  dst_search_reentry0_args0.k0 = args.k0;
  dst_search_reentry0_args0.rd = args.rd;
  dst_search_reentry0_args0.ri = args.ri;
  dst_search_reentry0_args0.cd = args.cd;
  dst_search_reentry0_args0.ci = args.ci;
  dst_search_reentry0_args0.bloom = args.bloom;
  dst_search_reentry0_args0.slot_ids = args.slot_ids;
  dst_search_reentry0_args0.slot_dists = args.slot_dists;
  dst_search_reentry0_args0.slot_cnt = args.slot_cnt;
  dst_search_reentry0_args0.meta = args.meta;
  dst_search_reentry0_args0.out = args.out;
  dst_search_reentry0_args0.slot_cap = slot_cap;
  dst_search_reentry0_args0.i = i;
  dst_search_reentry0_args0.i0 = i0;
  dst_search_reentry0_args0.d0 = d0;
  dst_search_reentry0_args0.cur = cur;
  dst_search_reentry0_args0.active = active;
  dst_search_reentry0_args0.ps = ps;
  dst_search_reentry0_args0.cs = cs;
  dst_search_reentry0_args0.popped = popped;
  dst_search_reentry0_args0.i1 = i1;
  taskGlobalOut_dst_search_reentry0.write(dst_search_reentry0_args0);

}

void dst_search_exit0 (
  void *mem,
  hls::stream<dst_search_exit0_task> &taskIn,
  hls::stream<uint64_t> &argOut,
  hls::stream<uint32_t_arg_out> &argDataOut
) {

#pragma HLS INTERFACE mode = axis port = taskIn
#pragma HLS INTERFACE mode = axis port = argOut
#pragma HLS INTERFACE mode = axis port = argDataOut
#pragma HLS INTERFACE mode = m_axi port = mem latency = 16
#pragma HLS INTERFACE ap_ctrl_none port = return
#pragma HLS PIPELINE II = 1 style = flp

  uint32_t i1;
  dst_search_exit0_task args = taskIn.read();

  for (i1 = 0;(i1 < args.k0);i1 = (i1 + 1)) {
    MEM_ARR_OUT(mem, args.out, i1, int, (-1));
    if ((i1 < MEM_ARR_IN(mem, args.meta, 0, int))) {
      uint32_t_arg_out a0;
      a0.addr = ((addr_t)(args.out) + (i1 * sizeof(int)));
      a0.data = MEM_ARR_IN(mem, args.ri, i1, int);
      a0.size = 2;
      a0.allow = 0;
      argDataOut.write(a0);
    }
  }
  uint32_t_arg_out a1;
  a1.addr = 0x3FFFFFFFF;
  a1.data = 0;
  a1.size = 0;
  a1.allow = 1;
  argDataOut.write(a1);
  argOut.write(args._cont);
}

void dst_search_reentry0 (
  void *mem,
  hls::stream<dst_search_reentry0_task> &taskIn,
  hls::stream<dst_search_exit0_task> &taskGlobalOut_dst_search_exit0,
  hls::stream<eval_range_task> &taskGlobalOut_eval_range_depends_dst_search_reentry0_cont0,
  hls::stream<uint64_t> &closureIn,
  hls::stream<dst_search_reentry0_cont0_spawn_next> &spawnNext_dst_search_reentry0_cont0
) {

#pragma HLS INTERFACE mode = axis port = taskIn
#pragma HLS INTERFACE mode = axis port = taskGlobalOut_dst_search_exit0
#pragma HLS INTERFACE mode = axis port = taskGlobalOut_eval_range_depends_dst_search_reentry0_cont0
#pragma HLS INTERFACE mode = axis port = closureIn
#pragma HLS INTERFACE mode = axis port = spawnNext_dst_search_reentry0_cont0
#pragma HLS INTERFACE mode = m_axi port = mem latency = 16
#pragma HLS INTERFACE ap_ctrl_none port = return
#pragma HLS PIPELINE II = 1 style = flp

  uint32_t launched;
  uint32_t g;
  dst_search_reentry0_task args = taskIn.read();

  if ((args.active == 1)) {
    launched = 0;
    uint32_t SN_dst_search_reentry0_cont0c_cnt = (0 + ((args.mg / 1) * 1));
    dst_search_reentry0_cont0_task SN_dst_search_reentry0_cont0c;
    SN_dst_search_reentry0_cont0c._cont = args._cont;
    SN_dst_search_reentry0_cont0c._counter = SN_dst_search_reentry0_cont0c_cnt;
    addr_t SN_dst_search_reentry0_cont0c_k = closureIn.read();

    SN_dst_search_reentry0_cont0c.i1 = args.i1;
    SN_dst_search_reentry0_cont0c.popped = args.popped;
    SN_dst_search_reentry0_cont0c.cs = args.cs;
    SN_dst_search_reentry0_cont0c.ps = args.ps;
    SN_dst_search_reentry0_cont0c.launched = launched;
    SN_dst_search_reentry0_cont0c.k0 = args.k0;
    SN_dst_search_reentry0_cont0c.grain = args.grain;
    SN_dst_search_reentry0_cont0c.mg = args.mg;
    SN_dst_search_reentry0_cont0c.mc = args.mc;
    SN_dst_search_reentry0_cont0c.cand_cap = args.cand_cap;
    SN_dst_search_reentry0_cont0c.ef = args.ef;
    SN_dst_search_reentry0_cont0c.entry = args.entry;
    SN_dst_search_reentry0_cont0c.maxDegree = args.maxDegree;
    SN_dst_search_reentry0_cont0c.d = args.d;
    SN_dst_search_reentry0_cont0c.links = args.links;
    SN_dst_search_reentry0_cont0c.degree = args.degree;
    SN_dst_search_reentry0_cont0c.vecs = args.vecs;
    SN_dst_search_reentry0_cont0c.q = args.q;
    SN_dst_search_reentry0_cont0c.rd = args.rd;
    SN_dst_search_reentry0_cont0c.ri = args.ri;
    SN_dst_search_reentry0_cont0c.cd = args.cd;
    SN_dst_search_reentry0_cont0c.ci = args.ci;
    SN_dst_search_reentry0_cont0c.bloom = args.bloom;
    SN_dst_search_reentry0_cont0c.slot_ids = args.slot_ids;
    SN_dst_search_reentry0_cont0c.slot_dists = args.slot_dists;
    SN_dst_search_reentry0_cont0c.slot_cnt = args.slot_cnt;
    SN_dst_search_reentry0_cont0c.meta = args.meta;
    SN_dst_search_reentry0_cont0c.out = args.out;
    SN_dst_search_reentry0_cont0c.slot_cap = args.slot_cap;
    SN_dst_search_reentry0_cont0c.i = args.i;
    SN_dst_search_reentry0_cont0c.i0 = args.i0;
    SN_dst_search_reentry0_cont0c.d0 = args.d0;
    SN_dst_search_reentry0_cont0c.cur = args.cur;
    SN_dst_search_reentry0_cont0c.active = args.active;
    uint8_t *SN_dst_search_reentry0_cont0_bytes = (uint8_t *)&SN_dst_search_reentry0_cont0c;
    dst_search_reentry0_cont0_spawn_next SN_dst_search_reentry0_cont00;
    SN_dst_search_reentry0_cont00.addr = SN_dst_search_reentry0_cont0c_k + 0;
    memcpy(SN_dst_search_reentry0_cont00.data, SN_dst_search_reentry0_cont0_bytes + 0, 64);
    SN_dst_search_reentry0_cont00.size = 6;
    SN_dst_search_reentry0_cont00.allow = 0;
    spawnNext_dst_search_reentry0_cont0.write(SN_dst_search_reentry0_cont00);
    dst_search_reentry0_cont0_spawn_next SN_dst_search_reentry0_cont01;
    SN_dst_search_reentry0_cont01.addr = SN_dst_search_reentry0_cont0c_k + 64;
    memcpy(SN_dst_search_reentry0_cont01.data, SN_dst_search_reentry0_cont0_bytes + 64, 64);
    SN_dst_search_reentry0_cont01.size = 6;
    SN_dst_search_reentry0_cont01.allow = 0;
    spawnNext_dst_search_reentry0_cont0.write(SN_dst_search_reentry0_cont01);
    dst_search_reentry0_cont0_spawn_next SN_dst_search_reentry0_cont02;
    SN_dst_search_reentry0_cont02.addr = SN_dst_search_reentry0_cont0c_k + 128;
    memcpy(SN_dst_search_reentry0_cont02.data, SN_dst_search_reentry0_cont0_bytes + 128, 64);
    SN_dst_search_reentry0_cont02.size = 6;
    SN_dst_search_reentry0_cont02.allow = 0;
    spawnNext_dst_search_reentry0_cont0.write(SN_dst_search_reentry0_cont02);
    dst_search_reentry0_cont0_spawn_next SN_dst_search_reentry0_cont03;
    SN_dst_search_reentry0_cont03.addr = SN_dst_search_reentry0_cont0c_k + 192;
    memcpy(SN_dst_search_reentry0_cont03.data, SN_dst_search_reentry0_cont0_bytes + 192, 64);
    SN_dst_search_reentry0_cont03.size = 6;
    SN_dst_search_reentry0_cont03.allow = SN_dst_search_reentry0_cont0c_cnt;
    spawnNext_dst_search_reentry0_cont0.write(SN_dst_search_reentry0_cont03);

    for (g = 0;(g < args.mg);g = (g + 1)) {
      args.ps = (((1 - args.cur) * args.mg) + g);
      args.cs = ((args.cur * args.mg) + g);
      merge_group(mem, args.rd, args.ri, args.cd, args.ci, args.meta, args.ef, args.cand_cap, (args.slot_ids + ((args.ps * args.slot_cap)) * sizeof(int)), (args.slot_dists + ((args.ps * args.slot_cap)) * sizeof(float)), MEM_ARR_IN(mem, args.slot_cnt, args.ps, int));
      MEM_ARR_OUT(mem, args.slot_cnt, args.ps, int, 0);
      args.popped = select_group(mem, args.rd, args.cd, args.ci, args.meta, args.ef, args.mc, args.degree, args.links, args.maxDegree, args.bloom, (args.slot_ids + ((args.cs * args.slot_cap)) * sizeof(int)), (args.slot_cnt + (args.cs) * sizeof(int)));
      launched = (launched + args.popped);
      eval_range_task eval_range_args1;
      eval_range_args1._cont = SN_dst_search_reentry0_cont0c_k;
      eval_range_args1.q = args.q;
      eval_range_args1.vecs = args.vecs;
      eval_range_args1.d = args.d;
      eval_range_args1.ids = (args.slot_ids + ((args.cs * args.slot_cap)) * sizeof(int));
      eval_range_args1.dists = (args.slot_dists + ((args.cs * args.slot_cap)) * sizeof(float));
      eval_range_args1.lo = 0;
      eval_range_args1.hi = MEM_ARR_IN(mem, args.slot_cnt, args.cs, int);
      eval_range_args1.grain = args.grain;
      taskGlobalOut_eval_range_depends_dst_search_reentry0_cont0.write(eval_range_args1);

    }
  } else {
    dst_search_exit0_task dst_search_exit0_args2;
    dst_search_exit0_args2._cont = args._cont;
    dst_search_exit0_args2.k0 = args.k0;
    dst_search_exit0_args2.ri = args.ri;
    dst_search_exit0_args2.meta = args.meta;
    dst_search_exit0_args2.out = args.out;
    taskGlobalOut_dst_search_exit0.write(dst_search_exit0_args2);

  }
}

void dst_search_reentry0_cont0 (
  hls::stream<dst_search_reentry0_cont0_task> &taskIn,
  hls::stream<dst_search_reentry0_task> &taskGlobalOut_dst_search_reentry0
) {

#pragma HLS INTERFACE mode = axis port = taskIn
#pragma HLS INTERFACE mode = axis port = taskGlobalOut_dst_search_reentry0
#pragma HLS INTERFACE ap_ctrl_none port = return
#pragma HLS PIPELINE II = 1 style = flp

  dst_search_reentry0_cont0_task args = taskIn.read();

  args.cur = (1 - args.cur);
  if ((args.launched == 0)) {
    args.active = 0;
  }
  dst_search_reentry0_task dst_search_reentry0_args3;
  dst_search_reentry0_args3._cont = args._cont;
  dst_search_reentry0_args3.q = args.q;
  dst_search_reentry0_args3.vecs = args.vecs;
  dst_search_reentry0_args3.degree = args.degree;
  dst_search_reentry0_args3.links = args.links;
  dst_search_reentry0_args3.d = args.d;
  dst_search_reentry0_args3.maxDegree = args.maxDegree;
  dst_search_reentry0_args3.entry = args.entry;
  dst_search_reentry0_args3.ef = args.ef;
  dst_search_reentry0_args3.cand_cap = args.cand_cap;
  dst_search_reentry0_args3.mc = args.mc;
  dst_search_reentry0_args3.mg = args.mg;
  dst_search_reentry0_args3.grain = args.grain;
  dst_search_reentry0_args3.k0 = args.k0;
  dst_search_reentry0_args3.rd = args.rd;
  dst_search_reentry0_args3.ri = args.ri;
  dst_search_reentry0_args3.cd = args.cd;
  dst_search_reentry0_args3.ci = args.ci;
  dst_search_reentry0_args3.bloom = args.bloom;
  dst_search_reentry0_args3.slot_ids = args.slot_ids;
  dst_search_reentry0_args3.slot_dists = args.slot_dists;
  dst_search_reentry0_args3.slot_cnt = args.slot_cnt;
  dst_search_reentry0_args3.meta = args.meta;
  dst_search_reentry0_args3.out = args.out;
  dst_search_reentry0_args3.slot_cap = args.slot_cap;
  dst_search_reentry0_args3.i = args.i;
  dst_search_reentry0_args3.i0 = args.i0;
  dst_search_reentry0_args3.d0 = args.d0;
  dst_search_reentry0_args3.cur = args.cur;
  dst_search_reentry0_args3.active = args.active;
  dst_search_reentry0_args3.ps = args.ps;
  dst_search_reentry0_args3.cs = args.cs;
  dst_search_reentry0_args3.popped = args.popped;
  dst_search_reentry0_args3.i1 = args.i1;
  taskGlobalOut_dst_search_reentry0.write(dst_search_reentry0_args3);

}

void eval_range (
  void *mem,
  hls::stream<eval_range_task> &taskIn,
  hls::stream<eval_range_task> &taskOut,
  hls::stream<eval_range_afterif0_task> &taskGlobalOut_eval_range_afterif0,
  hls::stream<uint64_t> &closureIn,
  hls::stream<eval_range_cont0_spawn_next> &spawnNext_eval_range_cont0
) {

#pragma HLS INTERFACE mode = axis port = taskIn
#pragma HLS INTERFACE mode = axis port = taskOut
#pragma HLS INTERFACE mode = axis port = taskGlobalOut_eval_range_afterif0
#pragma HLS INTERFACE mode = axis port = closureIn
#pragma HLS INTERFACE mode = axis port = spawnNext_eval_range_cont0
#pragma HLS INTERFACE mode = m_axi port = mem latency = 16
#pragma HLS INTERFACE ap_ctrl_none port = return
#pragma HLS PIPELINE II = 1 style = flp

  uint32_t i;
  uint32_t mid;
  eval_range_task args = taskIn.read();

  uint8_t _cont_tag = CONT_TAG(args._cont);

  if (((args.hi - args.lo) <= args.grain)) {
    for (i = args.lo;(i < args.hi);i = (i + 1)) {
      MEM_ARR_OUT(mem, args.dists, i, float, l2sq(mem, args.q, (args.vecs + ((MEM_ARR_IN(mem, args.ids, i, const int) * args.d)) * sizeof(const float)), args.d));
    }
    eval_range_afterif0_task eval_range_afterif0_args4;
    eval_range_afterif0_args4._cont = args._cont;
    taskGlobalOut_eval_range_afterif0.write(eval_range_afterif0_args4);

  } else {
    mid = (args.lo + ((args.hi - args.lo) / 2));
    uint32_t SN_eval_range_cont0c_cnt = 2;
    eval_range_cont0_task SN_eval_range_cont0c;
    SN_eval_range_cont0c._cont = args._cont;
    SN_eval_range_cont0c._counter = SN_eval_range_cont0c_cnt;
    addr_t SN_eval_range_cont0c_k = closureIn.read();

    eval_range_cont0_spawn_next SN_eval_range_cont0;
    SN_eval_range_cont0.addr = SN_eval_range_cont0c_k;
    SN_eval_range_cont0.data = SN_eval_range_cont0c;
    SN_eval_range_cont0.size = 4;
    SN_eval_range_cont0.allow = SN_eval_range_cont0c_cnt;
    spawnNext_eval_range_cont0.write(SN_eval_range_cont0);

    eval_range_task eval_range_args5;
    eval_range_args5._cont = SN_eval_range_cont0c_k;
    eval_range_args5.q = args.q;
    eval_range_args5.vecs = args.vecs;
    eval_range_args5.d = args.d;
    eval_range_args5.ids = args.ids;
    eval_range_args5.dists = args.dists;
    eval_range_args5.lo = args.lo;
    eval_range_args5.hi = mid;
    eval_range_args5.grain = args.grain;
    taskOut.write(eval_range_args5);

    eval_range_task eval_range_args6;
    eval_range_args6._cont = SN_eval_range_cont0c_k;
    eval_range_args6.q = args.q;
    eval_range_args6.vecs = args.vecs;
    eval_range_args6.d = args.d;
    eval_range_args6.ids = args.ids;
    eval_range_args6.dists = args.dists;
    eval_range_args6.lo = mid;
    eval_range_args6.hi = args.hi;
    eval_range_args6.grain = args.grain;
    taskOut.write(eval_range_args6);

  }
}

void eval_range_afterif0 (
  hls::stream<eval_range_afterif0_task> &taskIn,
  hls::stream<uint64_t> &argOut_dst_search_reentry0_cont0,
  hls::stream<uint64_t> &argOut_eval_range_cont0
) {

#pragma HLS INTERFACE mode = axis port = taskIn
#pragma HLS INTERFACE mode = axis port = argOut_dst_search_reentry0_cont0
#pragma HLS INTERFACE mode = axis port = argOut_eval_range_cont0
#pragma HLS INTERFACE ap_ctrl_none port = return
#pragma HLS PIPELINE II = 1 style = flp

  eval_range_afterif0_task args = taskIn.read();

  uint8_t _cont_tag = CONT_TAG(args._cont);

  if (_cont_tag == DST_SEARCH_REENTRY0_CONT0_TAG) {
    argOut_dst_search_reentry0_cont0.write(args._cont);
  } else if (_cont_tag == EVAL_RANGE_CONT0_TAG) {
    argOut_eval_range_cont0.write(args._cont);
  }
}

void eval_range_cont0 (
  hls::stream<eval_range_cont0_task> &taskIn,
  hls::stream<eval_range_afterif0_task> &taskGlobalOut_eval_range_afterif0
) {

#pragma HLS INTERFACE mode = axis port = taskIn
#pragma HLS INTERFACE mode = axis port = taskGlobalOut_eval_range_afterif0
#pragma HLS INTERFACE ap_ctrl_none port = return
#pragma HLS PIPELINE II = 1 style = flp

  eval_range_cont0_task args = taskIn.read();

  uint8_t _cont_tag = CONT_TAG(args._cont);

  eval_range_afterif0_task eval_range_afterif0_args7;
  eval_range_afterif0_args7._cont = args._cont;
  taskGlobalOut_eval_range_afterif0.write(eval_range_afterif0_args7);

}

