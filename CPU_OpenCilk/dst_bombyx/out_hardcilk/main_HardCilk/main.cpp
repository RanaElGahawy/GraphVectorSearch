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

void dst_search_reentry0_cont0 (
  void *mem,
  hls::stream<dst_search_reentry0_cont0_task> &taskIn,
  hls::stream<dst_search_reentry0_task> &taskGlobalOut_dst_search_reentry0,
  hls::stream<uint64_t> &argOut
) {

#pragma HLS INTERFACE mode = axis port = taskIn
#pragma HLS INTERFACE mode = axis port = taskGlobalOut_dst_search_reentry0
#pragma HLS INTERFACE mode = axis port = argOut
#pragma HLS INTERFACE mode = m_axi port = mem
#pragma HLS INTERFACE ap_ctrl_none port = return
#pragma HLS PIPELINE II = 1 style = flp

  dst_search_reentry0_cont0_task args = taskIn.read();

  if (((args.pending > 0) && ((args.pending == args.mg) || (args.popped == 0)))) {
    merge_group(mem, args.rd, args.ri, args.cd, args.ci, args.meta, args.ef, args.cand_cap, (args.slot_ids + ((args.tail * args.slot_cap)) * sizeof(int)), (args.slot_dists + ((args.tail * args.slot_cap)) * sizeof(float)), MEM_ARR_IN(mem, args.slot_cnt, args.tail, int));
    args.tail = ((args.tail + 1) % args.mg);
    args.pending = (args.pending - 1);
  }
  if (((args.pending == 0) && ((MEM_ARR_IN(mem, args.meta, 1, int) == 0) || (MEM_ARR_IN(mem, args.cd, 0, float) > threshold(args.rd, MEM_ARR_IN(mem, args.meta, 0, int), args.ef))))) {
    args.active = 0;
  }
  dst_search_reentry0_task dst_search_reentry0_args0;
  dst_search_reentry0_args0._cont = args._cont;
  dst_search_reentry0_args0.q = args.q;
  dst_search_reentry0_args0.vecs = args.vecs;
  dst_search_reentry0_args0.degree = args.degree;
  dst_search_reentry0_args0.links = args.links;
  dst_search_reentry0_args0.d = args.d;
  dst_search_reentry0_args0.maxDegree = args.maxDegree;
  dst_search_reentry0_args0.ef = args.ef;
  dst_search_reentry0_args0.cand_cap = args.cand_cap;
  dst_search_reentry0_args0.mc = args.mc;
  dst_search_reentry0_args0.mg = args.mg;
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
  dst_search_reentry0_args0.slot_cap = args.slot_cap;
  dst_search_reentry0_args0.head = args.head;
  dst_search_reentry0_args0.tail = args.tail;
  dst_search_reentry0_args0.pending = args.pending;
  dst_search_reentry0_args0.active = args.active;
  dst_search_reentry0_args0.s = args.s;
  dst_search_reentry0_args0.popped = args.popped;
  dst_search_reentry0_args0.i0 = args.i0;
  dst_search_reentry0_args0.i1 = args.i1;
  taskGlobalOut_dst_search_reentry0.write(dst_search_reentry0_args0);

  argOut.write(args._cont);
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
#pragma HLS INTERFACE mode = m_axi port = mem
#pragma HLS INTERFACE ap_ctrl_none port = return
#pragma HLS PIPELINE II = 1 style = flp

  dst_search_exit0_task args = taskIn.read();

  for (args.i1 = 0;(args.i1 < args.k0);args.i1 = (args.i1 + 1)) {
    MEM_ARR_OUT(mem, args.out, args.i1, int, (-1));
    if ((args.i1 < MEM_ARR_IN(mem, args.meta, 0, int))) {
      uint32_t_arg_out a0;
      a0.addr = ((addr_t)(args.out) + (args.i1 * sizeof(int)));
      a0.data = MEM_ARR_IN(mem, args.ri, args.i1, int);
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

void dist_task (
  void *mem,
  hls::stream<dist_task_task> &taskIn,
  hls::stream<uint64_t> &argOut,
  hls::stream<float_arg_out> &argDataOut
) {

#pragma HLS INTERFACE mode = axis port = taskIn
#pragma HLS INTERFACE mode = axis port = argOut
#pragma HLS INTERFACE mode = axis port = argDataOut
#pragma HLS INTERFACE mode = m_axi port = mem
#pragma HLS INTERFACE ap_ctrl_none port = return
#pragma HLS PIPELINE II = 1 style = flp

  dist_task_task args = taskIn.read();

  float_arg_out a2;
  a2.addr = ((addr_t)(args.dists) + (args.i * sizeof(float)));
  a2.data = l2sq(mem, args.q, (args.vecs + ((MEM_ARR_IN(mem, args.ids, args.i, const int) * args.d)) * sizeof(const float)), args.d);
  a2.size = 2;
  a2.allow = 1;
  argDataOut.write(a2);
  argOut.write(args._cont);
}

void dst_search (
  void *mem,
  hls::stream<dst_search_task> &taskIn,
  hls::stream<dst_search_reentry0_task> &taskGlobalOut_dst_search_reentry0
) {

#pragma HLS INTERFACE mode = axis port = taskIn
#pragma HLS INTERFACE mode = axis port = taskGlobalOut_dst_search_reentry0
#pragma HLS INTERFACE mode = m_axi port = mem
#pragma HLS INTERFACE ap_ctrl_none port = return
#pragma HLS PIPELINE II = 1 style = flp

  uint32_t slot_cap;
  uint32_t i;
  float d0;
  uint32_t head;
  uint32_t tail;
  uint32_t pending;
  uint32_t active;
  uint32_t s;
  uint32_t popped;
  uint32_t i0;
  uint32_t i1;
  dst_search_task args = taskIn.read();

  slot_cap = (args.mc * args.maxDegree);
  for (i = 0;(i < 8192);i = (i + 1)) {
    MEM_ARR_OUT(mem, args.bloom, i, uint32_t, 0);
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
  head = 0;
  tail = 0;
  pending = 0;
  active = 1;
  dst_search_reentry0_task dst_search_reentry0_args1;
  dst_search_reentry0_args1._cont = args._cont;
  dst_search_reentry0_args1.q = args.q;
  dst_search_reentry0_args1.vecs = args.vecs;
  dst_search_reentry0_args1.degree = args.degree;
  dst_search_reentry0_args1.links = args.links;
  dst_search_reentry0_args1.d = args.d;
  dst_search_reentry0_args1.maxDegree = args.maxDegree;
  dst_search_reentry0_args1.ef = args.ef;
  dst_search_reentry0_args1.cand_cap = args.cand_cap;
  dst_search_reentry0_args1.mc = args.mc;
  dst_search_reentry0_args1.mg = args.mg;
  dst_search_reentry0_args1.k0 = args.k0;
  dst_search_reentry0_args1.rd = args.rd;
  dst_search_reentry0_args1.ri = args.ri;
  dst_search_reentry0_args1.cd = args.cd;
  dst_search_reentry0_args1.ci = args.ci;
  dst_search_reentry0_args1.bloom = args.bloom;
  dst_search_reentry0_args1.slot_ids = args.slot_ids;
  dst_search_reentry0_args1.slot_dists = args.slot_dists;
  dst_search_reentry0_args1.slot_cnt = args.slot_cnt;
  dst_search_reentry0_args1.meta = args.meta;
  dst_search_reentry0_args1.out = args.out;
  dst_search_reentry0_args1.slot_cap = slot_cap;
  dst_search_reentry0_args1.head = head;
  dst_search_reentry0_args1.tail = tail;
  dst_search_reentry0_args1.pending = pending;
  dst_search_reentry0_args1.active = active;
  dst_search_reentry0_args1.s = s;
  dst_search_reentry0_args1.popped = popped;
  dst_search_reentry0_args1.i0 = i0;
  dst_search_reentry0_args1.i1 = i1;
  taskGlobalOut_dst_search_reentry0.write(dst_search_reentry0_args1);

}

void dst_search_reentry0 (
  void *mem,
  hls::stream<dst_search_reentry0_task> &taskIn,
  hls::stream<dist_task_task> &taskGlobalOut_dist_task_depends_dst_search_reentry0_cont0,
  hls::stream<dst_search_exit0_task> &taskGlobalOut_dst_search_exit0,
  hls::stream<uint64_t> &closureIn,
  hls::stream<dst_search_reentry0_cont0_spawn_next> &spawnNext_dst_search_reentry0_cont0
) {

#pragma HLS INTERFACE mode = axis port = taskIn
#pragma HLS INTERFACE mode = axis port = taskGlobalOut_dist_task_depends_dst_search_reentry0_cont0
#pragma HLS INTERFACE mode = axis port = taskGlobalOut_dst_search_exit0
#pragma HLS INTERFACE mode = axis port = closureIn
#pragma HLS INTERFACE mode = axis port = spawnNext_dst_search_reentry0_cont0
#pragma HLS INTERFACE mode = m_axi port = mem
#pragma HLS INTERFACE ap_ctrl_none port = return
#pragma HLS PIPELINE II = 1 style = flp

  dst_search_reentry0_task args = taskIn.read();

  if ((args.active == 1)) {
    args.s = args.head;
    args.popped = select_group(mem, args.rd, args.cd, args.ci, args.meta, args.ef, args.mc, args.degree, args.links, args.maxDegree, args.bloom, (args.slot_ids + ((args.s * args.slot_cap)) * sizeof(int)), (args.slot_cnt + (args.s) * sizeof(int)));
    uint32_t SN_dst_search_reentry0_cont0c_cnt = (0 + ((MEM_ARR_IN(mem, args.slot_cnt, args.s, int) / 1) * 1));
    dst_search_reentry0_cont0_task SN_dst_search_reentry0_cont0c;
    SN_dst_search_reentry0_cont0c._cont = args._cont;
    SN_dst_search_reentry0_cont0c._counter = SN_dst_search_reentry0_cont0c_cnt;
    addr_t SN_dst_search_reentry0_cont0c_k = closureIn.read();

    SN_dst_search_reentry0_cont0c.i0 = args.i0;
    SN_dst_search_reentry0_cont0c.popped = args.popped;
    SN_dst_search_reentry0_cont0c.s = args.s;
    SN_dst_search_reentry0_cont0c.head = args.head;
    SN_dst_search_reentry0_cont0c.slot_cnt = args.slot_cnt;
    SN_dst_search_reentry0_cont0c.links = args.links;
    SN_dst_search_reentry0_cont0c.slot_dists = args.slot_dists;
    SN_dst_search_reentry0_cont0c.tail = args.tail;
    SN_dst_search_reentry0_cont0c.out = args.out;
    SN_dst_search_reentry0_cont0c.ri = args.ri;
    SN_dst_search_reentry0_cont0c.slot_ids = args.slot_ids;
    SN_dst_search_reentry0_cont0c.meta = args.meta;
    SN_dst_search_reentry0_cont0c.bloom = args.bloom;
    SN_dst_search_reentry0_cont0c.i1 = args.i1;
    SN_dst_search_reentry0_cont0c.ci = args.ci;
    SN_dst_search_reentry0_cont0c.k0 = args.k0;
    SN_dst_search_reentry0_cont0c.ef = args.ef;
    SN_dst_search_reentry0_cont0c.cd = args.cd;
    SN_dst_search_reentry0_cont0c.rd = args.rd;
    SN_dst_search_reentry0_cont0c.mg = args.mg;
    SN_dst_search_reentry0_cont0c.active = args.active;
    SN_dst_search_reentry0_cont0c.mc = args.mc;
    SN_dst_search_reentry0_cont0c.cand_cap = args.cand_cap;
    SN_dst_search_reentry0_cont0c.pending = args.pending;
    SN_dst_search_reentry0_cont0c.slot_cap = args.slot_cap;
    SN_dst_search_reentry0_cont0c.degree = args.degree;
    SN_dst_search_reentry0_cont0c.vecs = args.vecs;
    SN_dst_search_reentry0_cont0c.maxDegree = args.maxDegree;
    SN_dst_search_reentry0_cont0c.d = args.d;
    SN_dst_search_reentry0_cont0c.q = args.q;
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

    if ((args.popped > 0)) {
      args.pending = (args.pending + 1);
      args.head = ((args.head + 1) % args.mg);
    }
    for (args.i0 = 0;(args.i0 < MEM_ARR_IN(mem, args.slot_cnt, args.s, int));args.i0 = (args.i0 + 1)) {
      dist_task_task dist_task_args2;
      dist_task_args2._cont = SN_dst_search_reentry0_cont0c_k;
      dist_task_args2.q = args.q;
      dist_task_args2.vecs = args.vecs;
      dist_task_args2.d = args.d;
      dist_task_args2.ids = (args.slot_ids + ((args.s * args.slot_cap)) * sizeof(int));
      dist_task_args2.dists = (args.slot_dists + ((args.s * args.slot_cap)) * sizeof(float));
      dist_task_args2.i = args.i0;
      taskGlobalOut_dist_task_depends_dst_search_reentry0_cont0.write(dist_task_args2);

    }
  } else {
    dst_search_exit0_task dst_search_exit0_args3;
    dst_search_exit0_args3._cont = args._cont;
    dst_search_exit0_args3.k0 = args.k0;
    dst_search_exit0_args3.ri = args.ri;
    dst_search_exit0_args3.meta = args.meta;
    dst_search_exit0_args3.out = args.out;
    dst_search_exit0_args3.i1 = args.i1;
    taskGlobalOut_dst_search_exit0.write(dst_search_exit0_args3);

  }
}

