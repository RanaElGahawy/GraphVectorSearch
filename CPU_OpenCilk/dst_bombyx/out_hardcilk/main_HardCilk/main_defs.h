#pragma once
#include <cstdint>
#include <cstring>
#include <stddef.h>
#include <stdint.h>

#define MEM_OUT(mem_port, addr, type, value) \
  *((type(*))((uint8_t *)(mem_port) + (addr))) = (value)
#define MEM_IN(mem_port, addr, type) \
  *((type(*))((uint8_t *)(mem_port) + (addr)))

#define MEM_ARR_OUT(mem_port, addr, idx, type, value) \
  *((type(*))((uint8_t *)(mem_port) + (addr) + (idx) * sizeof(type))) = (value)
#define MEM_ARR_IN(mem_port, addr, idx, type) \
  *((type(*))((uint8_t *)(mem_port) + (addr) + (idx) * sizeof(type)))

#define MEM_STRUCT(mem_port, str, str_type, field) \
    (((str_type*)((uint8_t*)(mem_port) + (str)))->field)
#define MEM_STRUCT_ARR_OUT(mem_port, str, str_type, field, idx, type, value) \
  *((type *)((uint8_t *)(mem_port) + (str) + offsetof(str_type, field) +     \
             (idx) * sizeof(type))) = (value)
#define MEM_STRUCT_ARR_IN(mem_port, str, str_type, field, idx, type)         \
  *((type *)((uint8_t *)(mem_port) + (str) + offsetof(str_type, field) +     \
             (idx) * sizeof(type)))

using namespace std;

using addr_t = uint64_t;

// Continuation tag carried in the high 8 bits of a continuation closure address.
// A task that may send its argument to more than one continuation matches this
// against each candidate continuation's <NAME>_TAG to pick the right port.
#define CONT_TAG(cont) ((uint8_t)((cont) >> 56))

#define DST_SEARCH_REENTRY0_CONT0_TAG 1

struct __attribute__((packed))dst_search_reentry0_cont0_task {
  uint32_t _counter;
  addr_t _cont;
  addr_t q;
  addr_t links;
  uint32_t d;
  addr_t vecs;
  addr_t degree;
  uint32_t ef;
  uint32_t maxDegree;
  uint32_t pending;
  uint32_t active;
  uint32_t cand_cap;
  uint32_t mc;
  uint32_t mg;
  uint32_t k0;
  addr_t rd;
  addr_t ri;
  addr_t cd;
  addr_t ci;
  addr_t bloom;
  addr_t slot_ids;
  addr_t slot_dists;
  addr_t slot_cnt;
  addr_t meta;
  addr_t out;
  uint32_t slot_cap;
  uint32_t head;
  uint32_t tail;
  uint32_t s;
  uint32_t popped;
  uint32_t i0;
  uint32_t i1;
  uint8_t _padding[68];
};

struct dst_search_reentry0_cont0_spawn_next {
  addr_t addr;
  uint8_t data[64];
  uint32_t size;
  uint32_t allow;
  uint8_t _padding[48];
};

struct __attribute__((packed))dst_search_exit0_task {
  addr_t _cont;
  uint32_t k0;
  addr_t ri;
  addr_t meta;
  addr_t out;
  uint32_t i1;
  uint8_t _padding[24];
};

struct __attribute__((packed)) uint32_t_arg_out {
  addr_t addr;
  uint32_t data;
  uint32_t size;
  uint32_t allow;
  uint8_t _padding[12];
};

struct __attribute__((packed))dist_task_task {
  addr_t _cont;
  addr_t q;
  addr_t vecs;
  uint32_t d;
  addr_t ids;
  addr_t dists;
  uint32_t i;
  uint8_t _padding[16];
};

struct __attribute__((packed)) float_arg_out {
  addr_t addr;
  float data;
  uint32_t size;
  uint32_t allow;
  uint8_t _padding[12];
};

struct __attribute__((packed))dst_search_task {
  addr_t _cont;
  addr_t q;
  addr_t vecs;
  addr_t degree;
  addr_t links;
  uint32_t d;
  uint32_t maxDegree;
  uint32_t entry;
  uint32_t ef;
  uint32_t cand_cap;
  uint32_t mc;
  uint32_t mg;
  uint32_t k0;
  addr_t rd;
  addr_t ri;
  addr_t cd;
  addr_t ci;
  addr_t bloom;
  addr_t slot_ids;
  addr_t slot_dists;
  addr_t slot_cnt;
  addr_t meta;
  addr_t out;
  uint8_t _padding[104];
};

struct __attribute__((packed))dst_search_reentry0_task {
  addr_t _cont;
  addr_t q;
  addr_t vecs;
  addr_t degree;
  addr_t links;
  uint32_t d;
  uint32_t maxDegree;
  uint32_t ef;
  uint32_t cand_cap;
  uint32_t mc;
  uint32_t mg;
  uint32_t k0;
  addr_t rd;
  addr_t ri;
  addr_t cd;
  addr_t ci;
  addr_t bloom;
  addr_t slot_ids;
  addr_t slot_dists;
  addr_t slot_cnt;
  addr_t meta;
  addr_t out;
  uint32_t slot_cap;
  uint32_t head;
  uint32_t tail;
  uint32_t pending;
  uint32_t active;
  uint32_t s;
  uint32_t popped;
  uint32_t i0;
  uint32_t i1;
  uint8_t _padding[72];
};

