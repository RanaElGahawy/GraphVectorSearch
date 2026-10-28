#pragma once
#include <assert.h>
#include <cilk/cilk.h>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>
#include <vector>

struct closure;
class spawn_write_dest;

typedef void (*task_fn_t)(std::shared_ptr<closure> args);

struct cont {
private:
  spawn_write_dest *mySN;

public:
  closure *cls;
  void *ret;

  cont() : mySN(nullptr), cls(nullptr), ret(nullptr) {}
  void init(spawn_write_dest *mySN1, closure *cls1) {
    mySN = mySN1;
    cls = cls1;
    ret = nullptr;
  }

  spawn_write_dest *getSN() const { return mySN; };
};

class closure {
public:
  cont k;

  virtual task_fn_t getTask() { return nullptr; };
  closure(cont k) : k(k) {}
};

class spawn_write_dest {
public:
  std::vector<std::shared_ptr<closure>> toSpawn;
  int jc;

  spawn_write_dest() : jc(0) {}

  void child_done() { jc--; }
  void put_spawn() { jc++; }
};

void taskSpawn(task_fn_t fn, std::shared_ptr<closure> cls) { fn(cls); }

template <class C> class spawn_next : public spawn_write_dest {
public:
  std::shared_ptr<C> cls;

  spawn_next(const C &incls) {
    static_assert(std::is_base_of<closure, C>::value,
                  "spawn next parameter should be derived from closure.");
    C *cls_temp = new C(incls);
    cls = std::shared_ptr<C>(cls_temp);
  }

  ~spawn_next() {

    for (auto &sp : toSpawn) {
      cilk_spawn taskSpawn(sp->getTask(), sp);
    }
    cilk_sync;

    cilk_spawn taskSpawn(cls->getTask(), cls);
  }
};

template <class C> class spawn {
public:
  spawn(const C &cls) {
    static_assert(std::is_base_of<closure, C>::value,
                  "spawn parameter must derive from closure");

    auto *sn = cls.k.getSN();
    if (!sn) {
      printf("unsupported\n");
      exit(1);
    }

    auto newCls = std::make_shared<C>(cls);
    sn->put_spawn();
    sn->toSpawn.push_back(newCls);
  }
};

#define SEND_ARGUMENT(k, n)                                                    \
  {                                                                            \
    if ((k).ret)                                                               \
      *((typeof(n) *)((k).ret)) = (n);                                         \
    if ((k).getSN())                                                           \
      (k).getSN()->child_done();                                               \
    return;                                                                    \
  }

#define SN_BIND(sn, k, field)                                                  \
  {                                                                            \
    assert(sn.cls);                                                            \
    (k)->init(&sn, sn.cls.get());                                              \
    (k)->ret = (void *)&(sn.cls->field);                                       \
  }
#define SN_BIND_EXT(sn, k, ptr)                                                \
  {                                                                            \
    assert(sn.cls);                                                            \
    (k)->init(&sn, sn.cls.get());                                              \
    (k)->ret = (void *)ptr;                                                    \
  }
#define SN_BIND_VOID(sn, k)                                                    \
  {                                                                            \
    assert(sn.cls);                                                            \
    (k)->init(&sn, sn.cls.get());                                              \
    (k)->ret = nullptr;                                                        \
  }
#define THREAD(fn_name) void fn_name(std::shared_ptr<closure> args)
#define CLOSURE_DEF(name, ...)                                                 \
  struct name##_closure : public closure {                                     \
    __VA_ARGS__                                                                \
    using closure::closure;                                                    \
    task_fn_t getTask() override { return &name; }                             \
  };
#define CONT_DUMMY (cont{})

#define CLOSURE_DEF_SHARED(name, data_type)                                    \
  struct name##_closure : public closure, public data_type {                   \
    name##_closure(cont k, const data_type &d) : closure(k), data_type(d) {}   \
    using closure::closure;                                                    \
    task_fn_t getTask() override { return &name; }                             \
  };