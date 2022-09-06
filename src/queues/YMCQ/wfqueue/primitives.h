/** @file */

#ifndef PRIMITIVES_H
#define PRIMITIVES_H

#if defined(__GNUC__) && __GNUC__ >= 4 && __GNUC_MINOR__ > 7
/**
 * An atomic fetch-and-add.
 */
#define FAA(ptr, val) __atomic_fetch_add(ptr, val, __ATOMIC_RELAXED)
#define FAA_LONG FAA
/**
 * An atomic fetch-and-add that also ensures sequential consistency.
 */
#define FAAcs(ptr, val) __atomic_fetch_add(ptr, val, __ATOMIC_SEQ_CST)
#define FAAcs_LONG FAAcs
/**
 * An atomic compare-and-swap.
 */
#define CAS(ptr, cmp, val) __atomic_compare_exchange_n(ptr, cmp, val, 0, \
    __ATOMIC_RELAXED, __ATOMIC_RELAXED)
#define CAS_LONG CAS
#define CAS_PTR CAS
/**
 * An atomic compare-and-swap that also ensures sequential consistency.
 */
#define CAScs(ptr, cmp, val) __atomic_compare_exchange_n(ptr, cmp, val, 0, \
    __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)
/**
#define CAScs_PTR CAScs
 * An atomic compare-and-swap that ensures release semantic when succeed
 * or acquire semantic when failed.
 */
#define CASra(ptr, cmp, val) __atomic_compare_exchange_n(ptr, cmp, val, 0, \
    __ATOMIC_RELEASE, __ATOMIC_ACQUIRE)
/**
#define CASra_LONG CASra
#define CASra_PTR CASra
 * An atomic compare-and-swap that ensures acquire semantic when succeed
 * or relaxed semantic when failed.
 */
#define CASa(ptr, cmp, val) __atomic_compare_exchange_n(ptr, cmp, val, 0, \
    __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)
#define CASa_LONG CASa

/**
 * An atomic swap.
 */
#define SWAP(ptr, val) __atomic_exchange_n(ptr, val, __ATOMIC_RELAXED)

/**
 * An atomic swap that ensures acquire release semantics.
 */
#define SWAPra(ptr, val) __atomic_exchange_n(ptr, val, __ATOMIC_ACQ_REL)

/**
 * A memory fence to ensure sequential consistency.
 */
#define FENCE() __atomic_thread_fence(__ATOMIC_SEQ_CST)

/**
 * An atomic store.
 */
#define STORE(ptr, val) __atomic_store_n(ptr, val, __ATOMIC_RELAXED)

/**
 * A store with a preceding release fence to ensure all previous load
 * and stores completes before the current store is visiable.
 */
#define RELEASE(ptr, val) __atomic_store_n(ptr, val, __ATOMIC_RELEASE)
#define RELEASE_PTR RELEASE
#define RELEASE_LONG RELEASE
#define RELEASE_LONGLLONG RELEASE

/**
 * A load with a following acquire fence to ensure no following load and
 * stores can start before the current load completes.
 */
#define ACQUIRE(ptr) __atomic_load_n(ptr, __ATOMIC_ACQUIRE)
#define ACQUIRE_PTR ACQUIRE
#define ACQUIRE_LONG ACQUIRE
#define ACQUIRE_LONGLLONG ACQUIRE

#else /** Non-GCC or old GCC. */
#if defined(__x86_64__) || defined(_M_X64_)

#define FAA __sync_fetch_and_add
#define FAAcs __sync_fetch_and_add

static inline int
_compare_and_swap(void ** ptr, void ** expected, void * desired) {
  void * oldval = *expected;
  void * newval = __sync_val_compare_and_swap(ptr, oldval, desired);

  if (newval == oldval) {
    return 1;
  } else {
    *expected = newval;
    return 0;
  }
}
#define CAS(ptr, expected, desired) \
  _compare_and_swap((void **) (ptr), (void **) (expected), (void *) (desired))
#define CAScs CAS
#define CASra CAS
#define CASa  CAS

#define SWAP __sync_lock_test_and_set
#define SWAPra SWAP

#define ACQUIRE(p) ({ \
  __typeof__(*(p)) __ret = *p; \
  __asm__("":::"memory"); \
  __ret; \
})

#define RELEASE(p, v) do {\
  __asm__("":::"memory"); \
  *p = v; \
} while (0)
#define FENCE() __sync_synchronize()

#endif

#ifdef _MSC_VER

#include <intrin.h>

#define FAA_LONG(ptr, val) _InterlockedExchangeAdd((volatile long*)ptr, val)
#define FAAcs_LONG FAA_LONG

static inline int
_compare_and_swap_long(volatile long* ptr, long* expected, long desired) {
    long oldval = *expected;
    long newval = _InterlockedCompareExchange(ptr, desired, oldval);

    if (newval == oldval) {
        return 1;
    }
    else {
        *expected = newval;
        return 0;
    }
}

static inline int
_compare_and_swap_ptr(void* volatile * ptr, void** expected, void* desired) {
    void* oldval = *expected;
    void* newval = (void*)_InterlockedCompareExchange64((volatile long long*)ptr, (long long)desired, (long long)oldval);

    if (newval == oldval) {
        return 1;
    }
    else {
        *expected = newval;
        return 0;
    }
}

#define CAS_LONG(ptr, cmp, val) _compare_and_swap_long(ptr, cmp, val)
#define CAS_PTR(ptr, cmp, val) _compare_and_swap_ptr(ptr, cmp, val)
#define CAScs_PTR CAS_PTR
#define CASra_LONG CAS_LONG
#define CASra_PTR CAS_PTR
#define CASa_LONG CAS_LONG

#define ACQUIRE_LONG(ptr) _InterlockedExchangeAdd((volatile long*)ptr, 0)
#define ACQUIRE_LONGLONG(ptr) _InterlockedExchangeAdd64((volatile long long*)ptr, 0)
#define ACQUIRE_PTR(ptr) (void*)_InterlockedExchangeAdd64((volatile long long*)ptr, 0)

#define RELEASE_LONG(ptr, val) _InterlockedExchange(ptr, val)
#define RELEASE_LONGLONG(ptr, val) _InterlockedExchange64(ptr, val)
#define RELEASE_PTR(ptr, val) (void*)_InterlockedExchange64((volatile long long*)ptr, (long long)val)
#define FENCE() // noop

#endif

#endif

#if defined(__x86_64__) || defined(_M_X64_)
#define PAUSE() __asm__ ("pause")

static inline
int _CAS2(volatile long * ptr, long * cmp1, long * cmp2, long val1, long val2)
{
  char success;
  long tmp1 = *cmp1;
  long tmp2 = *cmp2;

  __asm__ __volatile__(
      "lock cmpxchg16b %1\n"
      "setz %0"
      : "=q" (success), "+m" (*ptr), "+a" (tmp1), "+d" (tmp2)
      : "b" (val1), "c" (val2)
      : "cc" );

  *cmp1 = tmp1;
  *cmp2 = tmp2;
  return success;
}
#define CAS2(p, o1, o2, n1, n2) \
  _CAS2((volatile long *) p, (long *) o1, (long *) o2, (long) n1, (long) n2)

#define BTAS(ptr, bit) ({ \
  char __ret; \
  __asm__ __volatile__( \
      "lock btsq %2, %0; setnc %1" \
      : "+m" (*ptr), "=r" (__ret) : "ri" (bit) : "cc" ); \
  __ret; \
})

#else
#define PAUSE()
#endif

#endif /* end of include guard: PRIMITIVES_H */
