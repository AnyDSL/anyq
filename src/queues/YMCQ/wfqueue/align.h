#ifndef ALIGN_H
#define ALIGN_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _MSC_VER
#include <malloc.h>
#endif

#define PAGE_SIZE 4096
#define CACHE_LINE_SIZE 64
#define DOUBLE_CACHE_LINE_SIZE 128
#ifdef _MSC_VER
//#define CACHE_ALIGNED __declspec(align(CACHE_LINE_SIZE))
//#define DOUBLE_CACHE_ALIGNED __declspec(align(DOUBLE_CACHE_LINE_SIZE))
#define CACHE_ALIGNED
#define DOUBLE_CACHE_ALIGNED
#else
#define CACHE_ALIGNED __attribute__((aligned(CACHE_LINE_SIZE)))
#define DOUBLE_CACHE_ALIGNED __attribute__((aligned(2 * CACHE_LINE_SIZE)))
#endif

static inline void * align_malloc(size_t align, size_t size)
{
#ifdef _MSC_VER
  return _aligned_malloc(size, align);
#else
  return aligned_alloc(align, size);
#endif

  // void * ptr;

  // int ret = posix_memalign(&ptr, align, size);
  // if (ret != 0) {
  //   fprintf(stderr, strerror(ret));
  //   abort();
  // }

  // return ptr;
}

static inline void align_free(void* ptr)
{
#ifdef _MSC_VER
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

#endif /* end of include guard: ALIGN_H */
