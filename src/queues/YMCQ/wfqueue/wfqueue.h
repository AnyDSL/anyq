#ifndef WFQUEUE_H
#define WFQUEUE_H

#ifdef WFQUEUE

#include "align.h"
#define EMPTY ((void *) 0)

#ifndef WFQUEUE_NODE_SIZE
#define WFQUEUE_NODE_SIZE ((1 << 10) - 2)
#endif

struct _enq_t {
  long volatile id;
  void * volatile val;
} CACHE_ALIGNED;

struct _deq_t {
  long volatile id;
  long volatile idx;
} CACHE_ALIGNED;

struct _cell_t {
  void * volatile val;
  struct _enq_t * volatile enq;
  struct _deq_t * volatile deq;
  void * pad[5];
};

struct _node_t {
  CACHE_ALIGNED struct _node_t * volatile next;
  CACHE_ALIGNED long id;
  CACHE_ALIGNED struct _cell_t cells[WFQUEUE_NODE_SIZE];
};

typedef struct DOUBLE_CACHE_ALIGNED {
  /**
   * Index of the next position for enqueue.
   */
  DOUBLE_CACHE_ALIGNED volatile long Ei;

  /**
   * Index of the next position for dequeue.
   */
  DOUBLE_CACHE_ALIGNED volatile long Di;

  /**
   * Index of the head of the queue.
   */
  DOUBLE_CACHE_ALIGNED volatile long Hi;

  /**
   * Pointer to the head node of the queue.
   */
  struct _node_t * volatile Hp;

  /**
   * Number of processors.
   */
  long nprocs;
#ifdef RECORD
  long slowenq;
  long slowdeq;
  long fastenq;
  long fastdeq;
  long empty;
#endif
} queue_t;

typedef struct _handle_t {
  /**
   * Pointer to the next handle.
   */
  struct _handle_t * next;

  /**
   * Hazard pointer.
   */
  struct _node_t * volatile Hp;

  /**
   * Pointer to the node for enqueue.
   */
  struct _node_t * volatile Ep;

  /**
   * Pointer to the node for dequeue.
   */
  struct _node_t * volatile Dp;

  /**
   * Enqueue request.
   */
  CACHE_ALIGNED struct _enq_t Er;

  /**
   * Dequeue request.
   */
  CACHE_ALIGNED struct _deq_t Dr;

  /**
   * Handle of the next enqueuer to help.
   */
  CACHE_ALIGNED struct _handle_t * Eh;

  long Ei;

  /**
   * Handle of the next dequeuer to help.
   */
  struct _handle_t * Dh;

  /**
   * Pointer to a spare node to use, to speedup adding a new node.
   */
  CACHE_ALIGNED struct _node_t * spare;

  /**
   * Count the delay rounds of helping another dequeuer.
   */
  int delay;

#ifdef RECORD
  long slowenq;
  long slowdeq;
  long fastenq;
  long fastdeq;
  long empty;
#endif
} handle_t;

#endif

#endif /* end of include guard: WFQUEUE_H */
