set(ConcurrentProducerConsumerQueue_short_name CPCQ)
set(ConcurrentProducerConsumerQueueGeneric_short_name CPCQg)

set(ConcurrentProducerConsumerQueue_sources ${CMAKE_CURRENT_LIST_DIR}/producer_consumer_queue.art)
set(ConcurrentProducerConsumerQueueGeneric_sources ${ConcurrentProducerConsumerQueue_sources})

set(ConcurrentProducerConsumerQueue_constructor "createConcurrentProducerConsumerQueueGeneric")
set(ConcurrentProducerConsumerQueue_constructor_u32 "createConcurrentProducerConsumerIndexQueue")