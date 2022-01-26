
add_library(moodycamel STATIC
	${CMAKE_CURRENT_LIST_DIR}/moodycamel/concurrentqueue.h
	${CMAKE_CURRENT_LIST_DIR}/moodycamel/c_api/concurrentqueue.h
	${CMAKE_CURRENT_LIST_DIR}/moodycamel/c_api/concurrentqueue.cpp
)

target_compile_definitions(moodycamel PRIVATE MOODYCAMEL_STATIC)


set(queue_types_cpu ConcurrentProducerConsumerQueue MichaelScottQueue MoodyCamelQueue)
set(queue_types_fiberless ConcurrentProducerConsumerQueue MichaelScottQueue MoodyCamelQueue)
set(queue_types_cuda )
set(queue_types_nvvm ConcurrentProducerConsumerQueue MichaelScottQueue)
set(queue_types_amdgpu ConcurrentProducerConsumerQueue MichaelScottQueue)

set(ConcurrentProducerConsumerQueue_short_name CPCQ)
set(MichaelScottQueue_short_name MSQ)
set(MoodyCamelQueue_short_name MOQ)
set(BrokerWorkDistributorQueueCUDA_short_name BWD)

set(ConcurrentProducerConsumerQueue_sources ${CMAKE_CURRENT_LIST_DIR}/producer_consumer_queue.art)
set(MichaelScottQueue_sources ${CMAKE_CURRENT_LIST_DIR}/michael_scott.art)
set(MoodyCamelQueue_sources ${CMAKE_CURRENT_LIST_DIR}/moodycamel.art)
set(BrokerWorkDistributorQueueCUDA_sources ${CMAKE_CURRENT_LIST_DIR}/broker_queue_cuda.art)

set(MoodyCamelQueue_configure_target MoodyCamelQueue_configure)

function (MoodyCamelQueue_configure target)
	target_link_libraries(${target} PRIVATE moodycamel)
endfunction()
