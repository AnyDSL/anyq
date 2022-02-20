
add_library(moodycamel STATIC
	${CMAKE_CURRENT_LIST_DIR}/moodycamel/concurrentqueue.h
	${CMAKE_CURRENT_LIST_DIR}/moodycamel/c_api/concurrentqueue.h
	${CMAKE_CURRENT_LIST_DIR}/moodycamel/c_api/concurrentqueue.cpp
)

target_compile_definitions(moodycamel PRIVATE MOODYCAMEL_STATIC)


set(queue_types_cpu ConcurrentProducerConsumerQueue BrokerWorkDistributorQueue MichaelScottQueue MoodyCamelQueue)
set(queue_types_fiberless ConcurrentProducerConsumerQueue BrokerWorkDistributorQueue MichaelScottQueue MoodyCamelQueue)
set(queue_types_cuda ConcurrentProducerConsumerQueue BrokerWorkDistributorQueue BrokerWorkDistributorQueueCUDA)
set(queue_types_nvvm ConcurrentProducerConsumerQueue BrokerWorkDistributorQueue MichaelScottQueue)
set(queue_types_amdgpu ConcurrentProducerConsumerQueue BrokerWorkDistributorQueue MichaelScottQueue)

set(ConcurrentProducerConsumerQueue_short_name CPCQ)
set(ConcurrentProducerConsumerQueueGeneric_short_name CPCQ_g)
set(BrokerWorkDistributorQueue_short_name BWD)
set(MichaelScottQueue_short_name MSQ)
set(MoodyCamelQueue_short_name MOQ)
set(BrokerWorkDistributorQueueCUDA_short_name BWD_cuda)

set(ConcurrentProducerConsumerQueue_sources ${CMAKE_CURRENT_LIST_DIR}/producer_consumer_queue.art)
set(ConcurrentProducerConsumerQueueGeneric_sources ${ConcurrentProducerConsumerQueue_sources})
set(BrokerWorkDistributorQueue_sources ${CMAKE_CURRENT_LIST_DIR}/broker_queue.art)
set(MichaelScottQueue_sources ${CMAKE_CURRENT_LIST_DIR}/michael_scott.art)
set(MoodyCamelQueue_sources ${CMAKE_CURRENT_LIST_DIR}/moodycamel.art)
set(BrokerWorkDistributorQueueCUDA_sources ${CMAKE_CURRENT_LIST_DIR}/broker_queue_cuda.art)

set(MoodyCamelQueue_configure_target MoodyCamelQueue_configure)

function (MoodyCamelQueue_configure target)
	target_link_libraries(${target} PRIVATE moodycamel)
endfunction()


set(BrokerWorkDistributorQueueCUDA_configure_target BrokerWorkDistributorQueueCUDA_configure)

set(BrokerWorkDistributorQueueCUDA_PATCH_INCLUDES "${CMAKE_CURRENT_LIST_DIR}/bwd.h")

function (BrokerWorkDistributorQueueCUDA_configure target)
	set(cuda_src "$<TARGET_PROPERTY:BINARY_DIR>/${CMAKE_CFG_INTDIR}/$<TARGET_PROPERTY:NAME>")

	add_custom_command(
		TARGET ${target}
		POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E echo "Patching ${cuda_src}.cu"
		COMMAND ${CMAKE_COMMAND} -E rename ${cuda_src}.cu ${cuda_src}.orig.cu
		COMMAND ${CMAKE_COMMAND} -E cat ${BrokerWorkDistributorQueueCUDA_PATCH_INCLUDES} ${cuda_src}.orig.cu > ${cuda_src}.patched.cu
		COMMAND ${CMAKE_COMMAND} -E copy ${cuda_src}.patched.cu ${cuda_src}.cu
		VERBATIM COMMAND_EXPAND_LISTS
	)

	set_property(TARGET ${target} PROPERTY LINK_DEPENDS ${BrokerWorkDistributorQueueCUDA_PATCH_INCLUDES})
endfunction()
