
add_library(moodycamel STATIC
	${CMAKE_CURRENT_LIST_DIR}/moodycamel/concurrentqueue.h
	${CMAKE_CURRENT_LIST_DIR}/moodycamel/c_api/concurrentqueue.h
	${CMAKE_CURRENT_LIST_DIR}/moodycamel/c_api/concurrentqueue.cpp
)

target_compile_definitions(moodycamel PRIVATE MOODYCAMEL_STATIC)


set(queue_types_cpu ConcurrentProducerConsumerQueue MichaelScottQueue MoodyCamelQueue)
set(queue_types_fiberless ConcurrentProducerConsumerQueue MichaelScottQueue MoodyCamelQueue)
set(queue_types_cuda BrokerWorkDistributorQueueCUDA)
set(queue_types_nvvm ConcurrentProducerConsumerQueue ConcurrentProducerConsumerQueueGeneric MichaelScottQueue)
set(queue_types_amdgpu ConcurrentProducerConsumerQueue MichaelScottQueue)

set(ConcurrentProducerConsumerQueue_short_name CPCQ)
set(ConcurrentProducerConsumerQueueGeneric_short_name CPCQ_g)
set(MichaelScottQueue_short_name MSQ)
set(MoodyCamelQueue_short_name MOQ)
set(BrokerWorkDistributorQueueCUDA_short_name BWD)

set(ConcurrentProducerConsumerQueue_sources ${CMAKE_CURRENT_LIST_DIR}/producer_consumer_queue.art)
set(ConcurrentProducerConsumerQueueGeneric_sources ${ConcurrentProducerConsumerQueue_sources})
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
	get_target_property(bin_dir ${target} BINARY_DIR)
	get_target_property(name ${target} NAME)

	set(cuda_src "${bin_dir}/${name}.cu")
	set(patched_cuda_src "${bin_dir}/${name}.patched.cu")

	add_custom_command(
		TARGET ${target}
		POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E rename ${cuda_src} ${cuda_src}.orig
		COMMAND ${CMAKE_COMMAND} -E cat ${BrokerWorkDistributorQueueCUDA_PATCH_INCLUDES} ${cuda_src}.orig > ${patched_cuda_src}
		COMMAND ${CMAKE_COMMAND} -E copy ${patched_cuda_src} ${cuda_src}
	)

	set_property(TARGET ${target} PROPERTY LINK_DEPENDS ${BrokerWorkDistributorQueueCUDA_PATCH_INCLUDES})
endfunction()
