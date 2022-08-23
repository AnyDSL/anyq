set(BrokerWorkDistributorQueue_short_name BWD)
set(BrokerWorkDistributorQueueStatic_short_name BWD_static)
set(BrokerWorkDistributorQueueOrig_short_name BWD_orig)
set(BrokerWorkDistributorQueueOrigCUDA_short_name BWD_cuda_orig)
set(BrokerWorkDistributorQueueCUDA_short_name BWD_cuda)
set(BrokerWorkDistributorQueueCUDAIndirect_short_name BWD_cuda_indirect)

set(BrokerWorkDistributorQueue_sources ${CMAKE_CURRENT_LIST_DIR}/broker_queue.art)
set(BrokerWorkDistributorQueueStatic_sources ${CMAKE_CURRENT_LIST_DIR}/broker_queue.art)
set(BrokerWorkDistributorQueueOrig_sources ${CMAKE_CURRENT_LIST_DIR}/broker_queue.art)
set(BrokerWorkDistributorQueueOrigCUDA_sources ${CMAKE_CURRENT_LIST_DIR}/broker_queue_cuda.art)
set(BrokerWorkDistributorQueueCUDA_sources ${CMAKE_CURRENT_LIST_DIR}/broker_queue_cuda.art)
set(BrokerWorkDistributorQueueCUDAIndirect_sources ${CMAKE_CURRENT_LIST_DIR}/broker_queue_cuda_indirect.art)

set(BrokerWorkDistributorQueueOrigCUDA_constructor "")
set(BrokerWorkDistributorQueueOrigCUDA_constructor_u32 "createBrokerWorkDistributorQueueCUDA")
set(BrokerWorkDistributorQueueCUDA_constructor "")
set(BrokerWorkDistributorQueueCUDA_constructor_u32 "createBrokerWorkDistributorQueueCUDA")
set(BrokerWorkDistributorQueueCUDAIndirect_constructor "")
set(BrokerWorkDistributorQueueCUDAIndirect_constructor_u32 "createBrokerWorkDistributorQueueCUDAIndirect")

set(BrokerWorkDistributorQueueOrigCUDA_configure_target BrokerWorkDistributorQueueOrigCUDA_configure)
set(BrokerWorkDistributorQueueCUDA_configure_target BrokerWorkDistributorQueueCUDA_configure)
set(BrokerWorkDistributorQueueCUDAIndirect_configure_target BrokerWorkDistributorQueueCUDAIndirect_configure)

function (BrokerWorkDistributorQueueCUDA_configure_target target patch_includes)
	get_target_property(_bin_dir ${target} ANYDSL_BINARY_DIR)
	get_target_property(_name ${target} NAME)

	set(cuda_src "${_bin_dir}/${_name}")

	add_custom_command(
		OUTPUT ${cuda_src}.ll
		COMMAND ${CMAKE_COMMAND} -E echo "Patching ${cuda_src}.cu"
		COMMAND ${CMAKE_COMMAND} -E rename ${cuda_src}.cu ${cuda_src}.orig.cu
		COMMAND ${CMAKE_COMMAND} -E cat ${patch_includes} ${cuda_src}.orig.cu > ${cuda_src}.patched.cu
		COMMAND ${CMAKE_COMMAND} -E copy ${cuda_src}.patched.cu ${cuda_src}.cu
		DEPENDS ${patch_includes}
		VERBATIM APPEND COMMAND_EXPAND_LISTS
	)
endfunction()

set(BrokerWorkDistributorQueueOrigCUDA_PATCH_INCLUDES "${CMAKE_CURRENT_LIST_DIR}/bwd_orig.h")

function (BrokerWorkDistributorQueueOrigCUDA_configure target)
	BrokerWorkDistributorQueueCUDA_configure_target(${target} ${BrokerWorkDistributorQueueOrigCUDA_PATCH_INCLUDES})
endfunction()

set(BrokerWorkDistributorQueueCUDA_PATCH_INCLUDES "${CMAKE_CURRENT_LIST_DIR}/bwd.h")

function (BrokerWorkDistributorQueueCUDA_configure target)
	BrokerWorkDistributorQueueCUDA_configure_target(${target} ${BrokerWorkDistributorQueueCUDA_PATCH_INCLUDES})
endfunction()

set(BrokerWorkDistributorQueueCUDAIndirect_PATCH_INCLUDES "${CMAKE_CURRENT_LIST_DIR}/bwd_indirect.h")

function (BrokerWorkDistributorQueueCUDAIndirect_configure target)
	BrokerWorkDistributorQueueCUDA_configure_target(${target} ${BrokerWorkDistributorQueueCUDAIndirect_PATCH_INCLUDES})
endfunction()
