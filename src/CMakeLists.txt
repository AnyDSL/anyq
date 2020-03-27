cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

source_group("Impala Files" REGULAR_EXPRESSION "[.]impala$")

function(offline_graph _target)
	set(_infiles ${ARGN})
	list(FILTER _infiles INCLUDE REGEX "[.]impala$")
	anydsl_runtime_wrap(_${_target}_PROGRAM
		CLANG_FLAGS -march=native
		IMPALA_FLAGS --log-level info
		NAME ${_target}
		FILES ${_infiles})

	add_executable(${_target} ${ARGN} ${_${_target}_PROGRAM})
	target_link_libraries(${_target} PRIVATE ${AnyDSL_runtime_LIBRARIES})
	set_target_properties(${_target} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
	# addanydslruntimedllcopy(${_target})
endfunction()

set(COMMON_IMPALA_FILES
	${CMAKE_CURRENT_SOURCE_DIR}/gpu.impala
	${CMAKE_CURRENT_SOURCE_DIR}/framework.impala
	${CMAKE_CURRENT_SOURCE_DIR}/producer_consumer_queue.impala
	${CMAKE_CURRENT_SOURCE_DIR}/utils.impala
)


function(add_schedule_test _testfile _mapping)
	get_filename_component(_testname ${_testfile} NAME_WE)
	set(_target "${_testname}-${_mapping}")
	offline_graph(${_target} ${COMMON_IMPALA_FILES} "${CMAKE_CURRENT_SOURCE_DIR}/mapping_${_mapping}.impala" ${_testfile})
	set_target_properties(${_target} PROPERTIES FOLDER "tests/${_mapping}" )
	if (BUILD_TESTING)
		add_test(NAME "${_target}" COMMAND ${_target} WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
	endif()
endfunction()


if (AnyDSL_runtime_HAS_CUDA_SUPPORT)
	add_schedule_test(test-basics.impala cuda)
	add_schedule_test(test-basics.impala nvvm)
	add_schedule_test(test-producer-consumer-queue.impala cuda)
endif()

if (AnyDSL_runtime_HAS_OPENCL_SUPPORT)
	add_schedule_test(test-basics.impala opencl)
	add_schedule_test(test-producer-consumer-queue.impala opencl)
endif()
