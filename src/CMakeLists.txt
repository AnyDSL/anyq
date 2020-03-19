cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

source_group("Impala Files" REGULAR_EXPRESSION "[.]impala$")

function(offline_graph _target)
	set(_infiles ${ARGN})
	list(FILTER _infiles INCLUDE REGEX "[.]impala$")
	anydsl_runtime_wrap(_${_target}_PROGRAM
		CLANG_FLAGS -march=native
		IMPALA_FLAGS --log-level info
		FILES ${_infiles})

	add_executable(${_target} ${ARGN} ${_${_target}_PROGRAM})
	target_link_libraries(${_target} PRIVATE ${AnyDSL_runtime_LIBRARIES})
	set_target_properties(${_target} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
	# addanydslruntimedllcopy(${_target})
endfunction()

set(COMMON_IMPALA_FILES
	${CMAKE_CURRENT_SOURCE_DIR}/cuda_utils.impala
	${CMAKE_CURRENT_SOURCE_DIR}/framework.impala
	${CMAKE_CURRENT_SOURCE_DIR}/cuda_producer_consumer_queue.impala
	${CMAKE_CURRENT_SOURCE_DIR}/utils.impala
)


option(BUILD_TESTING OFF)
if(BUILD_TESTING)
	include(CTest)
endif()

function(add_schedule_test _testfile)
	get_filename_component(_testname ${_testfile} NAME_WE)
	set(_target ${_testname})
	offline_graph(${_target} ${COMMON_IMPALA_FILES} ${_testfile})
	set_target_properties(${_target} PROPERTIES FOLDER "tests" )
	if(BUILD_TESTING)
		add_test(NAME "${_testfile}" COMMAND ${_target} WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
	endif()
endfunction()


add_schedule_test(test-cuda-producer-consumer-queue.impala)
add_schedule_test(test-cuda-utils.impala)
