
function(offline_graph _target)
	set(_infiles ${ARGN})
	list(FILTER _infiles INCLUDE REGEX "[.]art$")
	anydsl_runtime_wrap(_${_target}_PROGRAM
		FRONTEND artic
		CLANG_FLAGS -march=native
		ARTIC_FLAGS --log-level info --max-errors 5
		NAME ${_target}
		FILES ${_infiles})

	add_executable(${_target} ${ARGN} ${_${_target}_PROGRAM})
	add_anydsl_runtime(${_target})
	set_target_properties(${_target} PROPERTIES
		LINKER_LANGUAGE CXX
		VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
	)
endfunction()

set(COMMON_ARTIC_FILES
	${CMAKE_CURRENT_SOURCE_DIR}/gpu.art
	${CMAKE_CURRENT_SOURCE_DIR}/producer_consumer_queue.art
	# ${CMAKE_CURRENT_SOURCE_DIR}/ringbuffer_allocator.art
	# ${CMAKE_CURRENT_SOURCE_DIR}/pipeline.art
)

set(COMMON_TEST_FILES
	${CMAKE_CURRENT_SOURCE_DIR}/framework.art
	${CMAKE_CURRENT_SOURCE_DIR}/test-producer-consumer-queue-common.art
	${CMAKE_CURRENT_SOURCE_DIR}/demo.art
	${CMAKE_CURRENT_SOURCE_DIR}/utils.art
)

source_group("Source Files" FILES ${COMMON_ARTIC_FILES} ${COMMON_TEST_FILES} ${CMAKE_CURRENT_SOURCE_DIR}/test-suite.art)
source_group("Runtime Files" REGULAR_EXPRESSION ".*/runtime/platforms/.+[.](impala|art)$")

find_package(Boost 1.65 COMPONENTS fiber)

if (Boost_FOUND)
	find_package(Threads REQUIRED)
	add_library(fiber-support STATIC fibers.h fibers.cpp)
	target_include_directories(fiber-support
		PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
		PRIVATE ${Boost_INCLUDE_DIRS}
	)
	target_link_libraries(fiber-support PRIVATE ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

	set(AnyQ_FIBER_SUPPORT TRUE)
	set(ADDITIONAL_LINK_DEPENDENCY_cpu fiber-support)
else()
	message(STATUS "Boost is required for fiber-support used in the CPU mapping.")
endif()

set(AnyQ_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

function(add_testcase _target _testfile _mapping)
	if(_mapping STREQUAL "cpu" AND Python3_FOUND)
		# message(STATUS "add advanced test for ${_testfile}")
		get_filename_component(_srcfile ${_testfile} ABSOLUTE)
		get_target_property(_source_files ${_target} SOURCES)
		list(FILTER _source_files INCLUDE REGEX "\.o$")
		list(GET _source_files 0 _obj_file)
		get_filename_component(_llvmname ${_obj_file} NAME_WE)
		set(_llvmfile ${_llvmname}.ll)
		# message(STATUS "${_llvmfile}")
		set(_test_command
			Python3::Interpreter ${AnyQ_SOURCE_DIR}/perform-cpu-test.py
			--src-file ${_srcfile}
			--config $<CONFIG>
			--llvm-file ${_llvmfile}
			--bin-file $<TARGET_FILE:${_target}>
		)
	else()
		set(_test_command ${_target})
	endif()
	add_test(NAME "${_target}" COMMAND ${_test_command} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endfunction()

function(add_schedule_test _testfile _mapping)
	get_filename_component(_testname ${_testfile} NAME_WE)
	set(_target "${_testname}-${_mapping}")
	offline_graph(${_target} ${COMMON_TEST_FILES} ${COMMON_ARTIC_FILES} "${AnyQ_SOURCE_DIR}/mapping_${_mapping}.art" ${_testfile} ${ARGN})
	set_target_properties(${_target} PROPERTIES FOLDER "tests/${_mapping}" )
	set(ADDITIONAL_LINK_DEPENDENCY ${ADDITIONAL_LINK_DEPENDENCY_${_mapping}})
	# message(STATUS "ADDITIONAL_LINK_DEPENDENCY: ${ADDITIONAL_LINK_DEPENDENCY}")
	if(ADDITIONAL_LINK_DEPENDENCY)
		target_link_libraries(${_target} PRIVATE ${ADDITIONAL_LINK_DEPENDENCY})
	endif()
	if (BUILD_TESTING)
		add_testcase(${_target} ${_testfile} ${_mapping})
	endif()
endfunction()

set(TEST_PLATFORMS)

if (AnyDSL_runtime_HAS_CUDA_SUPPORT)
	list(APPEND TEST_PLATFORMS cuda nvvm)
	# add_schedule_test(test-ringbuffer-allocator.art cuda)
	# add_schedule_test(test-ringbuffer-allocator.art nvvm)
	# add_schedule_test(test-pipeline.art cuda)
	# add_schedule_test(test-pipeline.art nvvm)
endif()

if (AnyDSL_runtime_HAS_HSA_SUPPORT)
	list(APPEND TEST_PLATFORMS amdgpu)
endif()

if (Thorin_HAS_RV_SUPPORT AND AnyQ_FIBER_SUPPORT)
	list(APPEND TEST_PLATFORMS cpu)
	# add_schedule_test(test-rv-intrinsics.art cpu)
	# offline_graph(test-rv-atomic-minimal test-rv-atomic-minimal.art)
	find_package(Python3 COMPONENTS Interpreter)
endif()

set(TEST_SCHEDULE_FILES
	test-basics.art
	test-shuffles.art
	test-barriers.art
	test-producer-consumer-queue-push-pop-generic.art
	test-producer-consumer-queue-push-pop-generic-struct.art
	test-producer-consumer-queue-push-pop-index.art
)

set(TEST_SUITE_FILES
	test-suite-atomic_add.art
	test-suite-atomic_sub.art
	test-suite-atomic_sub_u32.art
	test-suite-atomic_min.art
	test-suite-atomic_max.art
	test-suite-atomic_or.art
	test-suite-atomic_cas.art
	test-suite-atomic_exch.art
	test-suite-atomic_exch-uniform.art
	test-suite-atomic_inc.art
	test-suite-scatter_writes.art
	test-suite-shfl_bfly.art
)

foreach(_test_platform ${TEST_PLATFORMS})
	foreach(_test_schedule ${TEST_SCHEDULE_FILES})
		add_schedule_test(${_test_schedule} ${_test_platform})
	endforeach()

	foreach(_test_case ${TEST_SUITE_FILES})
		add_schedule_test(${_test_case} ${_test_platform} ${AnyQ_SOURCE_DIR}/test-suite.art)
	endforeach()
endforeach()


if (Thorin_HAS_RV_SUPPORT)
	add_subdirectory(rv-test-suite)
endif()

if (AnyQ_FIBER_SUPPORT)
	add_subdirectory(fiber-test-suite)
endif()
