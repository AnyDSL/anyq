

set(COMMON_TEST_FILES
	${CMAKE_CURRENT_SOURCE_DIR}/../framework.art
	${CMAKE_CURRENT_SOURCE_DIR}/../utils.art
	${CMAKE_CURRENT_SOURCE_DIR}/../random.art
	${CMAKE_CURRENT_SOURCE_DIR}/test-producer-consumer-queue-common.art
	${CMAKE_CURRENT_SOURCE_DIR}/test-producer-consumer-queue-push-and-then-pop.art
	${CMAKE_CURRENT_SOURCE_DIR}/test-producer-consumer-queue-concurrent-push-pop.art
)

source_group("Source Files" FILES ${COMMON_TEST_FILES} ${CMAKE_CURRENT_SOURCE_DIR}/test-suite.art)


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
	add_anydsl_executable(${_target} ${COMMON_TEST_FILES} ${COMMON_ARTIC_FILES} "${AnyQ_SOURCE_DIR}/mapping_${_mapping}.art" ${_testfile} ${ARGN})
	set_target_properties(${_target} PROPERTIES FOLDER "tests/${_mapping}" )
	set(ADDITIONAL_LINK_DEPENDENCY ${ADDITIONAL_LINK_DEPENDENCY_${_mapping}})
	# message(STATUS "ADDITIONAL_LINK_DEPENDENCY: ${ADDITIONAL_LINK_DEPENDENCY}")
	if(ADDITIONAL_LINK_DEPENDENCY)
		target_link_libraries(${_target} PRIVATE ${ADDITIONAL_LINK_DEPENDENCY})
	endif()
	add_testcase(${_target} ${_testfile} ${_mapping})
	add_dependencies(test-${_mapping} ${_target})
endfunction()


if (Thorin_HAS_RV_SUPPORT AND AnyQ_FIBER_SUPPORT)
	find_package(Python3 COMPONENTS Interpreter)
endif()


set(TEST_SCHEDULE_FILES
	test-basics.art
	test-shuffles.art
	test-barriers.art
	test-queue-sequential-generic.art
	test-queue-sequential-generic-struct.art
	test-queue-sequential-generic-array.art
	test-queue-sequential-index.art
	test-queue-concurrent-generic-small.art
	test-queue-concurrent-generic-large.art
	test-queue-concurrent-index-small.art
	test-queue-concurrent-index-large.art
)

set(TEST_SUITE_FILES
	test-suite-atomic_add.art
	test-suite-atomic_sub.art
	test-suite-atomic_sub_u32.art
	test-suite-atomic_min.art
	test-suite-atomic_max.art
	test-suite-atomic_or.art
	test-suite-atomic_cas.art
	test-suite-atomic_cas-uniform.art
	test-suite-atomic_cas-reverse.art
	test-suite-atomic_exch.art
	test-suite-atomic_exch-uniform.art
	test-suite-atomic_inc.art
	test-suite-memory_barrier.art
	test-suite-scatter_writes.art
	test-suite-shfl_bfly.art
)


foreach(_test_platform ${AnyQ_PLATFORMS})
	add_custom_target(test-${_test_platform})

	foreach(_test_schedule ${TEST_SCHEDULE_FILES})
		add_schedule_test(${_test_schedule} ${_test_platform})
	endforeach()

	foreach(_test_case ${TEST_SUITE_FILES})
		add_schedule_test(${_test_case} ${_test_platform} ${CMAKE_CURRENT_SOURCE_DIR}/test-suite.art)
	endforeach()
endforeach()


if (Thorin_HAS_RV_SUPPORT)
	add_subdirectory(rv-test-suite)
endif()

if (AnyQ_FIBER_SUPPORT)
	add_subdirectory(fiber-test-suite)
endif()