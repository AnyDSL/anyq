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
	${CMAKE_CURRENT_SOURCE_DIR}/pipeline.impala
	${CMAKE_CURRENT_SOURCE_DIR}/demo.impala
	${CMAKE_CURRENT_SOURCE_DIR}/utils.impala
)

find_package(Boost 1.65 COMPONENTS fiber)

if (Boost_FOUND)
	find_package(Threads REQUIRED)
	add_library(fiber-support STATIC fibers.h fibers.cpp)
	set_target_properties(fiber-support PROPERTIES INCLUDE_DIRECTORIES ${Boost_INCLUDE_DIRS})
	target_link_libraries(fiber-support PRIVATE Boost::fiber ${CMAKE_THREAD_LIBS_INIT})

	add_executable(fiber-test test-fibers.cpp)
	set_target_properties(fiber-test PROPERTIES
		FOLDER "tests/cpu"
		VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
	target_link_libraries(fiber-test PRIVATE fiber-support)

	offline_graph(fiber-integration fibers.impala)
	set_target_properties(fiber-integration PROPERTIES FOLDER "tests/cpu")
	target_link_libraries(fiber-integration PRIVATE fiber-support)

	set(AnyQ_FIBER_SUPPORT TRUE)
	set(ADDITIONAL_LINK_DEPENDENCY_cpu fiber-support)
else()
	message(STATUS "Boost is required for fiber-support used in the CPU mapping.")
endif()


function(add_schedule_test _testfile _mapping)
	get_filename_component(_testname ${_testfile} NAME_WE)
	set(_target "${_testname}-${_mapping}")
	offline_graph(${_target} ${COMMON_IMPALA_FILES} "${CMAKE_CURRENT_SOURCE_DIR}/mapping_${_mapping}.impala" ${_testfile})
	set_target_properties(${_target} PROPERTIES FOLDER "tests/${_mapping}" )
	set(ADDITIONAL_LINK_DEPENDENCY ${ADDITIONAL_LINK_DEPENDENCY_${_mapping}})
	# message(STATUS "ADDITIONAL_LINK_DEPENDENCY: ${ADDITIONAL_LINK_DEPENDENCY}")
	if(ADDITIONAL_LINK_DEPENDENCY)
		target_link_libraries(${_target} PRIVATE ${ADDITIONAL_LINK_DEPENDENCY})
	endif()
	if (BUILD_TESTING)
		add_test(NAME "${_target}" COMMAND ${_target} WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
	endif()
endfunction()


if (AnyDSL_runtime_HAS_CUDA_SUPPORT)
	add_schedule_test(test-basics.impala cuda)
	add_schedule_test(test-basics.impala nvvm)
	add_schedule_test(test-shuffles.impala cuda)
	add_schedule_test(test-shuffles.impala nvvm)
	add_schedule_test(test-barriers.impala cuda)
	add_schedule_test(test-barriers.impala nvvm)
	add_schedule_test(test-producer-consumer-queue.impala cuda)
	add_schedule_test(test-producer-consumer-queue.impala nvvm)
endif()

if (AnyDSL_runtime_HAS_HSA_SUPPORT)
	add_schedule_test(test-basics.impala amdgpu)
	add_schedule_test(test-shuffles.impala amdgpu)
	add_schedule_test(test-barriers.impala amdgpu)
	add_schedule_test(test-producer-consumer-queue.impala amdgpu)
endif()

if (Thorin_HAS_RV_SUPPORT AND AnyQ_FIBER_SUPPORT)
	add_schedule_test(test-basics.impala cpu)
	# add_schedule_test(test-shuffles.impala cpu)
	add_schedule_test(test-barriers.impala cpu)
	# add_schedule_test(test-producer-consumer-queue.impala cpu)

	add_schedule_test(test-rv-intrinsics.impala cpu)
endif()

if (Thorin_HAS_RV_SUPPORT)
	add_subdirectory(rv-test-suite)
endif()
