
function(add_anydsl_executable _target)
	set(_infiles ${ARGN})
	list(FILTER _infiles INCLUDE REGEX "[.]art$")
	if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm")
		set(ARCH_FLAG "-mcpu=native")
	else()
		set(ARCH_FLAG "-march=native")
	endif()
	anydsl_runtime_wrap(_${_target}_PROGRAM
		FRONTEND artic
		CLANG_FLAGS ${ARCH_FLAG}
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
	${CMAKE_CURRENT_SOURCE_DIR}/atomic.art
	${CMAKE_CURRENT_SOURCE_DIR}/producer_consumer_queue.art
	${CMAKE_CURRENT_SOURCE_DIR}/queues/michael_scott.art
	${CMAKE_CURRENT_SOURCE_DIR}/queues/moodycamel.art
	# ${CMAKE_CURRENT_SOURCE_DIR}/ringbuffer_allocator.art
	# ${CMAKE_CURRENT_SOURCE_DIR}/pipeline.art
	${CMAKE_CURRENT_SOURCE_DIR}/utils-$<LOWER_CASE:$<CONFIG>>.art
)

source_group("Source Files" FILES ${COMMON_ARTIC_FILES})
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

	option(AnyQ_VERBOSE OFF)
	if (AnyQ_VERBOSE)
		target_compile_definitions(fiber-support PRIVATE ANYQ_VERBOSE)
	endif()

	set(AnyQ_FIBER_SUPPORT TRUE)
	set(ADDITIONAL_LINK_DEPENDENCY_cpu fiber-support)
else()
	message(STATUS "Boost is required for fiber-support used in the CPU mapping.")
endif()

set(AnyQ_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(AnyQ_PLATFORMS)

if (AnyDSL_runtime_HAS_CUDA_SUPPORT)
	find_package(CUDAToolkit REQUIRED)
	list(APPEND AnyQ_PLATFORMS cuda nvvm)
endif()

if (AnyDSL_runtime_HAS_HSA_SUPPORT)
	list(APPEND AnyQ_PLATFORMS amdgpu)
endif()

if (Thorin_HAS_RV_SUPPORT AND AnyQ_FIBER_SUPPORT)
	list(APPEND AnyQ_PLATFORMS cpu)
endif()


if (BUILD_TESTING)
	add_subdirectory(test)
endif()

add_subdirectory(benchmark)

add_subdirectory(queues)
