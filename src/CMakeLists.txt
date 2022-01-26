
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
	${CMAKE_CURRENT_SOURCE_DIR}/queue.art
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
	set(AnyQ_CUDA_ALL_SANITIZER_TOOLS memcheck racecheck initcheck synccheck)
	set(AnyQ_CUDA_SANITIZER_TOOLS "all" CACHE STRING "values from all|${AnyQ_CUDA_ALL_SANITIZER_TOOLS} to select cuda sanitizer tools for AnyQ tests on CUDA")
	if(AnyQ_CUDA_SANITIZER_TOOLS STREQUAL "all")
		set(AnyQ_CUDA_SANITIZER_TOOLS ${AnyQ_CUDA_ALL_SANITIZER_TOOLS})
	endif()
	message(STATUS "AnyQ_CUDA_SANITIZER_TOOLS: ${AnyQ_CUDA_SANITIZER_TOOLS}")
	find_program(CUDAToolkit_COMPUTE_SANITIZER_EXECUTABLE
		NAMES compute-sanitizer
		PATHS ${CUDAToolkit_TARGET_DIR} ${CUDAToolkit_LIBRARY_ROOT} ${CUDAToolkit_BIN_DIR}
		PATH_SUFFIXES compute-sanitizer)
	find_path(CUDA_cupti_INCLUDE_DIR cupti.h
		PATHS ${CUDAToolkit_INCLUDE_DIRS} ${CUDAToolkit_TARGET_DIR}/include ${CUDAToolkit_TARGET_DIR}/extras/CUPTI/include)
endif()

if (AnyDSL_runtime_HAS_HSA_SUPPORT)
	list(APPEND AnyQ_PLATFORMS amdgpu)
endif()

if (Thorin_HAS_RV_SUPPORT)
	list(APPEND AnyQ_PLATFORMS fiberless)
	if (AnyQ_FIBER_SUPPORT)
		list(APPEND AnyQ_PLATFORMS cpu)
	endif()
endif()


include(queues/queues.cmake)

if (BUILD_TESTING)
	add_subdirectory(test)
endif()

add_subdirectory(benchmark)
