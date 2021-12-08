

if (cpu IN_LIST AnyQ_PLATFORMS OR fiberless IN_LIST AnyQ_PLATFORMS)
	add_library(instrumentation_cpu STATIC)
	if (WIN32)
		target_sources(instrumentation_cpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpu/instrumentation_win32.cpp)
	elseif(APPLE)
		target_sources(instrumentation_cpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpu/instrumentation_darwin.cpp)
	else()
		target_sources(instrumentation_cpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpu/instrumentation_linux.cpp)
	endif()
	target_include_directories(instrumentation_cpu INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/cpu)

	add_library(instrumentation_fiberless INTERFACE)
	target_link_libraries(instrumentation_fiberless INTERFACE instrumentation_cpu)
endif()

if (cuda IN_LIST AnyQ_PLATFORMS)
	add_library(instrumentation_cuda STATIC
		${CMAKE_CURRENT_SOURCE_DIR}/cuda/CUDA/error.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/cuda/CUDA/event.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/cuda/instrumentation.cpp
	)
	target_include_directories(instrumentation_cuda INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/cuda)
	target_link_libraries(instrumentation_cuda PUBLIC CUDA::cuda_driver)
endif()

if (nvvm IN_LIST AnyQ_PLATFORMS)
	add_library(instrumentation_nvvm INTERFACE)
	target_link_libraries(instrumentation_nvvm INTERFACE instrumentation_cuda)
endif()

if (amdgpu IN_LIST AnyQ_PLATFORMS)
	add_library(instrumentation_amdgpu STATIC
		${CMAKE_CURRENT_SOURCE_DIR}/amdgpu/instrumentation.cpp
	)
	target_include_directories(instrumentation_amdgpu INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/amdgpu)
endif()
