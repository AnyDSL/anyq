

if (cpu IN_LIST AnyQ_PLATFORMS)
	add_library(instrumentation_cpu INTERFACE)
	target_include_directories(instrumentation_cpu INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/cpu)
endif()

if (cuda IN_LIST AnyQ_PLATFORMS)
	add_library(instrumentation_cuda STATIC
		${CMAKE_CURRENT_SOURCE_DIR}/cuda/CUDA/error.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/cuda/CUDA/event.cpp
	)
	target_include_directories(instrumentation_cuda INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/cuda)
	target_link_libraries(instrumentation_cuda PUBLIC CUDA::cuda_driver)
endif()

if (nvvm IN_LIST AnyQ_PLATFORMS)
	add_library(instrumentation_nvvm INTERFACE)
	target_link_libraries(instrumentation_nvvm INTERFACE instrumentation_cuda)
endif()

if (amdgpu IN_LIST AnyQ_PLATFORMS)
	add_library(instrumentation_amdgpu INTERFACE)
	target_include_directories(instrumentation_amdgpu INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/amdgpu)
endif()
