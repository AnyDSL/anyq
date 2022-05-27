
add_library(cuda_utils STATIC
	${CMAKE_CURRENT_SOURCE_DIR}/CUDA/error.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/CUDA/error.h
	${CMAKE_CURRENT_SOURCE_DIR}/CUDA/event.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/CUDA/event.h
	${CMAKE_CURRENT_SOURCE_DIR}/CUDA/unique_handle.h
)
target_include_directories(cuda_utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cuda_utils PUBLIC CUDA::cuda_driver CUDA::cupti)


add_library(instrumentation_cuda_common STATIC
	${CMAKE_CURRENT_SOURCE_DIR}/instrumentation.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/instrumentation.h
)
target_include_directories(instrumentation_cuda_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(instrumentation_cuda_common PUBLIC cuda_utils)

add_library(instrumentation_cuda STATIC
	${CMAKE_CURRENT_SOURCE_DIR}/instrumentation_noits.cpp
)
target_link_libraries(instrumentation_cuda PUBLIC instrumentation_cuda_common)

add_library(instrumentation_cudaits STATIC
	${CMAKE_CURRENT_SOURCE_DIR}/instrumentation_its.cpp
)
target_link_libraries(instrumentation_cudaits PUBLIC instrumentation_cuda_common)


project(measure_globaltimer_res CUDA)

add_executable(measure_globaltimer_res
	${CMAKE_CURRENT_SOURCE_DIR}/measure_globaltimer_res.cu
)
target_link_libraries(measure_globaltimer_res PUBLIC CUDA::cudart CUDA::cupti)

set_target_properties(measure_globaltimer_res PROPERTIES
	CUDA_ARCHITECTURES "61;70;75"
)


# sometimes FindCUDAToolkit's `CUDA::cupti` target is broken
target_include_directories(instrumentation_cuda_common PRIVATE ${CUDA_cupti_INCLUDE_DIR})
target_include_directories(measure_globaltimer_res PRIVATE ${CUDA_cupti_INCLUDE_DIR})
