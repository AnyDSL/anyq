
add_library(instrumentation_cuda STATIC
	${CMAKE_CURRENT_SOURCE_DIR}/CUDA/error.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/CUDA/event.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/instrumentation.cpp
)
target_include_directories(instrumentation_cuda INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(instrumentation_cuda PUBLIC CUDA::cuda_driver CUDA::cupti)


project(measure_globaltimer_res CUDA)

add_executable(measure_globaltimer_res
	${CMAKE_CURRENT_SOURCE_DIR}/measure_globaltimer_res.cu
)
target_link_libraries(measure_globaltimer_res PUBLIC CUDA::cudart CUDA::cupti)
