
add_library(instrumentation_cuda STATIC
	${CMAKE_CURRENT_SOURCE_DIR}/CUDA/error.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/CUDA/error.h
	${CMAKE_CURRENT_SOURCE_DIR}/CUDA/event.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/CUDA/event.h
	${CMAKE_CURRENT_SOURCE_DIR}/CUDA/unique_handle.h
	${CMAKE_CURRENT_SOURCE_DIR}/instrumentation.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/instrumentation.h
)
target_include_directories(instrumentation_cuda INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
# sometimes FindCUDAToolkit's `CUDA::cupti` target is broken
target_include_directories(instrumentation_cuda PRIVATE ${CUDA_cupti_INCLUDE_DIR})
target_link_libraries(instrumentation_cuda PUBLIC CUDA::cuda_driver CUDA::cupti)

project(measure_globaltimer_res CUDA)

add_executable(measure_globaltimer_res
	${CMAKE_CURRENT_SOURCE_DIR}/measure_globaltimer_res.cu
)
target_link_libraries(measure_globaltimer_res PUBLIC CUDA::cudart CUDA::cupti)

set_target_properties(measure_globaltimer_res PROPERTIES
	CUDA_ARCHITECTURES "61;70;75"
)
