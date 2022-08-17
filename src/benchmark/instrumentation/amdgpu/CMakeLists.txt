
add_library(instrumentation_amdgpu INTERFACE
	${CMAKE_CURRENT_SOURCE_DIR}/instrumentation.h
)
target_include_directories(instrumentation_amdgpu INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
