
add_library(instrumentation_amdgpu STATIC
	${CMAKE_CURRENT_SOURCE_DIR}/instrumentation.cpp
)
target_include_directories(instrumentation_amdgpu INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
