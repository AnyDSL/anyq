
add_library(instrumentation_cpu STATIC
	${CMAKE_CURRENT_SOURCE_DIR}/instrumentation.cpp
)
target_include_directories(instrumentation_cpu INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
