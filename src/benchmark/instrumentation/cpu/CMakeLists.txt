
add_library(instrumentation_cpu INTERFACE
	${CMAKE_CURRENT_SOURCE_DIR}/instrumentation.h
)
target_include_directories(instrumentation_cpu INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
