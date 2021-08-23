
source_group("Source Files" FILES ${COMMON_ARTIC_FILES})
source_group("Runtime Files" REGULAR_EXPRESSION ".*/runtime/platforms/.+[.](impala|art)$")
source_group("/" REGULAR_EXPRESSION "test-.+[.]art$")


add_executable(fiber-test test-fibers.cpp)
set_target_properties(fiber-test PROPERTIES
	FOLDER "tests/fibers"
	VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
target_link_libraries(fiber-test PRIVATE fiber-support)

offline_graph(fiber-integration fibers.art)
set_target_properties(fiber-integration PROPERTIES FOLDER "tests/fibers")
target_link_libraries(fiber-integration PRIVATE fiber-support)
