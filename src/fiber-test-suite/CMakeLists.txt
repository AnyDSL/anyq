
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

set(TEST_SUITE_FILES
	fiber-test-idx-thread.art
	fiber-test-idx-group.art
	fiber-test-idx-group-thread.art
	fiber-test-idx-wave.art
	fiber-test-idx-wave-atomic.art
	fiber-test-idx-group-wave-thread.art
)

foreach(_test_case ${TEST_SUITE_FILES})
	get_filename_component(_testname ${_test_case} NAME_WE)
	add_schedule_test(${_test_case} cpu ${CMAKE_CURRENT_SOURCE_DIR}/fiber-test-suite.art)
	set_target_properties(${_testname}-cpu PROPERTIES FOLDER "tests/fibers")
endforeach()
