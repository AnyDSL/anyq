
source_group("/" REGULAR_EXPRESSION "test-.+[.]art$")
source_group("Source Files" FILES ${COMMON_ARTIC_FILES} ${COMMON_TEST_FILES} REGULAR_EXPRESSION "utils-(Debug|Release)[.]art$")
source_group("Runtime Files" REGULAR_EXPRESSION ".*/runtime/platforms/.+[.](impala|art)$")

add_executable(fiber-test test-fibers.cpp)
set_target_properties(fiber-test PROPERTIES
	FOLDER "tests/fibers"
	VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
target_link_libraries(fiber-test PRIVATE fiber-support)

add_anydsl_executable(fiber-integration fibers.art)
set_target_properties(fiber-integration PROPERTIES FOLDER "tests/fibers")
target_link_libraries(fiber-integration PRIVATE fiber-support)

set(TEST_SUITE_FILES
	fiber-test-idx-thread.art
	fiber-test-idx-group.art
	fiber-test-idx-group-thread.art
	fiber-test-idx-wave-barrier.art
	fiber-test-idx-wave-atomic.art
	fiber-test-idx-wave-shuffle.art
	fiber-test-idx-wave-constant-shuffle.art
	fiber-test-idx-wave-thread.art
	fiber-test-idx-group-wave.art
	fiber-test-idx-group-wave-thread.art
	fiber-test-idx-group-2xthread.art
	fiber-test-idx-group-wave-1xthread-2xthread.art
	fiber-test-gid-group-wave-thread.art
	fiber-test-gid-group-thread.art
	fiber-test-gid-wave-thread.art
	fiber-test-gid-thread.art
)

foreach(_test_case ${TEST_SUITE_FILES})
	get_filename_component(_testname ${_test_case} NAME_WE)
	add_test_executable(${_testname}-cpu cpu ${_test_case} ${CMAKE_CURRENT_SOURCE_DIR}/fiber-test-suite.art)
	set_target_properties(${_testname}-cpu PROPERTIES FOLDER "tests/fibers")
	add_test(NAME "${_testname}-cpu" COMMAND ${_testname}-cpu WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endforeach()
