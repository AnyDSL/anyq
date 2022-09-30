
function(disable_tests)
	foreach(_test ${ARGN})
		# message(STATUS "disable test: ${_test}")
		if(TEST ${_test})
			set_tests_properties(${_test} PROPERTIES DISABLED TRUE)
		endif()
	endforeach()
endfunction()


disable_tests(
	test-suite-atomic_cas-reverse-tbb
	test-suite-atomic_cas-reverse-cpu-scalar
	test-suite-atomic_cas-reverse-cuda
	test-suite-atomic_cas-reverse-nvvm
	test-suite-atomic_cas-reverse-amdgpu

	test-suite-basics-tbb
	test-suite-shfl_bfly-tbb
	test-barriers-tbb
	test-shuffles-tbb

	test-suite-basics-cpu-scalar
	test-suite-shfl_bfly-cpu-scalar
	test-barriers-cpu-scalar
	test-shuffles-cpu-scalar
)
