
function(disable_tests)
	foreach(_test ${ARGN})
		# message(STATUS "disable test: ${_test}")
		if(TEST ${_test})
			set_tests_properties(${_test} PROPERTIES DISABLED TRUE)
		endif()
	endforeach()
endfunction()


disable_tests(
	test-suite-atomic_cas-reverse-cpu
	test-suite-atomic_cas-reverse-fiberless
	test-suite-atomic_cas-reverse-cuda
	test-suite-atomic_cas-reverse-nvvm
	test-suite-atomic_cas-reverse-amdgpu
)
