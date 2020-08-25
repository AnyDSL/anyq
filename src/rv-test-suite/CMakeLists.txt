
file(GLOB _intrinsic_sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "intrinsic-*-*.impala" )
file(GLOB _test_sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "test-*-*.impala" )

if (Thorin_HAS_RV_SUPPORT)

	foreach(_test_filename ${_test_sources})
		string(REGEX REPLACE "^test-(.+)-.+\.impala$" "\\1" _test_component ${_test_filename})
		
		set(_test_implementations ${_intrinsic_sources})
		list(FILTER _test_implementations INCLUDE REGEX "^intrinsic-${_test_component}-.+\.impala$")
		foreach(_test_impl ${_test_implementations})
			string(REGEX REPLACE "^intrinsic-${_test_component}-(.+)\.impala$" "\\1" _test_impl_name ${_test_impl})
			get_filename_component(_test_name ${_test_filename} NAME_WE)
			set(_target "rv-${_test_name}--${_test_impl_name}")

			offline_graph(${_target} "${CMAKE_CURRENT_SOURCE_DIR}/main.impala" ${_test_filename} ${_test_impl})
			set_target_properties(${_target} PROPERTIES FOLDER "tests/rv" )
			if (BUILD_TESTING)
				add_test(NAME "${_target}" COMMAND ${_target} WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
			endif()
		endforeach()
	endforeach()

endif()
