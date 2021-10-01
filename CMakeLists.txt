cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

set(CMAKE_CONFIGURATION_TYPES Debug Release)

project(AnyQ CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (MSVC)
	add_compile_definitions(_CRT_SECURE_NO_WARNINGS _SCL_SECURE_NO_WARNINGS)
endif()


find_package(AnyDSL_runtime REQUIRED)
include_directories(${AnyDSL_runtime_INCLUDE_DIRS})
# if(NOT AnyDSL_runtime_HAS_JIT_SUPPORT)
	# message(FATAL_ERROR "AnyDSL runtime must be compiled with flag RUNTIME_JIT enabled")
# endif()

function(add_copy_dll_command target dependency)
	add_custom_command(TARGET ${target} POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
			$<IF:$<CONFIG:Debug>,$<TARGET_PROPERTY:${dependency},IMPORTED_LOCATION_DEBUG>,$<TARGET_PROPERTY:${dependency},IMPORTED_LOCATION_RELEASE>>
			$<TARGET_FILE_DIR:${target}>
	)
endfunction()

function(add_anydsl_runtime target)
	target_link_libraries(${target} PRIVATE ${AnyDSL_runtime_LIBRARIES})
	if (WIN32)
		add_copy_dll_command(${target} runtime)
		add_copy_dll_command(${target} runtime_jit_artic)
	endif()
endfunction()

option(BUILD_TESTING OFF)

if (BUILD_TESTING)
	include(CTest)
endif()

add_subdirectory(src)
