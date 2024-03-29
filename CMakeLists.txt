cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

set(CMAKE_CONFIGURATION_TYPES Debug Release)

project(AnyQ C CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(AnyQ_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

if (MSVC)
	add_compile_definitions(_CRT_SECURE_NO_WARNINGS _SCL_SECURE_NO_WARNINGS)
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT error)
if (ipo_supported)
	option(AnyQ_ENABLE_LTO "enable IPO/LTO for all targets" ON)
	message(STATUS "IPO/LTO: ${AnyQ_ENABLE_LTO}")
	set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ${AnyQ_ENABLE_LTO})
else()
	message(STATUS "IPO/LTO: not supported: <${error}>")
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
