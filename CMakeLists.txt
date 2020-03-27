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

option(BUILD_TESTING OFF)
if (BUILD_TESTING)
	include(CTest)
endif()

add_subdirectory(src)

# set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT minimal)
