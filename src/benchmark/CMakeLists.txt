

set(COMMON_BENCHMARK_FILES
	${CMAKE_CURRENT_SOURCE_DIR}/../framework.art
	${CMAKE_CURRENT_SOURCE_DIR}/../utils.art
	${CMAKE_CURRENT_SOURCE_DIR}/../random.art
	${CMAKE_CURRENT_SOURCE_DIR}/benchmark-common.art
	${CMAKE_CURRENT_SOURCE_DIR}/benchmark-producer-consumer-queue-concurrent-push-pop.art
)

source_group("Source Files" FILES ${COMMON_BENCHMARK_FILES})


add_library(benchmark_harness STATIC
	${CMAKE_CURRENT_SOURCE_DIR}/benchmark.cpp
)

set_target_properties(benchmark_harness PROPERTIES
	CXX_STANDARD 17
	CXX_STANDARD_REQUIRED ON
	CXX_EXTENSIONS OFF
)

# if (MSVC)
# 	target_compile_options(benchmark_harness PRIVATE /W4 /WX /permissive-)
# 	target_compile_definitions(benchmark_harness PRIVATE -D_CRT_SECURE_NO_WARNINGS -D_SCL_SECURE_NO_WARNINGS)
# endif ()


function(add_benchmark _testfile _mapping)
	get_filename_component(_testname ${_testfile} NAME_WE)
	set(_target "${_testname}-${_mapping}")
	add_anydsl_executable(${_target} ${COMMON_BENCHMARK_FILES} ${COMMON_ARTIC_FILES} "${AnyQ_SOURCE_DIR}/mapping_${_mapping}.art" ${_testfile} ${ARGN})
	set_target_properties(${_target} PROPERTIES FOLDER "benchmarks/${_mapping}" )
	set(ADDITIONAL_LINK_DEPENDENCY ${ADDITIONAL_LINK_DEPENDENCY_${_mapping}})
	if(ADDITIONAL_LINK_DEPENDENCY)
		target_link_libraries(${_target} PRIVATE ${ADDITIONAL_LINK_DEPENDENCY})
	endif()
	target_link_libraries(${_target} PRIVATE benchmark_harness)
	add_dependencies(benchmark-${_mapping} ${_target})
endfunction()


set(BENCHMARK_FILES
	${CMAKE_CURRENT_SOURCE_DIR}/benchmark-queue-concurrent-small.art
)

foreach(_platform ${AnyQ_PLATFORMS})
	add_custom_target(benchmark-${_platform})

	foreach(_benchmark ${BENCHMARK_FILES})
		add_benchmark(${_benchmark} ${_platform})
	endforeach()
endforeach()
