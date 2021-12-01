

add_subdirectory(instrumentation)

foreach(_platform ${AnyQ_PLATFORMS})
	set(_target benchmark_harness_${_platform})

	add_library(${_target} STATIC
		${CMAKE_CURRENT_SOURCE_DIR}/benchmark.cpp
	)

	target_link_libraries(${_target} PRIVATE instrumentation_${_platform})

	set_target_properties(${_target} PROPERTIES
		CXX_STANDARD 17
		CXX_STANDARD_REQUIRED ON
		CXX_EXTENSIONS OFF
		FOLDER "benchmarks"
	)

	target_compile_definitions(${_target} PRIVATE -DPLATFORM=\"${_platform}\")

	# if (MSVC)
	# 	target_compile_options(${_target} PRIVATE /W4 /WX /permissive-)
	# 	target_compile_definitions(${_target} PRIVATE -D_CRT_SECURE_NO_WARNINGS -D_SCL_SECURE_NO_WARNINGS)
	# endif ()
endforeach()


set(COMMON_BENCHMARK_FILES
	${AnyQ_SOURCE_DIR}/framework.art
	${AnyQ_SOURCE_DIR}/utils.art
	${AnyQ_SOURCE_DIR}/random.art
	${CMAKE_CURRENT_SOURCE_DIR}/benchmark-common.art
	${CMAKE_CURRENT_SOURCE_DIR}/benchmark-producer-consumer-queue-concurrent-push-pop.art
)

source_group("Source Files" FILES ${COMMON_BENCHMARK_FILES})

function(add_benchmark _target _testfile _mapping)
	add_anydsl_executable(${_target} ${COMMON_BENCHMARK_FILES} ${COMMON_ARTIC_FILES} "${AnyQ_SOURCE_DIR}/mapping_${_mapping}.art" ${_testfile} ${ARGN})
	set_target_properties(${_target} PROPERTIES FOLDER "benchmarks/${_mapping}" )
	set(ADDITIONAL_LINK_DEPENDENCY ${ADDITIONAL_LINK_DEPENDENCY_${_mapping}})
	if(ADDITIONAL_LINK_DEPENDENCY)
		target_link_libraries(${_target} PRIVATE ${ADDITIONAL_LINK_DEPENDENCY})
	endif()
	target_link_libraries(${_target} PRIVATE benchmark_harness_${_mapping})
	add_dependencies(benchmark-${_mapping} ${_target})
endfunction()

set(queue_types_cpu ConcurrentProducerConsumerQueue MichaelScottQueue MoodyCamelQueue)
set(queue_types_cuda ConcurrentProducerConsumerQueue MichaelScottQueue)
set(queue_types_nvvm ${queue_types_cuda})
set(queue_types_amdgpu ConcurrentProducerConsumerQueue MichaelScottQueue)

set(ConcurrentProducerConsumerQueue_constructor "createConcurrentProducerConsumerIndexQueue")

function(generate_benchmarks template_file platform)
	get_filename_component(benchmark_name ${template_file} NAME_WE)

	foreach(queue_type ${queue_types_${platform}})
		foreach(queue_size 10000 100000 1000000)
			set(queue_short_name "${${queue_type}_short_name}")
			set(variant_name "${benchmark_name}-${${queue_type}_short_name}-${queue_size}")
			set(dest_file "${CMAKE_CURRENT_BINARY_DIR}/${variant_name}.art")

			if (DEFINED ${queue_type}_constructor)
				set(queue_constructor_name ${${queue_type}_constructor})
			else()
				set(queue_constructor_name "create${queue_type}[u32]")
			endif()

			configure_file(${template_file} ${dest_file})

			set(target_name "${variant_name}-${platform}")
			add_benchmark(${target_name} ${dest_file} ${platform})

			if (DEFINED ${queue_type}_link_LIBRARIES)
				target_link_libraries(${target_name} PRIVATE ${${queue_type}_link_LIBRARIES})
			endif()
		endforeach()
	endforeach()
endfunction()



set(BENCHMARK_FILES
	${CMAKE_CURRENT_SOURCE_DIR}/benchmark-queue-concurrent.art.in
)

option(AnyQ_BENCHMARK_AUTO OFF)

if(AnyQ_BENCHMARK_AUTO)
	find_package(Python3 COMPONENTS Interpreter)
endif()

foreach(_platform ${AnyQ_PLATFORMS})
	if(AnyQ_BENCHMARK_AUTO)
		set(AnyQ_BENCHMARK_ARGS_${_platform} "" CACHE STRING "additional arguments for the benchmark script")
		string(REPLACE " " ";" BENCHMARK_ARGS "${AnyQ_BENCHMARK_ARGS_${_platform}}")
		set(BENCHMARK_COMMAND COMMAND Python3::Interpreter ${AnyQ_ROOT_DIR}/benchmark.py run
			--bin-dir ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}
			${BENCHMARK_ARGS} ".*-${_platform}")
		add_custom_target(benchmark-${_platform} ${BENCHMARK_COMMAND} WORKING_DIRECTORY ${AnyQ_ROOT_DIR} VERBATIM COMMAND_EXPAND_LISTS)
	else()
		add_custom_target(benchmark-${_platform})
	endif()

	set_target_properties(benchmark-${_platform} PROPERTIES FOLDER "benchmarks")

	foreach(_benchmark ${BENCHMARK_FILES})
		generate_benchmarks(${_benchmark} ${_platform})
	endforeach()
endforeach()
