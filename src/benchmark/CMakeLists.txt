

find_package(Python3 COMPONENTS Interpreter)


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

	# if (MSVC)
	# 	target_compile_options(${_target} PRIVATE /W4 /WX /permissive-)
	# 	target_compile_definitions(${_target} PRIVATE -D_CRT_SECURE_NO_WARNINGS -D_SCL_SECURE_NO_WARNINGS)
	# endif ()
endforeach()


set(COMMON_BENCHMARK_FILES
	${AnyQ_SOURCE_DIR}/framework.art
	${AnyQ_SOURCE_DIR}/utils.art
	${AnyQ_SOURCE_DIR}/random.art
	${CMAKE_CURRENT_SOURCE_DIR}/csv.art
	${CMAKE_CURRENT_SOURCE_DIR}/benchmark.art
	${CMAKE_CURRENT_SOURCE_DIR}/instrumentation.art
	${CMAKE_CURRENT_SOURCE_DIR}/queue-benchmark.art
	${CMAKE_CURRENT_SOURCE_DIR}/queue-instrumentation.art
)

source_group("Source Files" FILES ${COMMON_BENCHMARK_FILES})

function(add_benchmark _target _benchmark_file _mapping)
	add_anyq_executable(${_target} ${_mapping} ${COMMON_BENCHMARK_FILES} ${_benchmark_file} ${ARGN})

	get_target_property(_target_sources ${_target} SOURCES)
	list(APPEND _target_sources
		$<TARGET_PROPERTY:benchmark_harness_${_mapping},SOURCES>
		$<TARGET_PROPERTY:instrumentation_${_mapping},SOURCES>
	)

	set_target_properties(${_target} PROPERTIES FOLDER "benchmarks/${_mapping}" )
	set(ADDITIONAL_LINK_DEPENDENCY ${ADDITIONAL_LINK_DEPENDENCY_${_mapping}})
	if(ADDITIONAL_LINK_DEPENDENCY)
		foreach(_dep ${ADDITIONAL_LINK_DEPENDENCY})
			list(APPEND _target_sources $<TARGET_PROPERTY:${_dep},SOURCES>)
		endforeach()
		target_link_libraries(${_target} PRIVATE ${ADDITIONAL_LINK_DEPENDENCY})
	endif()
	target_link_libraries(${_target} PRIVATE benchmark_harness_${_mapping})
	add_dependencies(benchmark-${_mapping} ${_target})

	set(_target_fingerprint ${CMAKE_CURRENT_BINARY_DIR}/${_target}-fingerprint.cpp)
	set(FINGERPRINT_COMMAND Python3::Interpreter ${AnyQ_ROOT_DIR}/fingerprint.py -o ${_target_fingerprint})
	add_custom_command(OUTPUT ${_target_fingerprint}
		COMMAND ${FINGERPRINT_COMMAND} ${_target_sources}
		DEPENDS ${_target_sources}
		COMMENT "Generating fingerprint of ${_target}"
		WORKING_DIRECTORY ${AnyQ_ROOT_DIR} VERBATIM COMMAND_EXPAND_LISTS
	)
	target_sources(${_target} PRIVATE ${_target_fingerprint})
endfunction()

function(generate_benchmarks template_file queue_types queue_sizes platform)
	get_filename_component(benchmark_name ${template_file} NAME_WE)

	foreach(queue_type ${queue_types})
		foreach(queue_size ${queue_sizes})
			set(queue_short_name "${${queue_type}_short_name}")
			set(variant_name "${benchmark_name}-${${queue_type}_short_name}-${queue_size}")
			set(dest_file "${CMAKE_CURRENT_BINARY_DIR}/${variant_name}.art")

			get_queue_constructor(queue_constructor_name ${queue_type} u32)

			if (DEFINED queue_constructor_name)
				configure_file(${template_file} ${dest_file})

				set(target_name "${variant_name}-${platform}")
				add_benchmark(${target_name} ${dest_file} ${platform} ${${queue_type}_sources} ${ARGN})

				if (DEFINED ${queue_type}_configure_target)
					cmake_language(CALL ${${queue_type}_configure_target} ${target_name})
				endif()
			endif()
		endforeach()
	endforeach()
endfunction()

function(add_sass_output target)
	get_target_property(_bin_dir ${target} ANYDSL_BINARY_DIR)
	get_target_property(_name ${target} NAME)

	set(cuda_src "${_bin_dir}/${_name}")

	add_custom_command(
		OUTPUT "${cuda_src}.asm"
		COMMAND ${CMAKE_COMMAND} -E echo "SASSing up ${cuda_src}.cu"
		COMMAND nvcc -arch=sm_75 --cubin -o "${cuda_src}.cubin" "${cuda_src}.cu"
		COMMAND nvdisasm -c "${cuda_src}.cubin" > "${cuda_src}.asm"
		DEPENDS ${target}
	)

	add_custom_target(${target}_sass ALL DEPENDS "${cuda_src}.asm")
endfunction()


option(AnyQ_BENCHMARK_AUTO OFF)

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

	generate_benchmarks(${CMAKE_CURRENT_SOURCE_DIR}/benchmark-queue-concurrent.art.in "${queue_types_${_platform}}" "16384;1048576" ${_platform} ${CMAKE_CURRENT_SOURCE_DIR}/benchmark-producer-consumer-queue-concurrent-push-pop.art)

	if (${_platform} STREQUAL "cuda" OR ${_platform} STREQUAL "nvvm" OR ${_platform} STREQUAL "cpu-scalar" OR ${_platform} STREQUAL "tbb-scalar")
		generate_benchmarks(${CMAKE_CURRENT_SOURCE_DIR}/benchmark-queue-concurrent-its.art.in "${queue_types_${_platform}}" "16384;1048576" ${_platform} ${CMAKE_CURRENT_SOURCE_DIR}/benchmark-producer-consumer-queue-concurrent-push-pop.art)
	endif()

	if (${_platform} STREQUAL "cpu" OR ${_platform} STREQUAL "cpu-scalar" OR ${_platform} STREQUAL "tbb" OR ${_platform} STREQUAL "tbb-scalar")
		generate_benchmarks(${CMAKE_CURRENT_SOURCE_DIR}/benchmark-vectorization.art.in "${queue_types_${_platform}}" "16384;1048576" ${_platform} ${CMAKE_CURRENT_SOURCE_DIR}/benchmark-producer-consumer-queue-concurrent-push-pop.art)
	endif()

	if (${_platform} STREQUAL "cpu-scalar" OR ${_platform} STREQUAL "tbb-scalar")
		generate_benchmarks(${CMAKE_CURRENT_SOURCE_DIR}/benchmark-vectorization-its.art.in "${queue_types_${_platform}}" "16384;1048576" ${_platform} ${CMAKE_CURRENT_SOURCE_DIR}/benchmark-producer-consumer-queue-concurrent-push-pop.art)
	endif()

	generate_benchmarks(${CMAKE_CURRENT_SOURCE_DIR}/benchmark-pipeline-simple.art.in "${queue_types_${_platform}}" "16384;1048576" ${_platform} ${CMAKE_CURRENT_SOURCE_DIR}/benchmark-pipeline-simple.art)

	if (${_platform} STREQUAL "cuda" AND ${CMAKE_GENERATOR} STREQUAL "Ninja Multi-Config")
		generate_benchmarks(${CMAKE_CURRENT_SOURCE_DIR}/benchmark-bwd-comparison.art.in "BrokerWorkDistributorQueue;BrokerWorkDistributorQueueOrig;BrokerWorkDistributorQueueOrigCUDA" "131072" ${_platform} ${CMAKE_CURRENT_SOURCE_DIR}/benchmark-bwd-compare.art)
		#add_sass_output(benchmark-bwd-comparison-BWD_cuda_orig-131072-cuda)
		#add_sass_output(benchmark-bwd-comparison-BWD_orig-131072-cuda)
		#add_sass_output(benchmark-bwd-comparison-BWD-131072-cuda)
	endif()

	if (${_platform} STREQUAL "cpu-scalar" OR ${_platform} STREQUAL "tbb-scalar")
		generate_benchmarks(${CMAKE_CURRENT_SOURCE_DIR}/benchmark-bwd-comparison.art.in "YangMellorCrummeyQueue;YangMellorCrummeyQueue_ref" "131072" ${_platform} ${CMAKE_CURRENT_SOURCE_DIR}/benchmark-bwd-compare.art)
	endif()
endforeach()
