#!/usr/bin/env python3.9


def generate_benchmark_variants(test_name):
	num_threads_min = 1
	num_threads_max = 1 << 21

	if test_name.startswith("benchmark-queue-concurrent"):
		for workload_size in [1, 512]:
			for p_enq in [0.25, 0.5, 1.0]:
				for p_deq in [0.25, 0.5, 1.0]:
					for block_size in [32, 512]:
						yield (
							{
								"num_threads_min": num_threads_min,
								"num_threads_max": num_threads_max,
								"block_size": block_size,
								"p_enq": p_enq,
								"p_deq": p_deq,
								"workload_size": workload_size
							},
							f"{block_size}-{int(p_enq * 100)}-{int(p_deq * 100)}-{workload_size}"
						)

	elif test_name == "benchmark-pipeline-simple":
		for workload_size_producer in [1, 512]:
			for workload_size_consumer in [1, 512]:
				for num_input_elements in [1000, 1000000]:
					for block_size in [32, 512]:
						yield (
							{
								"num_threads_min": num_threads_min,
								"num_threads_max": num_threads_max,
								"block_size": block_size,
								"num_input_elements": num_input_elements,
								"workload_size_producer": workload_size_producer,
								"workload_size_consumer": workload_size_consumer
							},
							f"{block_size}-{num_input_elements}-{workload_size_producer}-{workload_size_consumer}"
						)

	elif test_name == "benchmark-bwd-comparison":
		for block_size in [32, 512]:
			yield (
				{
					"num_threads_min": num_threads_min,
					"num_threads_max": num_threads_max,
					"block_size": block_size,
				},
				f"{block_size}"
			)
