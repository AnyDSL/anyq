
num_threads_min = 1
num_threads_max = 1 << 21

workload_sizes = [1, 512]
p_enqs = [0.25, 0.5, 1.0]
p_deqs = [0.25, 0.5, 1.0]
block_sizes = [32, 512]


def generate_benchmark_variants(test_name):
	def generate(num_threads_min, num_threads_max, workload_sizes, p_enqs, p_deqs, block_sizes):
		for workload_size in workload_sizes:
			for p_enq in p_enqs:
				for p_deq in p_deqs:
					for block_size in block_sizes:
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

	if test_name.startswith("benchmark-queue-concurrent"):
		yield from generate(num_threads_min, num_threads_max, workload_sizes, p_enqs, p_deqs, block_sizes)

	elif test_name.startswith("benchmark-vectorization"):
		yield from generate(1, 1 << 18, [1], [0.5], [0.5], [32])

	elif test_name == "benchmark-pipeline-simple":
		pass
		# for workload_size_producer in workload_sizes:
		# 	for workload_size_consumer in workload_sizes:
		# 		for num_input_elements in [1000, 1000000]:
		# 			for block_size in block_sizes:
		# 				yield (
		# 					{
		# 						"num_threads_min": num_threads_min,
		# 						"num_threads_max": num_threads_max,
		# 						"block_size": block_size,
		# 						"num_input_elements": num_input_elements,
		# 						"workload_size_producer": workload_size_producer,
		# 						"workload_size_consumer": workload_size_consumer
		# 					},
		# 					f"{block_size}-{num_input_elements}-{workload_size_producer}-{workload_size_consumer}"
		# 				)

	elif test_name == "benchmark-bwd-comparison":
		for block_size in block_sizes:
			yield (
				{
					"num_threads_min": num_threads_min,
					"num_threads_max": num_threads_max,
					"block_size": block_size,
				},
				f"{block_size}"
			)
