

class LineDataScope:
	def __init__(self, file):
		self.file = file

	def result(self, num_threads, t_avg, t_min, t_max):
		self.file.write(f"new Result({num_threads},{t_avg},{t_min},{t_max}),")

	def write_queue_op_stats(self, stats):
		self.file.write(f",new QueueOperationStatistics({stats.num_operations},{stats.t_total},{stats.t_min},{stats.t_max})")

	def queue_op_stats_result(self, num_threads, t, n, enqueue_stats_succ, enqueue_stats_fail, dequeue_stats_succ, dequeue_stats_fail):
		self.file.write(f"new QueueOpStatsResult({num_threads},{t},{n}")
		self.write_queue_op_stats(enqueue_stats_succ)
		self.write_queue_op_stats(enqueue_stats_fail)
		self.write_queue_op_stats(dequeue_stats_succ)
		self.write_queue_op_stats(dequeue_stats_fail)
		self.file.write("),")


class LineDataWriter:
	def __init__(self, file, params):
		self.file = file
		self.params = params

	def __enter__(self):
		self.file.write(f"""
	new LineData(new LineParams("{self.params.device}-{self.params.platform}","{self.params.queue_type}",{self.params.queue_size},{self.params.block_size},{self.params.p_enq},{self.params.p_deq},{self.params.workload_size}),[""")
		return LineDataScope(self.file)

	def __exit__(self, exc_type, exc_value, traceback):
		self.file.write("]),")


class Scope:
	def __init__(self, file):
		self.file = file

	def line_data(self, params):
		return LineDataWriter(self.file, params)


class Writer:
	def __init__(self, file, var_name):
		self.file = file
		self.var_name = var_name

	def __enter__(self):
		self.file.write(f"var {self.var_name} = [")
		return Scope(self.file)

	def __exit__(self, exc_type, exc_value, traceback):
		self.file.write("\n];\n")
