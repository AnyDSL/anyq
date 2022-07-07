import json

class LineDataScope:
	def __init__(self, file):
		self.file = file
		self.append = False

	def result(self, num_threads, t_avg, t_min, t_max):
		if self.append:
			self.file.write(',')
		self.file.write(f"[{num_threads},{t_avg},{t_min},{t_max}]")
		self.append = True

	def queue_op_stats_result(self, num_threads, t, n, enqueue_stats_succ, enqueue_stats_fail, dequeue_stats_succ, dequeue_stats_fail):
		if self.append:
			self.file.write(',')
		self.file.write(f'{{ "num": {num_threads}, "t": {t}, "n": {n}, "stats": [')

		def write_queue_op_stats(stats):
			self.file.write(f"[{stats.num_operations},{stats.t_total},{stats.t_min},{stats.t_max}]")

		write_queue_op_stats(enqueue_stats_succ)
		self.file.write(',')
		write_queue_op_stats(enqueue_stats_fail)
		self.file.write(',')
		write_queue_op_stats(dequeue_stats_succ)
		self.file.write(',')
		write_queue_op_stats(dequeue_stats_fail)

		self.file.write('] }')

		self.append = True


class LineDataWriter:
	def __init__(self, file, params):
		self.file = file
		self.params = params

	def __enter__(self):
		params = self.params.properties.copy()
		params['device'] = f'{self.params.device}-{self.params.platform}'
		params['foo'] = 'bar'
		self.file.write(f'{{ "params": {json.dumps(params)}, "results": [')
		return LineDataScope(self.file)

	def __exit__(self, exc_type, exc_value, traceback):
		self.file.write("] }")


class Scope:
	def __init__(self, file):
		self.file = file
		self.first = True

	def line_data(self, params):
		if not self.first:
			self.file.write(",\n")
		self.first = False
		return LineDataWriter(self.file, params)


class Writer:
	def __init__(self, file, var_name):
		self.file = file
		self.var_name = var_name

	def __enter__(self):
		self.file.write(f"[")
		return Scope(self.file)

	def __exit__(self, exc_type, exc_value, traceback):
		self.file.write("\n]\n")

