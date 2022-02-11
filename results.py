import codecs


class QueueBenchmarkParams:
	def __init__(self, queue_type, queue_size, block_size, p_enq, p_deq, workload_size, platform, device, fingerprint):
		self.queue_type = queue_type
		self.queue_size = queue_size
		self.block_size = block_size
		self.p_enq = p_enq
		self.p_deq = p_deq
		self.workload_size = workload_size
		self.platform = platform
		self.device = device
		self.fingerprint = fingerprint

	def __repr__(self):
		return f"QueueBenchmarkParams(queue_type={self.queue_type}, queue_size={self.queue_size}, block_size={self.block_size}, p_enq={self.p_enq}, p_deq={self.p_deq}, workload_size={self.workload_size}, platform='{self.platform}', device='{self.device}', fingerprint='{self.fingerprint}')"

class EnqueueDequeueStatistics:
	def __init__(self, num_enqueues, num_enqueue_attempts, num_dequeues, num_dequeue_attempts):
		self.num_enqueues = num_enqueues
		self.num_enqueue_attempts = num_enqueue_attempts
		self.num_dequeues = num_dequeues,
		self.num_dequeue_attempts = num_dequeue_attempts

class QueueOperationTimings:
	def __init__(self, t_enqueue, t_enqueue_min, t_enqueue_max, t_dequeue, t_dequeue_min, t_dequeue_max):
		self.t_enqueue = t_enqueue
		self.t_enqueue_min = t_enqueue_min
		self.t_enqueue_max = t_enqueue_max
		self.t_dequeue = t_dequeue
		self.t_dequeue_min = t_dequeue_min
		self.t_dequeue_max = t_dequeue_max

class QueueOperationStatistics:
	def __init__(self, num_operations, t_total, t_min, t_max):
		self.num_operations = num_operations
		self.t_total = t_total
		self.t_min = t_min
		self.t_max = t_max

	def __add__(self, other):
		return QueueOperationStatistics(
			self.num_operations + other.num_operations,
			self.t_total + other.t_total,
			min(self.t_min, other.t_min),
			max(self.t_max, other.t_max))

	def __mul__(self, other):
		return QueueOperationStatistics(
			self.num_operations,
			self.t_total * other,
			self.t_min * other,
			self.t_max * other)


class DataVisitorKernelTimes:
	def visit(self, num_threads, t):
		pass

	def leave(self):
		pass

class DataVisitorEqDqStats:
	def visit(self, num_threads, t, queue_stats):
		pass

	def leave(self):
		pass

class DataVisitorOpTimes:
	def visit(self, num_threads, t, queue_stats, queue_timings):
		pass

	def leave(self):
		pass

class DataVisitorOpStats:
	def visit(self, num_threads, t, enqueue_stats_succ, enqueue_stats_fail, dequeue_stats_succ, dequeue_stats_fail):
		pass

	def leave(self):
		pass

class DatasetVisitor:
	def visit_kernel_times(self):
		return DataVisitorKernelTimes()

	def visit_eqdq_stats(self):
		return DataVisitorEqDqStats()

	def visit_op_times(self):
		return DataVisitorOpTimes()

	def visit_op_stats(self):
		return DataVisitorOpStats()

class Dataset:
	def __init__(self, params, path, data_offset):
		self.params = params
		self.path = path
		self.data_offset = data_offset

	def __repr__(self):
		return f"Dataset({self.params})"

	@staticmethod
	def make_sink(visitor, header):
		parse_kernel_times = lambda cols: (int(cols[0]), float(cols[1]))
		parse_eqdq_stats = lambda cols: EnqueueDequeueStatistics(int(cols[2]), int(cols[3]), int(cols[4]), int(cols[5]))
		parse_op_times = lambda cols: QueueOperationTimings(int(cols[6]), int(cols[7]), int(cols[8]), int(cols[9]), int(cols[10]), int(cols[11]))

		if len(header) == 2:
			data_visitor = visitor.visit_kernel_times()
			return lambda cols: data_visitor.visit(*parse_kernel_times(cols)), data_visitor

		if len(header) == 6:
			data_visitor = visitor.visit_eqdq_stats()
			return lambda cols: data_visitor.visit(*parse_kernel_times(cols), parse_eqdq_stats(cols)), data_visitor

		if len(header) == 12:
			data_visitor = visitor.visit_op_times()
			return lambda cols: data_visitor.visit(*parse_kernel_times(cols), parse_eqdq_stats(cols), parse_op_times(cols)), data_visitor

		if len(header) == 18:
			data_visitor = visitor.visit_op_stats()

			def parse(cols):
				enqueue_stats_succ = QueueOperationStatistics(int(cols[ 2]), int(cols[ 3]), int(cols[ 4]), int(cols[ 5]))
				enqueue_stats_fail = QueueOperationStatistics(int(cols[ 6]), int(cols[ 7]), int(cols[ 8]), int(cols[ 9]))
				dequeue_stats_succ = QueueOperationStatistics(int(cols[10]), int(cols[11]), int(cols[12]), int(cols[13]))
				dequeue_stats_fail = QueueOperationStatistics(int(cols[14]), int(cols[15]), int(cols[16]), int(cols[17]))
				return data_visitor.visit(*parse_kernel_times(cols), enqueue_stats_succ, enqueue_stats_fail, dequeue_stats_succ, dequeue_stats_fail)

			return parse, data_visitor

		raise Exception("invalid file format")

	def visit(self, visitor):
		with open(self.path, "rt") as file:
			file.seek(self.data_offset)

			header = next(file).split(';')

			sink, data_visitor = Dataset.make_sink(visitor, header)

			for l in file:
				cols = l.split(';')
				sink(cols)

			data_visitor.leave()


def parse_benchmark_output_header(file):
	try:
		next(file)
		params = codecs.decode(next(file)).strip().split(';')
		next(file)
		next(file)
		config = codecs.decode(next(file)).split(';')
		platform, device = config[0].strip(), config[1].strip()
		fingerprint = config[2].strip() if len(config) > 2 else None
		next(file)
		return QueueBenchmarkParams(params[0], int(params[1]), int(params[2]), float(params[3]), float(params[4]), int(params[5]), platform, device, fingerprint)
	except (StopIteration, ValueError, TypeError):
		raise Exception("failed to parse benchmark output")
