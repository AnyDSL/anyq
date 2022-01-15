#!/usr/bin/env python3.9

import sys
import io
import codecs
import asyncio
import re
from pathlib import Path
import argparse


default_bin_dir = Path(__file__).parent / "build" / "bin"


def device_id(name):
	return name.translate(str.maketrans(" \t\n\v@-/\\?", "_________"))


class BenchmarkError(Exception):
	pass


class QueueBenchmarkBinary:
	def __init__(self, path, test_name, queue_type, queue_size, platform):
		self.path = path
		self.test_name = test_name
		self.queue_type = queue_type
		self.queue_size = queue_size
		self.platform = platform

	def __repr__(self):
		return f"QueueBenchmarkBinary(path={self.path}, test_name={self.test_name}, queue_type={self.queue_type}, queue_size={self.queue_size}, platform={self.platform}')"

	async def capture_benchmark_info(self, p):
		device_name = None

		async for l in p.stdout:
			device_name = codecs.decode(l).strip()

		if not device_name:
			raise Exception("failed to parse device name")

		return device_name

	def info(self, device):
		async def run():
			p = await asyncio.create_subprocess_exec(
				self.path.as_posix(),
				"info",
				str(device),
				stdout=asyncio.subprocess.PIPE)

			device_name = await self.capture_benchmark_info(p)

			if await p.wait() != 0:
				raise Exception("benchmark info failed to run")

			return device_name
		return asyncio.run(run())

	async def capture_benchmark_output(self, dest, p):
		def write(l):
			# sys.stdout.write(codecs.decode(l))
			sys.stdout.write('.')
			sys.stdout.flush()
			dest.write(l)

		async for l in p.stdout:
			write(l)

		sys.stdout.write('\n')

	def run(self, dest, *, device, num_threads_min, num_threads_max, block_size, p_enq, p_deq, workload_size, timeout = None):
		async def run():
			p = await asyncio.create_subprocess_exec(
				self.path.as_posix(),
				str(device),
				str(num_threads_min),
				str(num_threads_max),
				str(block_size),
				str(p_enq),
				str(p_deq),
				str(workload_size),
				stdout=asyncio.subprocess.PIPE)

			try:
				await asyncio.wait_for(self.capture_benchmark_output(dest, p), timeout)
			except asyncio.TimeoutError:
				p.kill()
				raise
			finally:
				await p.wait()

			if p.returncode != 0:
				raise BenchmarkError("benchmark failed to run")

		return asyncio.run(run())


class QueueBenchmarkParams:
	def __init__(self, queue_type, queue_size, block_size, p_enq, p_deq, workload_size, platform, device):
		self.queue_type = queue_type
		self.queue_size = queue_size
		self.block_size = block_size
		self.p_enq = p_enq
		self.p_deq = p_deq
		self.workload_size = workload_size
		self.platform = platform
		self.device = device

	def __repr__(self):
		return f"QueueBenchmarkParams(queue_type={self.queue_type}, queue_size={self.queue_size}, block_size={self.block_size}, p_enq={self.p_enq}, p_deq={self.p_deq}, workload_size={self.workload_size}, platform='{self.platform}', device='{self.device}')"

def parse_benchmark_output_header(file):
	try:
		next(file)
		params = codecs.decode(next(file)).strip().split(';')
		next(file)
		next(file)
		platform, device = codecs.decode(next(file)).strip().split(';')
		next(file)
		return QueueBenchmarkParams(params[0], int(params[1]), int(params[2]), float(params[3]), float(params[4]), int(params[5]), platform, device)
	except (StopIteration, ValueError, TypeError):
		raise Exception("failed to parse benchmark output")


def benchmark_binaries(bin_dir, include):
	for f in bin_dir.iterdir():
		if f.name.startswith("benchmark-") and include.match(f.name):
			test_name, queue_type, queue_size, platform = f.stem.rsplit('-', 3)
			yield QueueBenchmarkBinary(f, test_name, queue_type, int(queue_size), platform)

def results_file_name(test_name, queue_type, queue_size, block_size, p_enq, p_deq, workload_size, device_name, platform):
	return f"{test_name}--{queue_type}-{queue_size}-{block_size}-{int(p_enq * 100)}-{int(p_deq * 100)}-{workload_size}-{device_id(device_name)}-{platform}.csv"

class QueueBenchmarkRun:
	def __init__(self, output_path, device_name, binary, **args):
		self.output_path = output_path
		self.device_name = device_name
		self.binary = binary
		self.args = args

	def __str__(self):
		return f"{self.binary.path.stem} -- {' '.join([str(v) for v in self.args.values()])}"

def run(results_dir, bin_dir, include, devices, *, rerun = False, dryrun = False, timeout = None, verbose = False):
	results_dir.mkdir(exist_ok=True, parents=True)

	def result_outdated(results_file, binary):
		try:
			return results_file.stat().st_mtime <= binary.stat().st_mtime
		except:
			return True

	skipped = []
	scheduled = []

	device_name_cache = dict()

	for workload_size in (1, 8, 32, 128, 2048):
		for binary in benchmark_binaries(bin_dir, include):
			for device in devices.get(binary.platform):
				device_name = device_name_cache.get((binary.platform, device))

				if not device_name:
					device_name = binary.info(device)
					device_name_cache[(binary.platform, device)] = device_name

				for p_enq in (0.25, 0.5, 1.0):
					for p_deq in (0.25, 0.5, 1.0):
						for block_size in [32, 256, 1024]:
							num_threads_min = 1
							num_threads_max = 1 << 21

							output_path = results_dir/results_file_name(binary.test_name, binary.queue_type, binary.queue_size, block_size, p_enq, p_deq, workload_size, device_name, binary.platform)
							bm = QueueBenchmarkRun(output_path, device_name, binary, device=device, num_threads_min=num_threads_min, num_threads_max=num_threads_max, block_size=block_size, p_enq=p_enq, p_deq=p_deq, workload_size=workload_size)

							if rerun or result_outdated(output_path, binary):
								scheduled.append(bm)
							else:
								if verbose:
									print("skipping", output_path)
								skipped.append(bm)

	print("run", len(scheduled), "benchmarks")
	failed = []
	timeouted = []
	succeeded = []
	total = len(scheduled)

	def num_digits(n):
		d = 1
		while n > 10:
			n /= 10
			d += 1
		return d

	def pad(n, width):
		return " " * (width - num_digits(n)) + str(n)

	max_digits = num_digits(total)

	try:
		for i, bm in enumerate(scheduled):
			try:
				with io.BytesIO() as buffer:
					print("[", pad(i+1, max_digits), "/", total, "]", bm)
					bm.binary.run(buffer, **bm.args, timeout=timeout)

					buffer.seek(0)
					params_reported = parse_benchmark_output_header(buffer)

					if params_reported.device != bm.device_name:
						raise Exception("benchmark reported inconsistent device name")

					if params_reported.platform != bm.binary.platform:
						raise Exception("benchmark reported inconsistent platform")

					if params_reported.queue_type != bm.binary.queue_type:
						raise Exception("benchmark reported inconsistent queue type")

					if params_reported.queue_size != bm.binary.queue_size:
						raise Exception("benchmark reported inconsistent queue size")

					if not dryrun:
						with open(bm.output_path, "wb") as file:
							file.write(buffer.getbuffer())

					succeeded.append(bm)

			except asyncio.TimeoutError:
				print("TIMEOUT")
				timeouted.append(bm)

			except BenchmarkError:
				print("BENCHMARK FAILED")
				failed.append(bm)

	except KeyboardInterrupt:
		print("\n[CTRL+C detected]")

	print("\nSummary on benchmarks:")
	padding = " " * (max_digits + 4)
	print(padding, pad(len(succeeded), max_digits), "successfully executed")
	print(padding, pad(len(skipped), max_digits), "skipped")
	print(padding, pad(len(failed), max_digits), "failed")
	print(padding, pad(len(timeouted), max_digits), "ran into timeout")

	if len(failed) > 0:
		print("\nThe following benchmarks FAILED:")
		for bm in failed:
			print("  -", bm)

	if len(timeouted) > 0:
		print("\nThe following benchmarks ran into TIMEOUT:")
		for bm in timeouted:
			print("  -", bm)


class EnqueueDequeueStatistics:
	def __init__(self, num_enqueues, num_enqueue_attempts, num_dequeues, num_dequeue_attempts):
		self.num_enqueues = num_enqueues
		self.num_enqueue_attempts = num_enqueue_attempts
		self.num_dequeues = num_dequeues,
		self.num_dequeue_attempts = num_dequeue_attempts

class QueueOperationStatistics:
	def __init__(self, num_operations, t_total, t_min, t_max):
		self.num_operations = num_operations
		self.t_total = t_total
		self.t_min = t_min
		self.t_max = t_max

class QueueOperationTimings:
	def __init__(self, t_enqueue, t_enqueue_min, t_enqueue_max, t_dequeue, t_dequeue_min, t_dequeue_max):
		self.t_enqueue = t_enqueue
		self.t_enqueue_min = t_enqueue_min
		self.t_enqueue_max = t_enqueue_max
		self.t_dequeue = t_dequeue
		self.t_dequeue_min = t_dequeue_min
		self.t_dequeue_max = t_dequeue_max

class Dataset:
	def __init__(self, params, path, data_offset):
		self.params = params
		self.path = path
		self.data_offset = data_offset

	def __repr__(self):
		return f"Dataset({self.params}, device='{self.device}')"

	@staticmethod
	def parse_row_old_format(cols):
		num_threads, t = int(cols[0]), float(cols[1])

		if len(cols) == 2:  # original format
			return num_threads, t, None, None

		queue_stats = EnqueueDequeueStatistics(int(cols[2]), int(cols[3]), int(cols[4]), int(cols[5]))

		if len(cols) == 6:  # with enqueue/dequeue stats
			return num_threads, t, queue_stats, None

		queue_timings = QueueOperationTimings(int(cols[6]), int(cols[7]), int(cols[8]), int(cols[9]), int(cols[10]), int(cols[11]))

		if len(cols) == 12: # with individual operation timings
			return num_threads, t, queue_stats, queue_timings

	@staticmethod
	def parse_old_format(file):
		for l in file:
			cols = l.split(';')
			yield Dataset.parse_row_old_format(cols)

	@staticmethod
	def parse_new_format(file):
		for l in file:
			cols = l.split(';')
			num_threads, t = int(cols[0]), float(cols[1])

			enqueue_stats_succ = QueueOperationStatistics(int(cols[ 2]), int(cols[ 3]), int(cols[ 4]), int(cols[ 5]))
			enqueue_stats_fail = QueueOperationStatistics(int(cols[ 6]), int(cols[ 7]), int(cols[ 8]), int(cols[ 9]))
			dequeue_stats_succ = QueueOperationStatistics(int(cols[10]), int(cols[11]), int(cols[12]), int(cols[13]))
			dequeue_stats_fail = QueueOperationStatistics(int(cols[14]), int(cols[15]), int(cols[16]), int(cols[17]))

			queue_stats = EnqueueDequeueStatistics(
				enqueue_stats_succ.num_operations,
				enqueue_stats_succ.num_operations + enqueue_stats_fail.num_operations,
				dequeue_stats_succ.num_operations,
				dequeue_stats_succ.num_operations + dequeue_stats_fail.num_operations)

			queue_timings = QueueOperationTimings(
				enqueue_stats_succ.t_total + enqueue_stats_fail.t_total,
				min(enqueue_stats_succ.t_min, enqueue_stats_fail.t_min),
				max(enqueue_stats_succ.t_max, enqueue_stats_fail.t_max),
				dequeue_stats_succ.t_total + dequeue_stats_fail.t_total,
				min(dequeue_stats_succ.t_min, dequeue_stats_fail.t_min),
				max(dequeue_stats_succ.t_max, dequeue_stats_fail.t_max)
			)

			yield num_threads, t, queue_stats, queue_timings

	def read(self):
		with open(self.path, "rt") as file:
			file.seek(self.data_offset)
			header = next(file).split(';')
			if len(header) in [2, 6, 12]:  # old format
				yield from Dataset.parse_old_format(file)
			elif len(header) == 18:        # new format
				yield from Dataset.parse_new_format(file)
			else:
				raise Exception("invalid file format")



def collect_datasets(results_dir, include):
	for f in results_dir.iterdir():
		if f.suffix == ".csv" and include.match(f.name):
			with open(f, "rb") as file:
				yield Dataset(parse_benchmark_output_header(file), f, file.tell())

def plot(results_dir, include):
	import numpy as np
	import plotutils
	import matplotlib.lines
	import matplotlib.legend

	datasets = [d for d in collect_datasets(results_dir, include)]
	platforms = sorted({(d.params.platform, d.params.device) for d in datasets})
	p_enqs = sorted({d.params.p_enq for d in datasets})
	p_deqs = sorted({d.params.p_deq for d in datasets})
	queue_sizes = sorted({d.params.queue_size for d in datasets})
	workload_sizes = sorted({d.params.workload_size for d in datasets})

	for platform, device in platforms:
		for p_enq in p_enqs:
			for p_deq in p_deqs:
				print(platform, p_enq, p_deq)

				plot_datasets = [d for d in filter(lambda d: d.params.platform == platform and d.params.p_enq == p_enq and d.params.p_deq == p_deq, datasets)]

				fig, canvas = plotutils.createFigure()

				ax = fig.add_subplot(1, 1, 1)

				ax.set_title(f"{device} ({platform})  $p_{{enq}}={p_enq}$  $p_{{deq}}={p_deq}$")

				for queue_size, color in zip(queue_sizes, plotutils.getBaseColorCycle()):
					for workload_size, style in zip(workload_sizes, plotutils.getBaseStyleCycle()):
						for d in filter(lambda d: d.params.queue_size == queue_size and d.params.workload_size == workload_size, plot_datasets):
							print(d)
							t = np.asarray([e for e in d.read()])
							ax.plot(t[:,0], t[:,1], color=color, linestyle=style)

				ax.set_xlabel("number of threads")
				ax.set_ylabel("average run time/ms")

				ax.add_artist(matplotlib.legend.Legend(parent=ax, handles=[matplotlib.lines.Line2D([], [], color=color) for _, color in zip(queue_sizes, plotutils.getBaseColorCycle())], labels=[f"queue size = {queue_size}" for queue_size in queue_sizes], loc="upper left", bbox_to_anchor=(0.0, 1.0)))
				ax.add_artist(matplotlib.legend.Legend(parent=ax, handles=[matplotlib.lines.Line2D([], [], linestyle=style) for _, style in zip(workload_sizes, plotutils.getBaseStyleCycle())], labels=[f"workload = {workload_size}" for workload_size in workload_sizes], loc="upper left", bbox_to_anchor=(0.0, 0.8)))

				canvas.print_figure(results_dir/f"{device_id(device)}-{platform}-{int(p_enq * 100)}-{int(p_deq * 100)}.pdf")

class Statistics:
	def __init__(self, t):
		self.t_avg = self.t_min = self.t_max = t
		self.n = 1

	def get(self):
		return self.t_avg / self.n, self.t_min, self.t_max, self.n

	def add(self, t):
		self.t_avg += t
		self.n += 1
		self.add_min(t)
		self.add_max(t)

	def add_min(self, t):
		self.t_min = min(self.t_min, t)

	def add_max(self, t):
		self.t_max = max(self.t_max, t)

def skip(data, n):
	for _ in range(n):
		next(data)
	yield from data

def results(data):
	burn_in = 3

	try:
		cur_num_threads, t, queue_stats, queue_timings = next(data)

		def reset_stats(t, queue_stats, queue_timings):
			if queue_stats and queue_timings:
				return Statistics(t), Statistics(queue_timings.t_enqueue / queue_stats.num_enqueue_attempts) if queue_stats.num_enqueue_attempts else None, Statistics(queue_timings.t_dequeue / queue_stats.num_dequeue_attempts) if queue_stats.num_dequeue_attempts else None
			return Statistics(t), None, None

		def add_stats(t, queue_stats, queue_timings):
			stats_t.add(t)
			if stats_enq:
				stats_enq.add(queue_timings.t_enqueue / queue_stats.num_enqueue_attempts)
				stats_enq.add_min(queue_timings.t_enqueue_min)
				stats_enq.add_max(queue_timings.t_enqueue_max)
			if stats_deq:
				stats_deq.add(queue_timings.t_dequeue / queue_stats.num_dequeue_attempts)
				stats_deq.add_min(queue_timings.t_dequeue_min)
				stats_deq.add_max(queue_timings.t_dequeue_max)

		def package_stats():
			return (stats_t.get(), stats_enq.get() if stats_enq else None, stats_deq.get() if stats_deq else None)

		stats_t, stats_enq, stats_deq = reset_stats(t, queue_stats, queue_timings)
		i = 1

		for num_threads, t, queue_stats, queue_timings in data:
			if num_threads == cur_num_threads:
				if i == burn_in:
					stats_t, stats_enq, stats_deq = reset_stats(t, queue_stats, queue_timings)
				elif i > burn_in:
					add_stats(t, queue_stats, queue_timings)
				i = i + 1
			else:
				if i >= burn_in:
					yield cur_num_threads, package_stats()
				cur_num_threads = num_threads
				stats_t, stats_enq, stats_deq = reset_stats(t, queue_stats, queue_timings)
				i = 1

		if i >= burn_in:
			yield cur_num_threads, package_stats()
	except StopIteration:
		pass

def export(file, results_dir, include):
	datasets = [d for d in collect_datasets(results_dir, include)]

	# platform > device > queue_type > queue_size > block_size > p_enq > p_deq > workload_size
	datasets.sort(key=lambda d: d.params.workload_size)
	datasets.sort(key=lambda d: d.params.p_deq)
	datasets.sort(key=lambda d: d.params.p_enq)
	datasets.sort(key=lambda d: d.params.block_size)
	datasets.sort(key=lambda d: d.params.queue_size)
	datasets.sort(key=lambda d: d.params.queue_type)
	datasets.sort(key=lambda d: d.params.device)
	datasets.sort(key=lambda d: d.params.platform)

	datasets = [(d.params, list(results(d.read()))) for d in datasets]

	def write_line_data(i):
		for params, dataset in datasets:
			if dataset:
				file.write(f"""
	new LineData(new LineParams("{params.device}-{params.platform}","{params.queue_type}",{params.queue_size},{params.block_size},{params.p_enq},{params.p_deq},{params.workload_size}),[""")
				for n, stats in dataset:
					if stats[i]:
						t_avg, t_min, t_max, _ = stats[i]
						file.write(f"new Result({n},{t_avg},{t_min},{t_max}),")
				file.write("]),")

	file.write("""class LineParams {
	constructor(device, queue_type, queue_size, block_size, p_enq, p_deq, workload_size) {
		this.device = device;
		this.queue_type = queue_type;
		this.queue_size = queue_size;
		this.block_size = block_size;
		this.p_enq = p_enq;
		this.p_deq = p_deq;
		this.workload_size = workload_size;
	}
};

class Result {
	constructor(num_threads, t_avg, t_min, t_max) {
		this.num_threads = num_threads;
		this.t_avg = t_avg;
		this.t_min = t_min;
		this.t_max = t_max;
	}
};

class LineData {
	constructor(params, results) {
		this.params = params;
		this.results = results;
	}
};

var kernel_run_time = [""")
	write_line_data(0)
	file.write("""
];

var enqueue_time = [""")
	write_line_data(1)
	file.write("""
];

var dequeue_time = [""")
	write_line_data(2)
	file.write("""
];
""")


def fixup(results_dir, include):
	for d in collect_datasets(results_dir, include):
		test_name = d.path.name.split('--')[0]
		filename = results_file_name(test_name, d.params.queue_type, d.params.queue_size, d.params.block_size, d.params.p_enq, d.params.p_deq, d.params.workload_size, d.params.device, d.params.platform)
		dest = d.path.with_name(filename)

		if d.path != dest:
			print(d.path.stem, "->", filename)
			d.path.rename(dest)



def main(args):
	this_dir = Path(__file__).parent.absolute()
	results_dir = this_dir/"results"

	include = re.compile(args.include)

	if args.command == run:
		bin_dir = args.bin_dir
		if not bin_dir.is_dir():
			raise IOError("Could not find path {} - please specify --bin-dir pointing to the location of benchmark binaries.".format(bin_dir))

		def device_list(args):
			return args if args else [0]

		devices = {
			"cpu": [0],
			"fiberless": [0],
			"cuda": device_list(args.cuda_device),
			"nvvm": device_list(args.cuda_device),
			"amdgpu": device_list(args.amdgpu_device)
		}

		run(results_dir, bin_dir, include, devices, rerun=args.rerun, dryrun=args.dryrun, timeout=args.timeout, verbose=args.verbose)

	elif args.command == plot:
		plot(results_dir, include)
	elif args.command == export:
		export(sys.stdout, results_dir, include)
	elif args.command == fixup:
		fixup(results_dir, include)


if __name__ == "__main__":
	args = argparse.ArgumentParser()
	sub_args = args.add_subparsers(dest='command', required=True)

	def add_command(name, function, **kwargs):
		args = sub_args.add_parser(name, **kwargs)
		args.add_argument("include", nargs="?", type=str, default=".*")
		args.set_defaults(command=function)
		return args

	run_cmd = add_command("run", run, help="run benchmarks")
	run_cmd.add_argument("--bin-dir", type=Path, default=default_bin_dir)
	run_cmd.add_argument("-dev-cuda", "--cuda-device", type=int, action="append")
	run_cmd.add_argument("-dev-amdgpu", "--amdgpu-device", type=int, action="append")
	run_cmd.add_argument("-rerun", "--rerun-all", dest="rerun", action="store_true")
	run_cmd.add_argument("-dryrun", "--dryrun", dest="dryrun", action="store_true")
	run_cmd.add_argument("--timeout", type=int, default=20)
	run_cmd.add_argument("-v", "--verbose", action="store_true")

	plot_cmd = add_command("plot", plot, help="generate plots from benchmark data")

	export_cmd = add_command("export", export, help="export benchmark data as JavaScript")

	fixup_cmd = add_command("fixup", fixup, help="restore result file names based on data contained in each file")

	try:
		main(args.parse_args())
	except Exception:
		import traceback
		traceback.print_exc()
		exit(-1)
