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

async def capture_benchmark_output(dest, p):
	def write(l):
		# sys.stdout.write(codecs.decode(l))
		sys.stdout.write('.')
		sys.stdout.flush()
		dest.write(l)

	async for l in p.stdout:
		write(l)
		if l.isspace():
			break

	device_name = ""
	async for l in p.stdout:
		write(l)
		if l.isspace():
			break
		device_name = codecs.decode(l).strip()

	async for l in p.stdout:
		write(l)

	sys.stdout.write('\n')
	return device_name.split(';')

class BenchmarkError(Exception):
	pass

def run_benchmark(dest, binary, *, device = 0, num_threads_min = 1, num_threads_max = 1 << 18, block_size = 256, p_enq = 0.5, p_deq = 0.5, workload_size = 8, timeout = None):
	async def run():
		p = await asyncio.create_subprocess_exec(
			binary.as_posix(), 
			str(device),
			str(num_threads_min),
			str(num_threads_max),
			str(block_size),
			str(p_enq),
			str(p_deq),
			str(workload_size),
			stdout=asyncio.subprocess.PIPE)

		try:
			platform, device_name = await asyncio.wait_for(capture_benchmark_output(dest, p), timeout)
		except asyncio.TimeoutError:
			p.kill()
			raise
		finally:
			await p.wait()

		if p.returncode != 0:
			raise BenchmarkError("benchmark failed to run")

		return platform, device_name

	return asyncio.run(run())


def benchmark_binaries(bin_dir, include):
	for f in bin_dir.iterdir():
		if f.name.startswith("benchmark-") and include.match(f.name):
			test_name, queue_type, queue_size, platform = f.stem.rsplit('-', 3)
			yield f, test_name, queue_type, queue_size, platform

def results_file_name(test_name, queue_type, queue_size, block_size, p_enq, p_deq, workload_size, device_name, platform):
	return f"{test_name}--{queue_type}-{queue_size}-{block_size}-{int(p_enq * 100)}-{int(p_deq * 100)}-{workload_size}-{device_id(device_name)}-{platform}.csv"

def run(results_dir, bin_dir, include, devices, *, rerun = False, dryrun = False, timeout = None, verbose = False):
	results_dir.mkdir(exist_ok=True, parents=True)

	def result_outdated(results_file, binary):
		try:
			return results_file.stat().st_mtime <= binary.stat().st_mtime
		except:
			return True

	skipped = 0
	scheduled = []

	for binary, test_name, queue_type, queue_size, platform in benchmark_binaries(bin_dir, include):
		for device in devices.get(platform):
			# TODO: invoke `$> binary device_id info` to retrieve device_name
			device_name = "AMD Ryzen 9 5950X 16-Core Processor"

			for p_enq in (0.25, 0.5, 1.0):
				for p_deq in (0.25, 0.5, 1.0):
					for workload_size in (1, 8, 32, 128, 2048):
						for block_size in [32, 256, 1024]:
							num_threads_min = 1
							num_threads_max = 1 << 21

							results_file_path = lambda device_name: results_dir/results_file_name(test_name, queue_type, queue_size, block_size, p_enq, p_deq, workload_size, device_name, platform)

							# print(device_name)
							# print(results_file_path(device_name))

							if rerun or not device_name or result_outdated(results_file_path(device_name), binary):
								scheduled.append({
									'binary': binary,
									'args': dict(device=device, num_threads_min=num_threads_min, num_threads_max=num_threads_max, block_size=block_size, p_enq=p_enq, p_deq=p_deq, workload_size=workload_size, timeout=timeout),
									'device_name': device_name,
									'output': results_file_path(device_name)
								})
							else:
								if verbose:
									print("skipping", results_file_path(device_name))
								skipped += 1

	print("execute", len(scheduled), "benchmarks")
	failed = []
	timeout = []
	success = 0
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
					print("[", pad(i+1, max_digits), "/", total, "]", bm['binary'].stem, "--", list(bm['args'].values()))
					platform_reported, device_name_reported = run_benchmark(buffer, bm['binary'], **bm['args'])

					if platform_reported != platform:
						raise Exception("benchmark platform doesn't match binary name")

					if bm['device_name'] != device_name_reported:
						raise Exception("device name doesn't match previously reported device name")

					results_file = bm['output']

					if not dryrun:
						with open(results_file, "wb") as file:
							file.write(buffer.getbuffer())

					success += 1

			except asyncio.TimeoutError:
				print("TIMEOUT")
				timeout.append(bm)

			except BenchmarkError:
				print("BENCHMARK FAILED")
				failed.append(bm)

	except KeyboardInterrupt:
		print()
		print("[CTRL+C detected]")

	print()
	print("Summary on benchmarks:")
	padding = " " * (max_digits + 4);
	print(padding, pad(success, max_digits), "successfully executed")
	print(padding, pad(skipped, max_digits), "skipped")
	print(padding, pad(len(failed), max_digits), "failed")
	print(padding, pad(len(timeout), max_digits), "ran into timeout")

	if len(failed) > 0:
		print()
		print("The following benchmarks FAILED:")
		for bm in failed:
			print("  -", bm['binary'].stem, "--", list(bm['args'].values()))

	if len(timeout) > 0:
		print()
		print("The following benchmarks ran into TIMEOUT:")
		for bm in timeout:
			print("  -", bm['binary'].stem, "--", list(bm['args'].values()))


class Dataset:
	def __init__(self, queue_type, queue_size, block_size, p_enq, p_deq, workload_size, platform, device, path, data_offset):
		self.queue_type = queue_type
		self.queue_size = queue_size
		self.block_size = block_size
		self.p_enq = p_enq
		self.p_deq = p_deq
		self.workload_size = workload_size
		self.platform = platform
		self.device = device
		self.path = path
		self.data_offset = data_offset

	def __repr__(self):
		return f"Dataset(queue_type={self.queue_type}, queue_size={self.queue_size}, block_size={self.block_size}, p_enq={self.p_enq}, p_deq={self.p_deq}, workload_size={self.workload_size}, platform='{self.platform}', device='{self.device}')"

	def read(self):
		with open(self.path, "rt") as file:
			file.seek(self.data_offset)
			for l in file:
				cols = l.split(';')
				num_threads, t, num_enqueues, num_enqueue_attempts, num_dequeues, num_dequeue_attempts = (*cols[:2], 0, 0, 0, 0) if len(cols) == 2 else cols
				yield int(num_threads), float(t), int(num_enqueues), int(num_enqueue_attempts), int(num_dequeues), int(num_dequeue_attempts)

def collect_datasets(results_dir, include):
	for f in results_dir.iterdir():
		if f.suffix == ".csv" and include.match(f.name):
			with open(f, "rb") as file:
				next(file)
				parts = codecs.decode(next(file)).strip().split(';')
				next(file)
				next(file)
				platform, device = codecs.decode(next(file)).strip().split(';')
				next(file)
				next(file)
				yield Dataset(parts[0], int(parts[1]), int(parts[2]), float(parts[3]), float(parts[4]), int(parts[5]), platform, device, f, file.tell())

def plot(results_dir, include):
	import numpy as np
	import plotutils
	import matplotlib.lines
	import matplotlib.legend

	datasets = [d for d in collect_datasets(results_dir, include)]
	platforms = sorted({(d.platform, d.device) for d in datasets})
	p_enqs = sorted({d.p_enq for d in datasets})
	p_deqs = sorted({d.p_deq for d in datasets})
	queue_sizes = sorted({d.queue_size for d in datasets})
	workload_sizes = sorted({d.workload_size for d in datasets})

	for platform, device in platforms:
		for p_enq in p_enqs:
			for p_deq in p_deqs:
				print(platform, p_enq, p_deq)

				plot_datasets = [d for d in filter(lambda d: d.platform == platform and d.p_enq == p_enq and d.p_deq == p_deq, datasets)]

				fig, canvas = plotutils.createFigure()

				ax = fig.add_subplot(1, 1, 1)

				ax.set_title(f"{device} ({platform})  $p_{{enq}}={p_enq}$  $p_{{deq}}={p_deq}$")

				for queue_size, color in zip(queue_sizes, plotutils.getBaseColorCycle()):
					for workload_size, style in zip(workload_sizes, plotutils.getBaseStyleCycle()):
						for d in filter(lambda d: d.queue_size == queue_size and d.workload_size == workload_size, plot_datasets):
							print(d)
							t = np.asarray([e for e in d.read()])
							ax.plot(t[:,0], t[:,1], color=color, linestyle=style)

				ax.set_xlabel("number of threads")
				ax.set_ylabel("average run time/ms")

				ax.add_artist(matplotlib.legend.Legend(parent=ax, handles=[matplotlib.lines.Line2D([], [], color=color) for _, color in zip(queue_sizes, plotutils.getBaseColorCycle())], labels=[f"queue size = {queue_size}" for queue_size in queue_sizes], loc="upper left", bbox_to_anchor=(0.0, 1.0)))
				ax.add_artist(matplotlib.legend.Legend(parent=ax, handles=[matplotlib.lines.Line2D([], [], linestyle=style) for _, style in zip(workload_sizes, plotutils.getBaseStyleCycle())], labels=[f"workload = {workload_size}" for workload_size in workload_sizes], loc="upper left", bbox_to_anchor=(0.0, 0.8)))

				canvas.print_figure(results_dir/f"{device_id(device)}-{platform}-{int(p_enq * 100)}-{int(p_deq * 100)}.pdf")

class Statistics:
	def reset(self, t):
		self.t_avg = self.t_min = self.t_max = t
		self.n = 1

	def get(self):
		return self.t_avg / self.n, self.t_min, self.t_max, self.n

	def add(self, t):
		self.t_avg += t
		self.t_min = min(self.t_min, t)
		self.t_max = max(self.t_max, t)
		self.n += 1

def skip(data, n):
	for _ in range(n):
		next(data)
	yield from data

def results(data):
	burn_in = 3

	try:
		stats = Statistics()
		cur_num_threads, t, num_enqueues, num_enqueue_attempts, num_dequeues, num_dequeue_attempts = next(data)
		stats.reset(t)
		i = 1

		for num_threads, t, num_enqueues, num_enqueue_attempts, num_dequeues, num_dequeue_attempts in data:
			if num_threads == cur_num_threads:
				if i == burn_in:
					stats.reset(t)
				elif i > burn_in:
					stats.add(t)
				i = i + 1
			else:
				if i >= burn_in:
					yield cur_num_threads, *stats.get()
				cur_num_threads = num_threads
				stats.reset(t)
				i = 1

		if i >= burn_in:
			yield cur_num_threads, *stats.get()
	except StopIteration:
		pass

def export(file, results_dir, include):
	datasets = [d for d in collect_datasets(results_dir, include)]

	# platform > device > queue_type > queue_size > block_size > p_enq > p_deq > workload_size
	datasets.sort(key=lambda d: d.workload_size)
	datasets.sort(key=lambda d: d.p_deq)
	datasets.sort(key=lambda d: d.p_enq)
	datasets.sort(key=lambda d: d.block_size)
	datasets.sort(key=lambda d: d.queue_size)
	datasets.sort(key=lambda d: d.queue_type)
	datasets.sort(key=lambda d: d.device)
	datasets.sort(key=lambda d: d.platform)

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

const data = [""")

	for d in datasets:
		file.write(f"""
	new LineData(new LineParams("{d.device}-{d.platform}","{d.queue_type}",{d.queue_size},{d.block_size},{d.p_enq},{d.p_deq},{d.workload_size}),[""")
		for n, t_avg, t_min, t_max, _ in results(d.read()):
			file.write(f"new Result({n},{t_avg},{t_min},{t_max}),")
		file.write("]),")

	file.write("\n];\n")


def fixup(results_dir, include):
	for d in collect_datasets(results_dir, include):
		test_name = d.path.name.split('--')[0]
		filename = results_file_name(test_name, d.queue_type, d.queue_size, d.block_size, d.p_enq, d.p_deq, d.workload_size, d.device, d.platform)
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
