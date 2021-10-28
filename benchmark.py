#!/usr/bin/env python3.9

import codecs
import subprocess
import re
import numpy as np
import plotutils
import matplotlib.lines
import matplotlib.legend
from pathlib import Path
import argparse


def capture_output(p):
	for l in p.stdout:
		if l == b'--------\n':
			break

	for l in p.stdout:
		num_threads, t = codecs.decode(l).split(';')
		yield int(num_threads), float(t)

def run_benchmark(dest, binary):
	p = subprocess.Popen([binary.as_posix()], stdout=subprocess.PIPE)
	for num_threads, t in capture_output(p):
		print(num_threads, t)
		dest.write(f"{num_threads};{t}\n")
	p.wait()


def benchmark_binaries(bin_dir, include):
	for f in bin_dir.iterdir():
		if f.name.startswith("benchmark-") and include.match(f.name):
			yield f

def run(results_dir, bin_dir, include):
	results_dir.mkdir(exist_ok=True, parents=True)

	for b in benchmark_binaries(bin_dir, include):
		results_file_path = results_dir/f"{b.stem}.csv"
		with open(results_file_path, "wt") as file:
			print(results_file_path)
			run_benchmark(file, b)


class Dataset:
	def __init__(self, queue, queue_size, p_enq, p_deq, workload_size, platform, filename):
		self.queue = queue
		self.queue_size = queue_size
		self.p_enq = p_enq
		self.p_deq = p_deq
		self.workload_size = workload_size
		self.platform = platform
		self.filename = filename

	def __repr__(self):
		return f"Dataset(queue={self.queue}, queue_size={self.queue_size}, p_enq={self.p_enq}, p_deq={self.p_deq}, workload_size={self.workload_size}, platform='{self.platform}')"

	def read(self):
		with open(self.filename, "rt") as file:
			return np.asarray([d for d in results(file)])

def collect_datasets(results_dir, include):
	for f in results_dir.iterdir():
		if f.suffix == ".csv" and include.match(f.name):
			parts = f.stem.split('-')
			yield Dataset(parts[-6], int(parts[-5]), int(parts[-4])/100, int(parts[-3])/100, int(parts[-2]), parts[-1], f)

def results(file):
	for l in file:
		num_threads, t = l.split(';')
		yield int(num_threads), float(t)

def plot(results_dir, bin_dir, include):
	datasets = [d for d in collect_datasets(results_dir, include)]
	platforms = sorted({d.platform for d in datasets})
	p_enqs = sorted({d.p_enq for d in datasets})
	p_deqs = sorted({d.p_deq for d in datasets})
	queue_sizes = sorted({d.queue_size for d in datasets})
	workload_sizes = sorted({d.workload_size for d in datasets})

	for platform in platforms:
		for p_enq in p_enqs:
			for p_deq in p_deqs:
				print(platform, p_enq, p_deq)

				plot_datasets = [d for d in filter(lambda d: d.platform == platform and d.p_enq == p_enq and d.p_deq == p_deq, datasets)]

				fig, canvas = plotutils.createFigure()

				ax = fig.add_subplot(1, 1, 1)

				ax.set_title(f"{platform}  $p_{{enq}}={p_enq}$  $p_{{deq}}={p_deq}$")

				for queue_size, color in zip(queue_sizes, plotutils.getBaseColorCycle()):
					for workload_size, style in zip(workload_sizes, plotutils.getBaseStyleCycle()):
						for d in filter(lambda d: d.queue_size == queue_size and d.workload_size == workload_size, plot_datasets):
							print(d)
							t = d.read()
							ax.plot(t[:,0], t[:,1], color=color, linestyle=style)

				ax.set_xlabel("number of threads")
				ax.set_ylabel("average run time/ms")

				ax.add_artist(matplotlib.legend.Legend(parent=ax, handles=[matplotlib.lines.Line2D([], [], color=color) for _, color in zip(queue_sizes, plotutils.getBaseColorCycle())], labels=[f"queue size = {queue_size}" for queue_size in queue_sizes], loc="upper left", bbox_to_anchor=(0.0, 1.0)))
				ax.add_artist(matplotlib.legend.Legend(parent=ax, handles=[matplotlib.lines.Line2D([], [], linestyle=style) for _, style in zip(workload_sizes, plotutils.getBaseStyleCycle())], labels=[f"workload = {workload_size}" for workload_size in workload_sizes], loc="upper left", bbox_to_anchor=(0.0, 0.8)))

				canvas.print_figure(results_dir/f"{platform}-{int(p_enq * 100)}-{int(p_deq * 100)}.pdf")



def main(args):
	this_dir = Path(__file__).parent.absolute()
	bin_dir = this_dir/"build"/"bin"
	results_dir = this_dir/"results"

	args.command(results_dir, bin_dir, re.compile(args.include))


if __name__ == "__main__":
	args = argparse.ArgumentParser()
	sub_args = args.add_subparsers(required=True)

	def add_command(name, function):
		args = sub_args.add_parser(name)
		args.set_defaults(command=function)
		args.add_argument("include", nargs="?", default=".*")
		return args

	run_cmd = add_command("run", run)
	plot_cmd = add_command("plot", plot)

	main(args.parse_args())
