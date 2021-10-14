#!/usr/bin/env python3.9

import codecs
import subprocess
import re
import numpy as np
import plotutils
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


def result_files(results_dir, include):
	for f in results_dir.iterdir():
		if f.suffix == ".csv" and include.match(f.name):
			yield f

def results(file):
	for l in file:
		num_threads, t = l.split(';')
		yield int(num_threads), float(t)

def plot(results_dir, bin_dir, include):
	fig, canvas = plotutils.createFigure()

	ax = fig.add_subplot(1, 1, 1)

	for f in result_files(results_dir, include):
		with open(f, "rt") as file:
			arr = np.asarray([d for d in results(file)])

			ax.plot(arr[:,0], arr[:,1])

	canvas.print_figure(results_dir/"results.pdf")



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
