#!/usr/bin/env python3

import sys
import argparse
# from pathlib import Path


def main(args):
	with open(args.source_file, "rt") as file:
		if args.include:
			for include in args.include:
				sys.stdout.write(f"#include \"{include}\"\n")
		sys.stdout.write(file.read())


if __name__ == "__main__":
	args = argparse.ArgumentParser()
	args.add_argument("source_file")
	args.add_argument("-I", "--include", action="append")
	main(args.parse_args())
