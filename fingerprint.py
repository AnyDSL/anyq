#!/usr/bin/env python3

from hashlib import sha1 as hashalgo
from pathlib import Path


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument("-o", "--out", type=Path)
	parser.add_argument("-v", "--verbose", action='store_true')
	parser.add_argument('files', type=Path, nargs='+', help='one or multiple source files')

	args = parser.parse_args()

	fingerprint = hashalgo()
	buffersize = 8192

	def filter_files(files):
		for f in args.files:
			if f.suffix[1:] not in ['h', 'cpp', 'art', 'impala']:
				if args.verbose:
					print('ignore', f)
				continue
			if not f.is_file():
				if args.verbose:
					print('ignore', f)
				continue
			if f == args.out:
				if args.verbose:
					print('ignore', f)
				continue
			yield f

	files = sorted(set(filter_files(args.files)))

	print('Processing', len(files), 'files ...')
	for filename in files:
		try:
			with open(filename, 'rb') as file:
				while chunk := file.read(buffersize):
					fingerprint.update(chunk)
			if args.verbose:
				print(filename)
		except:
			print('failed', filename)

	result = fingerprint.hexdigest()

	if args.out is None:
		print(result)
	else:
		with open(args.out, 'w') as file:
			file.write('\nextern "C" const char* FINGERPRINT() { return "' + result + '"; }\n')
			print('Saved', result, 'to', args.out)
