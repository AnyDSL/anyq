#!/usr/bin/env python3

import requests
import xml.sax


class reader(xml.sax.handler.ContentHandler):
	def __init__(self):
		super().__init__()
		self.module_map = dict()
		self.cur_mod = None

	def characters(self, data):
		if self.cur_mod:
			try:
				int(data)
			except:
				data = data.strip()

				if data:
					self.module_map[self.cur_mod].append(data)

	def startElement(self, name, attrs):
		if name == "h3":
			self.last_mod_name = attrs["id"]
		elif name == "p" and attrs["class"] == "primary-list":
			self.cur_mod = self.last_mod_name
			self.module_map[self.cur_mod] = []

	def endElement(self, name):
		if name == "p":
			self.cur_mod = None


def fetch_page():
	r = requests.get("https://pdimov.github.io/boostdep-report/boost-1.79.0/module-levels.html")

	if r.status_code != 200:
		raise Exception("failed to fetch page")

	return r.text

def map_modules():
	r = reader()
	xml.sax.parseString(fetch_page(), r)
	return r.module_map

def extract_deps(module, module_map):
	for dep in module_map[module]:
		yield from extract_deps(dep, module_map)
		yield dep
	yield module

def main():
	module_map = map_modules()

	deps = {d for d in extract_deps("fiber", module_map)}

	for d in sorted([d for d in deps]):
		print(f"\"libs/{d}\",")


if __name__ == "__main__":
	main()
