#!/usr/bin/env python3

import os
import subprocess
import shutil
from pathlib import Path


def cmd(*args, **kwargs):
	print(*args, kwargs)
	p = subprocess.Popen(args, **kwargs)
	return p.wait()

def git(*args):
	if cmd("git", *args) != 0:
		raise Exception(f"git {' '.join(args)} failed")

def ninja(*args):
	if cmd("ninja", *args) != 0:
		raise Exception(f"ninja {' '.join(args)} failed")

def rmtree(path):
	if path.is_dir():
		shutil.rmtree(path, ignore_errors=True)


class CMakeBuild:
	def remove(self, build_dir):
		rmtree(build_dir)
	
	def configure(self, build_dir, src_dir, *args):
		build_dir.mkdir(parents=True, exist_ok=True)
		if cmd("cmake", *args, str(src_dir), cwd=build_dir) != 0:
			raise Exception(f"CMake failed for {build_dir}")

class NinjaBuild(CMakeBuild):
	def configure(self, build_dir, src_dir, *args):
		super().configure(build_dir, src_dir, "-G", "Ninja", *args)

class VS2019Build(CMakeBuild):
	def configure(self, build_dir, src_dir, *args):
		super().configure(build_dir, src_dir, "-G", "Visual Studio 16 2019", "-A", "x64", "-T", "host=x64", *args)


def pull_git_dependency(dir, url, *args, branch = "master"):
	if dir.exists():
		git("-C", str(dir), "pull", "origin", branch)
	else:
		git("clone", *args, url, str(dir), "-b", branch)

def pull_svn_dependency(dir, url, *args):
	if dir.exists():
		git("-C", str(dir), "svn", "rebase")
	else:
		git("svn", "clone", *args, url, str(dir))


class Build:
	def __init__(self, buildsystem, build_dir = None):
		self.buildsystem = buildsystem
		self.build_dir = build_dir

	def pull(self):
		pass

	def configure(self):
		pass

	def remove(self):
		if self.build_dir:
			self.buildsystem.remove(self.build_dir)

	def build(self):
		pass

class MultiConfigBuild(Build):
	def __init__(self, buildsystem, build_dir, configs = ["Debug", "Release"]):
		super().__init__(buildsystem, build_dir)
		self.configs = configs

	def builds(self):
		for c in self.configs:
			yield self.build_dir/c, c

	def configure(self, *deps):
		for build_dir, build_type in self.builds():
			self.configure_config(build_dir, build_type, *deps)

	def remove(self):
		for build_dir, build_type in self.builds():
			rmtree(build_dir)

	def build(self):
		for build_dir, build_type in self.builds():
			self.build_config(build_dir, build_type)


def component(*, depends_on = ()):
	def wrap(C):
		C.dependencies = depends_on
		return C
	return wrap


def instantiate_dependencies(creator, components, visited = dict()):
	for C in components:
		yield from instantiate_dependencies(creator, C.dependencies, visited)

		if C not in visited:
			c = creator(C)
			c.dependencies = [visited[D] for D in C.dependencies]
			visited[C] = c
			yield c


def select_buildsystem():
	if os.name == "nt":
		return VS2019Build()
	raise Exception("non-windows build not there yet")


def copy_dlls(dir, components, attr_name):
	dir.mkdir(parents=True, exist_ok=True)

	for c in components:
		for d in getattr(c, attr_name, []):
			print(f"copying {d} to {dir}")
			shutil.copy(d, dir)
