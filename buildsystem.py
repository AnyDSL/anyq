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


def git_apply_patch(dir, patch):
	if cmd("git", "-C", dir, "apply", "--check", "--reverse", patch) != 0:
		git("-C", dir, "apply", "-v", patch)


class CMakeBuild:
	def configure(self, build_dir, configs, src_dir, *args):
		build_dir.mkdir(parents=True, exist_ok=True)
		if cmd("cmake", f"-DCMAKE_CONFIGURATION_TYPES={';'.join(configs)}", *args, str(src_dir), cwd=build_dir) != 0:
			raise Exception(f"CMake failed for {build_dir}")

	def build(self, build_dir, config, *targets):
		if cmd("cmake", "--build", ".", "--config", config, *[v for pair in zip(["-t"] * len(targets), targets) for v in pair], cwd=build_dir) != 0:
			raise Exception(f"build failed for {build_dir}")

	def remove_build(self, build_dir):
		rmtree(build_dir)

class NinjaBuild(CMakeBuild):
	def configure(self, build_dir, configs, src_dir, *args):
		super().configure(build_dir, configs, src_dir, "-G", "Ninja Multi-Config", *args)


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
	def __init__(self, buildsystem, build_dir):
		self.buildsystem = buildsystem
		self.build_dir = build_dir

	def pull(self):
		pass

	def configure(self, configs, *deps):
		pass

	def build(self, config):
		pass

	def remove_build(self):
		if self.build_dir:
			self.buildsystem.remove_build(self.build_dir)

class PhonyBuild:
	def __init__(self, buildsystem, build_dir):
		pass

	def pull(self):
		pass

	def configure(self, configs, *deps):
		pass

	def build(self, config):
		pass

	def remove_build(self):
		pass


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
		return NinjaBuild()
		return VS2019Build()
	raise Exception("non-windows build not there yet")
