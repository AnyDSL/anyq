#!/usr/bin/env python3

import os
import pathlib
import subprocess
import shutil
from pathlib import Path
from unicodedata import name


windows = os.name == "nt"

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
	def __definition_args(self, **options):
		yield from self.global_flags
		for key, value in options.items():
			if isinstance(value, pathlib.PurePath):
				yield f"-D{key}:PATH={value}"
			else:
				yield f"-D{key}={value}"

	def __init__(self, *, global_flags = []):
		self.global_flags = global_flags

	def configure(self, build_dir, src_dir, *args, **options):
		build_dir.mkdir(parents=True, exist_ok=True)
		if cmd("cmake", *args, *list(self.__definition_args(**options)), str(src_dir), cwd=build_dir) != 0:
			raise Exception(f"CMake failed for {build_dir}")

	def build(self, build_dir, config, *targets):
		if cmd("cmake", "--build", ".", "--config", config, *[v for pair in zip(["-t"] * len(targets), targets) for v in pair], cwd=build_dir) != 0:
			raise Exception(f"build failed for {build_dir}")

	def remove_build(self, build_dir):
		rmtree(build_dir)

class NinjaBuild(CMakeBuild):
	def configure(self, build_dir, configs, src_dir, **options):
		super().configure(build_dir, src_dir, "-G", "Ninja", f"-DCMAKE_BUILD_TYPE={configs[-1]}", **options)

class NinjaMultiConfigBuild(CMakeBuild):
	def configure(self, build_dir, configs, src_dir, **options):
		super().configure(build_dir, src_dir, "-G", "Ninja Multi-Config", f"-DCMAKE_CONFIGURATION_TYPES={';'.join(configs)}", **options)

# class VS2019Build(CMakeBuild):
# 	def configure(self, build_dir, configs, src_dir, **options):
# 		super().configure(build_dir, src_dir, configs, "-G", "Visual Studio 16 2019", "-A", "x64", **options)


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
