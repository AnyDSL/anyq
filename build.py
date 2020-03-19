#!/usr/bin/env python3

import os
import subprocess
import shutil
from pathlib import Path
import argparse


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


class CMakeBuild:
	def __init__(self):
		pass

	def remove(self, build_dir):
		shutil.rmtree(build_dir)

	def configure(self, build_dir, src_dir, *args):
		if cmd("cmake", *args, str(src_dir), cwd=build_dir) != 0:
			raise Exception(f"CMake failed for {build_dir}")

class NinjaBuild(CMakeBuild):
	def __init__(self):
		pass

	def configure(self, build_dir, src_dir, *args):
		super().configure(build_dir, src_dir, "-G", "Ninja", *args)

class VS2019Build(CMakeBuild):
	def __init__(self):
		pass

	def configure(self, build_dir, src_dir, *args):
		super().configure(build_dir, src_dir, "-G", "Visual Studio 16 2019", "-A", "x64", "-T", "host=x64", *args)


def pullGitDependency(dir, url, branch = "master", *args):
	if dir.exists():
		git("-C", str(dir), "pull", "origin", branch)
	else:
		git("clone", *args, url, str(dir), "-b", branch)


class LLVM:
	def __init__(self, dir):
		self.source_dir = dir/"source"
		self.build_dir = dir/"build"

	def pull(self):
		pullGitDependency(self.source_dir, "https://github.com/llvm/llvm-project.git", "llvmorg-8.0.1")
		pullGitDependency(self.source_dir/"llvm"/"tools"/"rv", "https://github.com/cdl-saarland/rv.git", "release_80", "--recurse-submodules")

	def configure(self, buildsystem):
		self.build_dir.mkdir(parents=True, exist_ok=True)

		flags = [
			"-DCMAKE_BUILD_TYPE=Release",
			"-DLLVM_OPTIMIZED_TABLEGEN=ON",
			"-DLLVM_TARGETS_TO_BUILD=X86;NVPTX",
			"-DLLVM_ENABLE_RTTI=ON",
			"-DLLVM_INCLUDE_UTILS=OFF",
			"-DLLVM_INCLUDE_TESTS=OFF",
			"-DLLVM_INCLUDE_EXAMPLES=OFF",
			"-DLLVM_INCLUDE_BENCHMARKS=OFF",
			"-DLLVM_INCLUDE_DOCS=OFF",
			"-DLLVM_ENABLE_PROJECTS=clang;lld",
			"-DLLVM_TOOL_CLANG_BUILD=ON",
			"-DLLVM_TOOL_LLD_BUILD=ON",
			"-DCLANG_INCLUDE_DOCS=OFF",
			"-DCLANG_INCLUDE_TESTS=OFF",
		]

		buildsystem.configure(self.build_dir, self.source_dir/"llvm", *flags)

	def build(self):
		ninja("-C", str(self.build_dir), "clang", "llvm-as", "lld", "RV", "LLVMRV", "LLVMMCJIT", "LLVMExecutionEngine", "LLVMRuntimeDyld")
		return []

class AnyDSL:
	def __init__(self, dir):
		self.source_dir = dir
		self.build_dir = dir/"build"

	def pull(self):
		pullGitDependency(self.source_dir, "https://github.com/AnyDSL/anydsl.git", "cmake-based-setup")

	def configure(self, buildsystem, llvm):
		self.build_dir.mkdir(parents=True, exist_ok=True)

		flags = [
			 "-DCMAKE_BUILD_TYPE=Release",
			f"-DLLVM_DIR:PATH={llvm.build_dir/'lib'/'cmake'/'llvm'}",
			 "-DRUNTIME_JIT=ON",
			f"-DRV_INCLUDE_DIR={llvm.source_dir/'llvm'/'tools'/'rv'/'include'}",
			f"-DRV_LIBRARY={llvm.build_dir/'lib'/'RV.lib'}",
			f"-DRV_SLEEF_LIBRARY={llvm.build_dir/'lib'/'gensleef.lib'}",
			 "-DBUILD_SHARED_LIBS=OFF",
			 "-DAnyDSL_runtime_BUILD_SHARED=ON"
		]

		buildsystem.configure(self.build_dir, self.source_dir, *flags)

	def buildRuntime(self):
		ninja("-C", str(self.build_dir), "runtime")
		return [self.build_dir/'bin'/'runtime.dll']

	def buildImpala(self):
		ninja("-C", str(self.build_dir), "impala")
		return []

class MultiConfigBuild:
	def __init__(self, build_dir):
		self.build_dir = build_dir
		self.configs = ["Debug", "Release"]
	
	def builds(self):
		for c in self.configs:
			yield self.build_dir/c, c

class ZLIB(MultiConfigBuild):
	def __init__(self, dir):
		super().__init__(dir/"build")
		self.install_dir = dir
		self.source_dir = dir/"source"

	def pull(self):
		pullGitDependency(self.source_dir, "https://github.com/madler/zlib.git")

	def configure(self, buildsystem):
		for build_dir, build_type in self.builds():
			build_dir.mkdir(parents=True, exist_ok=True)
			flags = [
				f"-DCMAKE_BUILD_TYPE={build_type}",
				f"-DCMAKE_INSTALL_PREFIX:PATH={self.install_dir}",
			]
			buildsystem.configure(build_dir, self.source_dir, *flags)

	def build(self):
		for build_dir, build_type in self.builds():
			ninja("-C", str(build_dir), "install")
		return [self.install_dir/'bin'/'zlibd.dll', self.install_dir/'bin'/'zlib.dll']

class LPNG(MultiConfigBuild):
	def __init__(self, dir):
		super().__init__(dir/"build")
		self.install_dir = dir
		self.source_dir = dir/"source"

	def pull(self):
		pullGitDependency(self.source_dir, "git://git.code.sf.net/p/libpng/code")

	def configure(self, buildsystem, zlib):
		for build_dir, build_type in self.builds():
			build_dir.mkdir(parents=True, exist_ok=True)
			flags = [
				f"-DCMAKE_BUILD_TYPE={build_type}",
				f"-DCMAKE_INSTALL_PREFIX:PATH={self.install_dir}",
				f"-DZLIB_ROOT:PATH={zlib.install_dir}",
			]
			buildsystem.configure(build_dir, self.source_dir, *flags)

	def build(self):
		for build_dir, build_type in self.builds():
			ninja("-C", str(build_dir), "install")
		return [self.install_dir/'bin'/'libpng16d.dll', self.install_dir/'bin'/'libpng16.dll']

def buildDependencies(dependencies_dir, pull):
	dependencies_dir.mkdir(parents=True, exist_ok=True)

	llvm = LLVM(dependencies_dir/"llvm")
	anydsl = AnyDSL(dependencies_dir/"anydsl")
	zlib = ZLIB(dependencies_dir/"zlib")
	libpng = LPNG(dependencies_dir/"libpng")

	if pull:
		llvm.pull()
		anydsl.pull()
		zlib.pull()
		libpng.pull()

	buildsystem = NinjaBuild()

	dlls = []
	llvm.configure(buildsystem)
	dlls.extend(llvm.build())

	anydsl.configure(buildsystem, llvm)
	dlls.extend(anydsl.buildRuntime())
	dlls.extend(anydsl.buildImpala())

	zlib.configure(buildsystem)
	dlls.extend(zlib.build())

	libpng.configure(buildsystem, zlib)
	dlls.extend(libpng.build())

	return dlls

def copyDlls(build_dir, deps):
	for bin_dir in [build_dir/"bin"/"Debug", build_dir/"bin"/"Release"]:
		bin_dir.mkdir(parents=True, exist_ok=True)
		for d in deps:
			shutil.copy(d, bin_dir)


def selectBuildsystem():
	if os.name == "nt":
		return VS2019Build()
	raise Exception("non-windows build not there yet")

def main(args):
	this_dir = Path(__file__).parent
	dependencies_dir = this_dir/"dependencies"
	build_dir = this_dir/"build"
	src_dir = this_dir

	build_dir.mkdir(parents=True, exist_ok=True)


	if args.dependencies:
		dlls = buildDependencies(dependencies_dir, args.pull)
		# copyDlls(build_dir, dlls)


	buildsystem = selectBuildsystem()

	if args.remove:
		buildsystem.remove(build_dir)

	if args.configure:
		flags = [
			f"-DZLIB_ROOT={dependencies_dir/'zlib'}",
			f"-DPNG_ROOT={dependencies_dir/'libpng'}",
			# f"-DPNG_CONFIG={dependencies_dir/'libpng'/'lib'/'libpng'/'libpng16.cmake'}",
			f"-DAnyDSL_runtime_DIR={dependencies_dir/'anydsl'/'build'/'share'/'anydsl'/'cmake'}",
		]
		buildsystem.configure(build_dir, src_dir, *flags)



if __name__ == "__main__":
	args = argparse.ArgumentParser()
	args.add_argument("-pull", "--pull", action="store_true")
	args.add_argument("-deps", "--dependencies", action="store_true")
	# args.add_argument("-dep", "--dependency", action="append", choices={"llvm", "runtime", "impala", "zlib", "libpng"})
	args.add_argument("-conf", "--configure", action="store_true")
	args.add_argument("-rm", "--remove", action="store_true")
	main(args.parse_args())
