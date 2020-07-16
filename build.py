#!/usr/bin/env python3

from pathlib import Path
import argparse
from buildsystem import *


@component()
class AnyDSL(Build):
	def __init__(self, dir, buildsystem):
		super().__init__(buildsystem)
		self.source_dir = dir/"anydsl"

	def pull(self):
		pull_git_dependency(self.source_dir, "https://github.com/AnyDSL/anydsl.git")

@component(depends_on=(AnyDSL,))
class LLVM(Build):
	def __init__(self, dir, buildsystem):
		super().__init__(buildsystem, dir/"llvm"/"build")
		self.source_dir = dir/"llvm"/"source"
		self.rv_source_dir = self.source_dir/"rv"

	def pull(self):
		pull_git_dependency(self.source_dir, "https://github.com/llvm/llvm-project.git", branch="llvmorg-10.0.0")
		pull_git_dependency(self.rv_source_dir, "https://github.com/cdl-saarland/rv.git", "--recurse-submodules", branch="release/10.x")

	def configure(self, anydsl):
		patch = anydsl.source_dir/"nvptx_feature_ptx60.patch"

		if cmd("git", "-C", str(self.source_dir), "apply", "--check", "--reverse", str(patch)) != 0:
			git("-C", str(self.source_dir), "apply", "-v", str(patch))

		self.buildsystem.configure(self.build_dir, self.source_dir/"llvm",
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
			 "-DLLVM_EXTERNAL_PROJECTS=RV",
			f"-DLLVM_EXTERNAL_RV_SOURCE_DIR:PATH={self.rv_source_dir}",
			 "-DLLVM_TOOL_CLANG_BUILD=ON",
			 "-DLLVM_TOOL_LLD_BUILD=ON",
			 "-DLLVM_TOOL_RV_BUILD=ON",
			 "-DCLANG_INCLUDE_DOCS=OFF",
			 "-DCLANG_INCLUDE_TESTS=OFF"
		)

	def build(self):
		ninja("-C", str(self.build_dir), "clang", "llvm-as", "lld", "RV", "LLVMRV", "LLVMMCJIT", "LLVMExecutionEngine", "LLVMRuntimeDyld")

@component(depends_on=(LLVM,))
class Thorin(Build):
	def __init__(self, dir, buildsystem):
		super().__init__(buildsystem, dir/"anydsl"/"build"/"thorin")
		self.source_dir = dir/"anydsl"/"thorin"
		self.half_source_dir = self.source_dir/"half"

	def pull(self):
		pull_git_dependency(self.source_dir, "https://github.com/AnyDSL/thorin.git")
		pull_svn_dependency(self.half_source_dir, "https://svn.code.sf.net/p/half/code/tags/release-1.11.0")

	def configure(self, llvm):
		self.buildsystem.configure(self.build_dir, self.source_dir,
			 "-DCMAKE_BUILD_TYPE=Release",
			f"-DLLVM_DIR:PATH={llvm.build_dir/'lib'/'cmake'/'llvm'}",
			f"-DRV_INCLUDE_DIR:PATH={llvm.rv_source_dir/'include'}",
			f"-DRV_LIBRARY:PATH={llvm.build_dir/'lib'/'RV.lib'}",
			f"-DRV_SLEEF_LIBRARY:PATH={llvm.build_dir/'lib'/'gensleef.lib'}",
			f"-DHalf_INCLUDE_DIR:PATH={self.half_source_dir/'include'}",
			 "-DBUILD_SHARED_LIBS=OFF"
		)

	def build(self):
		ninja("-C", str(self.build_dir), "thorin")

@component(depends_on=(Thorin,))
class Impala(Build):
	def __init__(self, dir, buildsystem):
		super().__init__(buildsystem, dir/"anydsl"/"build"/"impala")
		self.source_dir = dir/"anydsl"/"impala"

	def pull(self):
		pull_git_dependency(self.source_dir, "https://github.com/AnyDSL/impala.git")

	def configure(self, thorin):
		self.buildsystem.configure(self.build_dir, self.source_dir,
			 "-DCMAKE_BUILD_TYPE=Release",
			f"-DThorin_DIR:PATH={thorin.build_dir/'share'/'anydsl'/'cmake'}",
			 "-DBUILD_SHARED_LIBS=OFF"
		)

	def build(self):
		ninja("-C", str(self.build_dir), "impala")

@component(depends_on=(LLVM, Thorin, Impala))
class AnyDSLRuntime(Build):
	def __init__(self, dir, buildsystem):
		super().__init__(buildsystem, dir/"anydsl"/"build"/"runtime")
		self.source_dir = dir/"anydsl"/"runtime"
		self.debug_dlls = [self.build_dir/'bin'/'runtime.dll']
		self.release_dlls = [self.build_dir/'bin'/'runtime.dll']

	def pull(self):
		pull_git_dependency(self.source_dir, "https://github.com/AnyDSL/runtime.git")

	def configure(self, llvm, thorin, impala):
		self.buildsystem.configure(self.build_dir, self.source_dir,
			 "-DCMAKE_BUILD_TYPE=Release",
			f"-DThorin_DIR:PATH={thorin.build_dir/'share'/'anydsl'/'cmake'}",
			f"-DImpala_DIR:PATH={impala.build_dir/'share'/'anydsl'/'cmake'}",
			 "-DRUNTIME_JIT=ON",
			 "-DAnyDSL_runtime_BUILD_SHARED=ON"
		)

	def build(self):
		ninja("-C", str(self.build_dir), "runtime")

@component()
class ZLIB(MultiConfigBuild):
	def __init__(self, dir, buildsystem):
		super().__init__(buildsystem, dir/"zlib"/"build")
		self.install_dir = dir/"zlib"
		self.source_dir = dir/"zlib"/"source"
		self.debug_dlls = [self.install_dir/'bin'/'zlibd.dll']
		self.release_dlls = [self.install_dir/'bin'/'zlib.dll']

	def pull(self):
		pull_git_dependency(self.source_dir, "https://github.com/madler/zlib.git")

	def configure_config(self, build_dir, build_type):
		build_dir.mkdir(parents=True, exist_ok=True)
		self.buildsystem.configure(build_dir, self.source_dir,
			f"-DCMAKE_BUILD_TYPE={build_type}",
			f"-DCMAKE_INSTALL_PREFIX:PATH={self.install_dir}"
		)

	def build_config(self, build_dir, build_type):
		ninja("-C", str(build_dir), "install")

@component(depends_on=(ZLIB,))
class LPNG(MultiConfigBuild):
	def __init__(self, dir, buildsystem):
		super().__init__(buildsystem, dir/"libpng"/"build")
		self.install_dir = dir/"libpng"
		self.source_dir = dir/"libpng"/"source"
		self.debug_dlls = [self.install_dir/'bin'/'libpng16d.dll']
		self.release_dlls = [self.install_dir/'bin'/'libpng16.dll']

	def pull(self):
		pull_git_dependency(self.source_dir, "git://git.code.sf.net/p/libpng/code")

	def configure_config(self, build_dir, build_type, zlib):
		build_dir.mkdir(parents=True, exist_ok=True)
		self.buildsystem.configure(build_dir, self.source_dir,
			f"-DCMAKE_BUILD_TYPE={build_type}",
			f"-DCMAKE_INSTALL_PREFIX:PATH={self.install_dir}",
			f"-DZLIB_ROOT:PATH={zlib.install_dir}"
		)

	def build_config(self, build_dir, build_type):
		ninja("-C", str(build_dir), "install")

@component(depends_on=(ZLIB, LPNG, AnyDSLRuntime))
class AnyQ(Build):
	def __init__(self, dir, buildsystem):
		super().__init__(buildsystem, dir/"build")
		self.source_dir = dir

	def configure(self, zlib, libpng, runtime):
		self.buildsystem.configure(self.build_dir, self.source_dir,
			f"-DZLIB_ROOT={zlib.install_dir}",
			f"-DPNG_ROOT={libpng.install_dir}",
			f"-DAnyDSL_runtime_DIR={runtime.build_dir/'share'/'anydsl'/'cmake'}"
		)


dependency_name_map = { "zlib": ZLIB, "libpng": LPNG, "llvm": LLVM, "thorin": Thorin, "impala": Impala, "runtime": AnyDSLRuntime, "anydsl": AnyDSL }

def lookup_dependency(name):
	d = dependency_name_map.get(name)
	if not d:
		raise Exception(f"unknown dependency '{name}'")
	return d

def main(args):
	this_dir = Path(__file__).parent

	def collect_components(dir, args):
		dependencies_dir = dir/"dependencies"

		def create_component(C):
			if C is AnyQ:
				return AnyQ(dir, select_buildsystem())
			return C(dependencies_dir, NinjaBuild())

		yield from instantiate_dependencies(create_component, map(lookup_dependency, args.dependencies) if args.dependencies else [AnyQ])


	components = list(collect_components(this_dir, args))

	if args.remove:
		for c in components:
			c.remove()
		return

	if args.pull:
		for c in components:
			c.pull()
		return

	for c in components:
		c.configure(*c.dependencies)
		c.build()

	copy_dlls(this_dir/"build"/"bin"/"Debug", components, "debug_dlls")
	copy_dlls(this_dir/"build"/"bin"/"Release", components, "release_dlls")



if __name__ == "__main__":
	args = argparse.ArgumentParser()
	args.add_argument("-pull", "--pull", action="store_true")
	args.add_argument("-deps", "--dependencies", action="store_const", const=list(dependency_name_map.keys()), default=[])
	args.add_argument("-dep", "--dependency", action="append", dest="dependencies", choices=dependency_name_map.keys())
	# args.add_argument("-conf", "--configure", action="store_true")
	args.add_argument("-rm", "--remove", action="store_true")
	main(args.parse_args())
