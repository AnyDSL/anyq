#!/usr/bin/env python3

from pathlib import Path
import argparse
from buildsystem import *


def _git_apply_patch(dir, patch):
	if cmd("git", "-C", dir, "apply", "--check", "--reverse", patch) != 0:
		git("-C", dir, "apply", "-v", patch)

@component()
class Boost:
	def __init__(self, dir, buildsystem):
		self.source_dir = dir/"boost"
		self.build_dir = self.source_dir/"build"

	def pull(self):
		pull_git_dependency(self.source_dir, "https://github.com/boostorg/boost.git", branch="boost-1.72.0")
		git("-C", self.source_dir,"submodule", "update", "--init",
			"tools", "libs/config", "libs/predef", "libs/headers", "libs/core", "libs/detail",
			"libs/type_traits",
			"libs/preprocessor",
			"libs/assert",
			"libs/static_assert",
			"libs/mpl",
			"libs/iterator",
			"libs/move",
			"libs/smart_ptr",
			"libs/intrusive",
			"libs/functional",
			"libs/container_hash",
			"libs/context",
			"libs/system",
			"libs/winapi",
			"libs/io",
			"libs/filesystem",
			"libs/fiber",
			"libs/context",
			"libs/filesystem"
		)

	def configure(self):
		if (self.source_dir/"b2.exe").exists():
			return

		if cmd("cmd", "/C", "bootstrap.bat", cwd=self.source_dir) != 0:
			raise Exception("failed to bootstrap boost")

	def remove(self):
		rmtree(self.build_dir)

	def build(self):
		if cmd(self.source_dir/"b2", "install", f"--prefix={self.build_dir}", "--with-fiber", cwd=self.source_dir) != 0:
			raise Exception("failed to build boost")

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
		pull_git_dependency(self.source_dir, "https://github.com/llvm/llvm-project.git", branch="llvmorg-11.0.0")
		pull_git_dependency(self.rv_source_dir, "https://github.com/cdl-saarland/rv.git", "--recurse-submodules", branch="release/11.x")

	def configure(self, anydsl):
		patches = [
			anydsl.source_dir/"nvptx_feature_ptx60.patch"
		]

		for patch in patches:
			_git_apply_patch(self.source_dir, patch)

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
			 "-DCLANG_INCLUDE_TESTS=OFF",
			 "-DLLVM_RVPLUG_LINK_INTO_TOOLS:BOOL=ON"
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
		pull_git_dependency(self.source_dir, "https://github.com/AnyDSL/thorin.git", branch="llvm/11.x")
		pull_svn_dependency(self.half_source_dir, "https://svn.code.sf.net/p/half/code/tags/release-1.11.0")

	def configure(self, llvm):
		self.buildsystem.configure(self.build_dir, self.source_dir,
			 "-DCMAKE_BUILD_TYPE=Release",
			f"-DLLVM_DIR:PATH={llvm.build_dir/'lib'/'cmake'/'llvm'}",
			f"-DRV_INCLUDE_DIR:PATH={llvm.rv_source_dir/'include'}",
			f"-DRV_LIBRARY:PATH={llvm.build_dir/'lib'/'RV.lib'}",
			# f"-DRV_SLEEF_LIBRARY:PATH={llvm.build_dir/'lib'/'gensleef.lib'}",
			f"-DRV_SLEEF_LIBRARY:PATH=",
			f"-DHalf_INCLUDE_DIR:PATH={self.half_source_dir/'include'}",
			 "-DBUILD_SHARED_LIBS=OFF"
		)

	def build(self):
		ninja("-C", str(self.build_dir), "thorin")

@component(depends_on=(Thorin,))
class Artic(Build):
	def __init__(self, dir, buildsystem):
		super().__init__(buildsystem, dir/"anydsl"/"build"/"artic")
		self.source_dir = dir/"anydsl"/"artic"

	def pull(self):
		pull_git_dependency(self.source_dir, "https://github.com/AnyDSL/artic.git")

	def configure(self, thorin):
		self.buildsystem.configure(self.build_dir, self.source_dir,
			 "-DCMAKE_BUILD_TYPE=Release",
			f"-DThorin_DIR:PATH={thorin.build_dir/'share'/'anydsl'/'cmake'}",
			 "-DBUILD_SHARED_LIBS=OFF"
		)

	def build(self):
		ninja("-C", str(self.build_dir), "artic")

@component(depends_on=(LLVM, Thorin, Artic))
class AnyDSLRuntime(Build):
	def __init__(self, dir, buildsystem):
		super().__init__(buildsystem, dir/"anydsl"/"build"/"runtime")
		self.source_dir = dir/"anydsl"/"runtime"
		self.debug_dlls = [self.build_dir/'bin'/'runtime.dll']
		self.release_dlls = [self.build_dir/'bin'/'runtime.dll']

	def pull(self):
		pull_git_dependency(self.source_dir, "https://github.com/AnyDSL/runtime.git", branch="artic")

	def configure(self, llvm, thorin, artic):
		self.buildsystem.configure(self.build_dir, self.source_dir,
			 "-DCMAKE_BUILD_TYPE=Release",
			f"-DThorin_DIR:PATH={thorin.build_dir/'share'/'anydsl'/'cmake'}",
			f"-DImpala_DIR:PATH={artic.build_dir/'share'/'anydsl'/'cmake'}",
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

@component(depends_on=(ZLIB, LPNG, Boost, Thorin, Artic, AnyDSLRuntime))
class AnyQ(Build):
	def __init__(self, dir, buildsystem):
		super().__init__(buildsystem, dir/"build")
		self.source_dir = dir

	def configure(self, zlib, libpng, boost, thorin, artic, runtime):
		self.buildsystem.configure(self.build_dir, self.source_dir,
			f"-DZLIB_ROOT={zlib.install_dir}",
			f"-DPNG_ROOT={libpng.install_dir}",
			f"-DBoost_ROOT={boost.build_dir}",
			# f"-DThorin_DIR:PATH={thorin.build_dir/'share'/'anydsl'/'cmake'}",
			f"-DArtic_BINARY_DIR:PATH={artic.build_dir/'bin'}",
			f"-DAnyDSL_runtime_DIR={runtime.build_dir/'share'/'anydsl'/'cmake'}"
		)


dependency_name_map = { "boost": Boost, "zlib": ZLIB, "libpng": LPNG, "llvm": LLVM, "thorin": Thorin, "artic": Artic, "runtime": AnyDSLRuntime, "anydsl": AnyDSL }

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
			print("--------------------------------")
			c.remove()
			print()
		return

	if args.pull:
		for c in components:
			print("--------------------------------")
			c.pull()
			print()
		return

	for c in components:
		print("--------------------------------")
		c.configure(*c.dependencies)
		c.build()
		print()

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
