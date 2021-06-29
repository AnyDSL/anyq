#!/usr/bin/env python3

from pathlib import Path
import argparse

from buildsystem import *


@component()
class Boost:
	def __init__(self, dir, buildsystem):
		self.source_dir = dir/"dependencies"/"boost"
		self.build_dir = self.source_dir/"build"

	def pull(self):
		pull_git_dependency(self.source_dir, "https://github.com/boostorg/boost.git", branch="boost-1.76.0")
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

	def configure(self, configs):
		if (self.source_dir/"b2.exe").exists():
			return

		if cmd("cmd", "/C", "bootstrap.bat", cwd=self.source_dir) != 0:
			raise Exception("failed to bootstrap boost")

	def remove_build(self):
		rmtree(self.build_dir)

	def build(self, config):
		if cmd(self.source_dir/"b2", "install", f"--prefix={self.build_dir}", "--with-fiber", cwd=self.source_dir) != 0:
			raise Exception("failed to build boost")

@component()
class AnyDSL(Build):
	def __init__(self, dir, buildsystem):
		super().__init__(buildsystem, None)
		self.source_dir = dir/"dependencies"/"anydsl"

	def pull(self):
		pull_git_dependency(self.source_dir, "https://github.com/AnyDSL/anydsl.git")

@component(depends_on=(AnyDSL,))
class LLVM(Build):
	def __init__(self, dir, buildsystem):
		super().__init__(buildsystem, dir/"dependencies"/"llvm"/"build")
		self.source_dir = dir/"dependencies"/"llvm"/"source"
		self.rv_source_dir = self.source_dir/"rv"

	def pull(self):
		pull_git_dependency(self.source_dir, "https://github.com/llvm/llvm-project.git", branch="llvmorg-12.0.0")
		pull_git_dependency(self.rv_source_dir, "https://github.com/cdl-saarland/rv.git", "--recurse-submodules", branch="release/12.x")

	def configure(self, configs, anydsl):
		for patch in [
			anydsl.source_dir/"nvptx_feature_ptx60.patch"
		]:
			git_apply_patch(self.source_dir, patch)

		self.buildsystem.configure(self.build_dir, configs, self.source_dir/"llvm",
			 "-DLLVM_OPTIMIZED_TABLEGEN=ON",
			 "-DLLVM_TARGETS_TO_BUILD=X86;NVPTX",
			 "-DLLVM_ENABLE_RTTI=ON",
			 "-DLLVM_INCLUDE_UTILS=OFF",
			 "-DLLVM_INCLUDE_TESTS=OFF",
			 "-DLLVM_INCLUDE_EXAMPLES=OFF",
			 "-DLLVM_INCLUDE_BENCHMARKS=OFF",
			 "-DLLVM_INCLUDE_DOCS=OFF",
			 "-DLLVM_ENABLE_BINDINGS=OFF",
			 "-DLLVM_ENABLE_PROJECTS=clang;lld",
			 "-DLLVM_EXTERNAL_PROJECTS=RV",
			f"-DLLVM_EXTERNAL_RV_SOURCE_DIR:PATH={self.rv_source_dir}",
			 "-DLLVM_TOOL_CLANG_BUILD=ON",
			 "-DLLVM_TOOL_LLD_BUILD=ON",
			 "-DLLVM_TOOL_RV_BUILD=ON",
			 "-DCLANG_INCLUDE_DOCS=OFF",
			 "-DCLANG_INCLUDE_TESTS=OFF",
			 "-DLLVM_RVPLUG_LINK_INTO_TOOLS:BOOL=OFF"
		)

	def build(self, config):
		self.buildsystem.build(self.build_dir, config, "clang", "llvm-as", "lld", "LLVMExecutionEngine", "LLVMRuntimeDyld")
		self.buildsystem.build(self.build_dir, config, "RV", "LLVMRV", "LLVMMCJIT")

@component(depends_on=(LLVM,))
class Thorin(Build):
	def __init__(self, dir, buildsystem):
		super().__init__(buildsystem, dir/"dependencies"/"anydsl"/"build"/"thorin")
		self.source_dir = dir/"dependencies"/"anydsl"/"thorin"
		self.half_source_dir = self.source_dir/"half"

	def pull(self):
		pull_git_dependency(self.source_dir, "https://github.com/AnyDSL/thorin.git")
		pull_svn_dependency(self.half_source_dir, "https://svn.code.sf.net/p/half/code/tags/release-2.2.0")

	def configure(self, configs, llvm):
		self.buildsystem.configure(self.build_dir, configs, self.source_dir,
			f"-DLLVM_DIR:PATH={llvm.build_dir/'lib'/'cmake'/'llvm'}",
			f"-DHalf_INCLUDE_DIR:PATH={self.half_source_dir/'include'}",
			 "-DBUILD_SHARED_LIBS=OFF"
		)

	def build(self, config):
		self.buildsystem.build(self.build_dir, config, "thorin")

@component(depends_on=(Thorin,))
class Artic(Build):
	def __init__(self, dir, buildsystem):
		super().__init__(buildsystem, dir/"dependencies"/"anydsl"/"build"/"artic")
		self.source_dir = dir/"dependencies"/"anydsl"/"artic"

	def pull(self):
		pull_git_dependency(self.source_dir, "https://github.com/AnyDSL/artic.git")

	def configure(self, configs, thorin):
		self.buildsystem.configure(self.build_dir, configs, self.source_dir,
			f"-DThorin_DIR:PATH={thorin.build_dir/'share'/'anydsl'/'cmake'}",
			 "-DBUILD_SHARED_LIBS=OFF"
		)

	def build(self, config):
		self.buildsystem.build(self.build_dir, config, "artic")

@component(depends_on=(LLVM, Thorin, Artic))
class AnyDSLRuntime(Build):
	def __init__(self, dir, buildsystem):
		super().__init__(buildsystem, dir/"dependencies"/"anydsl"/"build"/"runtime")
		self.source_dir = dir/"dependencies"/"anydsl"/"runtime"

	def pull(self):
		pull_git_dependency(self.source_dir, "https://github.com/AnyDSL/runtime.git")

	def configure(self, configs, llvm, thorin, artic):
		self.buildsystem.configure(self.build_dir, configs, self.source_dir,
			f"-DLLVM_DIR:PATH={llvm.build_dir/'lib'/'cmake'/'llvm'}",
			f"-DThorin_DIR:PATH={thorin.build_dir/'share'/'anydsl'/'cmake'}",
			f"-DArtic_DIR:PATH={artic.build_dir/'share'/'anydsl'/'cmake'}",
			 "-DRUNTIME_JIT=ON",
			 "-DAnyDSL_runtime_BUILD_SHARED=ON"
		)

	def build(self, config):
		self.buildsystem.build(self.build_dir, config, "runtime", "runtime_jit_artic")

@component(depends_on=(Thorin, Artic, AnyDSLRuntime))
class AnyDSL(PhonyBuild):
	pass

@component()
class ZLIB(Build):
	def __init__(self, dir, buildsystem):
		super().__init__(buildsystem, dir/"dependencies"/"zlib"/"build")
		self.install_dir = dir/"dependencies"/"zlib"
		self.source_dir = dir/"dependencies"/"zlib"/"source"

	def pull(self):
		pull_git_dependency(self.source_dir, "https://github.com/madler/zlib.git")

	def configure(self, configs):
		self.buildsystem.configure(self.build_dir, configs, self.source_dir,
			f"-DCMAKE_INSTALL_PREFIX:PATH={self.install_dir}"
		)

	def build_config(self, config):
		self.buildsystem.build(self.build_dir, config, "install")

@component(depends_on=(ZLIB,))
class LPNG(Build):
	def __init__(self, dir, buildsystem):
		super().__init__(buildsystem, dir/"dependencies"/"libpng"/"build")
		self.install_dir = dir/"dependencies"/"libpng"
		self.source_dir = dir/"dependencies"/"libpng"/"source"

	def pull(self):
		pull_git_dependency(self.source_dir, "git://git.code.sf.net/p/libpng/code")

	def configure(self, configs, zlib):
		self.buildsystem.configure(self.build_dir, configs, self.source_dir,
			f"-DCMAKE_INSTALL_PREFIX:PATH={self.install_dir}",
			f"-DZLIB_ROOT:PATH={zlib.install_dir}"
		)

	def build(self, config):
		self.buildsystem.build(self.build_dir, config, "install")

@component(depends_on=(ZLIB, LPNG, Boost, Artic, AnyDSLRuntime))
class AnyQ(Build):
	def __init__(self, dir, buildsystem):
		super().__init__(buildsystem, dir/"build")
		self.source_dir = dir

	def configure(self, configs, zlib, libpng, boost, artic, runtime):
		self.buildsystem.configure(self.build_dir, configs, self.source_dir,
			f"-DZLIB_ROOT={zlib.install_dir}",
			f"-DPNG_ROOT={libpng.install_dir}",
			f"-DBoost_ROOT={boost.build_dir}",
			f"-DArtic_BINARY_DIR:PATH={artic.build_dir/'bin'}",
			f"-DAnyDSL_runtime_DIR={runtime.build_dir/'share'/'anydsl'/'cmake'}"
		)

@component(depends_on=(AnyQ,))
class All(PhonyBuild):
	pass


dependency_name_map = { "boost": Boost, "zlib": ZLIB, "libpng": LPNG, "llvm": LLVM, "thorin": Thorin, "artic": Artic, "runtime": AnyDSLRuntime, "anydsl": AnyDSL, "anyq": AnyQ, "all": All }

def lookup_dependency(name):
	d = dependency_name_map.get(name)
	if not d:
		raise Exception(f"unknown component '{name}'")
	return d

def dependencies(dir, components):
	def create_component(C):
		return C(dir, NinjaBuild())

	yield from instantiate_dependencies(create_component, components)


def pull(components, configs):
	for c in components:
		c.pull()
		print()

def build(components, configs):
	for c in components:
		c.configure(configs, *c.dependencies)
		for config in configs:
			c.build(config)
		print()

def remove(components, configs):
	for c in components:
		c.remove_build()
		print()

def main(args):
	this_dir = Path(__file__).parent
	components = dependencies(this_dir, map(lookup_dependency, args.components) if args.components else [All])

	def take_last(things):
		for thing in things:
			pass
		yield thing

	args.command(components if not args.no_deps else take_last(components), args.configs)


if __name__ == "__main__":
	args = argparse.ArgumentParser()
	sub_args = args.add_subparsers(required=True)

	def add_command(name, function):
		args = sub_args.add_parser(name)
		args.set_defaults(command=function)
		args.add_argument("components", nargs="*")#, choices=dependency_name_map.keys())
		args.add_argument("-cfg", "--config", action="append", dest="configs", default=["Debug", "Release"])
		args.add_argument("--no-deps", action="store_true")
		return args

	add_command("pull", pull)
	add_command("build", build)
	add_command("rm", remove)

	main(args.parse_args())
