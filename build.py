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
		pull_git_dependency(self.source_dir, "https://github.com/boostorg/boost.git", branch="boost-1.80.0")
		git("-C", self.source_dir,"submodule", "update", "--init",
			"tools",
			"libs/algorithm",
			"libs/align",
			"libs/array",
			"libs/assert",
			"libs/atomic",
			"libs/bind",
			"libs/concept_check",
			"libs/config",
			"libs/container_hash",
			"libs/context",
			"libs/conversion",
			"libs/core",
			"libs/detail",
			"libs/exception",
			"libs/fiber",
			"libs/filesystem",
			"libs/format",
			"libs/function",
			"libs/function_types",
			"libs/fusion",
			"libs/headers",
			"libs/integer",
			"libs/intrusive",
			"libs/io",
			"libs/iterator",
			"libs/move",
			"libs/mp11",
			"libs/mpl",
			"libs/optional",
			"libs/pool",
			"libs/predef",
			"libs/preprocessor",
			"libs/range",
			"libs/regex",
			"libs/smart_ptr",
			"libs/static_assert",
			"libs/system",
			"libs/throw_exception",
			"libs/tuple",
			"libs/type_index",
			"libs/type_traits",
			"libs/typeof",
			"libs/unordered",
			"libs/utility",
			"libs/variant2",
			"libs/winapi",
		)

	def __bootstrap(self):
		if windows:
			return cmd("cmd", "/C", "bootstrap.bat", cwd=self.source_dir)
		return cmd("./bootstrap.sh", shell=True, cwd=self.source_dir)

	def configure(self, configs):
		print("-------- configuring", self.build_dir, "--------")
		if (self.source_dir/"b2").exists() or (self.source_dir/"b2.exe").exists():
			return

		if self.__bootstrap() != 0:
			raise Exception("failed to bootstrap boost")

	def remove_build(self):
		rmtree(self.build_dir)

	def build(self, config):
		print("-------- building", self.build_dir, "--------")
		if cmd(self.source_dir/"b2", "install", f"--prefix={self.build_dir}", cwd=self.source_dir) != 0:
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
		pull_git_dependency(self.source_dir, "https://github.com/llvm/llvm-project.git", branch="llvmorg-16.0.6")
		pull_git_dependency(self.rv_source_dir, "https://github.com/cdl-saarland/rv.git", "--recurse-submodules", branch="release/16.x")

	def configure(self, configs, anydsl):
		for patch in [
			anydsl.source_dir/"patches"/"llvm"/"nvptx_feature.patch"
		]:
			git_apply_patch(self.source_dir, patch)

		self.buildsystem.configure(self.build_dir, configs, self.source_dir/"llvm",
			LLVM_OPTIMIZED_TABLEGEN=True,
			LLVM_TARGETS_TO_BUILD="X86;NVPTX",
			LLVM_ENABLE_RTTI=True,
			LLVM_INCLUDE_UTILS=False,
			LLVM_INCLUDE_TESTS=False,
			LLVM_INCLUDE_EXAMPLES=False,
			LLVM_INCLUDE_BENCHMARKS=False,
			LLVM_INCLUDE_DOCS=False,
			LLVM_ENABLE_BINDINGS=False,
			LLVM_ENABLE_PROJECTS="clang;lld",
			LLVM_EXTERNAL_PROJECTS="RV",
			LLVM_EXTERNAL_RV_SOURCE_DIR=self.rv_source_dir,
			LLVM_TOOL_CLANG_BUILD=True,
			LLVM_TOOL_LLD_BUILD=True,
			LLVM_TOOL_RV_BUILD=True,
			CLANG_INCLUDE_DOCS=False,
			CLANG_INCLUDE_TESTS=False,
			RV_ENABLE_PLUGIN=False,
			LLVM_RVPLUG_LINK_INTO_TOOLS=False,
			LLVM_BUILD_LLVM_DYLIB=False if windows else True,
			LLVM_LINK_LLVM_DYLIB=False if windows else True
		)

	def build(self, config):
		self.buildsystem.build(self.build_dir, config, "clang", "llvm-as", "lld", "LLVMExecutionEngine", "LLVMRuntimeDyld")
		self.buildsystem.build(self.build_dir, config, "RV", "LLVMMCJIT")

@component(depends_on=(LLVM,))
class Thorin(Build):
	def __init__(self, dir, buildsystem):
		super().__init__(buildsystem, dir/"dependencies"/"anydsl"/"build"/"thorin")
		self.source_dir = dir/"dependencies"/"anydsl"/"thorin"
		self.half_source_dir = self.source_dir/"half"

	def pull(self):
		pull_git_dependency(self.source_dir, "https://github.com/AnyDSL/thorin.git", branch="master")
		pull_svn_dependency(self.half_source_dir, "https://svn.code.sf.net/p/half/code/tags/release-2.2.0")

	def configure(self, configs, llvm):
		self.buildsystem.configure(self.build_dir, configs, self.source_dir,
			LLVM_DIR=llvm.build_dir/"lib"/"cmake"/"llvm",
			Half_INCLUDE_DIR=self.half_source_dir/"include",
			THORIN_PROFILE=False,
			BUILD_SHARED_LIBS=False if windows else True
		)

	def build(self, config):
		self.buildsystem.build(self.build_dir, config, "thorin")

@component(depends_on=(Thorin,))
class Artic(Build):
	def __init__(self, dir, buildsystem):
		super().__init__(buildsystem, dir/"dependencies"/"anydsl"/"build"/"artic")
		self.source_dir = dir/"dependencies"/"anydsl"/"artic"

	def pull(self):
		pull_git_dependency(self.source_dir, "https://github.com/AnyDSL/artic.git", branch="master")

	def configure(self, configs, thorin):
		self.buildsystem.configure(self.build_dir, configs, self.source_dir,
			Thorin_DIR=thorin.build_dir/"share"/"anydsl"/"cmake"
		)

	def build(self, config):
		self.buildsystem.build(self.build_dir, config, "artic")

@component(depends_on=(AnyDSL, LLVM, Thorin, Artic))
class AnyDSLRuntime(Build):
	def __init__(self, dir, buildsystem):
		super().__init__(buildsystem, dir/"dependencies"/"anydsl"/"build"/"runtime")
		self.source_dir = dir/"dependencies"/"anydsl"/"runtime"

	def pull(self):
		pull_git_dependency(self.source_dir, "https://github.com/AnyDSL/runtime.git", branch="master")

	def configure(self, configs, anydsl, llvm, thorin, artic):
		if windows:
			for patch in [
				anydsl.source_dir/"patches"/"runtime"/"cmake_multiconfig.patch"
			]:
				git_apply_patch(self.source_dir, patch)

		self.buildsystem.configure(self.build_dir, configs, self.source_dir,
			LLVM_DIR=llvm.build_dir/"lib"/"cmake"/"llvm",
			Thorin_DIR=thorin.build_dir/"share"/"anydsl"/"cmake",
			Artic_DIR=artic.build_dir/"share"/"anydsl"/"cmake",
			RUNTIME_JIT=True,
			AnyDSL_runtime_CUDA_CXX_STANDARD=17
		)

	def build(self, config):
		self.buildsystem.build(self.build_dir, config, "runtime", "runtime_jit_artic")

@component(depends_on=(Thorin, Artic, AnyDSLRuntime))
class AnyDSL(PhonyBuild):
	pass

@component(depends_on=(Boost, AnyDSLRuntime))
class AnyQ(Build):
	def __init__(self, dir, buildsystem):
		super().__init__(buildsystem, dir/"build")
		self.source_dir = dir

	def configure(self, configs, boost, runtime):
		self.buildsystem.configure(self.build_dir, configs, self.source_dir,
			Boost_NO_SYSTEM_PATHS=True,
			Boost_ROOT=boost.build_dir,
			AnyDSL_runtime_DIR=runtime.build_dir/"share"/"anydsl"/"cmake",
			BUILD_TESTING=True,
			CMAKE_EXPORT_COMPILE_COMMANDS=True,
			AnyQ_CUDA_SANITIZER_TOOLS=""
		)

@component(depends_on=(AnyQ,))
class All(PhonyBuild):
	pass


dependency_name_map = { "boost": Boost, "llvm": LLVM, "thorin": Thorin, "artic": Artic, "runtime": AnyDSLRuntime, "anydsl": AnyDSL, "anyq": AnyQ, "all": All }

def lookup_dependency(name):
	d = dependency_name_map.get(name)
	if not d:
		raise Exception(f"unknown component '{name}'")
	return d

def dependencies(dir, components, buildsystem):
	def create_component(C):
		return C(dir, buildsystem)

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

	# avoid passing -DNDEBUG for Release builds
	global_flags = [
		"-DCMAKE_C_FLAGS_RELEASE=/O2", "-DCMAKE_CXX_FLAGS_RELEASE=/O2"
	] if windows else [
		"-DCMAKE_C_COMPILER=gcc-11", "-DCMAKE_CXX_COMPILER=g++-11",
		"-DCMAKE_C_FLAGS_RELEASE=-O3", "-DCMAKE_CXX_FLAGS_RELEASE=-O3"
	]

	buildsystem = NinjaMultiConfigBuild(global_flags=global_flags) if windows else NinjaBuild(global_flags=global_flags)

	components = dependencies(this_dir, map(lookup_dependency, args.components) if args.components else [All], buildsystem)

	def take_last(things):
		for thing in things:
			pass
		yield thing

	configs = args.configs if args.configs else ["Debug", "Release"] if windows else ["Debug"]
	args.command(components if not args.no_deps else take_last(components), configs)


if __name__ == "__main__":
	args = argparse.ArgumentParser()
	sub_args = args.add_subparsers(required=True)

	def add_command(name, function):
		args = sub_args.add_parser(name)
		args.set_defaults(command=function)
		args.add_argument("components", nargs="*")#, choices=dependency_name_map.keys())
		args.add_argument("-cfg", "--config", action="append", dest="configs")
		args.add_argument("--no-deps", action="store_true")
		return args

	add_command("pull", pull)
	add_command("build", build)
	add_command("rm", remove)

	main(args.parse_args())
