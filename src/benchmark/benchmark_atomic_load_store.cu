#include <cstddef>
#include <ostream>
#include <type_traits>
#include <exception>
#include <stdexcept>
#include <algorithm>
#include <string_view>
#include <charconv>
#include <memory>
#include <iostream>
#include <iomanip>

#include <cuda.h>

using namespace std::literals;


__device__ unsigned int buffer[256*1024*1024];


__device__ unsigned int laneid()
{
	unsigned int id;
	asm("mov.u32 %0, %laneid;" : "=r"(id));
	return id;
}

__device__ unsigned int load_volatile(const unsigned int* const location)
{
	unsigned int value;
	asm("ld.volatile.global.u32 %0, [%1];" : "=r"(value) : "l"(location) : "memory");
	return value;
}

__device__ void store_volatile(unsigned int* const location, const unsigned int value)
{
	asm("st.volatile.global.u32 [%0], %1;" : : "l"(location), "r"(value) : "memory");
}

__device__ unsigned int load_atomic_add(const unsigned int* const location)
{
	unsigned int value;
	asm("atom.global.add.u32 %0, [%1], 0;" : "=r"(value) : "l"(location) : "memory");
	return value;
}

__device__ void store_atomic_exch(unsigned int* const location, const unsigned int value)
{
	[[maybe_unused]] unsigned int _;
	asm("atom.global.exch.b32 %0, [%1], %2;" : "=r"(_) : "l"(location), "r"(value) : "memory");
}

__device__ unsigned int load_cg(const unsigned int* const location)
{
	unsigned int value;
	asm("ld.global.cg.u32 %0, [%1];" : "=r"(value) : "l"(location) : "memory");
	return value;
}

__device__ void store_cg(unsigned int* const location, const unsigned int value)
{
	asm("st.global.cg.u32 [%0], %1;" : : "l"(location), "r"(value) : "memory");
}

__device__ unsigned int load_relaxed(const unsigned int* const location)
{
#if __CUDA_ARCH__ >= 700
	unsigned int value;
	asm("ld.relaxed.gpu.u32 %0, [%1];" : "=r"(value) : "l"(location) : "memory");
	return value;
#else
	__trap();
#endif
}

__device__ void store_relaxed(unsigned int* const location, const unsigned int value)
{
#if __CUDA_ARCH__ >= 700
	asm("st.relaxed.gpu.u32 [%0], %1;" : : "l"(location), "r"(value) : "memory");
#else
	__trap();
#endif
}


struct single_element
{
	__device__ constexpr unsigned int operator ()() const
	{
		return 0;
	}
};

template <unsigned int N, unsigned int stride>
struct strided_access
{
	__device__ unsigned int operator ()() const
	{
		return ((blockIdx.x * blockDim.x + threadIdx.x) * stride) % N;
	}
};


struct all_lanes
{
	__device__ constexpr bool operator ()() const
	{
		return true;
	}
};

template <unsigned int stride>
struct skip_lanes
{
	__device__ bool operator ()() const
	{
		return laneid() % stride == 0;
	}
};

template <unsigned int N>
struct lower_lanes
{
	__device__ bool operator ()() const
	{
		return laneid() < N;
	}
};



template <unsigned int (&load)(const unsigned int*), void (&store)(unsigned int*, unsigned int), int N, typename AccessPattern, typename LanePattern>
__global__ void test()
{
	if (LanePattern{}())
	{
		auto& a = buffer[AccessPattern{}()];

		#pragma unroll
		for (int i = 0; i < N; ++i)
		{
			auto x = load(&a);
			store(&a, x + 1);
		}
	}
}


namespace
{
	class cuda_error : public std::exception
	{
		cudaError err;

	public:
		cuda_error(cudaError err) : err(err) {}

		const char* what() const noexcept override { return cudaGetErrorName(err); }
	};

	cudaError throw_error(cudaError err)
	{
		if (err != cudaSuccess)
			throw cuda_error(err);
		return err;
	}


	struct cuda_event_deleter
	{
		void operator ()(cudaEvent_t e) const
		{
			cudaEventDestroy(e);
		}
	};

	using unique_event = std::unique_ptr<std::remove_pointer_t<cudaEvent_t>, cuda_event_deleter>;

	auto create_event()
	{
		cudaEvent_t event;
		throw_error(cudaEventCreate(&event));
		return unique_event(event);
	}


	int get_current_device()
	{
		int device;
		throw_error(cudaGetDevice(&device));
		return device;
	}

	auto get_device_properties(int device = get_current_device())
	{
		cudaDeviceProp props;
		throw_error(cudaGetDeviceProperties(&props, device));
		return props;
	}

	template <cudaDeviceAttr attr>
	auto get_device_attribute(int device = get_current_device())
	{
		int value;
		throw_error(cudaDeviceGetAttribute(&value, attr, device));
		return value;
	}

	template <typename F>
	std::size_t max_smem_per_block(F func, int block_size, int blocks_per_multiprocessor)
	{
		std::size_t smem;
		throw_error(cudaOccupancyAvailableDynamicSMemPerBlock(&smem, func, blocks_per_multiprocessor, block_size));
		return smem;
	}

	template <typename F>
	int max_blocks_per_multiprocessor(F func, int block_size, std::size_t smem)
	{
		int num_blocks;
		throw_error(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, func, block_size, smem));
		return num_blocks;
	}


	class Benchmark
	{
		unique_event begin;
		unique_event end;

	public:
		Benchmark()
			: begin(create_event()),
			  end(create_event())
		{
		}

		template <typename K>
		float run(K&& kernel, int num_blocks, int block_size, int blocks_per_multiprocessor)
		{
			// dynamic shared memory allocation is used to prevent the GPU from putting more blocks onto each sm
			auto smem = max_smem_per_block(kernel, block_size, blocks_per_multiprocessor);

			if (int expected_blocks = max_blocks_per_multiprocessor(kernel, block_size, smem); expected_blocks != blocks_per_multiprocessor)
				std::cerr << "WARNING: expected occupancy ("sv << expected_blocks << " blocks) does not match requested occupancy ("sv << blocks_per_multiprocessor << " blocks)!\n"sv;

			constexpr int num_samples = 20;

			float t = 0.0f;

			for (int i = 0; i < num_samples; ++i)
			{
				throw_error(cudaEventRecord(begin.get()));
				// kernel<<<num_blocks, block_size, smem>>>();
				throw_error(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(kernel), dim3(num_blocks), dim3(block_size), nullptr, smem));
				throw_error(cudaEventRecord(end.get()));
				throw_error(cudaEventSynchronize(end.get()));

				float dt;
				throw_error(cudaEventElapsedTime(&dt, begin.get(), end.get()));
				t += dt;
			}

			return t / num_samples;
		}
	};


	constexpr int divup(int a, int b)
	{
		return (a + b - 1) / b;
	}


	class usage_error : public std::runtime_error { using std::runtime_error::runtime_error; };

	std::ostream& print_usage(std::ostream& out)
	{
		return out << "usage: benchmark <device>\n"sv;
	}

	template <typename T>
	int parse_argument(std::string_view arg)
	{
		T value;

		if (auto [end, ec] = std::from_chars(&arg[0], &arg[0] + arg.length(), value); ec == std::errc::invalid_argument)
			throw usage_error("argument must be a number");
		else if (ec == std::errc::result_out_of_range)
			throw usage_error("argument out of range");
		else if (end != &arg[0] + arg.length())
			throw usage_error("invalid argument");

		return value;
	}

	template <int N, typename AccessPattern, typename LanePattern, typename F>
	void kernels(bool try_relaxed_atomics, F&& f)
	{
		f(test<load_volatile, store_volatile, N, AccessPattern, LanePattern>, "v+v"sv);
		f(test<load_volatile, store_atomic_exch, N, AccessPattern, LanePattern>, "v+a"sv);
		f(test<load_atomic_add, store_volatile, N, AccessPattern, LanePattern>, "a+v"sv);
		f(test<load_atomic_add, store_atomic_exch, N, AccessPattern, LanePattern>, "a+a"sv);

		if (try_relaxed_atomics)
		{
			f(test<load_relaxed, store_relaxed, N, AccessPattern, LanePattern>, "relaxed"sv);
		}
	}
}

int main(int argc, char** argv)
{
	try
	{
		int device = 0;

		if (argc == 2)
			device = parse_argument<int>(argv[1]);
		else if (argc != 1)
			throw usage_error("too many arguments");

		throw_error(cudaSetDevice(device));

		auto dev_props = get_device_properties(device);

		const bool try_relaxed_atomics = get_device_attribute<cudaDevAttrComputeCapabilityMajor>(device) >= 7;

		std::cout << "device "sv << device << ": "sv << dev_props.name << " ("sv << dev_props.multiProcessorCount << " multiprocessors)\n"sv;


		Benchmark benchmark;

		constexpr int N = 256;

		constexpr int col_width = 10;

		std::cout << "\n                    "sv;

		kernels<N, single_element, all_lanes>(try_relaxed_atomics, []([[maybe_unused]] auto&& k, auto n)
		{
			int pad = col_width - n.length();

			for (int i = 4 + (pad + 1) / 2; i > 0; --i) std::cout.put(' ');
			std::cout << n;
			for (int i = pad / 2; i > 0; --i) std::cout.put(' ');
		});

		std::cout << '\n';

		for (int occupancy : {1, 5, 10, 25, 50, 75, 100})
		{
			const int min_blocks_per_multiprocessor = divup(dev_props.maxThreadsPerMultiProcessor, dev_props.maxThreadsPerBlock);
			const int max_warps_per_multiprocessor = dev_props.maxThreadsPerMultiProcessor / dev_props.warpSize;
			const int max_warps_per_block = dev_props.maxThreadsPerBlock / dev_props.warpSize;

			const int warps_per_multiprocessor = divup(max_warps_per_multiprocessor * occupancy, 100);
			const int blocks_per_multiprocessor = std::max(divup(warps_per_multiprocessor, max_warps_per_block), min_blocks_per_multiprocessor);
			const int warps_per_block = divup(warps_per_multiprocessor,  blocks_per_multiprocessor);
			const int block_size = warps_per_block * dev_props.warpSize;
			const int num_blocks = dev_props.multiProcessorCount * blocks_per_multiprocessor;

			std::cout << std::fixed << std::setprecision(2) << std::setw(6) << blocks_per_multiprocessor * block_size * 100.0 / dev_props.maxThreadsPerMultiProcessor << "% ("sv
			          << std::setw(2) << (blocks_per_multiprocessor * block_size / dev_props.warpSize) << '/' << max_warps_per_multiprocessor << " warps "sv
			          << blocks_per_multiprocessor << " blocks)"sv;

			auto print_results = [&](auto&& k, [[maybe_unused]] auto n)
			{
				auto t = benchmark.run(std::forward<decltype(k)>(k), num_blocks, block_size, blocks_per_multiprocessor);
				std::cout << std::fixed << std::setprecision(2) << std::setw(col_width) << t << " ms "sv;
			};

			std::cout << "\n           [1] full:"sv;
			kernels<N, single_element, all_lanes>(try_relaxed_atomics, print_results);
			std::cout << '\n';

			std::cout << "\n           [1] <  1:"sv;
			kernels<N, single_element, lower_lanes< 1>>(try_relaxed_atomics, print_results);
			std::cout << "\n           [1] <  2:"sv;
			kernels<N, single_element, lower_lanes< 2>>(try_relaxed_atomics, print_results);
			std::cout << "\n           [1] <  4:"sv;
			kernels<N, single_element, lower_lanes< 4>>(try_relaxed_atomics, print_results);
			std::cout << "\n           [1] <  8:"sv;
			kernels<N, single_element, lower_lanes< 8>>(try_relaxed_atomics, print_results);
			std::cout << "\n           [1] < 16:"sv;
			kernels<N, single_element, lower_lanes<16>>(try_relaxed_atomics, print_results);
			std::cout << "\n           [1] < 32:"sv;
			kernels<N, single_element, lower_lanes<32>>(try_relaxed_atomics, print_results);
			std::cout << '\n';

			std::cout << "\n           [1] %  1:"sv;
			kernels<N, single_element, skip_lanes< 1>>(try_relaxed_atomics, print_results);
			std::cout << "\n           [1] %  2:"sv;
			kernels<N, single_element, skip_lanes< 2>>(try_relaxed_atomics, print_results);
			std::cout << "\n           [1] %  4:"sv;
			kernels<N, single_element, skip_lanes< 4>>(try_relaxed_atomics, print_results);
			std::cout << "\n           [1] %  8:"sv;
			kernels<N, single_element, skip_lanes< 8>>(try_relaxed_atomics, print_results);
			std::cout << "\n           [1] % 16:"sv;
			kernels<N, single_element, skip_lanes<16>>(try_relaxed_atomics, print_results);
			std::cout << "\n           [1] % 32:"sv;
			kernels<N, single_element, skip_lanes<32>>(try_relaxed_atomics, print_results);
			std::cout << '\n';

			std::cout << "\n   [1024 :  1] <  1:"sv;
			kernels<N, strided_access<1024,  1>, lower_lanes< 1>>(try_relaxed_atomics, print_results);
			std::cout << "\n   [1024 :  2] <  2:"sv;
			kernels<N, strided_access<1024,  2>, lower_lanes< 2>>(try_relaxed_atomics, print_results);
			std::cout << "\n   [1024 :  4] <  4:"sv;
			kernels<N, strided_access<1024,  4>, lower_lanes< 4>>(try_relaxed_atomics, print_results);
			std::cout << "\n   [1024 :  8] <  8:"sv;
			kernels<N, strided_access<1024,  8>, lower_lanes< 8>>(try_relaxed_atomics, print_results);
			std::cout << "\n   [1024 : 16] < 16:"sv;
			kernels<N, strided_access<1024, 16>, lower_lanes<16>>(try_relaxed_atomics, print_results);
			std::cout << "\n   [1024 : 32] < 32:"sv;
			kernels<N, strided_access<1024, 32>, lower_lanes<32>>(try_relaxed_atomics, print_results);
			std::cout << '\n';

			std::cout << "\n   [1024 :  1] %  1:"sv;
			kernels<N, strided_access<1024,  1>, skip_lanes< 1>>(try_relaxed_atomics, print_results);
			std::cout << "\n   [1024 :  2] %  2:"sv;
			kernels<N, strided_access<1024,  2>, skip_lanes< 2>>(try_relaxed_atomics, print_results);
			std::cout << "\n   [1024 :  4] %  4:"sv;
			kernels<N, strided_access<1024,  4>, skip_lanes< 4>>(try_relaxed_atomics, print_results);
			std::cout << "\n   [1024 :  8] %  8:"sv;
			kernels<N, strided_access<1024,  8>, skip_lanes< 8>>(try_relaxed_atomics, print_results);
			std::cout << "\n   [1024 : 16] % 16:"sv;
			kernels<N, strided_access<1024, 16>, skip_lanes<16>>(try_relaxed_atomics, print_results);
			std::cout << "\n   [1024 : 32] % 32:"sv;
			kernels<N, strided_access<1024, 32>, skip_lanes<32>>(try_relaxed_atomics, print_results);
			std::cout << '\n';

			std::cout << "\n   [1024 :  1] full:"sv;
			kernels<N, strided_access<1024,  1>, all_lanes>(try_relaxed_atomics, print_results);
			std::cout << "\n   [1024 :  2] full:"sv;
			kernels<N, strided_access<1024,  2>, all_lanes>(try_relaxed_atomics, print_results);
			std::cout << "\n   [1024 :  4] full:"sv;
			kernels<N, strided_access<1024,  4>, all_lanes>(try_relaxed_atomics, print_results);
			std::cout << "\n   [1024 :  8] full:"sv;
			kernels<N, strided_access<1024,  8>, all_lanes>(try_relaxed_atomics, print_results);
			std::cout << "\n   [1024 : 16] full:"sv;
			kernels<N, strided_access<1024, 16>, all_lanes>(try_relaxed_atomics, print_results);
			std::cout << "\n   [1024 : 32] full:"sv;
			kernels<N, strided_access<1024, 32>, all_lanes>(try_relaxed_atomics, print_results);
			std::cout << '\n';

			std::cout << '\n' << std::flush;
		}
	}
	catch (const usage_error& e)
	{
		std::cerr << "ERROR: "sv << e.what() << '\n' << print_usage;
		return -1;
	}
	catch (const std::exception& e)
	{
		std::cerr << "ERROR: "sv << e.what() << '\n';
		return -1;
	}
	catch (...)
	{
		std::cerr << "ERROR: unknown exception\n"sv;
		return -128;
	}

	return 0;
}
