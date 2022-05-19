#include <cstdio>
#include <stdexcept>
#include <iostream>

#include <cuda.h>

#include <cupti.h>
#include <cupti_profiler_target.h>


__device__ unsigned long long timestamp()
{
	unsigned long long timestamp;
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timestamp) ::);
	return timestamp;
}

__device__ unsigned long long wait_next_timestamp(unsigned long long t)
{
	unsigned long long t_next;

	do
	{
		t_next = timestamp();
	} while (t_next == t);

	return t_next;
}

__global__ void test()
{
	for (int i = 0; i < 128; ++i)
	{
		unsigned long long t_1 = wait_next_timestamp(timestamp());
		unsigned long long t_2 = wait_next_timestamp(t_1);

		printf("%llu ", t_2 - t_1);
	}

	printf("\n");
}

int main()
{
	try
	{
		CUpti_Profiler_Initialize_Params params = {
			.structSize = sizeof(params),
			.pPriv = nullptr
		};

		if (cuptiProfilerInitialize(&params) != CUPTI_SUCCESS)
			throw std::runtime_error("cuptiProfilerInitialize() failed");

		test<<<1,1>>>();

		auto err = cudaDeviceSynchronize();

		std::cerr << cudaGetErrorString(err) << '\n';
	}
	catch (const std::exception& e)
	{
		std::cerr << "ERROR: " << e.what() << '\n';
		return -1;
	}
	catch (...)
	{
		std::cerr << "ERROR: unknown exception\n";
		return -128;
	}
}
