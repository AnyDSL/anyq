#include <exception>
#include <iostream>

#include "instrumentation_cpu.h"


extern "C"
{
	int run_benchmark();

	void* benchmark_create()
	{
		return new Instrumentation;
	}

	void benchmark_begin(void* ctx, int N)
	{
		static_cast<Instrumentation*>(ctx)->begin(N);
	}

	void benchmark_end(void* ctx, int N)
	{
		static_cast<Instrumentation*>(ctx)->end(N);
	}

	void benchmark_destroy(void* ctx)
	{
		delete static_cast<Instrumentation*>(ctx);
	}
}

int main()
{
	try
	{
		return run_benchmark();
	}
	catch (std::exception& e)
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
