#ifndef INCLUDED_INSTRUMENTATION_CPU
#define INCLUDED_INSTRUMENTATION_CPU

#include <chrono>
#include <iostream>
#include <iomanip>


class Instrumentation
{
	std::chrono::steady_clock::time_point t_begin;

public:
	void begin(int N)
	{
		t_begin = std::chrono::steady_clock::now();
	}

	void end(int N)
	{
		auto dt = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t_begin);
		std::cout /*<< std::fixed << std::setprecision(2)*/ << dt.count() * 0.001 << '\n';
	}
};

#endif
