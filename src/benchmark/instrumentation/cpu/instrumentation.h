#ifndef INCLUDED_INSTRUMENTATION_CPU
#define INCLUDED_INSTRUMENTATION_CPU

#include <chrono>


class Instrumentation
{
	std::chrono::steady_clock::time_point t_begin;

	int device;

public:
	Instrumentation(int device)
		: device(device)
	{
	}

	void begin()
	{
		t_begin = std::chrono::steady_clock::now();
	}

	float end()
	{
		return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t_begin).count() * 0.001f;
	}
};

#endif
