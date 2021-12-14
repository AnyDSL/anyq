#ifndef INCLUDED_INSTRUMENTATION_CPU
#define INCLUDED_INSTRUMENTATION_CPU

#include <chrono>
#include <iosfwd>


class Instrumentation
{
	std::chrono::steady_clock::time_point t_begin;

	int device;

public:
	Instrumentation(int device)
		: device(device)
	{
	}

	std::ostream& print_device_info(std::ostream&);

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
