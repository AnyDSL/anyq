#ifndef INCLUDED_INSTRUMENTATION_CUDA
#define INCLUDED_INSTRUMENTATION_CUDA

#include <iosfwd>

#include "CUDA/error.h"
#include "CUDA/event.h"


class Instrumentation
{
	CU::unique_event event_begin = CU::create_event();
	CU::unique_event event_end = CU::create_event();

public:
	std::ostream& print_device_info(std::ostream&);

	void begin(int N)
	{
		throw_error(cuEventRecord(event_begin, nullptr));
	}

	float end(int N)
	{
		throw_error(cuEventRecord(event_end, nullptr));
		throw_error(cuEventSynchronize(event_end));

		float dt;
		throw_error(cuEventElapsedTime(&dt, event_begin, event_end));

		return dt / N;
	}
};

#endif
