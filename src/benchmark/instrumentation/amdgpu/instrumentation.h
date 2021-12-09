#ifndef INCLUDED_INSTRUMENTATION_AMDGPU
#define INCLUDED_INSTRUMENTATION_AMDGPU

#include <iosfwd>

#include <anydsl_runtime.hpp>


class Instrumentation
{
	uint64_t t_begin;
public:
	std::ostream& print_device_info(std::ostream&);

	void begin()
	{
		anydsl_synchronize(anydsl::make_device(anydsl::Platform::HSA, anydsl::Device(1)));
		t_begin = anydsl_get_kernel_time();
	}

	float end()
	{
		anydsl_synchronize(anydsl::make_device(anydsl::Platform::HSA, anydsl::Device(1)));
		return (float)(anydsl_get_kernel_time() - t_begin) / 1000.f;
	}
};

#endif
