#include <stdexcept>
#include <iostream>

#include <cupti.h>
#include <cupti_profiler_target.h>

#include <anydsl_runtime.hpp>

#include "instrumentation.h"


Instrumentation::Instrumentation(int device)
	: device(device)
{
	init_profiler();
}

void Instrumentation::init_profiler()
{
	// initialize CUPTI Profiling API in order to get increased %globaltimer resolution

	CUpti_Profiler_Initialize_Params params = {
		CUpti_Profiler_Initialize_Params_STRUCT_SIZE,
		nullptr
	};

	if (auto err = cuptiProfilerInitialize(&params); err != CUPTI_SUCCESS)
	{
		if (err == CUPTI_ERROR_OLD_PROFILER_API_INITIALIZED)
		{
			std::cerr << "WARNING: legacy CUPTI Profiling API was initialized\n";
			return;
		}

		throw std::runtime_error("cuptiProfilerInitialize() failed");
	}
}

std::ostream& Instrumentation::print_device_info(std::ostream& out)
{
	return out << anydsl_device_name(anydsl::make_device(anydsl::Platform::Cuda, anydsl::Device(device)));
}
