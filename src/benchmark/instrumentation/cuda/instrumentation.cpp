#include <stdexcept>
#include <iostream>

#include <cupti.h>
#include <cupti_profiler_target.h>

#include <anydsl_runtime.hpp>

#include "instrumentation.h"


void Instrumentation::init_profiler()
{
	// initialize CUPTI Profiling API in order to get increased %globaltimer resolution

	CUpti_Profiler_Initialize_Params params = {
		CUpti_Profiler_Initialize_Params_STRUCT_SIZE,
		nullptr
	};

	if (cuptiProfilerInitialize(&params) != CUPTI_SUCCESS)
		throw std::runtime_error("cuptiProfilerInitialize() failed");
}

std::ostream& Instrumentation::print_device_info(std::ostream& out)
{
	return out << anydsl_device_name(anydsl::make_device(anydsl::Platform::Cuda, anydsl::Device(device)));
}
