#include <stdexcept>
#include <iostream>

#include <cupti.h>
#include <cupti_profiler_target.h>

#include "instrumentation.h"


Instrumentation::Instrumentation(int device)
	: device(device)
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

		if (err == CUPTI_ERROR_NOT_INITIALIZED)
		{
			std::cerr << "WARNING: CUDA driver version is not compatible with the CUPTI version, CUPTI not initialized\n";
			return;
		}

		const char *err_str;
		cuptiGetResultString(err, &err_str);
		std::cerr << "CUPTI ERROR code " << err << ": " << err_str << std::endl;

		throw std::runtime_error("cuptiProfilerInitialize() failed");
	}
}
