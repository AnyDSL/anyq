#include <stdexcept>

#include <cuda.h>

#include <CUDA/error.h>

#include "instrumentation.h"


Instrumentation::Instrumentation(int device_id)
	: device(device_id)
{
	throw_error(cuInit(0));

	CUdevice device;
	throw_error(cuDeviceGet(&device, device_id));

	int cc_major;
	throw_error(cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));

	// if (cc_major < 7)
	// 	throw std::runtime_error("ITS not supported on this device");

	init_profiler();
}
