#include <iterator>
#include <iostream>

#include <cuda.h>

#include "CUDA/error.h"

#include "instrumentation.h"


namespace
{
	CUdevice get_device()
	{
		CUdevice device;
		throw_error(cuCtxGetDevice(&device));
		return device;
	}
}

std::ostream& Instrumentation::print_device_info(std::ostream& out)
{
	auto device = get_device();

	char name[256];
	throw_error(cuDeviceGetName(name, std::size(name), device));

	return out << name;
}
