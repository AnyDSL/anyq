#include <iostream>

#include "instrumentation.h"


std::ostream& Instrumentation::print_device_info(std::ostream& out)
{
	return out << anydsl_device_name(anydsl::make_device(anydsl::Platform::HSA, anydsl::Device(device)));
}
