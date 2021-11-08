#include <iostream>

#include "instrumentation.h"


std::ostream& Instrumentation::print_device_info(std::ostream& out)
{
	return out << "AMDGPU";  // TODO
}
