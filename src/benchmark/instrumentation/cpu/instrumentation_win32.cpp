#include <iostream>

#include <intrin.h>

#include "instrumentation.h"


std::ostream& Instrumentation::print_device_info(std::ostream& out)
{
	int cpu_brand_string[4*3] = {};

	__cpuid(cpu_brand_string + 0, 0x80000002);
	__cpuid(cpu_brand_string + 4, 0x80000003);
	__cpuid(cpu_brand_string + 8, 0x80000004);

	return std::cout << reinterpret_cast<char*>(&cpu_brand_string[0]);
}
