#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/sysctl.h>

#include "instrumentation.h"


std::ostream& Instrumentation::print_device_info(std::ostream& out)
{
	size_t buf_len;
	sysctlbyname("machdep.cpu.brand_string", nullptr, &buf_len, nullptr, 0);
	char buf[buf_len];
	sysctlbyname("machdep.cpu.brand_string", &buf, &buf_len, nullptr, 0);
	return out << std::string(buf);
}
