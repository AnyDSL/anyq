#include <stdexcept>
#include <string_view>
#include <string>
#include <algorithm>
#include <iterator>
#include <fstream>
#include <iostream>

#include "instrumentation.h"

using namespace std::literals;


namespace
{
	std::istream& skip_to(std::istream& in, std::string_view str)
	{
		std::search(std::istreambuf_iterator<char>(in), {}, begin(str), end(str));
		in.get();
		in >> std::ws;
		in.get();
		return in >> std::ws;
	}
}

std::ostream& Instrumentation::print_device_info(std::ostream& out)
{
	std::ifstream cpuinfo("/proc/cpuinfo");

	if (!cpuinfo)
		throw std::runtime_error("failed to open /proc/cpuinfo");

	// TODO: don't just output name of the first processor

	std::string model_name;
	skip_to(cpuinfo, "model name"sv);
	std::getline(cpuinfo, model_name);

	return out << model_name;
}
