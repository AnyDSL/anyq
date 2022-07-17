#include <cstdint>
#include <ostream>
#include <string_view>
#include <iostream>
#include <iomanip>

#include <instrumentation.h>

#include "benchmark_args.h"

using namespace std::literals;

extern const char* FINGERPRINT;


extern "C"
{
	std::int32_t parse_int_arg(void* args, std::int32_t i)
	{
		return parse_argument<int>(static_cast<char**>(args)[i]);
	}

	float parse_float_arg(void* args, std::int32_t i)
	{
		return parse_argument<float>(static_cast<char**>(args)[i]);
	}

	[[noreturn]] void throw_usage_error(const char* msg)
	{
		throw usage_error(msg);
	}

	void enum_int_arg(void* ctx, const char* name)
	{
		*static_cast<std::ostream*>(ctx) << " <" << name << '>';
	}

	void enum_float_arg(void* ctx, const char* name)
	{
		*static_cast<std::ostream*>(ctx) << " <" << name << '>';
	}

	int benchmark_enum_args(void* ctx);
	int benchmark_run(std::int32_t device, std::int32_t argc, void* argv);

	void* instrumentation_create(std::int32_t device)
	{
		return new Instrumentation(device);
	}

	void instrumentation_print_device_info(void* ctx)
	{
		std::cout << "platform;device_name;fingerprint\n";
		static_cast<Instrumentation*>(ctx)->print_device_info(std::cout << PLATFORM << ';') << ';' << FINGERPRINT << '\n' << std::flush;
	}

	void instrumentation_begin(void* ctx)
	{
		static_cast<Instrumentation*>(ctx)->begin();
	}

	void instrumentation_end(void* ctx)
	{
		float dt = static_cast<Instrumentation*>(ctx)->end();
		std::cout /*<< std::fixed << std::setprecision(2)*/ << dt; // << '\n' << std::flush;
	}

	void instrumentation_destroy(void* ctx)
	{
		delete static_cast<Instrumentation*>(ctx);
	}
}

namespace
{
	std::ostream& print_args(std::ostream& out)
	{
		benchmark_enum_args(&out);
		return out;
	}

	std::ostream& print_usage(std::ostream& out)
	{
		return out << "usage: benchmark <device>"sv << print_args << '\n'
		           << "       benchmark info <device>\n";
	}
}

int main(int argc, char* argv[])
{
	try
	{
		if (argc < 2)
			throw usage_error("expected at least 1 argument");

		if (argv[1] == "info"sv)
		{
			if (argc != 3)
				throw usage_error("too many arguments");

			int device = parse_argument<int>(argv[2]);

			Instrumentation(device).print_device_info(std::cout) << '\n' << FINGERPRINT << '\n' << std::flush;

			return 0;
		}

		int device = parse_argument<int>(argv[1]);

		return benchmark_run(device, argc - 2, argv + 2);
	}
	catch (const usage_error& e)
	{
		std::cerr << "ERROR: "sv << e.what() << '\n' << print_usage;
		return -1;
	}
	catch (const std::exception& e)
	{
		std::cerr << "ERROR: "sv << e.what() << '\n';
		return -1;
	}
	catch (...)
	{
		std::cerr << "ERROR: unknown exception\n"sv;
		return -128;
	}
}
