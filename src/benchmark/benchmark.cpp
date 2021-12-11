#include <cstdint>
#include <stdexcept>
#include <system_error>
#include <string_view>
#include <charconv>
#include <iostream>
#include <iomanip>

#include <instrumentation.h>

using namespace std::literals;


extern "C"
{
	int run(std::int32_t, std::int32_t, std::int32_t, std::int32_t, float, float, std::int32_t);

	void* instrumentation_create()
	{
		return new Instrumentation;
	}

	void instrumentation_print_device_info(void* ctx)
	{
		static_cast<Instrumentation*>(ctx)->print_device_info(std::cout << PLATFORM << ';');
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
	class usage_error : public std::runtime_error { using std::runtime_error::runtime_error; };

	std::ostream& print_usage(std::ostream& out)
	{
		return out << "usage: benchmark <device> <num-threads-min> <num-threads-max> <block-size> <p-enq> <p-deq> <workload-size>\n"sv;
	}

	template <typename T>
	T parse_argument(std::string_view arg)
	{
		T value;

		if (auto [end, ec] = std::from_chars(&arg[0], &arg[0] + arg.length(), value); ec == std::errc::invalid_argument)
			throw usage_error("argument must be a number");
		else if (ec == std::errc::result_out_of_range)
			throw usage_error("argument out of range");
		else if (end != &arg[0] + arg.length())
			throw usage_error("invalid argument");

		return value;
	}
#if defined(__GNUC__) && __GNUC__ < 11
// WORKAROUND for lack of std::from_chars support in GCC < 11
}
#include <cerrno>
#include <cstdlib>
namespace
{
	template <>
	float parse_argument<float>(std::string_view arg)
	{
		char* end;
		float value = std::strtof(&arg[0], &end);
		//                           ^ HACK!

		if (end == &arg[0])
			throw usage_error("argument must be a number");
		else if (errno == ERANGE)
			errno = 0, throw usage_error("argument out of range");
		else if (end != &arg[0] + arg.length())
			throw usage_error("invalid argument");

		return value;
	}
#endif
}

int main(int argc, char* argv[])
{
	try
	{
		if (argc == 3) {
			int device = parse_argument<int>(argv[1]);
			if (std::string_view(argv[2]) == "info") {
				Instrumentation::print_device_info(std::cout);
				return 0;
			}
		}
		else if (argc == 8) {
			int device = parse_argument<int>(argv[1]);
			int num_threads_min = parse_argument<int>(argv[2]);
			int num_threads_max = parse_argument<int>(argv[3]);
			int block_size = parse_argument<int>(argv[4]);
			float p_enq = parse_argument<float>(argv[5]);
			float p_deq = parse_argument<float>(argv[6]);
			int workload_size = parse_argument<int>(argv[7]);

			return run(device, num_threads_min, num_threads_max, block_size, p_enq, p_deq, workload_size);
		}
		else {
			throw usage_error("expected 7 arguments");
		}

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
