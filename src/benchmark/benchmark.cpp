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
	int run_queue_benchmark(std::int32_t, std::int32_t, std::int32_t, float, float, std::int32_t);

	void* benchmark_create()
	{
		return new Instrumentation;
	}

	void benchmark_begin(void* ctx, std::int32_t N)
	{
		static_cast<Instrumentation*>(ctx)->begin(N);
	}

	void benchmark_end(void* ctx, std::int32_t N)
	{
		float dt = static_cast<Instrumentation*>(ctx)->end(N);
		std::cout /*<< std::fixed << std::setprecision(2)*/ << dt << '\n' << std::flush;
	}

	void benchmark_destroy(void* ctx)
	{
		delete static_cast<Instrumentation*>(ctx);
	}
}

namespace
{
	class usage_error : public std::runtime_error { using std::runtime_error::runtime_error; };

	std::ostream& print_usage(std::ostream& out)
	{
		return out << "usage: benchmark <num-threads-min> <num-threads-max> <block-size> <p-enq> <p-deq> <workload-size>\n"sv;
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
		if (argc != 7)
			throw usage_error("expected 6 arguments");

		int num_threads_min = parse_argument<int>(argv[1]);
		int num_threads_max = parse_argument<int>(argv[2]);
		int block_size = parse_argument<int>(argv[3]);
		float p_enq = parse_argument<float>(argv[4]);
		float p_deq = parse_argument<float>(argv[5]);
		int workload_size = parse_argument<int>(argv[6]);

		std::cout << "num_threads_min;num_threads_max;block_size;p_enq;p_deq;workload_size\n"sv
		          << num_threads_min << ';'
		          << num_threads_max << ';'
		          << block_size << ';'
		          << p_enq << ';'
		          << p_deq << ';'
		          << workload_size << '\n' << '\n';

		return run_queue_benchmark(num_threads_min, num_threads_max, block_size, p_enq, p_deq, workload_size);
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
