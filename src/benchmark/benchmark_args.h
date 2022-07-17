#ifndef INCLUDED_BENCHMARK_ARGS
#define INCLUDED_BENCHMARK_ARGS

#include <stdexcept>
#include <system_error>
#include <string_view>
#include <charconv>


class usage_error : public std::runtime_error { using std::runtime_error::runtime_error; };


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
#include <cerrno>
#include <cstdlib>
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

#endif
