#include <thread>


extern "C" void anydsl_fiberless_yield(const char*)
{
	std::this_thread::yield();
}
