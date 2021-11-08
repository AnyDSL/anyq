#ifndef INCLUDED_INSTRUMENTATION_AMDGPU
#define INCLUDED_INSTRUMENTATION_AMDGPU

#include <iosfwd>


class Instrumentation
{
public:
	std::ostream& print_device_info(std::ostream&);

	void begin(int N)
	{
		// TODO
	}

	float end(int N)
	{
		return 0;  // TODO
	}
};

#endif
