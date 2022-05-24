#include "instrumentation.h"


Instrumentation::Instrumentation(int device)
	: device(device)
{
	init_profiler();
}
