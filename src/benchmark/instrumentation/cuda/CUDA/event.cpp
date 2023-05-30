#include "error.h"
#include "event.h"


namespace CU
{
	unique_event create_event(unsigned int flags)
	{
		CUevent event;
		succeed(cuEventCreate(&event, flags));
		return unique_event(event);
	}
}
