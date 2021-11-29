
add_library(moodycamel STATIC
	moodycamel/concurrentqueue.h
	moodycamel/c_api/concurrentqueue.h
	moodycamel/c_api/concurrentqueue.cpp
)

target_compile_definitions(moodycamel PRIVATE MOODYCAMEL_STATIC)
