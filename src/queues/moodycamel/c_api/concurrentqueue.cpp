#include "concurrentqueue.h"
#include "../concurrentqueue.h"

#include <iostream>


static bool locked_memory = false;

struct ConcurrentQueueTraits : public moodycamel::ConcurrentQueueDefaultTraits {
public:
	ConcurrentQueueTraits() { locked_memory = false; }

	static void* malloc(size_t size) {
		//std::cout << "malloc(" << size << ")" << std::endl;
		//if (locked_memory)
		//	return nullptr;
		return moodycamel::ConcurrentQueueDefaultTraits::malloc(size);
	}
	static inline void free(void* ptr) { return moodycamel::ConcurrentQueueDefaultTraits::free(ptr); }
};

typedef moodycamel::ConcurrentQueue<void*, ConcurrentQueueTraits> MoodycamelCQType, *MoodycamelCQPtr;
typedef moodycamel::ConcurrentQueue<uint32_t, ConcurrentQueueTraits> MoodycamelCQU32Type, *MoodycamelCQU32Ptr;

extern "C" {

int moodycamel_cq_create(uint32_t capacity, uint32_t num_producer, MoodycamelCQHandle* handle)
{
	if (num_producer == 0)
		num_producer = std::thread::hardware_concurrency();
	//std::cout << "moodycamel::ConcurrentQueue for " << num_producer << " implicit producers." << std::endl;

	//locked_memory = false;
	MoodycamelCQPtr retval = new MoodycamelCQType(capacity /* 6 * MoodycamelCQType::BLOCK_SIZE */, 0, num_producer);
	//locked_memory = true;

	if (retval == nullptr) {
		return 0;
	}
	*handle = retval;
	return 1;
}

int moodycamel_cq_destroy(MoodycamelCQHandle handle)
{
	delete reinterpret_cast<MoodycamelCQPtr>(handle);
	return 1;
}

int moodycamel_cq_try_enqueue(MoodycamelCQHandle handle, MoodycamelValue value)
{
	return reinterpret_cast<MoodycamelCQPtr>(handle)->try_enqueue(value) ? 1 : 0;
}

int moodycamel_cq_try_dequeue(MoodycamelCQHandle handle, MoodycamelValue* value)
{
	return reinterpret_cast<MoodycamelCQPtr>(handle)->try_dequeue(*value) ? 1 : 0;
}

size_t moodycamel_cq_size_approx(MoodycamelCQHandle handle)
{
    return reinterpret_cast<MoodycamelCQPtr>(handle)->size_approx();
}



int moodycamel_cq_create_u32(uint32_t capacity, uint32_t num_producer, MoodycamelCQHandle* handle)
{
	if (num_producer == 0)
		num_producer = std::thread::hardware_concurrency();
	//std::cout << "moodycamel::ConcurrentQueue for " << num_producer << " implicit producers." << std::endl;

	//locked_memory = false;
	MoodycamelCQU32Ptr retval = new MoodycamelCQU32Type(capacity /* 6 * MoodycamelCQU32Type::BLOCK_SIZE */, 0, num_producer);
	//locked_memory = true;
	if (retval == nullptr) {
		return 0;
	}
	*handle = retval;
	return 1;
}

int moodycamel_cq_destroy_u32(MoodycamelCQHandle handle)
{
	delete reinterpret_cast<MoodycamelCQU32Ptr>(handle);
	return 1;
}

int moodycamel_cq_try_enqueue_u32(MoodycamelCQHandle handle, uint32_t value)
{
	return reinterpret_cast<MoodycamelCQU32Ptr>(handle)->try_enqueue(value) ? 1 : 0;
}

int moodycamel_cq_try_dequeue_u32(MoodycamelCQHandle handle, uint32_t* value)
{
	return reinterpret_cast<MoodycamelCQU32Ptr>(handle)->try_dequeue(*value) ? 1 : 0;
}

uint64_t moodycamel_cq_size_approx_u32(MoodycamelCQHandle handle)
{
	return reinterpret_cast<MoodycamelCQU32Ptr>(handle)->size_approx();
}

}
