#include <cstddef>
#include <cstdio>


template <std::size_t buffer_size>
class ProducerConsumerQueue
{
	int size;
	unsigned int head;
	unsigned int tail;

	unsigned int buffer[buffer_size];

	static constexpr unsigned int FREE = ~0U;

public:
	template <typename F>
	__device__
	int push(F&& source)
	{
		if (atomicAdd(&size, 0) >= buffer_size) {
			return 0;
		}
		else {
			auto new_size = atomicAdd(&size, 1) + 1;

			if (new_size > buffer_size)
			{
				atomicSub(&size, 1);
				return 0;
			}
			else
			{
				auto i = atomicInc(&tail, (buffer_size - 1));

				auto value = source(i);

				while (atomicCAS(&buffer[i], FREE, value) != FREE)
				{
					__threadfence();
				}

				return new_size;
			}
		}
	}

	template <typename F>
	__device__
	int pop(F&& sink)
	{
		if (atomicAdd(&size, 0) <= 0) {
			return 0;
		}
		else {
			auto available = atomicSub(&size, 1);

			if (available <= 0)
			{
				atomicAdd(&size, 1);
				return 0;
			}
			else
			{
				auto i = atomicInc(&head, (buffer_size - 1));

				auto tryDequeue = [&](auto i)
				{
					auto el = atomicExch(&buffer[i], FREE);
					if (el != FREE)
					{
						sink(i, el);
						return true;
					}
					else
					{
						return false;
					}
				};

				while (!tryDequeue(i))
				{
					__threadfence();
				}

				return 1;
			}
		}
	}

	__device__
	int currentSize()
	{
		return atomicAdd(&size, 0);
	}

	__device__
	void reset()
	{
		if (blockIdx.x == 0 && threadIdx.x == 0)
		{
			size = 0;
			head = 0;
			tail = 0;
		}

		for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < buffer_size; i += gridDim.x * blockDim.x)
			buffer[i] = FREE;
	}

	__device__
	void validate()
	{
		for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < buffer_size; i += gridDim.x * blockDim.x)
		{
			if (buffer[i] != FREE)
			{
				printf("inconsistent queue state: buffer[%d] = %d\n", i, buffer[i]);
			}
		}
	}
};


constexpr int num_elements = 1024;
constexpr int  queue_size = 32;

__device__ ProducerConsumerQueue<queue_size> queue;

// __device__ int device_failed_flag;
__device__ int next_input;
__device__ int num_retired;

__global__ void resetKernel()
{
	if (blockIdx.x == 0 && threadIdx.x == 0)
	{
		// device_failed_flag = 0;
		next_input = 0;
		num_retired = 0;
	}

	queue.reset();
}

void reset()
{
	constexpr int block_dim = 256;
	constexpr int num_blocks = (queue_size + block_dim - 1) / block_dim;

	resetKernel<<<num_blocks, block_dim>>>();
}

__global__ void test()
{
	unsigned int wave_idx = gridDim.x * ((blockDim.x + 31) / 32) + threadIdx.x / 32;
	unsigned int thread_idx = threadIdx.x % 32;

	auto push = []
	{
		auto element = atomicAdd(&next_input, 1);

		auto not_done = element < num_elements;

		while (__any_sync(~0U, not_done) != 0)
		{
			if (not_done)
			{
				auto num_enqueued = queue.push([=](int i)
				{
					return element;
				});

				not_done = num_enqueued <= 0;
			}
		}

		if (element > num_elements)
		{
			atomicExch(&next_input, num_elements);
		}

		return element < num_elements;
	};

	auto pop = []
	{
		auto num_dequeued = queue.pop([](int i, unsigned int)
		{
			atomicAdd(&num_retired, 1);
		});

		return atomicAdd(&num_retired, 0) < num_elements;
	};


	bool drain = false;

	while (true)
	{
		auto queue_size = __shfl_sync(~0U, thread_idx == 0 ? queue.currentSize() : 0, 0);

		if (drain || queue_size >= warpSize)
		{
			bool done = !pop();

			if (__any_sync(~0U, done) != 0) break;
		}
		else
		{
			bool more_input = push();

			drain = __any_sync(~0U, !more_input) != 0;

			if (drain)
				printf("%d %d\n", wave_idx, thread_idx);
		}
	}
}

__global__ void validateKernel()
{
	queue.validate();

	if (blockIdx.x == 0 && threadIdx.x == 0)
	{
		if (next_input != num_elements)
		{
			printf("inconsistent next_input %d != %d\n", next_input, num_elements);
			// atomicExch(device_failed_flag, -1u32);
		}

		if (num_retired != num_elements)
		{
			printf("inconsistent num_retired %d != %d\n", num_retired, num_elements);
			// atomicExch(device_failed_flag, -2u32);
		}
	}
}

void validate()
{
	constexpr int block_dim = 256;
	constexpr int num_blocks = (queue_size + block_dim - 1) / block_dim;

	validateKernel<<<num_blocks, block_dim>>>();
}


int main()
{
	constexpr int block_dim = 256;

	reset();
	test<<<42, block_dim>>>();
	validate();

}
