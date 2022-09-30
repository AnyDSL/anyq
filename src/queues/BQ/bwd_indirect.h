#ifndef INCLUDED_QUEUE_BWD_INDIRECT
#define INCLUDED_QUEUE_BWD_INDIRECT

#pragma once

template <int N, typename T>
class BrokerWorkDistributorIndirect
{
	template <int>
	friend struct BWDIndexQueueIndirect;

	typedef unsigned int Ticket;
	typedef unsigned int HT_t;
	typedef unsigned long long int HT;

	volatile Ticket* tickets;
	T* ring_buffer;

	HT* head_tail;
	int* count;

private:
	// use nanosleep on Turing architecture, threadfence on all others
#if __CUDA_ARCH__ < 700
	__device__ static __forceinline__ void backoff()
	{
		__threadfence();
	}

	template <typename L>
	__device__ static __forceinline__ L uncachedLoad(const L* l)
	{
		return *l;
	}

	template <typename L>
	__device__ static __forceinline__ L atomicLoad(const L* l)
	{
		return *l;
	}
#else
	__device__ static __forceinline__ void backoff()
	{
		__threadfence();
	}

	template <typename L>
	__device__ static __forceinline__ L uncachedLoad(const L* l)
	{
		return *l;
	}

	template <typename L>
	__device__ static __forceinline__ L atomicLoad(const L* l)
	{
		return *l;
	}
#endif

	__device__ HT_t* head()
	{
		return reinterpret_cast<HT_t*>(head_tail) + 1;
	}

	__device__ HT_t* tail()
	{
		return reinterpret_cast<HT_t*>(head_tail);
	}

	__forceinline__ __device__ void waitForTicket(const unsigned int P, const Ticket number)
	{
		while (tickets[P] != number)
		{
			backoff(); // back off
		}
	}

	__forceinline__ __device__ bool ensureDequeue()
	{
		int Num = atomicLoad(count);
		bool ensurance = false;

		while (!ensurance && Num > 0)
		{
			if (atomicSub(count, 1) > 0)
			{
				ensurance = true;
			}
			else
			{
				Num = atomicAdd(count, 1) + 1;
			}
		}
		return ensurance;
	}

	__forceinline__ __device__ bool ensureEnqueue()
	{
		int Num = atomicLoad(count);
		bool ensurance = false;

		while (!ensurance && Num < N)
		{
			if (atomicAdd(count, 1) < N)
			{
				ensurance = true;
			}
			else
			{
				Num = atomicSub(count, 1) - 1;
			}
		}
		return ensurance;
	}

	__forceinline__ __device__ void readData(T& val)
	{
		const unsigned int Pos = atomicAdd(head(), 1);
		const unsigned int P = Pos % N;

		waitForTicket(P, 2 * (Pos / N) + 1);
		val = ring_buffer[P];
		__threadfence();
		tickets[P] = 2 * ((Pos + N) / N);
	}

	__forceinline__ __device__ void putData(const T data)
	{
		const unsigned int Pos = atomicAdd(tail(), 1);
		const unsigned int P = Pos % N;
		const unsigned int B = 2 * (Pos / N);

		waitForTicket(P, B);
		ring_buffer[P] = data;
		__threadfence();
		tickets[P] = B + 1;
	}

public:
	__device__ BrokerWorkDistributorIndirect(volatile Ticket* tickets, T* ring_buffer, HT* head_tail, int* count)
		: tickets(tickets), ring_buffer(ring_buffer), head_tail(head_tail), count(count)
	{
	}

	__device__ void init()
	{
		const int lid = threadIdx.x + blockIdx.x * blockDim.x;

		if (lid == 0)
		{
			*count = 0;
			*head_tail = 0x0ULL;
		}

		for (int v = lid; v < N; v += blockDim.x * gridDim.x)
		{
			// ring_buffer[v] = T(0x0);
			tickets[v] = 0x0;
		}
	}

	__device__ inline bool enqueue(const T& data)
	{
		bool writeData = ensureEnqueue();
		if (writeData)
		{
			putData(data);
		}
		return writeData;
	}

	__device__ inline void dequeue(bool& hasData, T& data)
	{
		hasData = ensureDequeue();
		if (hasData)
		{
			readData(data);
		}
	}

	__device__ int size() const
	{
		return atomicLoad(count);
	}
};

#endif


template <int N>
struct BWDIndexQueueIndirect
{
	__device__ static BrokerWorkDistributorIndirect<N, unsigned int> queue(void* q)
	{
		struct queue_data
		{
			volatile unsigned int tickets[N];
			unsigned int ring_buffer[N];

			unsigned long long head_tail;
			int count;
		};
		
		auto& queue = *static_cast<queue_data*>(q);

		return { queue.tickets, queue.ring_buffer, &queue.head_tail, &queue.count };
	}

	__device__ static void init(void* q)
	{
		queue(q).init();
	}

	__device__ static bool ensure_enqueue(void* q)
	{
		return queue(q).ensureEnqueue();
	}

	__device__ static void put_data(void* q, unsigned int value)
	{
		queue(q).putData(value);
	}

	__device__ static bool ensure_dequeue(void* q)
	{
		return queue(q).ensureDequeue();
	}

	__device__ static unsigned int read_data(void* q)
	{
		unsigned int dest;
		queue(q).readData(dest);
		return dest;
	}

	__device__ static int size(void* q)
	{
		return queue(q).size();
	}
};

template struct BWDIndexQueueIndirect<1024>;
template struct BWDIndexQueueIndirect<16384>;
template struct BWDIndexQueueIndirect<131072>;
template struct BWDIndexQueueIndirect<1048576>;
