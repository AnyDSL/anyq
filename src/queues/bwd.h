#ifndef INCLUDED_QUEUE_BWD
#define INCLUDED_QUEUE_BWD

#pragma once

template <int N, typename T>
class BrokerWorkDistributor
{
	typedef unsigned int Ticket;
	typedef unsigned int HT_t;
	typedef unsigned long long int HT;

	volatile Ticket tickets[N];
	T ring_buffer[N];

	HT head_tail;
	int count;

	// use nanosleep on Turing architecture, threadfence on all others
#if __CUDA_ARCH__ < 700
	__device__ __forceinline__ void backoff()
	{
		__threadfence();
	}

	template <typename L>
	__device__ __forceinline__ L uncachedLoad(L* l)
	{
		return *l;
	}

	template <typename L>
	__device__ __forceinline__ L atomicLoad(L* l)
	{
		return *l;
	}
#else
	__device__ __forceinline__ void backoff()
	{
		__threadfence();
	}

	template <typename L>
	__device__ __forceinline__ L uncachedLoad(L* l)
	{
		return *l;
	}

	template <typename L>
	__device__ __forceinline__ L atomicLoad(L* l)
	{
		return *l;
	}
#endif

	__device__ HT_t* head(HT* head_tail)
	{
		return reinterpret_cast<HT_t*>(head_tail) + 1;
	}

	__device__ HT_t* tail(HT* head_tail)
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
		int Num = atomicLoad(&count);
		bool ensurance = false;

		while (!ensurance && Num > 0)
		{
			if (atomicSub(&count, 1) > 0)
			{
				ensurance = true;
			}
			else
			{
				Num = atomicAdd(&count, 1) + 1;
			}
		}
		return ensurance;
	}

	__forceinline__ __device__ bool ensureEnqueue()
	{
		int Num = atomicLoad(&count);
		bool ensurance = false;

		while (!ensurance && Num < N)
		{
			if (atomicAdd(&count, 1) < N)
			{
				ensurance = true;
			}
			else
			{
				Num = atomicSub(&count, 1) - 1;
			}
		}
		return ensurance;
	}

	__forceinline__ __device__ void readData(T& val)
	{
		const unsigned int Pos = atomicAdd(head(const_cast<HT*>(&head_tail)), 1);
		const unsigned int P = Pos % N;

		waitForTicket(P, 2 * (Pos / N) + 1);
		val = ring_buffer[P];
		__threadfence();
		tickets[P] = 2 * ((Pos + N) / N);
	}

	__forceinline__ __device__ void putData(const T data)
	{
		const unsigned int Pos = atomicAdd(tail(const_cast<HT*>(&head_tail)), 1);
		const unsigned int P = Pos % N;
		const unsigned int B = 2 * (Pos / N);

		waitForTicket(P, B);
		ring_buffer[P] = data;
		__threadfence();
		tickets[P] = B + 1;
	}

public:

	__device__ void init()
	{
		const int lid = threadIdx.x + blockIdx.x * blockDim.x;

		if (lid == 0)
		{
			count = 0;
			head_tail = 0x0ULL;
		}

		for (int v = lid; v < N; v += blockDim.x * gridDim.x)
		{
			ring_buffer[v] = T(0x0);
			tickets[v] = 0x0;
		}
	}

	__device__ inline bool enqueue(const T& data)
	{
		bool writeData= ensureEnqueue();
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
};

#endif

template <int N, typename T>
__device__ BrokerWorkDistributor<N, T> broker_queue;

template <int N>
inline __device__ void bq_init()
{
	broker_queue<N, unsigned int>.init();
}

template <int N>
inline __device__ int bq_push(unsigned int value)
{
	return broker_queue<N, unsigned int>.enqueue(value);
}

template <int N>
inline __device__ int bq_pop(unsigned int* dest)
{
	bool succ;
	broker_queue<N, unsigned int>.dequeue(succ, *dest);
	return succ;
}

extern "C"
{
	inline __device__ void bq_init_1000()
	{
		bq_init<1000>();
	}
	inline __device__ void bq_init_10000()
	{
		bq_init<10000>();
	}
	inline __device__ void bq_init_100000()
	{
		bq_init<100000>();
	}
	inline __device__ void bq_init_1000000()
	{
		bq_init<1000000>();
	}

	inline __device__ int bq_push_1000(unsigned int value)
	{
		return bq_push<1000>(value);
	}
	inline __device__ int bq_push_10000(unsigned int value)
	{
		return bq_push<10000>(value);
	}
	inline __device__ int bq_push_100000(unsigned int value)
	{
		return bq_push<100000>(value);
	}
	inline __device__ int bq_push_1000000(unsigned int value)
	{
		return bq_push<1000000>(value);
	}

	inline __device__ int bq_pop_1000(unsigned int* value)
	{
		return bq_pop<1000>(value);
	}
	inline __device__ int bq_pop_10000(unsigned int* value)
	{
		return bq_pop<10000>(value);
	}
	inline __device__ int bq_pop_100000(unsigned int* value)
	{
		return bq_pop<100000>(value);
	}
	inline __device__ int bq_pop_1000000(unsigned int* value)
	{
		return bq_pop<1000000>(value);
	}
}


#ifdef __LP64__
inline __device__ unsigned long atomicAdd(unsigned long* loc, unsigned long value)
{
	return atomicAdd(reinterpret_cast<unsigned long long*>(loc), value);
}
#endif
