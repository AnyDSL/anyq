

#include <boost/assert.hpp>
#include <boost/fiber/all.hpp>
#include <thread>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <mutex>

#include "fibers.h"


typedef std::unique_lock<std::mutex> lock_type;

template<class CONDITION_TYPE>
class barrier {
public:
	typedef CONDITION_TYPE condition_type;
private:
	std::size_t     initial_;
	std::size_t     current_;
	bool            cycle_{ true };
	std::mutex      mtx_{};
	condition_type  cond_{};

public:
	explicit barrier(std::size_t initial) :
		initial_{ initial },
		current_{ initial_ } {
		BOOST_ASSERT(0 != initial);
	}

	barrier() : initial_(0), current_(0) { }
	barrier(barrier&& other) : initial_(other.initial_), current_(other.current_) { };

	barrier(barrier const&) = delete;
	barrier& operator=(barrier const&) = delete;

	bool wait() {
		lock_type lk(mtx_);
		const bool cycle = cycle_;
		if (0 == --current_) {
			cycle_ = ! cycle_;
			current_ = initial_;
			lk.unlock(); // no pessimization
			cond_.notify_all();
			return true;
		} else {
			cond_.wait(lk, [&](){ return cycle != cycle_; });
		}
		return false;
	}
};

typedef barrier<std::condition_variable> thread_barrier;
typedef barrier<boost::fibers::condition_variable_any> fiber_barrier;
//typedef boost::fibers::barrier fiber_barrier;

typedef void (*func_type) (void*, int32_t, int32_t);

struct Context {
	int num_threads{0};
	std::size_t fiber_count{0};
	std::mutex mtx_count{};
	boost::fibers::condition_variable_any cnd_count{};

	explicit Context(int num_threads_) : num_threads(num_threads_) { }

	Context(Context const&) = delete;
	Context& operator=(Context const&) = delete;
};


static std::vector<fiber_barrier> block_barriers;


void anydsl_fibers_sync_block(int32_t block) {
	std::ostringstream buffer;
	buffer << "wait for block barrier " << block << std::endl;
	std::cout << buffer.str() << std::flush;
	block_barriers[block].wait();
}

void anydsl_fibers_yield() {
	boost::this_fiber::yield();
}


void thread_fun(Context* ctx, thread_barrier* b) {
	std::ostringstream buffer;
	buffer << "thread started " << std::this_thread::get_id() << std::endl;
	std::cout << buffer.str() << std::flush;
	boost::fibers::use_scheduling_algorithm<boost::fibers::algo::work_stealing>(ctx->num_threads);

	b->wait();

	lock_type lk(ctx->mtx_count);
	ctx->cnd_count.wait(lk, [&ctx](){ return 0 == ctx->fiber_count; });

	BOOST_ASSERT( 0 == ctx->fiber_count);
}


void fiber_fun(Context& ctx, int block, int warp, void* args, func_type func) {
	try {
		std::thread::id my_thread = std::this_thread::get_id(); /*< get ID of initial thread >*/
		std::ostringstream buffer;
		buffer << "fiber " << block << "/" << warp << " started on thread " << my_thread << '\n';
		std::cout << buffer.str() << std::flush;

		// invoke the actual work function
		func(args, block, warp);
	} catch ( ... ) {
	}
	lock_type lk(ctx.mtx_count);
	if (0 == --ctx.fiber_count) {
		lk.unlock();
		ctx.cnd_count.notify_all(); /*< Notify all fibers waiting on `cnd_count`. >*/
	}
}



void anydsl_fibers_spawn(
	int32_t num_threads,
	int32_t num_blocks,
	int32_t num_warps,
	void* args, void* fun
) {
	Context ctx(num_threads);

	void (*fun_ptr) (void*, int32_t, int32_t) = reinterpret_cast<void (*) (void*, int32_t, int32_t)>(fun);


	boost::fibers::use_scheduling_algorithm<boost::fibers::algo::work_stealing>(ctx.num_threads);

	// TODO: incorporate num_blocks_in_flight to reuse fibers for multiple blocks
	block_barriers.clear();
	block_barriers.reserve(num_blocks);
	std::cout << "size of barrier " << sizeof(fiber_barrier) << " / size of block barriers " << ((block_barriers.capacity()*sizeof(fiber_barrier)) >> 10) << "KB" << std::endl;

	for (int block = 0; block < num_blocks; ++block) {
		block_barriers.emplace_back(num_warps);
		for (int warp = 0; warp < num_warps; ++warp) {
			// Launch a number of worker fibers
			// Each worker fiber gets detached.
			boost::fibers::fiber([&ctx, &args, &fun_ptr, block, warp](){ fiber_fun(ctx, block, warp, args, fun_ptr); }).detach();
			++ctx.fiber_count;
		}
	}
	BOOST_ASSERT(num_blocks*num_warps == ctx.fiber_count);

	thread_barrier b(ctx.num_threads);
	// Launch a couple of additional threads that join the work sharing.
	std::vector<std::thread> threads;
	for (int t = 1; t < ctx.num_threads; ++t) {
		threads.emplace_back(thread_fun, &ctx, &b);
	}

	// sync main thread with other threads
	b.wait();

	{
		lock_type lk(ctx.mtx_count);
		ctx.cnd_count.wait(lk, [&ctx](){ return 0 == ctx.fiber_count; });
		/*
			Suspend main fiber and resume worker fibers in the meanwhile.
			Main fiber gets resumed (e.g returns from `condition_variable_any::wait()`)
			if all worker fibers are complete.
		*/
	}
	BOOST_ASSERT(0 == ctx.fiber_count);

	for (std::thread & t : threads) {
		t.join();
	}
}

