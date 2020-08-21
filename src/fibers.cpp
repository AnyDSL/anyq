

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

	template<typename T>
	bool wait_for_result(T* output, T* result, T reset) {
		lock_type lk(mtx_);
		const bool cycle = cycle_;

		if (0 == --current_) {
			cycle_ = ! cycle_;
			current_ = initial_;

			// copy result for all waiting fibers
			*output = *result;
			// reset predicate for next use
			*result = reset;

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
	bool terminate{false};

	explicit Context(int num_threads_) : num_threads(num_threads_) { }

	Context(Context const&) = delete;
	Context& operator=(Context const&) = delete;
};


static std::vector<fiber_barrier> block_barriers;


void anydsl_fibers_sync_block(int32_t block) {
	//std::ostringstream buffer;
	//buffer << "wait for block barrier " << block << std::endl;
	//std::cout << buffer.str() << std::flush;
	block_barriers[block].wait();
}

void anydsl_fibers_sync_block_with_result(int32_t* output, int32_t* result, int32_t reset, int32_t block) {
	//std::ostringstream buffer;
	//buffer << "wait for block barrier " << block << " with current result " << *result << std::endl;
	//std::cout << buffer.str() << std::flush;
	block_barriers[block].wait_for_result<int32_t>(output, result, reset);
}

void anydsl_fibers_yield() {
	boost::this_fiber::yield();
}


void thread_fun(Context* ctx, thread_barrier* b) {
	if (false) {
		std::ostringstream buffer;
		buffer << "thread started " << std::this_thread::get_id() << std::endl;
		std::cerr << buffer.str() << std::flush;
	}

	static thread_local std::once_flag flag;
	std::call_once(flag, [&ctx] {
		if (false) {
			std::ostringstream buffer;
			buffer << "thread " << std::this_thread::get_id() << " joins work_stealing scheduler" << std::endl;
			std::cout << buffer.str() << std::flush;
		}
		boost::fibers::use_scheduling_algorithm<boost::fibers::algo::work_stealing>(ctx->num_threads);
	});
	b->wait();

	while (true) {
		b->wait();
		if (ctx->terminate) break;

		if (false) {
			std::ostringstream buffer;
			buffer << "thread start work " << std::this_thread::get_id() << std::endl;
			std::cerr << buffer.str() << std::flush;
		}

		{
			lock_type lk(ctx->mtx_count);
			ctx->cnd_count.wait(lk, [&ctx](){ return 0 == ctx->fiber_count; });
		}

		if (false) {
			std::ostringstream buffer;
			buffer << "thread finished work " << std::this_thread::get_id() << std::endl;
			std::cerr << buffer.str() << std::flush;
		}

		// this assertion can't be hold if new fibers are scheduled before reaching this point
		b->wait();
		//BOOST_ASSERT( 0 == ctx->fiber_count);
	}

	if (false) {
		std::ostringstream buffer;
		buffer << "thread terminated " << std::this_thread::get_id() << std::endl;
		std::cerr << buffer.str() << std::flush;
	}
}


void fiber_fun(Context& ctx, int block, int warp, void* args, func_type func) {
	try {
		// std::thread::id my_thread = std::this_thread::get_id(); /*< get ID of initial thread >*/
		// std::ostringstream buffer;
		// buffer << "fiber " << block << "/" << warp << " started on thread " << my_thread << '\n';
		// std::cout << buffer.str() << std::flush;

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


class thread_pool : public std::vector<std::thread> {
public:
	thread_pool(int num_threads, thread_barrier& barrier, bool& terminate) : _barrier(barrier), _terminate(terminate) {
		reserve(num_threads);
	}

	~thread_pool() {
		_terminate = true;
		_barrier.wait();

		for (std::thread & t : *this) {
			if (!t.joinable())
				std::cerr << "thread not joinable " << t.get_id() << std::endl;
			t.join();
		}
	}

private:
	thread_barrier& _barrier;
	bool& _terminate;
};

void anydsl_fibers_spawn(
	int32_t num_threads,
	int32_t num_blocks,
	int32_t num_warps,
	void* args, void* fun
) {
	// unfortunately, boost uses some static variables that do not allow clean shutdown
	// thus, we must use a static thread pool and context for proper work stealing scheduling
	static int fixed_num_threads = num_threads == 0 ? std::thread::hardware_concurrency() : num_threads;

	static Context ctx(fixed_num_threads);

	void (*fun_ptr) (void*, int32_t, int32_t) = reinterpret_cast<void (*) (void*, int32_t, int32_t)>(fun);

	static thread_barrier b(ctx.num_threads);
	// Launch a couple of additional threads that join the work sharing.
	static thread_pool threads(ctx.num_threads, b, ctx.terminate);
	static std::once_flag init_workers;
	std::call_once(init_workers, [&] {
		if (true) {
			std::ostringstream buffer;
			buffer << "using " << ctx.num_threads << " worker threads " << std::endl;
			std::cerr << buffer.str() << std::flush;
		}

		for (int t = 1; t < ctx.num_threads; ++t) {
			threads.emplace_back(thread_fun, &ctx, &b);
		}

		// wait for all threads to join work scheduling
		b.wait();

		// main thread must not join scheduling multiple times
		if (false) {
			std::ostringstream buffer;
			buffer << "main thread " << std::this_thread::get_id() << " joins work_stealing scheduler" << std::endl;
			std::cerr << buffer.str() << std::flush;
		}
		boost::fibers::use_scheduling_algorithm<boost::fibers::algo::work_stealing>(ctx.num_threads);
	});

	// TODO: incorporate num_blocks_in_flight to reuse fibers for multiple blocks
	block_barriers.clear();
	block_barriers.reserve(num_blocks);
	//std::cout << "size of barrier " << sizeof(fiber_barrier) << " / size of block barriers " << ((block_barriers.capacity()*sizeof(fiber_barrier)) >> 10) << "KB" << std::endl;

	for (int block = 0; block < num_blocks; ++block) {
		block_barriers.emplace_back(num_warps);
		for (int warp = 0; warp < num_warps; ++warp) {
			// Launch a number of worker fibers
			// Each worker fiber gets detached.
			boost::fibers::fiber([&args, &fun_ptr, block, warp](){ fiber_fun(ctx, block, warp, args, fun_ptr); }).detach();
			++ctx.fiber_count;
		}
	}
	BOOST_ASSERT(num_blocks*num_warps == ctx.fiber_count);

	// sync main thread with other threads before starting current work load
	b.wait();

	if (false) {
		std::ostringstream buffer;
		buffer << "main thread start work " << std::this_thread::get_id() << std::endl;
		std::cerr << buffer.str() << std::flush;
	}

	{
		lock_type lk(ctx.mtx_count);
		ctx.cnd_count.wait(lk, [](){ return 0 == ctx.fiber_count; });
		/*
			Suspend main fiber and resume worker fibers in the meanwhile.
			Main fiber gets resumed (e.g returns from `condition_variable_any::wait()`)
			if all worker fibers are complete.
		*/
	}

	if (false) {
		std::ostringstream buffer;
		buffer << "main thread finished work " << std::this_thread::get_id() << std::endl;
		std::cerr << buffer.str() << std::flush;
	}

	// sync main thread with other threads
	b.wait();
	BOOST_ASSERT(0 == ctx.fiber_count);
}


static std::mutex fiber_print_mtx{};

int32_t anyq_print_i32a(const char* format, int32_t val0, int32_t val1) {
	lock_type lk(fiber_print_mtx);

	fprintf(stdout, format, val0, val1);
	fflush(stdout);

	return 0;
}
