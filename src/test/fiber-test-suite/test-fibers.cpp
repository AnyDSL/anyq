
#include <iostream>
#include <cstddef>
#include <thread>

#include "fibers.h"


void my_func(void* args, int32_t block, int32_t warp) {
	printf("I am %d/%d at A\n", block, warp);

	// TODO sync
	anydsl_fibers_sync_block(block);

	printf("I am %d/%d at B\n", block, warp);
}


int main(int argc, char* argv[]) {

    std::cout << "main thread started " << std::this_thread::get_id() << std::endl;

	int num_threads = 1;
	int num_blocks = 4;
	int num_warps = 8;

	anydsl_fibers_spawn(num_threads, num_blocks, num_warps, NULL, reinterpret_cast<void*>(my_func));

    std::cout << "done." << std::endl;
    return EXIT_SUCCESS;
}
