#ifndef anyq_fibers_h
#define anyq_fibers_h


#ifdef __cplusplus
extern "C" {
#endif

void anydsl_fibers_spawn(int32_t, int32_t, int32_t, void*, void*);
void anydsl_fibers_sync_block(int32_t block);
void anydsl_fibers_sync_block_with_result(int32_t* output, int32_t* result, int32_t reset, int32_t block);
void anydsl_fibers_yield(const char* msg);

int32_t anyq_print_3xi32(const char* format, int32_t val0, int32_t val1, int32_t val2);

#ifdef __cplusplus
}
#endif

#endif // anyq_fibers_h
