#ifndef anyq_fibers_h
#define anyq_fibers_h


#ifdef __cplusplus
extern "C" {
#endif

void anydsl_fibers_spawn(int32_t, int32_t, int32_t, void*, void*);
void anydsl_fibers_sync_block(int32_t block);
void anydsl_fibers_yield();

#ifdef __cplusplus
}
#endif

#endif // anyq_fibers_h
