
## Building

prerequisites:
  * C++17 toolchain (gcc, clang, or msvc)
  * Git (with git-svn)
  * [Ninja](https://ninja-build.org/)
  * CMake 3.21+
  * Python 3.10+

optional dependencies:
  * CUDA Toolkit 11.x+ (for cuda/nvvm support)
  * ROCm (for amdgpu support)

Use
```
./build.py pull
```
to pull and then
```
./build.py build -cfg Release
```
to build dependencies and generate the AnyQ build.
Once complete, build AnyQ via
```
cmake --build ./build/ --config Release
```
Benchmark binaries will be generated in `build/bin/`.

---

## Adding a Queue

example: adding `MyQueue` (MYQ)

  * create `src/queues/MYQ/`
  * add queue implementation source file(s), e.g.: `src/queues/MYQ/my_queue.art`
    * provide queue constructor

```rust
fn @createMyQueue[T](device: AccDevice, queue_size: i32) {
    …

    ProducerConsumerQueue[T] {
        push = @|source|@|thread| {
           …
        },

        pop = @|sink|@|thread| {
           …
        },

        size = @|thread| {
           …
        },

        reset = @|grid| {
           …
        },

        validate = @|corrupted, grid| {
           …
        },

        release = @|| {
           …
        }
    }
}
```

  * create `src/queues/MYQ/queue.cmake`

```cmake
set(MyQueue_short_name MYQ)

set(MyQueue_sources ${CMAKE_CURRENT_LIST_DIR}/my_queue.art)
```

  * update `src/queues/queues.cmake`
    * `include(${CMAKE_CURRENT_LIST_DIR}/MYQ/queues.cmake)`
    * update `queue_types_<platform>` variables to include `MyQueue` for benchmarking on the respective platforms

```cmake
set(queue_types_cpu … MyQueue)
set(queue_types_cuda … MyQueue)
set(queue_types_amdgpu … MyQueue)
      ⋮
```
