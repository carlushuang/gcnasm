#include <hip/hip_runtime.h>
#include <stdint.h>
#include "common.h"

#define BLOCK_SIZE 256
template<typename T>
__global__
void memcpy_kernel(T* __restrict__ dst, const T* __restrict__ src, uint32_t n){
    int idx = (blockIdx.x * BLOCK_SIZE + threadIdx.x);
    if(idx < n)
        dst[idx] = src[idx];
}

template<typename T>
struct memcpy_with_type
{
    static void run(void * B, const void * A, int dwords){
        int bx = BLOCK_SIZE;
        int pixels = dwords * 4 / sizeof(T);
        int gx = (pixels + BLOCK_SIZE - 1) / BLOCK_SIZE;
        memcpy_kernel<T><<<gx, bx>>>(reinterpret_cast<T*>(B), reinterpret_cast<const T*>(A), pixels);
    }
};

extern "C"
void memcpy_fp32(void* dst, const void* src, uint32_t dwords)
{
    memcpy_with_type<fp32>::run(dst, src, dwords);
}

extern "C"
void memcpy_fp32x2(void* dst, const void* src, uint32_t dwords)
{
    memcpy_with_type<fp32x2>::run(dst, src, dwords);
}

extern "C"
void memcpy_fp32x4(void* dst, const void* src, uint32_t dwords)
{
    memcpy_with_type<fp32x4>::run(dst, src, dwords);
}

