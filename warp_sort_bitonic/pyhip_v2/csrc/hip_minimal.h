// Minimal HIP declarations for pyhip_v2 — no <<<>>> in USER code.
// Kernel launch is done entirely from Python via hipLaunchKernel.
//
// Note: hipcc still generates host stubs for __global__ functions internally,
// which reference dim3/hipLaunchKernel. We declare them here for the compiler
// but our C++ code never uses <<<>>> or dim3 directly.
//
// Device-side:
//   __shfl — cross-lane data exchange (uses ds_bpermute builtin)
// Builtins used directly in kernel:
//   threadIdx.x   → __builtin_amdgcn_workitem_id_x()
//   __syncthreads → __builtin_amdgcn_s_barrier()
//   warpSize      → __builtin_amdgcn_wavefrontsize()
#pragma once

#include <cstdint>
#include <cstddef>

#ifndef INFINITY
#define INFINITY __builtin_huge_valf()
#endif

// ─── dim3 + launch decls (required by compiler-generated host stubs) ─────────
struct dim3 {
    uint32_t x, y, z;
    constexpr __host__ __device__ dim3(uint32_t _x = 1, uint32_t _y = 1, uint32_t _z = 1)
        : x(_x), y(_y), z(_z) {}
};

typedef enum { hipSuccess = 0 } hipError_t;
typedef struct ihipStream_t* hipStream_t;

extern "C" hipError_t __hipPushCallConfiguration(
    dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, hipStream_t stream = 0);
extern "C" hipError_t __hipPopCallConfiguration(
    dim3* gridDim, dim3* blockDim, size_t* sharedMem, hipStream_t* stream);
extern "C" hipError_t hipLaunchKernel(
    const void* function_address, dim3 numBlocks, dim3 dimBlocks,
    void** args, size_t sharedMemBytes = 0, hipStream_t stream = 0);

// ─── __shfl (cross-lane shuffle via ds_bpermute) ─────────────────────────────
__device__ static inline unsigned int __lane_id() {
    if (__builtin_amdgcn_wavefrontsize() == 32)
        return __builtin_amdgcn_mbcnt_lo(-1, 0);
    return __builtin_amdgcn_mbcnt_hi(-1, __builtin_amdgcn_mbcnt_lo(-1, 0));
}

__device__ inline int __shfl(int var, int src_lane,
                             int width = __builtin_amdgcn_wavefrontsize()) {
    int self = __lane_id();
    int index = (src_lane & (width - 1)) + (self & ~(width - 1));
    return __builtin_amdgcn_ds_bpermute(index << 2, var);
}

__device__ inline float __shfl(float var, int src_lane,
                               int width = __builtin_amdgcn_wavefrontsize()) {
    union { int i; float f; } tmp;
    tmp.f = var;
    tmp.i = __shfl(tmp.i, src_lane, width);
    return tmp.f;
}
