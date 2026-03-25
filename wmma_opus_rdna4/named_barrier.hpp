#pragma once
#include <cstdint>

// [COMPUTE_FIX #5] Named barrier helpers using compiler builtins.
// The barrier variables must be declared inside the kernel function
// as __shared__, then passed by pointer to init/join helpers.
// This lets the compiler track named barrier usage and set NAMED_BAR_CNT.

// Macro to declare named barrier variables inside a kernel body.
// Usage: DECLARE_NAMED_BARRIERS(); at the top of the kernel function.
#define DECLARE_NAMED_BARRIERS() \
    __shared__ __amdgpu_named_workgroup_barrier_t __nbar_1; \
    __shared__ __amdgpu_named_workgroup_barrier_t __nbar_2; \
    __shared__ __amdgpu_named_workgroup_barrier_t __nbar_3;

__device__ __forceinline__ void s_barrier_init_ptr(
    __amdgpu_named_workgroup_barrier_t *bar, uint32_t member_cnt) {
    __builtin_amdgcn_s_barrier_init(bar, member_cnt);
}

__device__ __forceinline__ void s_barrier_join_ptr(
    __amdgpu_named_workgroup_barrier_t *bar) {
    __builtin_amdgcn_s_barrier_join(bar);
}