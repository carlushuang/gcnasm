// pyhip_v3: Truly minimal — ZERO host declarations.
// Compiled with --genco (device-only). No dim3, no hipLaunchKernel,
// no host stubs. Kernel launch done from Python via hipModuleLaunchKernel.
//
// Only device-side helpers that can't be replaced by builtins:
//   __shfl — cross-lane data exchange (ds_bpermute builtin)
#pragma once

#include <cstdint>

#ifndef INFINITY
#define INFINITY __builtin_huge_valf()
#endif

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
