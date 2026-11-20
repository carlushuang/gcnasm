// SPDX-License-Identifier: MIT
// v1 instantiation — 4 waves/WG, BLOCK_M = 64, BLOCK_N = 16, cooperative
// smem K/V. First perf pass over v0.
#include <hip/hip_runtime.h>
#include "attn_gfx1201_kernel_v1_template.hpp"

template __global__ void opus_attn_gfx1201_kernel_v1<opus_attn_traits<64, 16, 128>>(opus_attn_kargs);
