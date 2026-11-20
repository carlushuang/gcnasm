// SPDX-License-Identifier: MIT
// v0 instantiation — 1 wave/WG, BLOCK_M = BLOCK_N = 16. Correctness baseline.
#include <hip/hip_runtime.h>
#include "attn_gfx1201_kernel_v0_template.hpp"

template __global__ void opus_attn_gfx1201_kernel<opus_attn_traits<16, 16, 128>>(opus_attn_kargs);
