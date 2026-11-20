// SPDX-License-Identifier: MIT
// v9 instantiation — single wave, BLOCK_M=16, BLOCK_N=16, V is pre-transposed.
#include <hip/hip_runtime.h>
#include "attn_gfx1201_kernel_v9_template.hpp"

template __global__ void opus_attn_gfx1201_kernel_v9<opus_attn_traits<16, 16, 128>>(opus_attn_kargs);
