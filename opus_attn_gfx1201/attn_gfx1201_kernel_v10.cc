// SPDX-License-Identifier: MIT
// v10 instantiation — single wave, BLOCK_M=16, BLOCK_N=32, V is pre-transposed.
#include <hip/hip_runtime.h>
#include "attn_gfx1201_kernel_v10_template.hpp"

template __global__ void opus_attn_gfx1201_kernel_v10<opus_attn_traits<16, 32, 128>>(opus_attn_kargs);
