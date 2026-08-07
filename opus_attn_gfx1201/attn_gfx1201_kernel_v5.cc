// SPDX-License-Identifier: MIT
// v5 instantiation — single wave, BLOCK_M=16, BLOCK_N=32.
#include <hip/hip_runtime.h>
#include "attn_gfx1201_kernel_v5_template.hpp"

template __global__ void opus_attn_gfx1201_kernel_v5<opus_attn_traits<16, 32, 128>>(opus_attn_kargs);
