// SPDX-License-Identifier: MIT
// v7 instantiation — single wave, BLOCK_M=16, BLOCK_N=16, batched V load.
#include <hip/hip_runtime.h>
#include "attn_gfx1201_kernel_v7_template.hpp"

template __global__ void opus_attn_gfx1201_kernel_v7<opus_attn_traits<16, 16, 128>>(opus_attn_kargs);
