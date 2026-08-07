// SPDX-License-Identifier: MIT
// v8 instantiation — single wave, BLOCK_M=16, BLOCK_N=32, contig V load.
#include <hip/hip_runtime.h>
#include "attn_gfx1201_kernel_v8_template.hpp"

template __global__ void opus_attn_gfx1201_kernel_v8<opus_attn_traits<16, 32, 128>>(opus_attn_kargs);
