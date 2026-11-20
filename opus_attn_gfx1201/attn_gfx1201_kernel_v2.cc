// SPDX-License-Identifier: MIT
// v2 instantiation — v1 layout + BLOCK_N = 64 (4 wmma N-tiles per softmax).
#include <hip/hip_runtime.h>
#include "attn_gfx1201_kernel_v2_template.hpp"

template __global__ void opus_attn_gfx1201_kernel_v2<opus_attn_traits<64, 64, 128>>(opus_attn_kargs);
