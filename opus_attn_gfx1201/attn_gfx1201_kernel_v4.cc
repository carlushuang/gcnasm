// SPDX-License-Identifier: MIT
// v4 instantiation — single-wave (v0 geometry) + double-buffered K/V
// register prefetch.
#include <hip/hip_runtime.h>
#include "attn_gfx1201_kernel_v4_template.hpp"

template __global__ void opus_attn_gfx1201_kernel_v4<opus_attn_traits<16, 16, 128>>(opus_attn_kargs);
