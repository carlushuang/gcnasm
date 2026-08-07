// SPDX-License-Identifier: MIT
// v11 instantiation — swap_ab pattern, no S→P smem flip.
#include <hip/hip_runtime.h>
#include "attn_gfx1201_kernel_v11_template.hpp"

template __global__ void opus_attn_gfx1201_kernel_v11<opus_attn_traits<16, 16, 128>>(opus_attn_kargs);
