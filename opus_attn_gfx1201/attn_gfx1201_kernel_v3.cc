// SPDX-License-Identifier: MIT
// v3 instantiation — same geometry as v1, tighter launch_bounds + fused
// softmax + single wave_barrier. Goal: higher occupancy via lower VGPR use.
#include <hip/hip_runtime.h>
#include "attn_gfx1201_kernel_v3_template.hpp"

template __global__ void opus_attn_gfx1201_kernel_v3<opus_attn_traits<64, 16, 128>>(opus_attn_kargs);
