// SPDX-License-Identifier: MIT
#include "attn_common.h"
#include "attn_gfx1201_kernel_v13_template.hpp"

template __global__ void opus_attn_gfx1201_kernel_v13<opus_attn_traits<16, 32, 128>>(opus_attn_kargs);
