#include "attn_gfx1201_kernel_v32_template.hpp"

template __global__ void opus_attn_gfx1201_kernel_v32<opus_attn_traits<128, 16, 128>>(opus_attn_kargs);
