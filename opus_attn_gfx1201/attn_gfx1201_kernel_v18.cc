#include "attn_gfx1201_kernel_v18_template.hpp"

template __global__ void opus_attn_gfx1201_kernel_v18<opus_attn_traits<16, 64, 128>>(opus_attn_kargs);
