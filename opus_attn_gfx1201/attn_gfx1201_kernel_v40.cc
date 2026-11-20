#include "attn_gfx1201_kernel_v40_template.hpp"

template __global__ void opus_attn_gfx1201_kernel_v40<opus_attn_traits<128, 64, 128>>(opus_attn_kargs);
