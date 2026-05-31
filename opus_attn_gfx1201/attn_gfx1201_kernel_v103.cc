#include "attn_gfx1201_kernel_v103_template.hpp"

template __global__ void opus_attn_gfx1201_kernel_v103<opus_attn_traits<128, 32, 128>>(opus_attn_kargs);
