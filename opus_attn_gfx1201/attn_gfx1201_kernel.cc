// SPDX-License-Identifier: MIT
// Device-side instantiation of the gfx1201 attention kernel. Compiled with
// the standard hipcc front-end (no -D__HIPCC_RTC__); host-side launcher
// lives in attn_gfx1201_host.cc.
#include <hip/hip_runtime.h>
#include "attn_gfx1201_kernel_template.hpp"

template __global__ void opus_attn_gfx1201_kernel<opus_attn_traits<16, 16, 128>>(opus_attn_kargs);
