// GQA flash attention kernel — causal variant
// Host pass: empty stub for __device_stub__ generation
// Device pass: includes full kernel template
#include <opus/hip_minimal.hpp>
#include "gqa_common.h"
#ifndef __HIP_DEVICE_COMPILE__
template<typename Traits> __global__ void gqa_kernel(opus_gqa_kargs kargs) {}
template __global__ void gqa_kernel<opus_gqa_traits<32, 64, 128, 8, true>>(opus_gqa_kargs);
#else
#include "gqa_gfx950_kernel_template.hpp"
template __global__ void gqa_kernel<opus_gqa_traits<32, 64, 128, 8, true>>(opus_gqa_kargs);
#endif
