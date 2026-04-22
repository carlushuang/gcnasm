#include <opus/hip_minimal.hpp>

#include "gemm_a8w8_blockscale_common.h"

using GemmTraits = gemm_a8w8_blockscale_traits<>;

#ifndef __HIP_DEVICE_COMPILE__
template<typename Traits>
__global__ void gemm_a8w8_blockscale_kernel(opus_gemm_kargs kargs) {}

template __global__ void gemm_a8w8_blockscale_kernel<GemmTraits>(opus_gemm_kargs);
#else
#include "gemm_a8w8_blockscale_kernel_template.hpp"

template __global__ void gemm_a8w8_blockscale_kernel<GemmTraits>(opus_gemm_kargs);
#endif
