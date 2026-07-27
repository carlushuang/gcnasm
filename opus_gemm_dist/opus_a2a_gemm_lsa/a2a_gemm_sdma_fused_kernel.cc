#include <opus/hip_minimal.hpp>

#include "gemm_defs.h"

#ifndef __HIP_DEVICE_COMPILE__
template<typename Traits, int Mode, bool Persistent, typename Kargs>
__global__ void a2a_gemm_lsa_kernel(Kargs kargs) {}
#else
#include "a2a_gemm_kernel_template.hpp"
#endif

template __global__ void a2a_gemm_lsa_kernel<
    opus_gemm_traits<512, 256, 256, 64, bf16_t, bf16_t, bf16_t, float>,
    4,
    false,
    opus_a2a_gemm_sdma_fused_kargs>(
        opus_a2a_gemm_sdma_fused_kargs);

template __global__ void a2a_gemm_lsa_kernel<
    opus_gemm_traits<512, 256, 256, 64, bf16_t, bf16_t, bf16_t, float>,
    4,
    true,
    opus_a2a_gemm_sdma_fused_kargs>(
        opus_a2a_gemm_sdma_fused_kargs);
