#include <opus/hip_minimal.hpp>

#ifndef __HIP_DEVICE_COMPILE__
#include "gemm_defs.h"  // IWYU pragma: keep -- host-pass template types
#endif

// Mode 4 is the literal single-kernel SDMA experiment. Keep static and
// persistent instantiations separate so resource and scheduler A/B tests do
// not perturb the retained static path.
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
