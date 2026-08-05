#include <opus/hip_minimal.hpp>
#include "gemm_defs.h"

// Kernel mode map:
//   0: fused LSA copy + GEMM with XCC-spread communication WGs.
//   1: compute-only local source baseline.
//   2: compute from a fully received shard layout (split LSA / SDMA pipeline).
//   3: same-epoch SDMA GEMM with per-source ready polling.
#ifndef __HIP_DEVICE_COMPILE__
template<typename Traits, int Mode, bool Persistent,
         typename Kargs = opus_a2a_gemm_kargs>
__global__ void a2a_gemm_lsa_kernel(Kargs kargs) {}
template<typename Traits>
__global__ void a2a_lsa_comm_kernel(opus_a2a_gemm_kargs kargs) {}
template __global__ void a2a_lsa_comm_kernel<
    opus_gemm_traits<512, 256, 256, 64, bf16_t, bf16_t, bf16_t, float>>(opus_a2a_gemm_kargs);
template __global__ void a2a_gemm_lsa_kernel<
    opus_gemm_traits<512, 256, 256, 64, bf16_t, bf16_t, bf16_t, float>, 0, false>(opus_a2a_gemm_kargs);
template __global__ void a2a_gemm_lsa_kernel<
    opus_gemm_traits<512, 256, 256, 64, bf16_t, bf16_t, bf16_t, float>, 0, true>(opus_a2a_gemm_kargs);
template __global__ void a2a_gemm_lsa_kernel<
    opus_gemm_traits<512, 256, 256, 64, bf16_t, bf16_t, bf16_t, float>, 1, false>(opus_a2a_gemm_kargs);
template __global__ void a2a_gemm_lsa_kernel<
    opus_gemm_traits<512, 256, 256, 64, bf16_t, bf16_t, bf16_t, float>, 2, false>(opus_a2a_gemm_kargs);
template __global__ void a2a_gemm_lsa_kernel<
    opus_gemm_traits<512, 256, 256, 64, bf16_t, bf16_t, bf16_t, float>, 3, false>(opus_a2a_gemm_kargs);
#else
#include "a2a_gemm_kernel_template.hpp"
template __global__ void a2a_lsa_comm_kernel<
    opus_gemm_traits<512, 256, 256, 64, bf16_t, bf16_t, bf16_t, float>>(opus_a2a_gemm_kargs);
template __global__ void a2a_gemm_lsa_kernel<
    opus_gemm_traits<512, 256, 256, 64, bf16_t, bf16_t, bf16_t, float>, 0, false>(opus_a2a_gemm_kargs);
template __global__ void a2a_gemm_lsa_kernel<
    opus_gemm_traits<512, 256, 256, 64, bf16_t, bf16_t, bf16_t, float>, 0, true>(opus_a2a_gemm_kargs);
template __global__ void a2a_gemm_lsa_kernel<
    opus_gemm_traits<512, 256, 256, 64, bf16_t, bf16_t, bf16_t, float>, 1, false>(opus_a2a_gemm_kargs);
template __global__ void a2a_gemm_lsa_kernel<
    opus_gemm_traits<512, 256, 256, 64, bf16_t, bf16_t, bf16_t, float>, 2, false>(opus_a2a_gemm_kargs);
template __global__ void a2a_gemm_lsa_kernel<
    opus_gemm_traits<512, 256, 256, 64, bf16_t, bf16_t, bf16_t, float>, 3, false>(opus_a2a_gemm_kargs);
#endif
