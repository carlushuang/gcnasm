// Quad-subtile BF16 GEMM stub TU.
// Host pass: empty kernel body so the launch stub (`__device_stub__`) symbol
//            exists for linking against gemm_host.o.
// Device pass: include the full kernel template + explicit instantiation.
#include <opus/hip_minimal.hpp>
#include "gemm_defs.h"

// Optimized instance map:
//   <false,false,false>: Direct LSA / local-output baseline.
//   <false,false,true >: rank-rotated two-M-tile Direct LSA stripe.
//   <true, false,false>: compact local staging for split LSA / bulk SDMA.
//   <true, true, false>: chunk-ready in-kernel SDMA submission.
#ifndef __HIP_DEVICE_COMPILE__
template<typename Traits, bool LocalStaging, bool ChunkFused, bool DirectStriped>
__global__ void gemm_a16w16_quad_subtile_kernel(opus_gemm_kargs kargs) {}
template __global__ void gemm_a16w16_quad_subtile_kernel<
    opus_gemm_traits<512, 256, 256, 64, bf16_t, bf16_t, bf16_t, float>, false, false, false>(opus_gemm_kargs);
template __global__ void gemm_a16w16_quad_subtile_kernel<
    opus_gemm_traits<512, 256, 256, 64, bf16_t, bf16_t, bf16_t, float>, false, false, true>(opus_gemm_kargs);
template __global__ void gemm_a16w16_quad_subtile_kernel<
    opus_gemm_traits<512, 256, 256, 64, bf16_t, bf16_t, bf16_t, float>, true, false, false>(opus_gemm_kargs);
template __global__ void gemm_a16w16_quad_subtile_kernel<
    opus_gemm_traits<512, 256, 256, 64, bf16_t, bf16_t, bf16_t, float>, true, true, false>(opus_gemm_kargs);
#else
#include "gemm_a16w16_quad_subtile_kernel_template.hpp"
template __global__ void gemm_a16w16_quad_subtile_kernel<
    opus_gemm_traits<512, 256, 256, 64, bf16_t, bf16_t, bf16_t, float>, false, false, false>(opus_gemm_kargs);
template __global__ void gemm_a16w16_quad_subtile_kernel<
    opus_gemm_traits<512, 256, 256, 64, bf16_t, bf16_t, bf16_t, float>, false, false, true>(opus_gemm_kargs);
template __global__ void gemm_a16w16_quad_subtile_kernel<
    opus_gemm_traits<512, 256, 256, 64, bf16_t, bf16_t, bf16_t, float>, true, false, false>(opus_gemm_kargs);
template __global__ void gemm_a16w16_quad_subtile_kernel<
    opus_gemm_traits<512, 256, 256, 64, bf16_t, bf16_t, bf16_t, float>, true, true, false>(opus_gemm_kargs);
#endif
