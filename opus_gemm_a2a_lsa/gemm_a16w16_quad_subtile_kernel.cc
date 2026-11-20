// Quad-subtile BF16 GEMM stub TU.
// Host pass: empty kernel body so the launch stub (`__device_stub__`) symbol
//            exists for linking against gemm_host.o.
// Device pass: include the full kernel template + explicit instantiation.
#include <opus/hip_minimal.hpp>
#include "gemm_defs.h"

#ifndef __HIP_DEVICE_COMPILE__
template<typename Traits>
__global__ void gemm_a16w16_quad_subtile_kernel(opus_gemm_kargs kargs) {}
template __global__ void gemm_a16w16_quad_subtile_kernel<
    opus_gemm_traits<512, 256, 256, 64, bf16_t, bf16_t, bf16_t, float>>(opus_gemm_kargs);
#else
#include "gemm_a16w16_quad_subtile_kernel_template.hpp"
template __global__ void gemm_a16w16_quad_subtile_kernel<
    opus_gemm_traits<512, 256, 256, 64, bf16_t, bf16_t, bf16_t, float>>(opus_gemm_kargs);
#endif
