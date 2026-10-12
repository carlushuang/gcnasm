// 4wave_compute BF16 GEMM stub TU.
// Host pass: empty kernel body so the launch stub (`__device_stub__`) symbol
//            exists for linking against gemm_host.o.
// Device pass: include the full kernel template + explicit instantiation.
#include <opus/hip_minimal.hpp>
#include "gemm_defs.h"

#ifndef __HIP_DEVICE_COMPILE__
template<typename Traits>
__global__ void gemm_a16w16_4wave_compute_kernel(opus_gemm_kargs kargs) {}
#else
#include "gemm_a16w16_4wave_compute_kernel_template.hpp"
#endif

template __global__ void
gemm_a16w16_4wave_compute_kernel<gemm_traits_128x256x128>(opus_gemm_kargs);
