#include <opus/hip_minimal.hpp>

#include "gemm_a8w8_mxfp8_scale_common.h"
#include "gemm_a8w8_mxfp8_scale_kernel_template.hpp"

// Two instantiations: the persistent fixed-B tile (4 output tiles per workgroup)
// and the plain one-tile-per-workgroup form.  The host picks between them by
// grid size, so the choice stays compile-time inside the kernel and the main
// loop keeps its register allocation.
using GemmTraitsPersist = gemm_a8w8_mxfp8_scale_traits<256, 256, 128, 1, 128, 128, 4, false>;
using GemmTraitsSingle  = gemm_a8w8_mxfp8_scale_traits<256, 256, 128, 1, 128, 128, 1, false>;
using GemmTraitsPersistBF16 = gemm_a8w8_mxfp8_scale_traits<256, 256, 128, 1, 128, 128, 4, true>;
using GemmTraitsSingleBF16  = gemm_a8w8_mxfp8_scale_traits<256, 256, 128, 1, 128, 128, 1, true>;

template __global__ void gemm_a8w8_mxfp8_scale_kernel<GemmTraitsPersist>(opus_gemm_scale_kargs);
template __global__ void gemm_a8w8_mxfp8_scale_kernel<GemmTraitsSingle>(opus_gemm_scale_kargs);
template __global__ void gemm_a8w8_mxfp8_scale_kernel<GemmTraitsPersistBF16>(opus_gemm_scale_kargs);
template __global__ void gemm_a8w8_mxfp8_scale_kernel<GemmTraitsSingleBF16>(opus_gemm_scale_kargs);
