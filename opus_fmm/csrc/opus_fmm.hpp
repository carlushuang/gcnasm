#pragma once

#include "opus_fmm_kernel.hpp"

struct opus_gemm_arg {
    opus_gemm_rcr_f16_hargs h;
    hipStream_t s;
};

// pure C like host API
void opus_fmm(opus_gemm_arg);
