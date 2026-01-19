#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <random>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <numeric>
#include <list>
#include "ck_tile/core.hpp"

void opus_fmm(opus_gemm_arg a)  {
    // TODO: dispatch layer
    using traits = opus::opus_flatmm_traits<  opus::seq<256,   1,  1>,    // WG
                                              opus::seq<128, 128, 64>,    // Block
                                              opus::seq<  2,   2,  1>,    // Tiles
                                              opus::seq< 16,  16, 16>,    // Waves
                                              opus::tuple<opus::bf16_t, opus::bf16_t, opus::bf16_t, opus::fp32_t, opus::fp32_t>, // opus::tuple<d_a, d_b, d_c, d_acc, d_bias>, data type
                                              opus::seq<  8,   8,  8>,    // Vec
                                              1,                          // tile scheduler
                                              false                       // bias
                                              >;
    auto k = opus_fmm_kernel<traits>{};
    auto h = a.h;

    if(!k.is_applicable(h))
        return 0;   // TODO: assert?

    k.prepare(h)
    k();

}