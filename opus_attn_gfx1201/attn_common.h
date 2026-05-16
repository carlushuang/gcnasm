// SPDX-License-Identifier: MIT
// opus_attn_gfx1201 — shared types between device kernel and host driver.
#pragma once

using fp16_t = _Float16;
using fp32_t = float;

// Kernel arguments. Layout: Q,K,V,O are [B, H, N, D] row-major (head_dim D
// innermost). No GQA — H_q == H_kv.
struct opus_attn_kargs {
    const void* __restrict__ ptr_q;   // [B, H, N, D]
    const void* __restrict__ ptr_k;   // [B, H, N, D]
    const void* __restrict__ ptr_v;   // [B, H, N, D]
    void*       __restrict__ ptr_o;   // [B, H, N, D]
    int B;
    int H;
    int N;
    int D;
    fp32_t scale;                     // typically 1/sqrt(D)
};

// Kernel traits.
//
// Each version (v0..v5) instantiates this with its own BLOCK_M / BLOCK_N to
// describe per-workgroup geometry. NUM_WAVES is derived (each wave handles
// WAVE_M=16 M-rows independently for softmax).
template<int BLOCK_M_, int BLOCK_N_, int D_ = 128>
struct opus_attn_traits {
    static constexpr int BLOCK_M    = BLOCK_M_;
    static constexpr int BLOCK_N    = BLOCK_N_;
    static constexpr int D          = D_;
    static constexpr int W_M        = 16;
    static constexpr int W_N        = 16;
    static constexpr int W_K        = 16;
    static constexpr int WARP_SIZE  = 32;
    static constexpr int WAVE_M     = 16;                       // each wave does 16 M-rows
    static constexpr int NUM_WAVES  = BLOCK_M / WAVE_M;
    static constexpr int BLOCK_SIZE = NUM_WAVES * WARP_SIZE;
    static constexpr int D_TILES_K  = D / W_K;
    static constexpr int N_TILES    = BLOCK_N / W_N;            // wmma N-tiles per softmax

    static_assert(BLOCK_M % WAVE_M == 0, "BLOCK_M must be multiple of WAVE_M=16");
    static_assert(BLOCK_N % W_N    == 0, "BLOCK_N must be multiple of W_N=16");
    static_assert(D       % W_K    == 0, "D must be multiple of W_K=16");
};

__host__ __device__ inline int ceil_div(int a, int b) { return (a + b - 1) / b; }
