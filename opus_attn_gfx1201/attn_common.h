// SPDX-License-Identifier: MIT
// opus_attn_gfx1201 — shared types between device kernel and host driver.
#pragma once

using fp16_t = _Float16;
using fp32_t = float;

// Kernel arguments. Layout: Q,K,V,O are [B, H, N, D] row-major (head_dim D
// innermost). No GQA in v0 — H_q == H_kv.
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

// Kernel traits — fp16 inputs, fp32 accumulator, fp16 output, head_dim=128.
//
// Tile layout (each workgroup = 1 wave32, computes BLOCK_M rows of one
// (head, batch) for full N):
//   BLOCK_M = 16    one wmma M-tile per workgroup
//   BLOCK_N = 16    one wmma N-tile per KV-loop iteration (online softmax)
//   D       = 128   head dim, factored as D/16 = 8 wmma K-tiles
//
// Per wave register usage:
//   v_q     : 8 fp16/lane × (D/16) = 64 fp16/lane         (Q tile, persistent)
//   v_o     : 8 fp32/lane × (D/16) = 64 fp32/lane         (output acc, persistent)
//   v_s     : 8 fp32/lane                                 (current S tile)
//   m_row,
//   l_row,
//   rescale : 1 fp32/lane each                            (online softmax state)
template<int BLOCK_M_ = 16, int BLOCK_N_ = 16, int D_ = 128>
struct opus_attn_traits {
    static constexpr int BLOCK_M    = BLOCK_M_;
    static constexpr int BLOCK_N    = BLOCK_N_;
    static constexpr int D          = D_;
    static constexpr int W_M        = 16;   // wmma M
    static constexpr int W_N        = 16;   // wmma N
    static constexpr int W_K        = 16;   // wmma K
    static constexpr int WARP_SIZE  = 32;
    static constexpr int BLOCK_SIZE = WARP_SIZE;   // 1 wave per workgroup in v0
    static constexpr int D_TILES_K  = D / W_K;     // 8

    static_assert(BLOCK_M == W_M, "v0: BLOCK_M must equal wmma M tile (16)");
    static_assert(BLOCK_N == W_N, "v0: BLOCK_N must equal wmma N tile (16)");
    static_assert(D % W_K == 0,   "D must be a multiple of wmma K (16)");
};

__host__ __device__ inline int ceil_div(int a, int b) { return (a + b - 1) / b; }
