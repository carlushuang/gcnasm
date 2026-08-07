// SPDX-License-Identifier: MIT
// opus_attn_gfx1201 — shared types between device kernel and host driver.
// bf16 variant: in/out tensors are bf16, accumulators are fp32.
#pragma once

// bf16 stored as raw 16-bit bits. The wmma builtin signature takes
// `short __attribute__((ext_vector_type(8)))`, so we use `short` here too.
using bf16_t = short;
using fp32_t = float;

// fp32 ↔ bf16 conversions (host + device). RNE rounding.
__host__ __device__ inline bf16_t bf16_from_f32(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    // Preserve NaNs as quiet NaN
    if (((x >> 23) & 0xFF) == 0xFF && (x & 0x7FFFFF)) {
        return (bf16_t)((x >> 16) | 0x40);
    }
    unsigned int rb = 0x7FFF + ((x >> 16) & 1);
    return (bf16_t)((x + rb) >> 16);
}

__host__ __device__ inline fp32_t bf16_to_f32(bf16_t b) {
    // Zero-extend the 16 bits into the high half of a 32-bit fp32 layout.
    unsigned int u = (unsigned int)(unsigned short)b;
    return __builtin_bit_cast(fp32_t, u << 16);
}

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
