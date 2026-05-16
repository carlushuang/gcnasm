// SPDX-License-Identifier: MIT
// opus_attn_gfx1201 v4 — single-wave (v0) geometry + register-buffered K/V
// prefetch. Each KV iter pre-issues the global loads for iter+1's K and V
// while iter's wmma is in flight; lets the hardware overlap memory and
// compute without the multi-wave smem cooperation that didn't pay off in
// v1-v3.
//
//   workgroup = 1 wave × 32 lanes
//   BLOCK_M = BLOCK_N = 16, D = 128
//
// Per-lane register state for K/V prefetch:
//   v_k_cur[8] (8 fp16/lane per kt — total 64 fp16/lane for full D)
//   v_k_nxt[8] (same)  — total 64 fp16/lane
//   v_v_cur[8] (same) + v_v_nxt[8] (same)
//   Total prefetch ~128 fp16/lane = 64 fp32-eq VGPRs added on top of v0.
//   v0 used 165 VGPRs; adding 64 → ~230 VGPRs. Should still fit at occ 1+.
#include <hip/hip_runtime.h>
#include "attn_common.h"

using fp16x8_t = fp16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline fp32_t v4_fmaxf(fp32_t a, fp32_t b) { return a > b ? a : b; }

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v4(opus_attn_kargs k)
{
#if defined(__gfx1201__) || defined(__gfx1200__)
    constexpr int BLOCK_M = T::BLOCK_M;
    constexpr int BLOCK_N = T::BLOCK_N;
    constexpr int D       = T::D;
    constexpr int W_K     = T::W_K;
    constexpr int DK      = T::D_TILES_K;

    const int lane     = static_cast<int>(threadIdx.x);
    const int col16    = lane % 16;
    const int row_grp  = lane / 16;
    const int row8     = row_grp * 8;

    const int q_tile_id = blockIdx.x;
    const int h         = blockIdx.y;
    const int b         = blockIdx.z;

    const int stride_n = k.D;
    const int stride_h = k.N * k.D;
    const int stride_b = k.H * k.N * k.D;

    const fp16_t* __restrict__ Qp = reinterpret_cast<const fp16_t*>(k.ptr_q) + b * stride_b + h * stride_h;
    const fp16_t* __restrict__ Kp = reinterpret_cast<const fp16_t*>(k.ptr_k) + b * stride_b + h * stride_h;
    const fp16_t* __restrict__ Vp = reinterpret_cast<const fp16_t*>(k.ptr_v) + b * stride_b + h * stride_h;
    fp16_t*       __restrict__ Op = reinterpret_cast<fp16_t*>      (k.ptr_o) + b * stride_b + h * stride_h;

    __shared__ fp16_t s_p[16 * 16];

    // ── Q load (persistent) ─────────────────────────────────────────────
    fp16x8_t v_q[DK];
    {
        const int q_m_base = q_tile_id * BLOCK_M;
        const fp16_t* q_row = Qp + (q_m_base + col16) * stride_n;
        #pragma unroll
        for (int kt = 0; kt < DK; ++kt) {
            const int k_off = kt * W_K + row8;
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_q[kt][j] = q_row[k_off + j];
        }
    }
    constexpr fp32_t LOG2_E = 1.44269504088896340736f;
    const fp32_t qscale = k.scale * LOG2_E;
    #pragma unroll
    for (int kt = 0; kt < DK; ++kt) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_q[kt][j] = static_cast<fp16_t>(static_cast<fp32_t>(v_q[kt][j]) * qscale);
    }

    fp32x8_t v_o[DK];
    #pragma unroll
    for (int kt = 0; kt < DK; ++kt) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_o[kt][j] = 0.0f;
    }
    fp32_t m_row[8], l_row[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) { m_row[j] = -3.4e38f; l_row[j] = 0.0f; }

    const int num_kv_tiles = k.N / BLOCK_N;

    // Prefetch helpers: load lane's K and V fragment for a given n_base.
    // Layout: lane[i].K_kt[j] = K[n=col16, k=row8+j+kt*16] = Kp[(n_base+col16)*stride_n + kt*16 + row8 + j]
    auto load_k_tile = [&](int n_base, fp16x8_t (&v_k_buf)[DK]) {
        const fp16_t* k_row = Kp + (n_base + col16) * stride_n;
        #pragma unroll
        for (int kt = 0; kt < DK; ++kt) {
            const int k_off = kt * W_K + row8;
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_k_buf[kt][j] = k_row[k_off + j];
        }
    };
    auto load_v_tile = [&](int n_base, fp16x8_t (&v_v_buf)[DK]) {
        #pragma unroll
        for (int dt = 0; dt < DK; ++dt) {
            const fp16_t* v_col = Vp + (n_base + row8) * stride_n + dt * W_K + col16;
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_v_buf[dt][j] = v_col[j * stride_n];
        }
    };

    // Double-buffered K/V prefetch (two register sets)
    fp16x8_t v_k_buf[2][DK];
    fp16x8_t v_v_buf[2][DK];
    int cur = 0;

    // Prologue: load tile 0
    load_k_tile(0, v_k_buf[cur]);
    load_v_tile(0, v_v_buf[cur]);

    for (int n_tile = 0; n_tile < num_kv_tiles; ++n_tile) {
        const int nxt = 1 - cur;
        const int n_next = (n_tile + 1) * BLOCK_N;

        // Issue prefetch for next iter (hardware overlaps with mma below)
        if (n_tile + 1 < num_kv_tiles) {
            load_k_tile(n_next, v_k_buf[nxt]);
        }

        // 1) S = Q @ K^T
        fp32x8_t v_s = {0,0,0,0,0,0,0,0};
        #pragma unroll
        for (int kt = 0; kt < DK; ++kt) {
            v_s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v_q[kt], v_k_buf[cur][kt], v_s);
        }

        if (n_tile + 1 < num_kv_tiles) {
            load_v_tile(n_next, v_v_buf[nxt]);
        }

        // 2) Fused softmax
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            fp32_t v = v_s[j];
            v = v4_fmaxf(v, __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x041F)));
            v = v4_fmaxf(v, __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x081F)));
            v = v4_fmaxf(v, __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x101F)));
            v = v4_fmaxf(v, __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x201F)));
            const fp32_t new_m = v4_fmaxf(m_row[j], v);
            const fp32_t scale = __builtin_amdgcn_exp2f(m_row[j] - new_m);
            m_row[j] = new_m;
            v_s[j]   = __builtin_amdgcn_exp2f(v_s[j] - new_m);
            #pragma unroll
            for (int kt = 0; kt < DK; ++kt) v_o[kt][j] *= scale;
            fp32_t s = v_s[j];
            s += __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, s), 0x041F));
            s += __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, s), 0x081F));
            s += __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, s), 0x101F));
            s += __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, s), 0x201F));
            l_row[j] = l_row[j] * scale + s;
        }

        // Flip S → P via smem (single-wave: no cross-wave sync)
        #pragma unroll
        for (int j = 0; j < 8; ++j) s_p[(row8 + j) * 16 + col16] = static_cast<fp16_t>(v_s[j]);
        __builtin_amdgcn_wave_barrier();
        fp16x8_t v_p;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p[j] = s_p[col16 * 16 + row8 + j];

        // 3) O += P @ V
        #pragma unroll
        for (int dt = 0; dt < DK; ++dt) {
            v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v_p, v_v_buf[cur][dt], v_o[dt]);
        }

        cur = nxt;
    }

    // Normalize + write
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const fp32_t inv = (l_row[j] > 0.0f) ? (1.0f / l_row[j]) : 0.0f;
        const int q_m_base = q_tile_id * BLOCK_M;
        #pragma unroll
        for (int kt = 0; kt < DK; ++kt) {
            const int d_base = kt * W_K + col16;
            Op[(q_m_base + row8 + j) * stride_n + d_base] = static_cast<fp16_t>(v_o[kt][j] * inv);
        }
    }
#else
    (void)k;
#endif
}
