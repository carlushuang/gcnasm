// SPDX-License-Identifier: MIT
// opus_attn_gfx1201 v11 — eliminate S→P smem flip via mma swap_ab pattern.
//
// gfx950's opus_attn does S→P with a simple cast (no smem) because MFMA
// C-out and A-in layouts align under mfma_adaptor_swap_ab. On gfx12 wmma
// the layouts are asymmetric (C.lane=N, A.lane=M, with M/N axes flipped
// in the register dim), so a direct cast gives P^T not P.
//
// v11 sidesteps the smem flip by calling both mmas with swapped operand
// order — wmma(K, Q) and wmma(V, P) — and computing O^T = V^T @ P^T:
//
//   mma0_swap: wmma(v_k, v_q, 0)   → S in "swap" C layout
//                                     lane(c,r) reg j → S[M_q=c, N_kv=r*8+j]
//   softmax in this layout         (8-value per-lane reduce + 1 cross-half)
//   mma1_swap: wmma(v_v, v_p, v_o) → O^T in C layout
//                                     lane(c,r) reg j → O[M_q=c, D=r*8+j]
//   write O contiguously           (1 b128 per lane per D-tile, vs 8 small
//                                     strided writes in v0/v6)
//
// Tradeoff vs v6:
//   - LOSES: V load is strided again (back to 80+ small loads/iter vs v6's
//     8 b128). The B-fragment access pattern only vectorizes when V is in
//     [B,H,D,N] layout (v9), which is not production-realistic.
//   - WINS: no S→P smem flip (saves 1 barrier + ~9 ds-ops/iter), vectorized
//     output write (saves ~56 small writes total), and softmax row-reduce
//     drops from 4 ds_swizzle stages × 8j to 1 ds_bpermute per row.
//
// Net: usually loses to v6 on the V-load-bound common case, but a useful
// reference for "the smem flip CAN be eliminated on gfx12, here's the cost".
//
//   workgroup = 1 wave × 32 lanes, BLOCK_M=16, BLOCK_N=16, D=128
#include <hip/hip_runtime.h>
#include "attn_common.h"

using fp16x8_t = fp16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline fp32_t v11_fmaxf(fp32_t a, fp32_t b) { return a > b ? a : b; }

__device__ static inline fp32_t v11_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v11_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v11_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v11(opus_attn_kargs k)
{
#if defined(__gfx1201__) || defined(__gfx1200__)
    constexpr int BLOCK_M = T::BLOCK_M;
    constexpr int BLOCK_N = T::BLOCK_N;
    constexpr int W_K     = T::W_K;
    constexpr int DK      = T::D_TILES_K;

    const int lane    = static_cast<int>(threadIdx.x);
    const int col16   = lane % 16;
    const int row_grp = lane / 16;
    const int row8    = row_grp * 8;

    const int q_tile_id = blockIdx.x;
    const int h         = blockIdx.y;
    const int b         = blockIdx.z;

    const int stride_n = k.D;
    const int stride_h = k.N * k.D;
    const int stride_b = k.H * k.N * k.D;

    const fp16_t* __restrict__ Qp = reinterpret_cast<const fp16_t*>(k.ptr_q) + b * stride_b + h * stride_h;
    const fp16_t* __restrict__ Kp = reinterpret_cast<const fp16_t*>(k.ptr_k) + b * stride_b + h * stride_h;
    const fp16_t* __restrict__ Vp = reinterpret_cast<const fp16_t*>(k.ptr_v) + b * stride_b + h * stride_h;
    fp16_t*       __restrict__ Op = reinterpret_cast<fp16_t*>(k.ptr_o) + b * stride_b + h * stride_h;

    // ── Load Q (same pattern as v0) ────────────────────────────────────────
    // lane (col16, row_grp) reg j → Q[q_m_base + col16, kt*16 + row8 + j]
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

    // ── Output accumulator in v11 swap layout ──────────────────────────────
    // lane (col16=M_q, row_grp) reg j → O[col16, dt*16 + row8 + j]
    fp32x8_t v_o[DK];
    #pragma unroll
    for (int kt = 0; kt < DK; ++kt) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_o[kt][j] = 0.0f;
    }
    // m_row / l_row are now PER-LANE (one per M_q row = col16). Same value
    // in lane(c, r=0) and lane(c, r=1).
    fp32_t m_row = -3.4e38f;
    fp32_t l_row = 0.0f;

    const int num_kv_tiles = k.N / BLOCK_N;

    for (int n_tile = 0; n_tile < num_kv_tiles; ++n_tile) {
        const int n_base = n_tile * BLOCK_N;

        // ── mma0 SWAP: S = wmma(v_k, v_q, 0) ──────────────────────────────
        // K loaded same as v0; passing it as A operand makes the result
        // lane(c, r) reg j → S[M_q=c, N_kv=r*8+j].
        fp32x8_t v_s = {0,0,0,0,0,0,0,0};
        #pragma unroll
        for (int kt = 0; kt < DK; ++kt) {
            fp16x8_t v_k;
            const fp16_t* k_row = Kp + (n_base + col16) * stride_n + kt * W_K;
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_k[j] = k_row[row8 + j];
            v_s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v_k, v_q[kt], v_s);  // SWAPPED
        }

        // ── Online softmax in v11 layout ──────────────────────────────────
        // Per lane: row max over 8 j values, then cross-half via ds_bpermute.
        fp32_t row_max = v_s[0];
        #pragma unroll
        for (int j = 1; j < 8; ++j) row_max = v11_fmaxf(row_max, v_s[j]);
        row_max = v11_cross_half_max(row_max);

        const fp32_t new_m = v11_fmaxf(m_row, row_max);
        const fp32_t scale = __builtin_amdgcn_exp2f(m_row - new_m);
        m_row = new_m;

        // Rescale v_o by per-row scale (same for both halves of this lane's c)
        #pragma unroll
        for (int dt = 0; dt < DK; ++dt) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_o[dt][j] *= scale;
        }

        // exp2(s - m_new), sum row
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_s[j] = __builtin_amdgcn_exp2f(v_s[j] - new_m);
        fp32_t row_sum = v_s[0];
        #pragma unroll
        for (int j = 1; j < 8; ++j) row_sum += v_s[j];
        row_sum = v11_cross_half_sum(row_sum);
        l_row = l_row * scale + row_sum;

        // ── mma1 SWAP: O^T += wmma(v_v, v_p, v_o) ─────────────────────────
        // v_v loaded strided (giving B-layout of V); fed as A makes it V^T.
        // v_p in C layout from mma0_swap; fed as B is the right form.
        // Result lane (c, r) reg j → O[col16, dt*16 + r*8 + j].
        #pragma unroll
        for (int dt = 0; dt < DK; ++dt) {
            fp16x8_t v_v;
            const int d_base = dt * W_K + col16;
            const fp16_t* v_col = Vp + (n_base + row8) * stride_n + d_base;
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_v[j] = v_col[j * stride_n];
            // v_p is just v_s after exp; cast fp32 → fp16
            fp16x8_t v_p;
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_p[j] = static_cast<fp16_t>(v_s[j]);
            v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v_v, v_p, v_o[dt]);  // SWAPPED
        }
    }

    // ── Normalize and write back (vectorized) ──────────────────────────────
    // lane (col16, row_grp) reg j → O[col16, dt*16 + row_grp*8 + j].
    // Per lane writes 8 consecutive fp16 starting at O[col16, dt*16+row8].
    const fp32_t inv = (l_row > 0.0f) ? (1.0f / l_row) : 0.0f;
    const int q_m_base = q_tile_id * BLOCK_M;
    fp16_t* o_row = Op + (q_m_base + col16) * stride_n;
    #pragma unroll
    for (int dt = 0; dt < DK; ++dt) {
        const int d_off = dt * W_K + row8;
        #pragma unroll
        for (int j = 0; j < 8; ++j) o_row[d_off + j] = static_cast<fp16_t>(v_o[dt][j] * inv);
    }
#else
    (void)k;
#endif
}
