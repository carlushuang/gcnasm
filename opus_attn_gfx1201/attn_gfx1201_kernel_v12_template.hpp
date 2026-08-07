// SPDX-License-Identifier: MIT
// opus_attn_gfx1201 v12 — v11 swap_ab + v6 contiguous V load + vectorized O write.
//
// Combines v11's smem-free S->P pipeline (swap_ab, lean softmax) with v6's
// contiguous V load (1 global_load_b128 per lane per D-tile instead of 8
// global_load_u16). The V tile is transposed in smem into A-fragment layout
// (not B-fragment like v6) to match the swap_ab mma1 pattern.
//
// Also fixes the output write: v11 claimed vectorized stores but the compiler
// emitted 65 global_store_b16. v12 explicitly packs bf16x8 and stores via
// pointer cast to get global_store_b128.
//
//   workgroup = 1 wave x 32 lanes, BLOCK_M=16, BLOCK_N=16, D=128
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline fp32_t v12_fmaxf(fp32_t a, fp32_t b) { return a > b ? a : b; }

__device__ static inline fp32_t v12_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v12_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v12_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v12(opus_attn_kargs k)
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

    const bf16_t* __restrict__ Qp = reinterpret_cast<const bf16_t*>(k.ptr_q) + b * stride_b + h * stride_h;
    const bf16_t* __restrict__ Kp = reinterpret_cast<const bf16_t*>(k.ptr_k) + b * stride_b + h * stride_h;
    const bf16_t* __restrict__ Vp = reinterpret_cast<const bf16_t*>(k.ptr_v) + b * stride_b + h * stride_h;
    bf16_t*       __restrict__ Op = reinterpret_cast<bf16_t*>(k.ptr_o) + b * stride_b + h * stride_h;

    // smem for V transpose only (S->P flip eliminated by swap_ab)
    __shared__ bf16_t s_v[16 * 16];

    // Load Q (same as v11)
    bf16x8_t v_q[DK];
    {
        const int q_m_base = q_tile_id * BLOCK_M;
        const bf16_t* q_row = Qp + (q_m_base + col16) * stride_n;
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
        for (int j = 0; j < 8; ++j) v_q[kt][j] = bf16_from_f32(bf16_to_f32(v_q[kt][j]) * qscale);
    }

    // Output accumulator in v11 swap layout:
    // lane (col16=M_q, row_grp) reg j -> O[col16, dt*16 + row8 + j]
    fp32x8_t v_o[DK];
    #pragma unroll
    for (int kt = 0; kt < DK; ++kt) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_o[kt][j] = 0.0f;
    }
    fp32_t m_row = -3.4e38f;
    fp32_t l_row = 0.0f;

    const int num_kv_tiles = k.N / BLOCK_N;

    for (int n_tile = 0; n_tile < num_kv_tiles; ++n_tile) {
        const int n_base = n_tile * BLOCK_N;

        // mma0 SWAP: S = wmma(v_k, v_q, 0) — same as v11
        fp32x8_t v_s = {0,0,0,0,0,0,0,0};
        #pragma unroll
        for (int kt = 0; kt < DK; ++kt) {
            bf16x8_t v_k;
            const bf16_t* k_row = Kp + (n_base + col16) * stride_n + kt * W_K;
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_k[j] = k_row[row8 + j];
            v_s = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_k, v_q[kt], v_s);
        }

        // Online softmax in v11 layout — same as v11
        fp32_t row_max = v_s[0];
        #pragma unroll
        for (int j = 1; j < 8; ++j) row_max = v12_fmaxf(row_max, v_s[j]);
        row_max = v12_cross_half_max(row_max);

        const fp32_t new_m = v12_fmaxf(m_row, row_max);
        const fp32_t scale = __builtin_amdgcn_exp2f(m_row - new_m);
        m_row = new_m;

        #pragma unroll
        for (int dt = 0; dt < DK; ++dt) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_o[dt][j] *= scale;
        }

        #pragma unroll
        for (int j = 0; j < 8; ++j) v_s[j] = __builtin_amdgcn_exp2f(v_s[j] - new_m);
        fp32_t row_sum = v_s[0];
        #pragma unroll
        for (int j = 1; j < 8; ++j) row_sum += v_s[j];
        row_sum = v12_cross_half_sum(row_sum);
        l_row = l_row * scale + row_sum;

        // Cast v_s -> v_p (bf16) once, outside dt loop — same as v11
        bf16x8_t v_p;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p[j] = bf16_from_f32(v_s[j]);

        // mma1 SWAP with CONTIGUOUS V load + smem transpose
        // v11 loads V strided (8 global_load_u16 per D-tile per lane).
        // v12 loads V contiguously along D, transposes in smem to A-fragment layout.
        //
        // A-fragment for swap_ab mma1: wmma(v_v, v_p, v_o)
        //   v_v as A: lane(col16, row_grp) reg j -> V[n_base + row_grp*8+j, dt*16 + col16]
        //
        // Contiguous load: lane(col16, row_grp) loads V[n_base + col16, dt*16 + row8..row8+7]
        //   -> smem[col16][row8+j] = V[n_base + col16, dt*16 + row_grp*8+j]
        //
        // Transposed read into A-fragment:
        //   v_v[j] = smem[row_grp*8+j][col16] = V[n_base + row_grp*8+j, dt*16 + col16]
        #pragma unroll
        for (int dt = 0; dt < DK; ++dt) {
            bf16x8_t v_load;
            const bf16_t* v_row = Vp + (n_base + col16) * stride_n + dt * W_K + row8;
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_load[j] = v_row[j];

            #pragma unroll
            for (int j = 0; j < 8; ++j) s_v[col16 * 16 + row8 + j] = v_load[j];
            __builtin_amdgcn_wave_barrier();

            bf16x8_t v_v;
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_v[j] = s_v[(row8 + j) * 16 + col16];

            v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_v, v_p, v_o[dt]);
        }
    }

    // Normalize and write back — VECTORIZED
    // v11 layout: lane (col16, row_grp) reg j -> O[col16, dt*16 + row8 + j]
    // Per lane: 8 contiguous bf16 starting at O[col16, dt*16 + row8]
    const fp32_t inv = (l_row > 0.0f) ? (1.0f / l_row) : 0.0f;
    const int q_m_base = q_tile_id * BLOCK_M;
    bf16_t* o_row = Op + (q_m_base + col16) * stride_n;
    #pragma unroll
    for (int dt = 0; dt < DK; ++dt) {
        bf16x8_t o_pack;
        #pragma unroll
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_from_f32(v_o[dt][j] * inv);
        const int d_off = dt * W_K + row8;
        *reinterpret_cast<bf16x8_t*>(&o_row[d_off]) = o_pack;
    }
#else
    (void)k;
#endif
}
