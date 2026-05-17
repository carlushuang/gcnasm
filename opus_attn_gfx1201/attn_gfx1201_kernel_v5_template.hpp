// SPDX-License-Identifier: MIT
// opus_attn_gfx1201 v5 — combine the lessons learned from v1-v4:
//   * single wave per WG (v0/v4 showed this beats multi-wave; high
//     occupancy + L1 cache absorbs K/V re-reads anyway)
//   * BLOCK_N = 32 — TWO wmma N-tiles per outer iter, halving softmax
//     overhead vs v0's BLOCK_N=16
//   * process the 2 sub-tiles serially through online softmax (one v_s
//     in flight at a time, like the FA pattern in v2 — keeps register
//     pressure manageable)
//   * NO manual K/V prefetch (v4 showed it regresses; compiler schedules
//     better than hand-rolled buffers do)
//   * NO multi-wave smem cooperation (v1-v3 showed the L1 cache makes
//     this irrelevant on RDNA4)
//
//   workgroup = 1 wave × 32 lanes
//   BLOCK_M = 16, BLOCK_N = 32, D = 128
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline fp32_t v5_fmaxf(fp32_t a, fp32_t b) { return a > b ? a : b; }

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v5(opus_attn_kargs k)
{
#if defined(__gfx1201__) || defined(__gfx1200__)
    constexpr int BLOCK_M  = T::BLOCK_M;     // 16
    constexpr int BLOCK_N  = T::BLOCK_N;     // 32
    constexpr int D        = T::D;
    constexpr int W_K      = T::W_K;
    constexpr int DK       = T::D_TILES_K;   // 8
    constexpr int NT       = T::N_TILES;     // BLOCK_N / 16 = 2 sub-tiles
    constexpr int SUB_N    = 16;

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
    bf16_t*       __restrict__ Op = reinterpret_cast<bf16_t*>      (k.ptr_o) + b * stride_b + h * stride_h;

    __shared__ bf16_t s_p[16 * 16];

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

    fp32x8_t v_o[DK];
    #pragma unroll
    for (int kt = 0; kt < DK; ++kt) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_o[kt][j] = 0.0f;
    }
    fp32_t m_row[8], l_row[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) { m_row[j] = -3.4e38f; l_row[j] = 0.0f; }

    const int num_outer_tiles = k.N / BLOCK_N;

    for (int outer = 0; outer < num_outer_tiles; ++outer) {
        const int n_outer_base = outer * BLOCK_N;

        // Process NT sub-tiles serially through the online-softmax pattern
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int n_base = n_outer_base + nt * SUB_N;

            // 1) S = Q @ K^T (direct global reads — L1 cache absorbs)
            fp32x8_t v_s = {0,0,0,0,0,0,0,0};
            #pragma unroll
            for (int kt = 0; kt < DK; ++kt) {
                bf16x8_t v_k;
                const bf16_t* k_row = Kp + (n_base + col16) * stride_n + kt * W_K;
                #pragma unroll
                for (int j = 0; j < 8; ++j) v_k[j] = k_row[row8 + j];
                v_s = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_q[kt], v_k, v_s);
            }

            // 2) Fused softmax (max → rescale O → exp2 → sum → l update)
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                fp32_t v = v_s[j];
                v = v5_fmaxf(v, __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x041F)));
                v = v5_fmaxf(v, __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x081F)));
                v = v5_fmaxf(v, __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x101F)));
                v = v5_fmaxf(v, __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x201F)));
                const fp32_t new_m = v5_fmaxf(m_row[j], v);
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

            // 3) Flip S → P via smem (single wave, no inter-wave sync)
            #pragma unroll
            for (int j = 0; j < 8; ++j) s_p[(row8 + j) * 16 + col16] = bf16_from_f32(v_s[j]);
            __builtin_amdgcn_wave_barrier();
            bf16x8_t v_p;
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_p[j] = s_p[col16 * 16 + row8 + j];

            // 4) O += P @ V (direct global reads — L1 absorbs)
            #pragma unroll
            for (int dt = 0; dt < DK; ++dt) {
                bf16x8_t v_v;
                const bf16_t* v_col = Vp + (n_base + row8) * stride_n + dt * W_K + col16;
                #pragma unroll
                for (int j = 0; j < 8; ++j) v_v[j] = v_col[j * stride_n];
                v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_p, v_v, v_o[dt]);
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const fp32_t inv = (l_row[j] > 0.0f) ? (1.0f / l_row[j]) : 0.0f;
        const int q_m_base = q_tile_id * BLOCK_M;
        #pragma unroll
        for (int kt = 0; kt < DK; ++kt) {
            const int d_base = kt * W_K + col16;
            Op[(q_m_base + row8 + j) * stride_n + d_base] = bf16_from_f32(v_o[kt][j] * inv);
        }
    }
#else
    (void)k;
#endif
}
