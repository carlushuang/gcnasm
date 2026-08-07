// SPDX-License-Identifier: MIT
// opus_attn_gfx1201 v10 — v9 (pre-transposed V) + BLOCK_N=32 (2 sub-tiles).
//
// Combines:
//   * V_T in DRAM ([B,H,D,N]) — vectorized B-layout load, no smem flip for V
//   * BLOCK_N=32 — two 16-wide sub-tiles per outer iter through online softmax
//
//   workgroup = 1 wave × 32 lanes, BLOCK_M=16, BLOCK_N=32, D=128
//   k.ptr_v MUST point to V_T = [B, H, D, N], not the row-major V.
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline fp32_t v10_fmaxf(fp32_t a, fp32_t b) { return a > b ? a : b; }

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v10(opus_attn_kargs k)
{
#if defined(__gfx1201__) || defined(__gfx1200__)
    constexpr int BLOCK_M = T::BLOCK_M;
    constexpr int BLOCK_N = T::BLOCK_N;
    constexpr int W_K     = T::W_K;
    constexpr int DK      = T::D_TILES_K;
    constexpr int NT      = T::N_TILES;
    constexpr int SUB_N   = 16;

    const int lane    = static_cast<int>(threadIdx.x);
    const int col16   = lane % 16;
    const int row_grp = lane / 16;
    const int row8    = row_grp * 8;

    const int q_tile_id = blockIdx.x;
    const int h         = blockIdx.y;
    const int b         = blockIdx.z;

    const int stride_n  = k.D;
    const int stride_h  = k.N * k.D;
    const int stride_b  = k.H * k.N * k.D;
    const int vt_stride_d = k.N;
    const int vt_stride_h = k.D * k.N;
    const int vt_stride_b = k.H * k.D * k.N;

    const bf16_t* __restrict__ Qp  = reinterpret_cast<const bf16_t*>(k.ptr_q) + b * stride_b + h * stride_h;
    const bf16_t* __restrict__ Kp  = reinterpret_cast<const bf16_t*>(k.ptr_k) + b * stride_b + h * stride_h;
    const bf16_t* __restrict__ VTp = reinterpret_cast<const bf16_t*>(k.ptr_v) + b * vt_stride_b + h * vt_stride_h;
    bf16_t*       __restrict__ Op  = reinterpret_cast<bf16_t*>(k.ptr_o) + b * stride_b + h * stride_h;

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

        #pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int n_base = n_outer_base + nt * SUB_N;

            fp32x8_t v_s = {0,0,0,0,0,0,0,0};
            #pragma unroll
            for (int kt = 0; kt < DK; ++kt) {
                bf16x8_t v_k;
                const bf16_t* k_row = Kp + (n_base + col16) * stride_n + kt * W_K;
                #pragma unroll
                for (int j = 0; j < 8; ++j) v_k[j] = k_row[row8 + j];
                v_s = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_q[kt], v_k, v_s);
            }

            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                fp32_t v = v_s[j];
                v = v10_fmaxf(v, __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x041F)));
                v = v10_fmaxf(v, __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x081F)));
                v = v10_fmaxf(v, __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x101F)));
                v = v10_fmaxf(v, __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x201F)));
                const fp32_t new_m = v10_fmaxf(m_row[j], v);
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

            #pragma unroll
            for (int j = 0; j < 8; ++j) s_p[(row8 + j) * 16 + col16] = bf16_from_f32(v_s[j]);
            __builtin_amdgcn_wave_barrier();
            bf16x8_t v_p;
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_p[j] = s_p[col16 * 16 + row8 + j];

            #pragma unroll
            for (int dt = 0; dt < DK; ++dt) {
                bf16x8_t v_v;
                const bf16_t* vt_row = VTp + (dt * W_K + col16) * vt_stride_d + n_base + row8;
                #pragma unroll
                for (int j = 0; j < 8; ++j) v_v[j] = vt_row[j];

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
