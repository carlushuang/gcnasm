// SPDX-License-Identifier: MIT
// opus_attn_gfx1201 v22 — 8-wave, D-chunked (4+4), Q in regs per chunk
//
// 8 waves, BLOCK_M=128, BLOCK_N=16.
// 2 passes: each handles 4 D-tiles of Q and O.
// Q is kept in registers (pre-scaled) for the active 4 D-tiles.
// S = QK^T uses only the active 4 D-tiles (partial dot product).
// After pass 1, store partial O,m,l. After pass 2, merge and normalize.
//
// This means EACH PASS computes a partial attention with D/2 head dim.
// The two partials must be merged using the log-sum-exp trick.
//
// VGPRs: v_q(16) + v_o(32) + v_s(8) + temps(~15) ≈ 71 → 3 waves/SIMD
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast22(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v22_fmaxf(fp32_t a, fp32_t b) { return a > b ? a : b; }

__device__ static inline fp32_t v22_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v22_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v22_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v22(opus_attn_kargs k)
{
#if defined(__gfx1201__) || defined(__gfx1200__)
    constexpr int BLOCK_M = T::BLOCK_M;   // 128
    constexpr int BLOCK_N = T::BLOCK_N;   // 16
    constexpr int W_K     = T::W_K;       // 16
    constexpr int DK      = T::D_TILES_K; // 8
    constexpr int DK_HALF = DK / 2;       // 4

    const int tid     = static_cast<int>(threadIdx.x);
    const int wave_id = tid / 32;
    const int lane    = tid % 32;
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

    __shared__ bf16_t s_v[8][16 * 16];  // V transpose scratch, 4 KB

    const int q_m_base = q_tile_id * BLOCK_M + wave_id * 16;
    const bf16_t* q_row_base = Qp + (q_m_base + col16) * stride_n;
    bf16_t* o_row = Op + (q_m_base + col16) * stride_n;

    constexpr fp32_t LOG2_E = 1.44269504088896340736f;
    const fp32_t qscale = k.scale * LOG2_E;
    const int num_kv_tiles = k.N / BLOCK_N;

    // Pass 1 results stored temporarily in global O and in registers
    fp32_t m1_final, l1_final;

    for (int d_pass = 0; d_pass < 2; ++d_pass) {
        const int dt_base = d_pass * DK_HALF;

        // Load Q for this D-chunk (pre-scaled, 16 VGPRs)
        bf16x8_t v_q[DK_HALF];
        #pragma unroll
        for (int kt = 0; kt < DK_HALF; ++kt) {
            v_q[kt] = *reinterpret_cast<const bf16x8_t*>(
                &q_row_base[(dt_base + kt) * W_K + row8]);
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                v_q[kt][j] = bf16_fast22(bf16_to_f32(v_q[kt][j]) * qscale);
        }

        // O accumulator (32 VGPRs)
        fp32x8_t v_o[DK_HALF];
        #pragma unroll
        for (int dt = 0; dt < DK_HALF; ++dt) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_o[dt][j] = 0.0f;
        }
        fp32_t m_row = -3.4e38f;
        fp32_t l_row = 0.0f;

        for (int n_tile = 0; n_tile < num_kv_tiles; ++n_tile) {
            const int n_base = n_tile * BLOCK_N;

            // S = Q_chunk × K_chunk^T (partial dot product over D/2)
            fp32x8_t v_s = {0,0,0,0,0,0,0,0};
            #pragma unroll
            for (int kt = 0; kt < DK_HALF; ++kt) {
                const bf16_t* k_addr = Kp + (n_base + col16) * stride_n
                    + (dt_base + kt) * W_K + row8;
                bf16x8_t v_k = *reinterpret_cast<const bf16x8_t*>(k_addr);
                v_s = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                    v_k, v_q[kt], v_s);
            }

            // Online softmax (on partial S)
            fp32_t row_max = v_s[0];
            #pragma unroll
            for (int j = 1; j < 8; ++j) row_max = v22_fmaxf(row_max, v_s[j]);
            row_max = v22_cross_half_max(row_max);

            const fp32_t new_m = v22_fmaxf(m_row, row_max);
            const fp32_t rescale = __builtin_amdgcn_exp2f(m_row - new_m);
            m_row = new_m;

            #pragma unroll
            for (int dt = 0; dt < DK_HALF; ++dt) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) v_o[dt][j] *= rescale;
            }

            #pragma unroll
            for (int j = 0; j < 8; ++j) v_s[j] = __builtin_amdgcn_exp2f(v_s[j] - new_m);
            fp32_t row_sum = v_s[0];
            #pragma unroll
            for (int j = 1; j < 8; ++j) row_sum += v_s[j];
            row_sum = v22_cross_half_sum(row_sum);
            l_row = l_row * rescale + row_sum;

            // P × V_chunk
            bf16x8_t v_p;
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_p[j] = bf16_fast22(v_s[j]);

            #pragma unroll
            for (int dt = 0; dt < DK_HALF; ++dt) {
                const bf16_t* v_addr = Vp + (n_base + col16) * stride_n
                    + (dt_base + dt) * W_K + row8;
                bf16x8_t v_load = *reinterpret_cast<const bf16x8_t*>(v_addr);

                #pragma unroll
                for (int j = 0; j < 8; ++j)
                    s_v[wave_id][(row8 + j) * 16 + col16] = v_load[j];
                __builtin_amdgcn_wave_barrier();

                bf16x8_t v_v = *reinterpret_cast<const bf16x8_t*>(
                    &s_v[wave_id][col16 * 16 + row8]);

                v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                    v_v, v_p, v_o[dt]);
            }
        }

        if (d_pass == 0) {
            // Store pass 1 results: unnormalized O and m,l
            m1_final = m_row;
            l1_final = l_row;
            // Write unnormalized O1 to global (will be corrected later)
            #pragma unroll
            for (int dt = 0; dt < DK_HALF; ++dt) {
                bf16x8_t o_pack;
                #pragma unroll
                for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast22(v_o[dt][j]);
                *reinterpret_cast<bf16x8_t*>(&o_row[(dt_base + dt) * W_K + row8]) = o_pack;
            }
        } else {
            // Pass 2: merge with pass 1 results
            // O_final = (l1*exp(m1-m_max)*O1 + l2*exp(m2-m_max)*O2) / (l1*exp(m1-m_max) + l2*exp(m2-m_max))
            const fp32_t m_max = v22_fmaxf(m1_final, m_row);
            const fp32_t scale1 = l1_final * __builtin_amdgcn_exp2f(m1_final - m_max);
            const fp32_t scale2 = l_row * __builtin_amdgcn_exp2f(m_row - m_max);
            const fp32_t l_total = scale1 + scale2;
            const fp32_t inv = (l_total > 0.0f) ? (1.0f / l_total) : 0.0f;

            // Write normalized pass 2 O
            #pragma unroll
            for (int dt = 0; dt < DK_HALF; ++dt) {
                bf16x8_t o_pack;
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                    o_pack[j] = bf16_fast22(v_o[dt][j] * scale2 * inv);
                *reinterpret_cast<bf16x8_t*>(&o_row[(dt_base + dt) * W_K + row8]) = o_pack;
            }

            // Fix pass 1 O: reload, rescale, rewrite
            #pragma unroll
            for (int dt = 0; dt < DK_HALF; ++dt) {
                bf16x8_t o1 = *reinterpret_cast<const bf16x8_t*>(
                    &o_row[dt * W_K + row8]);
                bf16x8_t o_pack;
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                    o_pack[j] = bf16_fast22(bf16_to_f32(o1[j]) * scale1 * inv);
                *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
            }
        }
    }
#else
    (void)k;
#endif
}
