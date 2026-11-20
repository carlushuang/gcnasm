// SPDX-License-Identifier: MIT
// opus_attn_gfx1201 v20 — 8-wave workgroup, BLOCK_M=128, Q from L1 cache
//
// 8 waves × 32 lanes = 256 threads per workgroup.
// Each wave handles 16 Q rows. All waves access same K,V via L1 cache.
// Q is NOT held in registers — loaded per D-tile from global (L1 cached).
// This trades Q register pressure for massive K,V reuse (8× fewer WGs).
//
// D-chunked: 2 passes over N, each handling 4 D-tiles of O.
// QK^T needs all 8 D-tiles, so Q,K are loaded in full each pass (L1 cached).
// V is loaded only for the active D-chunk.
// Total K,V traffic per head: 4× less than BLOCK_M=16.
//
//   workgroup = 8 waves × 32 = 256 threads, BLOCK_M=128, BLOCK_N=16, D=128
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast20(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v20_fmaxf(fp32_t a, fp32_t b) { return a > b ? a : b; }

__device__ static inline fp32_t v20_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v20_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v20_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v20(opus_attn_kargs k)
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

    // Each wave has its own smem tile for V transpose
    __shared__ bf16_t s_v[8][16 * 16];  // 8 waves × 256 × 2 = 4 KB

    // Q base pointer for this wave
    const int q_m_base = q_tile_id * BLOCK_M + wave_id * 16;
    const bf16_t* q_row_base = Qp + (q_m_base + col16) * stride_n;

    // Prescale Q once and store in smem to avoid recomputing
    // Actually, we'll prescale Q on the fly since it's L1 cached

    constexpr fp32_t LOG2_E = 1.44269504088896340736f;
    const fp32_t qscale = k.scale * LOG2_E;

    const int num_kv_tiles = k.N / BLOCK_N;

    // 2 D-chunk passes: [0..3] and [4..7]
    for (int d_pass = 0; d_pass < 2; ++d_pass) {
        const int dt_base = d_pass * DK_HALF;

        // O accumulator for this D-chunk (4 D-tiles × 8 fp32 = 32 VGPRs)
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

            // 1. Compute S = QK^T over ALL 8 D-tiles
            //    Q loaded per D-tile from global (L1 cached)
            //    K loaded per D-tile from global (L1 shared across waves)
            fp32x8_t v_s = {0,0,0,0,0,0,0,0};
            #pragma unroll
            for (int kt = 0; kt < DK; ++kt) {
                // Load Q D-tile (hits L1 after first iteration)
                bf16x8_t v_q = *reinterpret_cast<const bf16x8_t*>(
                    &q_row_base[kt * W_K + row8]);

                // Prescale Q
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                    v_q[j] = bf16_fast20(bf16_to_f32(v_q[j]) * qscale);

                // Load K D-tile (L1 shared across 8 waves)
                const bf16_t* k_addr = Kp + (n_base + col16) * stride_n + kt * W_K + row8;
                bf16x8_t v_k = *reinterpret_cast<const bf16x8_t*>(k_addr);

                v_s = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_k, v_q, v_s);
            }

            // 2. Online softmax
            fp32_t row_max = v_s[0];
            #pragma unroll
            for (int j = 1; j < 8; ++j) row_max = v20_fmaxf(row_max, v_s[j]);
            row_max = v20_cross_half_max(row_max);

            const fp32_t new_m = v20_fmaxf(m_row, row_max);
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
            row_sum = v20_cross_half_sum(row_sum);
            l_row = l_row * rescale + row_sum;

            // 3. P × V for this D-chunk only
            bf16x8_t v_p;
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_p[j] = bf16_fast20(v_s[j]);

            #pragma unroll
            for (int dt = 0; dt < DK_HALF; ++dt) {
                const int abs_dt = dt_base + dt;
                const bf16_t* v_addr = Vp + (n_base + col16) * stride_n + abs_dt * W_K + row8;
                bf16x8_t v_load = *reinterpret_cast<const bf16x8_t*>(v_addr);

                // Smem transpose (write strided, read contiguous)
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                    s_v[wave_id][(row8 + j) * 16 + col16] = v_load[j];
                __builtin_amdgcn_wave_barrier();

                bf16x8_t v_v = *reinterpret_cast<const bf16x8_t*>(
                    &s_v[wave_id][col16 * 16 + row8]);

                v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_v, v_p, v_o[dt]);
            }
        }

        // Normalize and write this D-chunk
        const fp32_t inv = (l_row > 0.0f) ? (1.0f / l_row) : 0.0f;
        bf16_t* o_row = Op + (q_m_base + col16) * stride_n;
        #pragma unroll
        for (int dt = 0; dt < DK_HALF; ++dt) {
            const int abs_dt = dt_base + dt;
            bf16x8_t o_pack;
            #pragma unroll
            for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast20(v_o[dt][j] * inv);
            *reinterpret_cast<bf16x8_t*>(&o_row[abs_dt * W_K + row8]) = o_pack;
        }
    }
#else
    (void)k;
#endif
}
