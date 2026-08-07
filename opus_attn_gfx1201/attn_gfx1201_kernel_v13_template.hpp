// SPDX-License-Identifier: MIT
// opus_attn_gfx1201 v13 — v12 + vectorized K loads + BLOCK_N=32 + LDS store transpose.
//
// Over v12:
//   1. K loads via pointer cast to force global_load_b128 (24 → 8 instructions)
//   2. BLOCK_N=32: process 2 N-subtiles per outer iter, halving loop overhead
//   3. V smem write in transposed layout so the read side is contiguous (ds_load_b128)
//
//   workgroup = 1 wave x 32 lanes, BLOCK_M=16, BLOCK_N=32, D=128
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline fp32_t v13_fmaxf(fp32_t a, fp32_t b) { return a > b ? a : b; }

__device__ static inline fp32_t v13_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v13_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v13_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v13(opus_attn_kargs k)
{
#if defined(__gfx1201__) || defined(__gfx1200__)
    constexpr int BLOCK_M = T::BLOCK_M;
    constexpr int BLOCK_N = T::BLOCK_N;
    constexpr int W_K     = T::W_K;
    constexpr int DK      = T::D_TILES_K;
    constexpr int N_SUB   = BLOCK_N / 16;  // 2 for BLOCK_N=32

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

    // smem for V transpose: 16x16 tile, store transposed so read is contiguous.
    // Write: lane writes 8 bf16 at smem[(row8+j)*16 + col16], stride-16 per element.
    // Read:  lane reads 8 contiguous bf16 from smem[col16*16 + row8 .. row8+7].
    // This is the OPPOSITE of v12: here we write strided (scatter) and read contiguous.
    __shared__ bf16_t s_v[16 * 16];

    // Load Q — vectorized via pointer cast
    bf16x8_t v_q[DK];
    {
        const int q_m_base = q_tile_id * BLOCK_M;
        const bf16_t* q_row = Qp + (q_m_base + col16) * stride_n;
        #pragma unroll
        for (int kt = 0; kt < DK; ++kt) {
            v_q[kt] = *reinterpret_cast<const bf16x8_t*>(&q_row[kt * W_K + row8]);
        }
    }
    constexpr fp32_t LOG2_E = 1.44269504088896340736f;
    const fp32_t qscale = k.scale * LOG2_E;
    #pragma unroll
    for (int kt = 0; kt < DK; ++kt) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_q[kt][j] = bf16_from_f32(bf16_to_f32(v_q[kt][j]) * qscale);
    }

    // Output accumulator
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

        // Process N_SUB=2 sub-tiles of 16 KV positions each
        fp32x8_t v_s[N_SUB];

        // mma0 SWAP for both sub-tiles
        #pragma unroll
        for (int ns = 0; ns < N_SUB; ++ns) {
            v_s[ns] = (fp32x8_t){0,0,0,0,0,0,0,0};
            #pragma unroll
            for (int kt = 0; kt < DK; ++kt) {
                // Vectorized K load via pointer cast
                const bf16_t* k_addr = Kp + (n_base + ns * 16 + col16) * stride_n + kt * W_K + row8;
                bf16x8_t v_k = *reinterpret_cast<const bf16x8_t*>(k_addr);
                v_s[ns] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_k, v_q[kt], v_s[ns]);
            }
        }

        // Online softmax over all N_SUB*8 = 16 values
        // Find row max across both sub-tiles
        fp32_t row_max = v_s[0][0];
        #pragma unroll
        for (int ns = 0; ns < N_SUB; ++ns) {
            #pragma unroll
            for (int j = (ns == 0 ? 1 : 0); j < 8; ++j)
                row_max = v13_fmaxf(row_max, v_s[ns][j]);
        }
        row_max = v13_cross_half_max(row_max);

        const fp32_t new_m = v13_fmaxf(m_row, row_max);
        const fp32_t rescale = __builtin_amdgcn_exp2f(m_row - new_m);
        m_row = new_m;

        // Rescale v_o
        #pragma unroll
        for (int dt = 0; dt < DK; ++dt) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_o[dt][j] *= rescale;
        }

        // exp2 + row sum for both sub-tiles
        fp32_t row_sum = 0.0f;
        #pragma unroll
        for (int ns = 0; ns < N_SUB; ++ns) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                v_s[ns][j] = __builtin_amdgcn_exp2f(v_s[ns][j] - new_m);
                row_sum += v_s[ns][j];
            }
        }
        row_sum = v13_cross_half_sum(row_sum);
        l_row = l_row * rescale + row_sum;

        // mma1 SWAP for both sub-tiles with contiguous V load + smem transpose
        #pragma unroll
        for (int ns = 0; ns < N_SUB; ++ns) {
            // Cast v_s[ns] to bf16 once
            bf16x8_t v_p;
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_p[j] = bf16_from_f32(v_s[ns][j]);

            #pragma unroll
            for (int dt = 0; dt < DK; ++dt) {
                // Contiguous V load: 1 global_load_b128 per lane
                const bf16_t* v_addr = Vp + (n_base + ns * 16 + col16) * stride_n + dt * W_K + row8;
                bf16x8_t v_load = *reinterpret_cast<const bf16x8_t*>(v_addr);

                // Smem write: store TRANSPOSED so read side is contiguous.
                // Write each element to smem[(row8+j)*16 + col16] — strided per element.
                #pragma unroll
                for (int j = 0; j < 8; ++j) s_v[(row8 + j) * 16 + col16] = v_load[j];
                __builtin_amdgcn_wave_barrier();

                // Read contiguously: smem[col16*16 + row8 .. row8+7] = A-fragment layout
                bf16x8_t v_v = *reinterpret_cast<const bf16x8_t*>(&s_v[col16 * 16 + row8]);

                v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_v, v_p, v_o[dt]);
            }
        }
    }

    // Normalize and write back — vectorized
    const fp32_t inv = (l_row > 0.0f) ? (1.0f / l_row) : 0.0f;
    const int q_m_base = q_tile_id * BLOCK_M;
    bf16_t* o_row = Op + (q_m_base + col16) * stride_n;
    #pragma unroll
    for (int dt = 0; dt < DK; ++dt) {
        bf16x8_t o_pack;
        #pragma unroll
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_from_f32(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
