// SPDX-License-Identifier: MIT
// opus_attn_gfx1201 v14 — BLOCK_N=32 + v12 smem pattern (write-contiguous / read-strided)
//
// Over v13:
//   - Reverts to v12's smem layout: write contiguous (ds_store_b128), read strided (ds_load_u16)
//     This replaces v13's 128 ds_store_b16 with 16 ds_store_b128 — far fewer instructions.
//   - Keeps v13's BLOCK_N=32, vectorized K loads, and swap_ab pattern.
//
//   workgroup = 1 wave x 32 lanes, BLOCK_M=16, BLOCK_N=32, D=128
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline fp32_t v14_fmaxf(fp32_t a, fp32_t b) { return a > b ? a : b; }

__device__ static inline fp32_t v14_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v14_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v14_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v14(opus_attn_kargs k)
{
#if defined(__gfx1201__) || defined(__gfx1200__)
    constexpr int BLOCK_M = T::BLOCK_M;
    constexpr int BLOCK_N = T::BLOCK_N;
    constexpr int W_K     = T::W_K;
    constexpr int DK      = T::D_TILES_K;
    constexpr int N_SUB   = BLOCK_N / 16;  // 2

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

    // smem for V transpose: write contiguous along D, read strided for A-fragment
    // Write: s_v[col16 * 16 + row8 + j] = V[n, dt*16 + row8+j]  -> contiguous per lane
    // Read:  s_v[(row8+j) * 16 + col16] = V[row8+j, dt*16 + col16] -> strided per lane
    __shared__ bf16_t s_v[16 * 16];

    // Load Q — vectorized
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

        // mma0 SWAP for both sub-tiles
        fp32x8_t v_s[N_SUB];
        #pragma unroll
        for (int ns = 0; ns < N_SUB; ++ns) {
            v_s[ns] = (fp32x8_t){0,0,0,0,0,0,0,0};
            #pragma unroll
            for (int kt = 0; kt < DK; ++kt) {
                const bf16_t* k_addr = Kp + (n_base + ns * 16 + col16) * stride_n + kt * W_K + row8;
                bf16x8_t v_k = *reinterpret_cast<const bf16x8_t*>(k_addr);
                v_s[ns] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_k, v_q[kt], v_s[ns]);
            }
        }

        // Online softmax over N_SUB*8 = 16 values
        fp32_t row_max = v_s[0][0];
        #pragma unroll
        for (int ns = 0; ns < N_SUB; ++ns) {
            #pragma unroll
            for (int j = (ns == 0 ? 1 : 0); j < 8; ++j)
                row_max = v14_fmaxf(row_max, v_s[ns][j]);
        }
        row_max = v14_cross_half_max(row_max);

        const fp32_t new_m = v14_fmaxf(m_row, row_max);
        const fp32_t rescale = __builtin_amdgcn_exp2f(m_row - new_m);
        m_row = new_m;

        #pragma unroll
        for (int dt = 0; dt < DK; ++dt) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_o[dt][j] *= rescale;
        }

        fp32_t row_sum = 0.0f;
        #pragma unroll
        for (int ns = 0; ns < N_SUB; ++ns) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                v_s[ns][j] = __builtin_amdgcn_exp2f(v_s[ns][j] - new_m);
                row_sum += v_s[ns][j];
            }
        }
        row_sum = v14_cross_half_sum(row_sum);
        l_row = l_row * rescale + row_sum;

        // mma1 SWAP with contiguous V load + smem transpose (v12 pattern)
        #pragma unroll
        for (int ns = 0; ns < N_SUB; ++ns) {
            bf16x8_t v_p;
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_p[j] = bf16_from_f32(v_s[ns][j]);

            #pragma unroll
            for (int dt = 0; dt < DK; ++dt) {
                // Contiguous V load -> ds_store_b128
                const bf16_t* v_addr = Vp + (n_base + ns * 16 + col16) * stride_n + dt * W_K + row8;
                bf16x8_t v_load = *reinterpret_cast<const bf16x8_t*>(v_addr);

                // Write contiguous: s_v[col16*16 + row8+j] = v_load[j]
                *reinterpret_cast<bf16x8_t*>(&s_v[col16 * 16 + row8]) = v_load;
                __builtin_amdgcn_wave_barrier();

                // Read strided: A-fragment layout
                bf16x8_t v_v;
                #pragma unroll
                for (int j = 0; j < 8; ++j) v_v[j] = s_v[(row8 + j) * 16 + col16];

                v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_v, v_p, v_o[dt]);
            }
        }
    }

    // Normalize and write — vectorized
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
