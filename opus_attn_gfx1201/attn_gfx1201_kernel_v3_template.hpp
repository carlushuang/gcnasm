// SPDX-License-Identifier: MIT
// opus_attn_gfx1201 v3 — attack the v2 register-spill problem from a
// different angle: aggressively cap launch_bounds + simplify the kernel
// to reduce VGPR liveness.
//
// Pragmatic deviation from the original roadmap "cross-lane permute"
// idea: implementing the 16x16 column→row transpose via bpermute is
// ~64 bpermutes per transpose, roughly the same cost as the smem
// round-trip (~200 cycles either way), so the expected win was small.
// The REAL bottleneck observed in v2 is register spill (133 VGPRs to
// scratch, occupancy 3 vs 9) — addressing that is the bigger lever.
//
// Geometry: same as v1.
//   workgroup = 4 waves × 32 lanes = 128 threads
//   BLOCK_M   = 64
//   BLOCK_N   = 16
//   D         = 128
//
// Changes vs v1:
//   - launch_bounds(128, 4) forces VGPRs ≤ ~64 → occupancy 4 waves/SIMD
//   - merge separate {s_max, rescale, s_sum} arrays into a single
//     fused softmax loop, fewer live ranges
//   - skip the (redundant for 1-tile-per-iter) wave_barrier after P write
#include <hip/hip_runtime.h>
#include "attn_common.h"

using fp16x8_t = fp16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline fp32_t v3_fmaxf(fp32_t a, fp32_t b) { return a > b ? a : b; }

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 4)
__global__ void opus_attn_gfx1201_kernel_v3(opus_attn_kargs k)
{
#if defined(__gfx1201__) || defined(__gfx1200__)
    constexpr int BLOCK_M    = T::BLOCK_M;
    constexpr int BLOCK_N    = T::BLOCK_N;
    constexpr int D          = T::D;
    constexpr int W_K        = T::W_K;
    constexpr int DK         = T::D_TILES_K;
    constexpr int WAVE_M     = T::WAVE_M;
    constexpr int WARP_SIZE  = T::WARP_SIZE;

    const int tid     = static_cast<int>(threadIdx.x);
    const int wave_id = tid / WARP_SIZE;
    const int lane    = tid % WARP_SIZE;
    const int col16   = lane % 16;
    const int row_grp = lane / 16;
    const int row8    = row_grp * 8;

    const int q_block_id = blockIdx.x;
    const int h          = blockIdx.y;
    const int b          = blockIdx.z;

    const int stride_n = k.D;
    const int stride_h = k.N * k.D;
    const int stride_b = k.H * k.N * k.D;

    const fp16_t* __restrict__ Qp = reinterpret_cast<const fp16_t*>(k.ptr_q) + b * stride_b + h * stride_h;
    const fp16_t* __restrict__ Kp = reinterpret_cast<const fp16_t*>(k.ptr_k) + b * stride_b + h * stride_h;
    const fp16_t* __restrict__ Vp = reinterpret_cast<const fp16_t*>(k.ptr_v) + b * stride_b + h * stride_h;
    fp16_t*       __restrict__ Op = reinterpret_cast<fp16_t*>      (k.ptr_o) + b * stride_b + h * stride_h;

    constexpr int K_SMEM_ELEMS = BLOCK_N * D;
    constexpr int V_SMEM_ELEMS = BLOCK_N * D;
    constexpr int P_SMEM_ELEMS = T::NUM_WAVES * WAVE_M * BLOCK_N;
    __shared__ fp16_t s_k[K_SMEM_ELEMS];
    __shared__ fp16_t s_v[V_SMEM_ELEMS];
    __shared__ fp16_t s_p[P_SMEM_ELEMS];

    const int wave_m_base = q_block_id * BLOCK_M + wave_id * WAVE_M;
    fp16x8_t v_q[DK];
    {
        const fp16_t* q_row = Qp + (wave_m_base + col16) * stride_n;
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
    constexpr int FP16_PER_LD = K_SMEM_ELEMS / T::BLOCK_SIZE;

    for (int n_tile = 0; n_tile < num_kv_tiles; ++n_tile) {
        const int n_base = n_tile * BLOCK_N;

        const fp16_t* k_src = Kp + n_base * stride_n;
        const fp16_t* v_src = Vp + n_base * stride_n;
        #pragma unroll
        for (int i = 0; i < FP16_PER_LD; ++i) {
            const int idx = i * T::BLOCK_SIZE + tid;
            const int row = idx / D;
            const int col = idx % D;
            s_k[idx] = k_src[row * stride_n + col];
            s_v[idx] = v_src[row * stride_n + col];
        }
        __syncthreads();

        fp32x8_t v_s = {0,0,0,0,0,0,0,0};
        #pragma unroll
        for (int kt = 0; kt < DK; ++kt) {
            fp16x8_t v_k;
            const fp16_t* k_row = s_k + col16 * D + kt * W_K;
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_k[j] = k_row[row8 + j];
            v_s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v_q[kt], v_k, v_s);
        }

        // Fused softmax: compute max, rescale O, exp2, accumulate l — all
        // in a single pass per element when possible. Use scalar temps to
        // limit live ranges (no s_max[8] / s_sum[8] arrays).
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            fp32_t v = v_s[j];
            v = v3_fmaxf(v, __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x041F)));
            v = v3_fmaxf(v, __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x081F)));
            v = v3_fmaxf(v, __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x101F)));
            v = v3_fmaxf(v, __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x201F)));
            const fp32_t new_m = v3_fmaxf(m_row[j], v);
            const fp32_t scale = __builtin_amdgcn_exp2f(m_row[j] - new_m);
            m_row[j] = new_m;
            v_s[j]   = __builtin_amdgcn_exp2f(v_s[j] - new_m);
            // Rescale this row's O slots inline
            #pragma unroll
            for (int kt = 0; kt < DK; ++kt) v_o[kt][j] *= scale;
            // Cross-lane sum
            fp32_t s = v_s[j];
            s += __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, s), 0x041F));
            s += __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, s), 0x081F));
            s += __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, s), 0x101F));
            s += __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, s), 0x201F));
            l_row[j] = l_row[j] * scale + s;
        }

        // Flip S → P via per-wave smem (single wave_barrier instead of two)
        fp16_t* p_wave = s_p + wave_id * (WAVE_M * BLOCK_N);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            p_wave[(row8 + j) * BLOCK_N + col16] = static_cast<fp16_t>(v_s[j]);
        }
        __builtin_amdgcn_wave_barrier();

        fp16x8_t v_p;
        const int p_row = col16;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p[j] = p_wave[p_row * BLOCK_N + row8 + j];

        #pragma unroll
        for (int dt = 0; dt < DK; ++dt) {
            fp16x8_t v_v;
            const fp16_t* v_col = s_v + row8 * D + dt * W_K + col16;
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_v[j] = v_col[j * D];
            v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v_p, v_v, v_o[dt]);
        }

        __syncthreads();
    }

    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const fp32_t inv = (l_row[j] > 0.0f) ? (1.0f / l_row[j]) : 0.0f;
        #pragma unroll
        for (int kt = 0; kt < DK; ++kt) {
            const int d_base = kt * W_K + col16;
            Op[(wave_m_base + row8 + j) * stride_n + d_base] = static_cast<fp16_t>(v_o[kt][j] * inv);
        }
    }
#else
    (void)k;
#endif
}
