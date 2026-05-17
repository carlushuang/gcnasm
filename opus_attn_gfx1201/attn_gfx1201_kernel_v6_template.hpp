// SPDX-License-Identifier: MIT
// opus_attn_gfx1201 v6 — contiguous V load + smem transpose.
//
// Disasm of v0 shows V load emits 42 global_load_u16 + 40 global_load_d16_hi_b16
// per outer iter (~80 16-bit loads), because v0's V access pattern is strided
// by stride_n=128 fp16 per lane. Same data via vectorized load would be 8
// global_load_b128 per outer iter (10× fewer load instructions).
//
// CK's BlockFmhaPipelineQRKSVS does the same trick via MakeShuffledVRegBlockDescriptor:
// load V along its contiguous (D) axis, then redistribute to B-fragment layout.
// Here we use a 16×16 smem buffer for the transpose.
//
//   workgroup = 1 wave × 32 lanes, BLOCK_M=16, BLOCK_N=16, D=128
#include <hip/hip_runtime.h>
#include "attn_common.h"

using fp16x8_t = fp16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline fp32_t v6_fmaxf(fp32_t a, fp32_t b) { return a > b ? a : b; }

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v6(opus_attn_kargs k)
{
#if defined(__gfx1201__) || defined(__gfx1200__)
    constexpr int BLOCK_M = T::BLOCK_M;
    constexpr int BLOCK_N = T::BLOCK_N;
    constexpr int D       = T::D;
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

    __shared__ fp16_t s_p[16 * 16];
    __shared__ fp16_t s_v[16 * 16];

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

    for (int n_tile = 0; n_tile < num_kv_tiles; ++n_tile) {
        const int n_base = n_tile * BLOCK_N;

        fp32x8_t v_s = {0,0,0,0,0,0,0,0};
        #pragma unroll
        for (int kt = 0; kt < DK; ++kt) {
            fp16x8_t v_k;
            const fp16_t* k_row = Kp + (n_base + col16) * stride_n + kt * W_K;
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_k[j] = k_row[row8 + j];
            v_s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v_q[kt], v_k, v_s);
        }

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            fp32_t v = v_s[j];
            v = v6_fmaxf(v, __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x041F)));
            v = v6_fmaxf(v, __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x081F)));
            v = v6_fmaxf(v, __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x101F)));
            v = v6_fmaxf(v, __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x201F)));
            const fp32_t new_m = v6_fmaxf(m_row[j], v);
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
        for (int j = 0; j < 8; ++j) s_p[(row8 + j) * 16 + col16] = static_cast<fp16_t>(v_s[j]);
        __builtin_amdgcn_wave_barrier();
        fp16x8_t v_p;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p[j] = s_p[col16 * 16 + row8 + j];

        #pragma unroll
        for (int dt = 0; dt < DK; ++dt) {
            // Contiguous V load: lane (col16, row_grp) reads 8 consecutive fp16
            // from V[n_base+col16, dt*16+row8 .. row8+7]. Per-lane contiguous →
            // compiler emits 1 global_load_b128 wave instruction.
            fp16x8_t v_load;
            const fp16_t* v_row = Vp + (n_base + col16) * stride_n + dt * W_K + row8;
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_load[j] = v_row[j];

            // Smem store: smem[col16, row8+j] — same layout as load, 8 consecutive
            // fp16 starting at col16*16+row8. Per-lane contiguous → 1 ds_store_b128.
            #pragma unroll
            for (int j = 0; j < 8; ++j) s_v[col16 * 16 + row8 + j] = v_load[j];
            __builtin_amdgcn_wave_barrier();

            // Read in B-layout: lane (c, r) reg j = V[r*8+j, c] = smem[r*8+j, c].
            // Strided per lane (16 fp16 = 32 B between elements) — compiler emits
            // ~8 ds_load_u16 (cheap, single-cycle issue on RDNA4 LDS).
            fp16x8_t v_v;
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_v[j] = s_v[(row8 + j) * 16 + col16];

            v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v_p, v_v, v_o[dt]);
        }
    }

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
