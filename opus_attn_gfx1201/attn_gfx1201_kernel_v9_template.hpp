// SPDX-License-Identifier: MIT
// opus_attn_gfx1201 v9 — pre-transposed V (DRAM layout [B,H,D,N]).
//
// In v6 we load V row-major and use smem to transpose into B-fragment layout.
// If the host pre-transposes V → V_T (a one-time cost that is amortizable
// across decode steps), the B-layout load becomes naturally contiguous and
// we can drop the entire smem flip for V:
//
//   V_T[d, n] at byte offset (d*N + n)*2; B-layout requires lane (c, r) reg j
//   to read V[r*8+j, dt*16+c] = V_T[dt*16+c, r*8+j]. Per lane, that's 8
//   contiguous fp16 starting at V_T + (dt*16+c)*N + n_base + r*8 → 1
//   global_load_b128 wave instruction, in B-layout already.
//
// Tradeoffs:
//   - Pre-transpose is amortizable for prefill (1 kernel pass on V) and for
//     decode (V_T is appended one row per token, same as V append).
//   - Saves the per-D-tile smem store + barrier + strided smem read vs v6.
//
//   workgroup = 1 wave × 32 lanes, BLOCK_M=16, BLOCK_N=16, D=128
//   k.ptr_v MUST point to V_T = [B, H, D, N], not the row-major V.
#include <hip/hip_runtime.h>
#include "attn_common.h"

using fp16x8_t = fp16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline fp32_t v9_fmaxf(fp32_t a, fp32_t b) { return a > b ? a : b; }

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v9(opus_attn_kargs k)
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

    const int stride_n  = k.D;            // for Q, K: stride between sequence positions
    const int stride_h  = k.N * k.D;
    const int stride_b  = k.H * k.N * k.D;

    // V_T layout: [B, H, D, N]. Per-head stride: D*N = stride_h (same total size).
    const int vt_stride_d = k.N;          // bytes-per-D-row in V_T
    const int vt_stride_h = k.D * k.N;
    const int vt_stride_b = k.H * k.D * k.N;

    const fp16_t* __restrict__ Qp  = reinterpret_cast<const fp16_t*>(k.ptr_q) + b * stride_b + h * stride_h;
    const fp16_t* __restrict__ Kp  = reinterpret_cast<const fp16_t*>(k.ptr_k) + b * stride_b + h * stride_h;
    const fp16_t* __restrict__ VTp = reinterpret_cast<const fp16_t*>(k.ptr_v) + b * vt_stride_b + h * vt_stride_h;
    fp16_t*       __restrict__ Op  = reinterpret_cast<fp16_t*>(k.ptr_o) + b * stride_b + h * stride_h;

    __shared__ fp16_t s_p[16 * 16];

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
            v = v9_fmaxf(v, __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x041F)));
            v = v9_fmaxf(v, __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x081F)));
            v = v9_fmaxf(v, __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x101F)));
            v = v9_fmaxf(v, __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x201F)));
            const fp32_t new_m = v9_fmaxf(m_row[j], v);
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

        // V_T is [B, H, D, N]. For B-layout lane (c, r) reg j ← V[r*8+j, dt*16+c]
        // = V_T[dt*16+c, n_base+r*8+j]. Per lane: 8 contiguous fp16 starting at
        // VTp + (dt*16+c)*N + n_base+r*8. → 1 global_load_b128 per V tile.
        #pragma unroll
        for (int dt = 0; dt < DK; ++dt) {
            fp16x8_t v_v;
            const fp16_t* vt_row = VTp + (dt * W_K + col16) * vt_stride_d + n_base + row8;
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_v[j] = vt_row[j];

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

// Host-side V → V_T transpose kernel: V is [B,H,N,D], VT is [B,H,D,N].
__global__ void v_transpose_kernel(const fp16_t* __restrict__ V,
                                   fp16_t* __restrict__ VT,
                                   int B, int H, int N, int D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * N * D;
    if (idx >= total) return;
    int d = idx % D;
    int n = (idx / D) % N;
    int h = (idx / (D * N)) % H;
    int b =  idx / (D * N * H);
    VT[(b * H + h) * D * N + d * N + n] = V[idx];
}
