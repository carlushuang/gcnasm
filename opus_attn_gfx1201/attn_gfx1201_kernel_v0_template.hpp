// SPDX-License-Identifier: MIT
// opus_attn_gfx1201 — flash attention forward for gfx1201 (Navi 48 / RX 9070
// XT, RDNA4), fp16 in/out + fp32 acc, single-wave per Q-tile, online softmax.
//
// Layout: Q, K, V, O are [B, H, N, D=128] row-major (D innermost).
//
// Per gfx12 wmma_128b 16x16x16 fragment encoding (AMD RDNA4 ISA §7.12.2,
// confirmed against the lane-layout spreadsheet in our opus reference):
//
//   A (M×K, row-distributed):
//       lane i, register j ∈ [0,7] → A[i % 16, (i/16)*8 + j]
//   B (K×N, column-distributed):
//       lane i, register j ∈ [0,7] → B[(i/16)*8 + j, i % 16]
//   C (M×N, column-distributed):
//       lane i, register j ∈ [0,7] → C[(i/16)*8 + j, i % 16]
//
// Workgroup geometry (v0):
//   grid  = dim3(N / BLOCK_M, H, B)
//   block = dim3(WARP_SIZE = 32)            (1 wave per workgroup)
//   each workgroup writes BLOCK_M = 16 rows of O for one (b, h).
//
// Compile only on gfx1201 / gfx1200 — the kernel uses the _w32_gfx12 wmma
// builtins which require the wmma-128b-insts target feature.

// Standard HIP header — the kernel .cc compiles with hipcc's default flags
// (no -D__HIPCC_RTC__) so __global__, threadIdx, etc. are available.
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline fp32_t d_fmaxf(fp32_t a, fp32_t b) { return a > b ? a : b; }
__device__ static inline int    d_thread_x() { return threadIdx.x; }
__device__ static inline int    d_block_x()  { return blockIdx.x;  }
__device__ static inline int    d_block_y()  { return blockIdx.y;  }
__device__ static inline int    d_block_z()  { return blockIdx.z;  }

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel(opus_attn_kargs k)
{
#if defined(__gfx1201__) || defined(__gfx1200__)
    constexpr int BLOCK_M = T::BLOCK_M;
    constexpr int BLOCK_N = T::BLOCK_N;
    constexpr int D       = T::D;
    constexpr int W_K     = T::W_K;
    constexpr int DK      = T::D_TILES_K;   // D / 16

    const int lane     = static_cast<int>(d_thread_x());
    const int col16    = lane % 16;          // for B/C lane-mapping
    const int row_grp  = lane / 16;          // 0 or 1 (which 8-row block)
    const int row8     = row_grp * 8;        // 0 or 8

    const int q_tile_id = d_block_x();        // along N dim
    const int h         = d_block_y();
    const int b         = d_block_z();

    // Runtime strides ([B, H, N, D] row-major, D innermost):
    const int stride_n = k.D;
    const int stride_h = k.N * k.D;
    const int stride_b = k.H * k.N * k.D;

    const bf16_t* __restrict__ Qp = reinterpret_cast<const bf16_t*>(k.ptr_q)
                                    + b * stride_b + h * stride_h;
    const bf16_t* __restrict__ Kp = reinterpret_cast<const bf16_t*>(k.ptr_k)
                                    + b * stride_b + h * stride_h;
    const bf16_t* __restrict__ Vp = reinterpret_cast<const bf16_t*>(k.ptr_v)
                                    + b * stride_b + h * stride_h;
    bf16_t*       __restrict__ Op = reinterpret_cast<bf16_t*>(k.ptr_o)
                                    + b * stride_b + h * stride_h;

    // ── Load Q tile into registers ─────────────────────────────────────────
    // Q row-distributed: lane[i].A[j] = Q[m_base + i%16, (i/16)*8 + j]
    // For the FULL D=128 we iterate over D_TILES_K = 8 wmma K-tiles. We hold
    // all 8 Q fragments in registers throughout the KV loop (8 × 8 = 64 fp16
    // per lane).
    bf16x8_t v_q[DK];
    {
        const int q_m_base = q_tile_id * BLOCK_M;
        const bf16_t* q_row = Qp + (q_m_base + col16) * stride_n;
        #pragma unroll
        for (int kt = 0; kt < DK; ++kt) {
            const int k_base = kt * W_K + row8;     // K=0..7 for grp0, K=8..15 for grp1
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                v_q[kt][j] = q_row[k_base + j];
            }
        }
    }

    // ── Pre-scale Q by 1/sqrt(D) * LOG2_E so softmax can use exp2 ──────────
    constexpr fp32_t LOG2_E = 1.44269504088896340736f;
    const fp32_t qscale = k.scale * LOG2_E;
    #pragma unroll
    for (int kt = 0; kt < DK; ++kt) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            v_q[kt][j] = bf16_from_f32(bf16_to_f32(v_q[kt][j]) * qscale);
        }
    }

    // ── Output accumulator (fp32, full D=128 = 8 wmma C tiles per wave) ────
    fp32x8_t v_o[DK];
    #pragma unroll
    for (int kt = 0; kt < DK; ++kt) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_o[kt][j] = 0.0f;
    }

    // ── Online softmax state — per lane, but each lane owns ONE column of
    //   the (M×N)=16×16 result tile (since C is column-distributed). The
    //   "row" stats m_row[j], l_row[j] are SAME across all lanes that share
    //   the same M-row (lane group), so we store them per-register-index j
    //   and use a cross-lane reduction (permlane32_swap between lanes 0..15
    //   and 16..31) for max/sum over the N dimension.
    fp32_t m_row[8];                        // per row, per lane
    fp32_t l_row[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        m_row[j] = -3.4e38f;
        l_row[j] = 0.0f;
    }

    const int num_kv_tiles = k.N / BLOCK_N;

    // ── KV loop ─────────────────────────────────────────────────────────────
    for (int n_tile = 0; n_tile < num_kv_tiles; ++n_tile) {
        const int n_base = n_tile * BLOCK_N;

        // 1) S = Q @ K^T, shape (16×16), accumulator fp32
        //
        //    For wmma f32_16x16x16_f16: A=Q row-distributed, B=K^T col-distributed.
        //    "K^T" means we are looking at K with K-dim as rows, N-dim as cols.
        //    Lane i loads B[(i/16)*8 + j, i % 16] = K^T[k = row8+j, n = col16]
        //                                         = K[n = col16, k = row8+j]
        fp32x8_t v_s = {0,0,0,0,0,0,0,0};
        #pragma unroll
        for (int kt = 0; kt < DK; ++kt) {
            bf16x8_t v_k;
            const bf16_t* k_row = Kp + (n_base + col16) * stride_n + kt * W_K;
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_k[j] = k_row[row8 + j];
            v_s = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_q[kt], v_k, v_s);
        }

        // 2) Online softmax row update
        //
        //    Each lane holds an 8-element column slice of S; the M-row of
        //    element j is (lane/16)*8 + j. So m/l updates are per-j and
        //    require reducing v_s[j] across the 16 lanes that share row
        //    (lane/16)*8 + j  → that's the "column" reduction, done via
        //    a row-wise max across lanes 0..15 (or 16..31) for each j.
        //
        //    For wave32, lanes 0..15 own columns 0..15 of the 16×16 tile for
        //    rows 0..7, and lanes 16..31 own columns 0..15 for rows 8..15.
        //    Within each lane-group of 16, lanes own the same row-set with
        //    different N columns. So a 16-wide reduction inside the lane
        //    group gives the row max / row sum.

        // 2a) per-j row max across the 16 columns in this lane group
        fp32_t s_max[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            fp32_t v = v_s[j];
            // tree reduction inside 16-lane half-wave via DPP-style permutes.
            // RDNA4 wave32: __builtin_amdgcn_ds_swizzle_b32 patterns + max.
            v = d_fmaxf(v, __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x041F)));   // xor 1
            v = d_fmaxf(v, __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x081F)));   // xor 2
            v = d_fmaxf(v, __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x101F)));   // xor 4
            v = d_fmaxf(v, __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x201F)));   // xor 8
            s_max[j] = v;
        }

        // 2b) new m = max(old m, row max);  rescale = exp2(old m - new m)
        fp32_t rescale[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const fp32_t new_m = d_fmaxf(m_row[j], s_max[j]);
            rescale[j] = __builtin_amdgcn_exp2f(m_row[j] - new_m);
            m_row[j]  = new_m;
        }

        // 2c) p = exp2(s - m_new); accumulate row sum
        fp32_t s_sum[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            v_s[j] = __builtin_amdgcn_exp2f(v_s[j] - m_row[j]);
            fp32_t v = v_s[j];
            v += __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x041F));
            v += __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x081F));
            v += __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x101F));
            v += __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x201F));
            s_sum[j] = v;
            l_row[j] = l_row[j] * rescale[j] + s_sum[j];
        }

        // 2d) cast P to fp16 — but for P @ V's A operand we need row-distributed
        //     P, not the column-distributed S layout we currently have.
        //     Current v_s[j] holds S[(row_grp*8)+j, col16]. For the next wmma
        //     we want v_p[j] = P[row=(row_grp*8)+j, k=(row_grp*8)+j_alt] which
        //     is NOT what we have either.
        //
        //     The simplest correct approach: write P into shared memory in
        //     row-major (16×16), barrier, then re-read in A's row-distributed
        //     layout.
        __shared__ bf16_t s_p[16 * 16];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int m_row_global = row8 + j;
            s_p[m_row_global * 16 + col16] = bf16_from_f32(v_s[j]);
        }
        __builtin_amdgcn_s_barrier();

        // Also rescale v_o by `rescale[j]` for each row.
        // v_o[kt][j] holds O[row=row8+j, col=col16+kt_col_offset], where for
        // each wmma kt the lane's 8 elements are 8 rows of D-column col16
        // within tile kt. The `rescale[j]` indexed by j matches the same row.
        #pragma unroll
        for (int kt = 0; kt < DK; ++kt) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                v_o[kt][j] *= rescale[j];
            }
        }

        // 3) Load V tile from global (column-distributed for B operand)
        //    Then read P back from smem in A's row-distributed layout, do
        //    O += P @ V where P is (M=16, K=N_BLOCK=16) and V is (K=N_BLOCK=16, N=D=128).
        //    We unroll across D: for each kt ∈ [0, 8), do one wmma over
        //    the FULL K=16 of N_BLOCK, accumulating into v_o[kt].

        // Load P row-distributed from smem: lane i, register j → P[i%16, row8+j]
        bf16x8_t v_p;
        {
            const int p_row = col16;       // lane selects row
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_p[j] = s_p[p_row * 16 + row8 + j];
        }
        __builtin_amdgcn_s_barrier();

        #pragma unroll
        for (int kt = 0; kt < DK; ++kt) {
            const int d_base = kt * W_K + col16;   // V col = D coordinate
            const bf16_t* v_col = Vp + (n_base + row8) * stride_n + d_base;
            bf16x8_t v_v;
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_v[j] = v_col[j * stride_n];
            v_o[kt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_p, v_v, v_o[kt]);
        }
    }

    // ── Normalize by l_row and write back to O (column-distributed → row-major)
    const int q_m_base = q_tile_id * BLOCK_M;
    fp32_t inv_l[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        inv_l[j] = (l_row[j] > 0.0f) ? (1.0f / l_row[j]) : 0.0f;
    }
    #pragma unroll
    for (int kt = 0; kt < DK; ++kt) {
        const int d_base = kt * W_K + col16;   // lane's column within tile
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int m_row_global = row8 + j;
            const fp32_t out_val = v_o[kt][j] * inv_l[j];
            Op[(q_m_base + m_row_global) * stride_n + d_base] = bf16_from_f32(out_val);
        }
    }
#else
    (void)k;
#endif // gfx1201/gfx1200
}
