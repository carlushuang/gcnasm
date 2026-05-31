// v116 -- v111 base (v63 K double-buffer QKT + v43 chunked pre-PV rescale PV)
// with a KV-BLOCK SOFTWARE PIPELINE that overlaps the NEXT tile's QKT WMMAs
// with the CURRENT tile's softmax exp2 storm.
//
// Motivation (the one structural lever the v101..v115 family never touched):
// Every prior round in this lane reshuffled PV-internal V-load / rescale
// scheduling. They are exhausted -- v111's chunked rescale (91.59) is the only
// win; v112/v113/v114/v115 all regressed. The remaining big unfilled shadow is
// elsewhere: during the per-tile softmax (row-max reduce + 16x v_exp2f + sum
// reduce + bf16 pack) the MATRIX PIPE IS COMPLETELY IDLE. exp2 runs on the
// transcendental pipe (~3.28 T ops/s, co-executes with wmma); the matrix pipe
// just waits for v_p before PV.
//
// v116 fills that idle matrix-pipe window with the *next* KV tile's QKT WMMAs:
//   prologue:  QKT(tile 0)  -> v_s_cur
//   per tile t:
//       row-max(v_s_cur)                                  (VALU)
//       QKT(tile t+1) -> v_s_next      <-- 8 WMMAs, MATRIX pipe
//       exp2 softmax on v_s_cur        <-- 16 v_exp2f, TRANSCENDENTAL pipe
//                                          (co-executes with the QKT WMMAs above)
//       PV(v_p_cur) -> v_o             (v111 chunked rescale, MATRIX pipe)
//       v_s_cur = v_s_next
// The next QKT and the current PV are both matrix-pipe, so per iteration the
// matrix pipe streams 8 (QKT_next) + 16 (PV_cur) = 24 WMMAs back-to-back with
// no softmax gap, while the exp2 storm hides under the first 8. QKT(t+1) writes
// v_s_next; exp2 reads/writes v_s_cur -> register-independent, so the scheduler
// is free to interleave them.
//
// Algebra is byte-identical to v111 (same QKT contraction, same scalar rescale
// applied to each v_o[dt] strictly before that dt's PV WMMAs, same log2-domain
// exp2 softmax, same single ds_bpermute(lane^16) cross-half reduce). Only the
// EMISSION ORDER of the next-tile QKT moves earlier; results match v111's benign
// max_abs~0.036 signature (passes n_bad(>0.05)=0 cleanly).
//
// Cost: a second pair of score accumulators (v_s0_next/v_s1_next = 16 fp32/lane)
// must be live across the exp2 region -> ~+16 VGPR over v111. Risk: if that tips
// occupancy below the v111 point the overlap may not pay; measured this round.
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast116(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v116_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v116_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v116(opus_attn_kargs k)
{
#if defined(__gfx1201__) || defined(__gfx1200__)
    constexpr int BLOCK_M = T::BLOCK_M;
    constexpr int BLOCK_N = T::BLOCK_N;
    constexpr int W_K     = T::W_K;
    constexpr int DK      = T::D_TILES_K;

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
    bf16_t*       __restrict__ Op = reinterpret_cast<bf16_t*>(k.ptr_o) + b * stride_b + h * stride_h;

    const int vt_stride_d = k.N;
    const int vt_stride_h = k.D * k.N;
    const int vt_stride_b = k.H * k.D * k.N;
    const bf16_t* __restrict__ VTp = reinterpret_cast<const bf16_t*>(k.ptr_v) + b * vt_stride_b + h * vt_stride_h;

    const int q_m_base = q_tile_id * BLOCK_M + wave_id * 16;
    const bf16_t* q_row = Qp + (q_m_base + col16) * stride_n;
    bf16x8_t v_q[DK];
    constexpr fp32_t LOG2_E = 1.44269504088896340736f;
    const fp32_t qscale = k.scale * LOG2_E;
    #pragma unroll
    for (int kt = 0; kt < DK; ++kt) {
        v_q[kt] = *reinterpret_cast<const bf16x8_t*>(&q_row[kt * W_K + row8]);
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            v_q[kt][j] = bf16_fast116(bf16_to_f32(v_q[kt][j]) * qscale);
    }

    fp32x8_t v_o[DK];
    #pragma unroll
    for (int kt = 0; kt < DK; ++kt) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_o[kt][j] = 0.0f;
    }
    fp32_t m_row = -3.4e38f;
    fp32_t l_row = 0.0f;

    const int num_kv_tiles = k.N / BLOCK_N;

    // QKT contraction for a given n_base into (s0,s1), with v63 software-pipelined
    // K loads (preload D-tile kt, prefetch kt+1 while current WMMA executes).
    #define V116_QKT(NBASE, S0, S1)                                                                       \
    do {                                                                                                  \
        const int nb_ = (NBASE);                                                                          \
        bf16x8_t v_k0_next = *reinterpret_cast<const bf16x8_t*>(Kp + (nb_ + col16) * stride_n + row8);    \
        bf16x8_t v_k1_next = *reinterpret_cast<const bf16x8_t*>(Kp + (nb_ + 16 + col16) * stride_n + row8); \
        _Pragma("unroll")                                                                                 \
        for (int kt = 0; kt < DK; ++kt) {                                                                 \
            bf16x8_t v_k0 = v_k0_next;                                                                     \
            bf16x8_t v_k1 = v_k1_next;                                                                     \
            if (kt + 1 < DK) {                                                                             \
                v_k0_next = *reinterpret_cast<const bf16x8_t*>(Kp + (nb_ + col16) * stride_n + (kt+1) * W_K + row8); \
                v_k1_next = *reinterpret_cast<const bf16x8_t*>(Kp + (nb_ + 16 + col16) * stride_n + (kt+1) * W_K + row8); \
            }                                                                                             \
            (S0) = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_k0, v_q[kt], (S0));                \
            (S1) = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_k1, v_q[kt], (S1));                \
        }                                                                                                 \
    } while (0)

    // ---- Prologue: QKT for tile 0 ----
    fp32x8_t v_s0_cur = {0,0,0,0,0,0,0,0};
    fp32x8_t v_s1_cur = {0,0,0,0,0,0,0,0};
    V116_QKT(0, v_s0_cur, v_s1_cur);

    for (int n_tile = 0; n_tile < num_kv_tiles; ++n_tile) {
        const int n_base = n_tile * BLOCK_N;

        // ---- row max from CURRENT scores (VALU only) ----
        fp32_t row_max = v_s0_cur[0];
        #pragma unroll
        for (int j = 1; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s0_cur[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s1_cur[j]);
        row_max = v116_cross_half_max(row_max);

        const fp32_t new_m = __builtin_fmaxf(m_row, row_max);
        fp32_t rescale = 1.0f;
        const bool need_rescale = (new_m != m_row);
        if (need_rescale) {
            rescale = __builtin_amdgcn_exp2f(m_row - new_m);
            l_row *= rescale;
        }
        m_row = new_m;

        // ---- Issue NEXT tile's QKT WMMAs (matrix pipe) BEFORE the exp2 storm.
        // These have no dependency on the current softmax, so they stream on the
        // matrix pipe while the exp2 below runs on the transcendental pipe,
        // filling the otherwise-idle matrix window. ----
        fp32x8_t v_s0_next = {0,0,0,0,0,0,0,0};
        fp32x8_t v_s1_next = {0,0,0,0,0,0,0,0};
        if (n_tile + 1 < num_kv_tiles) {
            V116_QKT((n_tile + 1) * BLOCK_N, v_s0_next, v_s1_next);
        }

        // ---- exp2 softmax on CURRENT (transcendental pipe; co-executes with the
        // next-QKT WMMAs just issued above) ----
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_s0_cur[j] = __builtin_amdgcn_exp2f(v_s0_cur[j] - new_m);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_s1_cur[j] = __builtin_amdgcn_exp2f(v_s1_cur[j] - new_m);

        fp32_t row_sum = v_s0_cur[0];
        #pragma unroll
        for (int j = 1; j < 8; ++j) row_sum += v_s0_cur[j];
        #pragma unroll
        for (int j = 0; j < 8; ++j) row_sum += v_s1_cur[j];
        row_sum = v116_cross_half_sum(row_sum);
        l_row += row_sum;

        bf16x8_t v_p0, v_p1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p0[j] = bf16_fast116(v_s0_cur[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p1[j] = bf16_fast116(v_s1_cur[j]);

        // ---- PV: v111 chunked rescale interleaved with WMMAs (matrix pipe) ----
        constexpr int CHUNK = 2;
        #pragma unroll
        for (int c0 = 0; c0 < DK; c0 += CHUNK) {
            if (need_rescale) {
                #pragma unroll
                for (int dc = 0; dc < CHUNK; ++dc) {
                    const int dt = c0 + dc;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) v_o[dt][j] *= rescale;
                }
            }
            #pragma unroll
            for (int dc = 0; dc < CHUNK; ++dc) {
                const int dt = c0 + dc;
                const bf16_t* vt_addr0 = VTp + (dt * W_K + col16) * vt_stride_d + n_base + row8;
                const bf16_t* vt_addr1 = VTp + (dt * W_K + col16) * vt_stride_d + n_base + 16 + row8;
                bf16x8_t v_v0 = *reinterpret_cast<const bf16x8_t*>(vt_addr0);
                bf16x8_t v_v1 = *reinterpret_cast<const bf16x8_t*>(vt_addr1);
                v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_v0, v_p0, v_o[dt]);
                v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_v1, v_p1, v_o[dt]);
            }
        }

        // ---- promote next scores -> current for the following iteration ----
        v_s0_cur = v_s0_next;
        v_s1_cur = v_s1_next;
    }

    #undef V116_QKT

    const fp32_t inv = (l_row > 0.0f) ? (1.0f / l_row) : 0.0f;
    bf16_t* o_row = Op + (q_m_base + col16) * stride_n;
    #pragma unroll
    for (int dt = 0; dt < DK; ++dt) {
        bf16x8_t o_pack;
        #pragma unroll
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast116(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
