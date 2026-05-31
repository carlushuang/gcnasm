// v133 -- v132 (the 92.78 champion: uniform prio-2 on all WMMA + QK^T raise
// hoisted above K preloads) FUSED with v128's STEADY-STATE single-buffer V
// software-pipeline ring in the PV phase. ROUND-16 (lane B).
//
// ================= WHY THIS, WHY NOW ======================================
// Round-15 verdict on v124 (the last V-pipeline attempt) was explicit:
//   "prologue-only latency hiding ... does not address the dominant steady-state
//    loop. NEXT FOCUS: steady-state PV V-tile lookahead/interleaving with tight
//    live ranges, not another prologue-only variant."
//
// The champion v132 wins purely on wave-priority s_setprio scheduling, but its
// PV loop still loads each D-tile's V *inside* the WMMA loop body (chunked
// CHUNK=2), so every tile pays a head-of-tile global-load bubble: the PV WMMA
// for tile dt waits on the s_waitcnt for that tile's V load. v132 hides this
// only via raw issue priority, not via prefetch -- the V load and its consuming
// WMMA are still adjacent in the dependency chain.
//
// v133 applies the reviewer's mandated lever DIRECTLY in steady state: the EXACT
// single-buffer V ring already proven in this kernel's QK^T K-loop (and shipped
// bit-exact as v128). Preload D-tile 0's V, then while tile dt's two PV WMMAs
// run, prefetch tile dt+1's V into ONE next-pair. Only one extra V pair (2
// bf16x8 regs) is ever live -- identical footprint to the accepted K ring, and
// far below v127's CHUNK=2 double-buffer that doubled the V live set and sank to
// 83.94. Tight live ranges = the reviewer's exact ask.
//
// THE TWO LEVERS ARE ORTHOGONAL:
//   - v132's contribution = s_setprio wave-priority arbitration (which resident
//     wave wins the shared matrix-pipe issue slot). Kept VERBATIM: prio 2 spans
//     [K-preload .. end of QK^T] and [all of PV]; prio 0 spans softmax island
//     and epilogue. The PV prio-2 raise is issued ONCE before the PV ring and
//     held across the whole burst (no per-tile toggle -- lane A's v136 proved
//     toggling inside a WMMA burst regresses).
//   - v128's contribution = WHEN each V load is issued relative to its consuming
//     WMMA (prefetch dt+1 under dt's WMMAs). This reduces the s_waitcnt bubble
//     that priority alone cannot remove -- priority decides who issues a READY
//     WMMA, but a WMMA stalled on an un-arrived V load is not ready at all.
// Combining them attacks both the arbitration cost (v132) and the latency-
// exposure cost (v128) of the steady-state PV loop simultaneously.
//
// ================= WHY IT IS BIT-EXACT ====================================
// 1. s_setprio(N) emits one `s_setprio N` SALU op: changes only issue priority,
//    moves no data, reassociates no fp32 add/mul. (Same as v131/v132.)
// 2. The V ring changes only the SCHEDULE of the V global loads, not the
//    arithmetic. Per-tile rescale: each of the 8 independent fp32x8 O fragments
//    is multiplied exactly once by the same scalar `rescale` immediately before
//    its two WMMAs -- identical bits per fragment vs v132's 2-tile-chunk order
//    (independent accumulators, so chunk vs per-tile grouping is irrelevant to
//    rounding). Same WMMA operand order (v0 then v1) and same v_o[dt] feed.
//    -> max_abs stays 0.0001, n_bad=0. (v128 measured this exact PV body
//    bit-exact at 92.07 secondary.)
//
// ================= WIRING NOTE (inherited) =================================
// v133 MUST be in the host dVT (pre-transposed V [B,H,D,N]) allow-list; the
// kernel indexes V with vt_stride_d = k.N. Host registers v133 in BOTH the
// dispatch switch AND the dVT allow-list.
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast133(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v133_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v133_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v133(opus_attn_kargs k)
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
            v_q[kt][j] = bf16_fast133(bf16_to_f32(v_q[kt][j]) * qscale);
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

    for (int n_tile = 0; n_tile < num_kv_tiles; ++n_tile) {
        const int n_base = n_tile * BLOCK_N;

        fp32x8_t v_s0 = {0,0,0,0,0,0,0,0};
        fp32x8_t v_s1 = {0,0,0,0,0,0,0,0};

        // QK^T is bubble-prone (2 chains, depth DK, gated on K loads) -> prio 2.
        // Raised BEFORE the K preloads so the FIRST QK^T WMMA also arbitrates at
        // prio 2 (v132 lever A). s_setprio only affects instructions issued after.
        __builtin_amdgcn_s_setprio(2);

        // ---- QKT with software-pipelined K loads (v63) ----
        bf16x8_t v_k0_next = *reinterpret_cast<const bf16x8_t*>(Kp + (n_base + col16) * stride_n + row8);
        bf16x8_t v_k1_next = *reinterpret_cast<const bf16x8_t*>(Kp + (n_base + 16 + col16) * stride_n + row8);

        #pragma unroll
        for (int kt = 0; kt < DK; ++kt) {
            bf16x8_t v_k0 = v_k0_next;
            bf16x8_t v_k1 = v_k1_next;
            if (kt + 1 < DK) {
                v_k0_next = *reinterpret_cast<const bf16x8_t*>(Kp + (n_base + col16) * stride_n + (kt+1) * W_K + row8);
                v_k1_next = *reinterpret_cast<const bf16x8_t*>(Kp + (n_base + 16 + col16) * stride_n + (kt+1) * W_K + row8);
            }
            v_s0 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_k0, v_q[kt], v_s0);
            v_s1 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_k1, v_q[kt], v_s1);
        }

        // Softmax island = pure VALU, no WMMA -> prio 0 (never out-competes
        // another resident wave's ready WMMA). No WMMA adjacent -> honors the
        // "never lower priority next to a WMMA burst" rule.
        __builtin_amdgcn_s_setprio(0);
        fp32_t row_max = v_s0[0];
        #pragma unroll
        for (int j = 1; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s1[j]);
        row_max = v133_cross_half_max(row_max);

        const fp32_t new_m = __builtin_fmaxf(m_row, row_max);
        fp32_t rescale = 1.0f;
        const bool need_rescale = (new_m != m_row);
        if (need_rescale) {
            rescale = __builtin_amdgcn_exp2f(m_row - new_m);
            l_row *= rescale;
            // v_o rescale deferred into the per-tile PV ring below.
        }
        m_row = new_m;

        #pragma unroll
        for (int j = 0; j < 8; ++j) v_s0[j] = __builtin_amdgcn_exp2f(v_s0[j] - new_m);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_s1[j] = __builtin_amdgcn_exp2f(v_s1[j] - new_m);

        fp32_t row_sum = v_s0[0];
        #pragma unroll
        for (int j = 1; j < 8; ++j) row_sum += v_s0[j];
        #pragma unroll
        for (int j = 0; j < 8; ++j) row_sum += v_s1[j];
        row_sum = v133_cross_half_sum(row_sum);
        l_row += row_sum;

        bf16x8_t v_p0, v_p1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p0[j] = bf16_fast133(v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p1[j] = bf16_fast133(v_s1[j]);

        // ---- PV: STEADY-STATE single-buffer V SW-pipeline (v128 ring) under
        //         v132's uniform prio-2 (round-16 fusion) ----
        // Raise to prio 2 ONCE here, held across the entire PV burst (uniform
        // with QK^T, no per-tile toggle). The V ring preloads D-tile 0, then
        // while tile dt's two PV WMMAs execute it prefetches tile dt+1's V into
        // ONE next-pair (2 bf16x8 regs live -- identical to the K ring footprint,
        // tight live ranges per the round-15 mandate). The prefetched load
        // retires in the matrix-pipe shadow, removing the per-tile head-of-tile
        // s_waitcnt bubble that priority alone cannot hide.
        // BIT-EXACT: per-tile rescale (each independent fp32 O fragment multiplied
        // once by the same scalar `rescale`); same WMMA operand order.
        __builtin_amdgcn_s_setprio(2);
        bf16x8_t v_v0_next = *reinterpret_cast<const bf16x8_t*>(VTp + (0 * W_K + col16) * vt_stride_d + n_base + row8);
        bf16x8_t v_v1_next = *reinterpret_cast<const bf16x8_t*>(VTp + (0 * W_K + col16) * vt_stride_d + n_base + 16 + row8);
        #pragma unroll
        for (int dt = 0; dt < DK; ++dt) {
            bf16x8_t v_v0 = v_v0_next;
            bf16x8_t v_v1 = v_v1_next;
            if (dt + 1 < DK) {
                v_v0_next = *reinterpret_cast<const bf16x8_t*>(VTp + ((dt+1) * W_K + col16) * vt_stride_d + n_base + row8);
                v_v1_next = *reinterpret_cast<const bf16x8_t*>(VTp + ((dt+1) * W_K + col16) * vt_stride_d + n_base + 16 + row8);
            }
            if (need_rescale) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) v_o[dt][j] *= rescale;
            }
            v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_v0, v_p0, v_o[dt]);
            v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_v1, v_p1, v_o[dt]);
        }
    }

    // Epilogue is pure VALU + global store, no WMMA -> prio 0.
    __builtin_amdgcn_s_setprio(0);
    const fp32_t inv = (l_row > 0.0f) ? (1.0f / l_row) : 0.0f;
    bf16_t* o_row = Op + (q_m_base + col16) * stride_n;
    #pragma unroll
    for (int dt = 0; dt < DK; ++dt) {
        bf16x8_t o_pack;
        #pragma unroll
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast133(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
