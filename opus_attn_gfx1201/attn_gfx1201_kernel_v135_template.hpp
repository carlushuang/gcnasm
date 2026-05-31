// v135 -- v132 (the 92.78 champion) with the O-accumulator RESCALE HOISTED OUT
// of the prio-2 PV burst and folded into the prio-0 softmax island, so the PV
// region is a PURE load+WMMA matrix burst. ROUND-18 (lane B).
//
// ================= WHAT CHANGED vs v132 (the 92.78 champion) ===============
// v132 wins purely on the s_setprio policy: prio 2 spans [K-preload..QK^T] and
// [all of PV]; prio 0 spans the softmax island and epilogue. But inside the
// prio-2 PV region v132 still runs the chunked `v_o[dt][j] *= rescale` block --
// pure VALU work -- AT PRIO 2. Round-17's v134 probed the adjacent question
// (hoisting prio-2 OVER the bf16 P-pack) and measured 92.7 = neutral-to-slight
// regression; the reviewer's reading: "extra prio-2 coverage over VALU is at
// best neutral... promoting a short VALU region to the same priority as resident
// waves' matrix work can hurt arbitration fairness without improving the matrix
// critical path." The reviewer's explicit NEXT lever #2: "test prio2 only ...
// before the first actual PV WMMA/load pair, not before pack OR RESCALE ...
// prioritize the matrix burst while keeping VALU feeder work yieldable."
//
// v135 does exactly that WITHOUT any per-tile priority toggle (lane-A v136
// proved toggling inside a WMMA burst regresses): it MOVES the entire O rescale
// back to v100's original placement -- a single monolithic pass over all 8 O
// fragments INSIDE the prio-0 softmax island, right after `m_row = new_m`, where
// it overlaps the exp2 / row-sum VALU. The PV loop then becomes a clean
// load-then-WMMA burst entered at prio 2 with NO embedded VALU. Two consequences:
//   1. The rescale VALU now runs at prio 0, so it stays YIELDABLE -- it hides in
//      other resident waves' WMMA shadows (playbook lever 6) instead of competing
//      with their matrix work for the shared issue slot.
//   2. The prio-2 matrix arbitration path is no longer punctuated by prio-2 VALU,
//      so a ready PV WMMA from ANY wave wins the slot without contending against
//      this wave's rescale FMAs -- the exact pressure v134 hinted at, removed.
//
// Net policy (unchanged structure, cleaner VALU/matrix separation):
//   prio 2 spans [K-preload .. end of QK^T] and [PV load+WMMA only];
//   prio 0 spans [softmax island INCLUDING the O rescale] and [epilogue store].
// The QK^T hoist-above-K-preloads (v132 lever A) is kept verbatim.
//
// ================= WHY IT IS BIT-EXACT ====================================
// Two facts:
//  (1) s_setprio(N) emits one `s_setprio N` SALU op: changes only issue
//      priority, moves no data, reassociates no fp32 add/mul.
//  (2) The rescale arithmetic is IDENTICAL to v100's (the original monolithic
//      placement): each of the 8 independent fp32x8 O accumulators is multiplied
//      exactly ONCE by the same scalar `rescale`, BEFORE any PV WMMA reads it.
//      The accumulators are mutually independent, so doing all 8 in one pass
//      (v135/v100) vs in 2-tile chunks interleaved with PV (v132) produces
//      bit-identical results -- the multiply order WITHIN each fragment, the
//      scalar value, and the WMMA operand feed (v_v0,v_p0 then v_v1,v_p1) are
//      all unchanged. -> max_abs stays 0.0001, n_bad=0. (v100 measured this
//      exact rescale placement bit-exact.)
//
// ================= ROOT-CAUSE / WIRING NOTE (inherited from v126) ==========
// v135 MUST be in the host dVT (pre-transposed V, layout [B,H,D,N]) allow-list;
// the kernel indexes V with vt_stride_d = k.N. Host registers v135 in BOTH the
// dispatch switch AND the dVT allow-list.
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast135(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v135_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v135_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v135(opus_attn_kargs k)
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
            v_q[kt][j] = bf16_fast135(bf16_to_f32(v_q[kt][j]) * qscale);
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

        // QK^T is bubble-prone (2 chains, depth DK, gated on K loads) -> highest
        // priority so a ready QK^T WMMA wins the shared matrix-pipe issue slot.
        // v135: raise BEFORE the K preloads (v131 raised it after), so the FIRST
        // QK^T WMMA -- the one most exposed to K-load latency -- also issues at
        // prio 2 instead of the default priority. s_setprio only affects
        // instructions issued after it, so its placement matters for the head.
        __builtin_amdgcn_s_setprio(2);

        // ---- QKT with software-pipelined K loads (v63) ----
        // Preload D-tile 0 K data, then prefetch the next D-tile while the
        // current WMMA executes, hiding global K-load latency.
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

        // Softmax island = pure VALU (cross-lane max/sum, exp2, bf16 pack), no
        // WMMA -> drop to priority 0 so it never out-competes another resident
        // wave's ready WMMA. (No WMMA is adjacent here, so this honors the
        // "never lower priority next to a WMMA burst" rule.)
        __builtin_amdgcn_s_setprio(0);
        fp32_t row_max = v_s0[0];
        #pragma unroll
        for (int j = 1; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s1[j]);
        row_max = v135_cross_half_max(row_max);

        const fp32_t new_m = __builtin_fmaxf(m_row, row_max);
        fp32_t rescale = 1.0f;
        const bool need_rescale = (new_m != m_row);
        if (need_rescale) {
            rescale = __builtin_amdgcn_exp2f(m_row - new_m);
            l_row *= rescale;
            // v135: the O-accumulator rescale is done HERE -- a single monolithic
            // pass over all 8 fragments INSIDE the prio-0 softmax island (v100's
            // original placement), NOT deferred into the prio-2 PV burst (v132).
            // Each independent fp32x8 O fragment is multiplied exactly once by the
            // same scalar `rescale` before any PV WMMA reads it -> bit-identical to
            // v132's per-chunk multiply (independent accumulators). Running it at
            // prio 0 keeps this VALU yieldable (hides in other waves' WMMA shadows)
            // and leaves the PV region a pure prio-2 load+WMMA matrix burst.
            #pragma unroll
            for (int dt = 0; dt < DK; ++dt) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) v_o[dt][j] *= rescale;
            }
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
        row_sum = v135_cross_half_sum(row_sum);
        l_row += row_sum;

        bf16x8_t v_p0, v_p1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p0[j] = bf16_fast135(v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p1[j] = bf16_fast135(v_s1[j]);

        // ---- PV: PURE load+WMMA matrix burst at prio 2 (v135) ----
        // The O rescale has ALREADY been applied above in the prio-0 softmax
        // island, so this region contains NO VALU -- only V global loads and the
        // PV WMMAs. Entered at prio 2 once and held across the whole burst (no
        // per-tile toggle; lane A's v136 proved toggling inside a WMMA burst
        // regresses). Every PV WMMA is now matrix work competing only against
        // other waves' matrix work for the shared issue slot -- the prio-2 path
        // is no longer punctuated by this wave's rescale FMAs (the arbitration
        // pressure v134 hinted at, removed). Same WMMA operand feed as v132.
        __builtin_amdgcn_s_setprio(2);
        #pragma unroll
        for (int dt = 0; dt < DK; ++dt) {
            const bf16_t* vt_addr0 = VTp + (dt * W_K + col16) * vt_stride_d + n_base + row8;
            const bf16_t* vt_addr1 = VTp + (dt * W_K + col16) * vt_stride_d + n_base + 16 + row8;
            bf16x8_t v_v0 = *reinterpret_cast<const bf16x8_t*>(vt_addr0);
            bf16x8_t v_v1 = *reinterpret_cast<const bf16x8_t*>(vt_addr1);
            v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_v0, v_p0, v_o[dt]);
            v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_v1, v_p1, v_o[dt]);
        }
    }

    // Epilogue is pure VALU + global store, no WMMA -> priority 0.
    __builtin_amdgcn_s_setprio(0);
    const fp32_t inv = (l_row > 0.0f) ? (1.0f / l_row) : 0.0f;
    bf16_t* o_row = Op + (q_m_base + col16) * stride_n;
    #pragma unroll
    for (int dt = 0; dt < DK; ++dt) {
        bf16x8_t o_pack;
        #pragma unroll
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast135(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
