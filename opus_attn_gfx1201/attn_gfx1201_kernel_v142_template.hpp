// v142 -- v140 champion (93.38 TFLOPS) + ONE zero-liveness scheduling change:
// raise the softmax HEAD (cross-lane row-max ds_bpermute reduction + new_m + the
// head rescale exp2f + l_row scale) from prio 0 to prio 1, UNIFYING the entire
// softmax island at prio 1. ROUND-25 (lane B).
//
// ================= WHY THIS, AND WHY IT IS GENUINELY NEW =====================
// v140 (93.38, the running champion) proved the lever: the softmax island sits on
// this wave's QK^T->PV critical path, and at v132's uniform prio 0 it loses issue
// arbitration to every co-resident wave's prio-2 matrix work, stretching the
// QK->PV gap with no compensating matrix benefit. v140 fixed the TAIL (16 bulk
// exp2 + row-sum fold + bf16 P-pack) by raising it to the middle prio 1, above
// other waves' prio-0 bookkeeping but below prio-2 matrix so it never steals
// matrix slots (the v134 mistake). It WON.
//
// v141 then tried the OPPOSITE narrowing -- drop row-sum/P-pack BACK to prio 0,
// keep only exp2 at prio1 -- and REGRESSED to 93.1. The round-24 reviewer
// root-caused it precisely: the row-sum fold and bf16 P-pack are SHORT DEPENDENT
// FEEDERS on the direct path to the first PV WMMA; dropping them to prio 0 lets
// co-resident matrix waves out-arbitrate this wave's VALU while it still must
// finish those feeders before it can become PV-ready. So the lesson is:
// EVERYTHING on the QK->PV feeder chain wants to be AT LEAST prio 1; do not let
// any dependent feeder fall back to prio 0 between the two matrix bursts.
//
// v142 applies that lesson to the ONE part of the island v140 still leaves at
// prio 0: the HEAD. The head is NOT free bookkeeping -- it is:
//   1. a cross-lane row-max reduction (v142_cross_half_max = ds_bpermute(lane^16)
//      + fmaxf) -- the SAME LDS-permute cross-lane op as the tail's row-sum fold
//      that v141 proved must stay >= prio1, and
//   2. the head rescale path: new_m = fmax(...), and when need_rescale the
//      TRANSCENDENTAL exp2f(m_row - new_m) + l_row scale -- a transcendental on
//      the exact pipe v140 identified as arbitration-starved.
// And the head GATES the tail: exp2(s - new_m) cannot issue until new_m exists.
// So at prio 0 the head's ds_bpermute + exp2f lose arbitration to other waves'
// prio-2 matrix work for the same reason the tail did at v132 -- delaying new_m,
// which in turn delays EVERY tail exp2, which delays the first PV WMMA. v140's
// reasoning ("don't starve this wave's critical-path transcendental/cross-lane
// work") applies verbatim to the head; v140 simply stopped one step short.
//
// THE FIX (minimal extension of v140's proven lever):
//   * Raise the prio-1 setprio so it covers the WHOLE softmax island -- from the
//     cross-lane row-max reduction through new_m / head-rescale exp2f, then the
//     bulk exp2 loops, row-sum fold, and bf16 P-pack -- a single contiguous prio1
//     window between the prio-2 QK^T burst and the prio-2 PV burst.
//   * QK^T (incl. K preload) stays prio 2; PV stays prio 2 -- UNCHANGED from v140.
//   * Epilogue stays prio 0 -- UNCHANGED.
//
// Net policy: prio2 [K-preload..QK^T] and [PV]; prio1 [ENTIRE softmax island:
// row-max reduce .. head rescale .. exp2 .. row-sum .. P-pack]; prio0 [epilogue].
// This is the natural completion of v140: v140 raised the tail, v141 proved the
// feeder chain must not fall to prio0, and v142 raises the remaining head feeders
// (cross-lane reduce + head exp2f) that v140 left starved at prio0. No version has
// run the island as a single uniform prio-1 region; v132 ran it uniform prio0.
//
// WHY NOT just a no-op vs v140: in v140 the head's ds_bpermute row-max and the
// (taken-when-need_rescale) head exp2f still issue at prio0 and can be deferred by
// other waves' matrix work BEFORE new_m is produced; this stretches the gate that
// the prio1 tail then waits on. Raising the head closes that residual stall. It is
// strictly more of the lever that already won, on a region with the SAME pipe
// characteristics (cross-lane LDS + transcendental) v140 targeted.
//
// ================= WHY IT IS BIT-EXACT ====================================
// __builtin_amdgcn_s_setprio(N) emits a single `s_setprio N` SALU instruction
// that ONLY changes this wave's issue-arbitration priority. It moves no data,
// does no arithmetic, reassociates no fp32 add/mul. Every exp2 input/output,
// every row_max/row_sum fold order, every bf16 pack, every WMMA operand order is
// byte-identical to v140/v132. -> max_abs stays 0.0001, n_bad=0. Only priority
// VALUES changed; no instruction that produces a value moved relative to another.
//
// ================= WIRING NOTE (inherited from v126/v132/v140) =============
// v142 MUST be in the host dVT (pre-transposed V, layout [B,H,D,N]) allow-list;
// the kernel indexes V with vt_stride_d = k.N. Host registers v142 in BOTH the
// dispatch switch AND the dVT allow-list. (Omitting it -> max_abs=0.0360.)
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast142(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v142_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v142_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v142(opus_attn_kargs k)
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
            v_q[kt][j] = bf16_fast142(bf16_to_f32(v_q[kt][j]) * qscale);
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
        // v132: raise BEFORE the K preloads (v131 raised it after), so the FIRST
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

        // Softmax HEAD = cross-lane row-max reduction (ds_bpermute) + new_m +
        // head rescale exp2f + l_row scale. v140 left this at prio 0 calling it
        // "short, yieldable bookkeeping"; v141's regression proved the QK->PV
        // FEEDER chain must NOT fall to prio 0. The head IS a feeder: its cross-
        // lane row-max (same ds_bpermute as the tail's row-sum fold) and its
        // (need_rescale) transcendental exp2f GATE new_m, and new_m gates every
        // tail exp2. At prio 0 these lose arbitration to other waves' prio-2
        // matrix work, delaying new_m and thus the whole tail. v142 raises the
        // ENTIRE island to prio 1 here: above other waves' prio-0 bookkeeping so
        // this wave's gate-producing cross-lane/transcendental head work is not
        // starved, still below prio-2 matrix so it never steals matrix slots.
        __builtin_amdgcn_s_setprio(1);
        fp32_t row_max = v_s0[0];
        #pragma unroll
        for (int j = 1; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s1[j]);
        row_max = v142_cross_half_max(row_max);

        const fp32_t new_m = __builtin_fmaxf(m_row, row_max);
        fp32_t rescale = 1.0f;
        const bool need_rescale = (new_m != m_row);
        if (need_rescale) {
            rescale = __builtin_amdgcn_exp2f(m_row - new_m);
            l_row *= rescale;
            // NOTE: the v_o rescale is NOT done here (v100 did all 8 fragments in
            // one VALU block). v131 defers it into 2-tile chunks interleaved with
            // the PV loop below -> only 2 O fragments hot per VALU region.
        }
        m_row = new_m;

        // Softmax TAIL = the 16 bulk exp2 transcendentals + row-sum fold + bf16
        // P-pack. Already running at prio 1 (raised once at the head above) -- v142
        // keeps the whole island at a single uniform prio 1, so NO setprio toggle
        // is needed here (v140 re-raised to 1 at this point because its head ran at
        // prio 0). One contiguous prio-1 window now spans head+tail, ending only at
        // the prio-2 PV burst below.
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_s0[j] = __builtin_amdgcn_exp2f(v_s0[j] - new_m);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_s1[j] = __builtin_amdgcn_exp2f(v_s1[j] - new_m);

        fp32_t row_sum = v_s0[0];
        #pragma unroll
        for (int j = 1; j < 8; ++j) row_sum += v_s0[j];
        #pragma unroll
        for (int j = 0; j < 8; ++j) row_sum += v_s1[j];
        row_sum = v142_cross_half_sum(row_sum);
        l_row += row_sum;

        bf16x8_t v_p0, v_p1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p0[j] = bf16_fast142(v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p1[j] = bf16_fast142(v_s1[j]);

        // ---- PV: chunked rescale interleaved with WMMAs (v131) ----
        // CHUNK D-tiles per group: rescale just this chunk's O fragments (VALU),
        // then issue the chunk's V loads + PV WMMAs. The next chunk's rescale
        // overlaps this chunk's WMMAs in the matrix-pipe shadow, and no single
        // VALU region touches more than CHUNK fragments -> tighter live ranges.
        // PV is matrix-pipe work just like QK^T. v131 held it at prio 1 on the
        // theory its 8-chain ILP lets it yield; v132 raises it to prio 2 (uniform
        // with QK^T). Rationale: within a wave QK^T and PV are temporally disjoint
        // (softmax island at prio 0 between them), so there is no intra-wave
        // contention to break; ACROSS resident waves, demoting PV to 1 only makes
        // this wave's PV WMMAs lose the shared slot to other waves' equal-value
        // matrix work, lengthening the PV critical path for no compensating gain.
        // Uniform prio 2 on all WMMAs lets HW round-robin fairly. Raised ONCE here
        // and held across the whole PV region -- no toggle inside the WMMA burst
        // (lane A's v136 proved per-chunk toggling regresses).
        __builtin_amdgcn_s_setprio(2);
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
    }

    // Epilogue is pure VALU + global store, no WMMA -> priority 0.
    __builtin_amdgcn_s_setprio(0);
    const fp32_t inv = (l_row > 0.0f) ? (1.0f / l_row) : 0.0f;
    bf16_t* o_row = Op + (q_m_base + col16) * stride_n;
    #pragma unroll
    for (int dt = 0; dt < DK; ++dt) {
        bf16x8_t o_pack;
        #pragma unroll
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast142(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
