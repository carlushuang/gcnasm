// v137 -- v132 champion (92.78 TFLOPS, uniform prio-2 s_setprio schedule) with
// the round-20 reviewer's mandated NEXT-FOCUS lever applied in its minimal,
// strictly liveness-neutral form: ISSUE EACH PV TILE'S V GLOBAL LOAD *BEFORE*
// THAT TILE'S O-ACCUMULATOR RESCALE, keeping the two PV WMMAs after the rescale.
// ROUND-20 (lane B).
//
// ================= WHAT CHANGED vs v132 (the 92.78 champion) ================
// The round-20 reviewer named three follow-on levers after v136's prio-toggle
// regression. Lever #1 ("revert per-chunk prio0 windows, return PV to v132's
// uninterrupted prio2 region") is honored verbatim -- v137 keeps v132's single
// uninterrupted prio-2 PV region with NO extra s_setprio toggles. v137 then adds
// ONLY lever #3, the one that was never tested on the fast v132 schedule:
//
//   "Test moving V loads for chunk c BEFORE that chunk's rescale while keeping
//    WMMA after rescale. This may overlap V-load latency under the rescale FMAs
//    without adding VGPR-heavy V rings or extra priority toggles."
//
// THE TRAP every prior V lever fell into (v127->83.94, v133->71, v124 FAIL) was
// adding REGISTER LIVENESS: a double-buffer snapshot (cv0/cv1) or a single-buffer
// prefetch ring (v_v*_next) holds 1-2 EXTRA V pairs live across the PV WMMAs, on
// top of the dominant eight fp32x8 O accumulators (64 fp32/lane). On gfx1201 the
// occupancy wall is razor-thin: even +1 V pair (~8 VGPR) crossed the 7-wave cliff
// and lost ~20 TFLOPS. So this lever MUST be implemented with ZERO added liveness.
//
// v137 does exactly that. It collapses v132's CHUNK=2 "rescale-group then
// WMMA-group" into a PER-TILE pipeline:
//     for dt in 0..DK:
//         v_v0 = load(addr0); v_v1 = load(addr1);   // <-- load FIRST (latency starts)
//         if need_rescale: v_o[dt][j] *= rescale;    // <-- 8 FMAs hide the load
//         v_o[dt] = wmma(v_v0, v_p0, v_o[dt]);       // <-- consume V after rescale
//         v_o[dt] = wmma(v_v1, v_p1, v_o[dt]);
// At every instant ONLY ONE tile's V (v_v0,v_v1 = 2 fragments, 16 VGPR) is live --
// IDENTICAL to v132's peak V live-set inside its WMMA sub-loop. No ring, no
// snapshot, no next-tile prefetch buffer => v137's VGPR footprint == v132's
// (still 7 waves/SIMD, no occupancy-cliff risk). The ONLY structural difference
// is the intra-tile instruction ORDER: the V load is issued before the rescale
// FMAs instead of after them, so the ~L2-latency of the V fetch is covered by the
// 8 dependent rescale multiplies (pure VALU, no V dependency) rather than being
// exposed right in front of the first PV WMMA.
//
// This also makes the first PV WMMA fire EARLIER than v132, not later: v132's
// CHUNK=2 rescales BOTH tiles' O fragments before issuing ANY V load; v137 issues
// tile-0's V load and rescales only tile-0 before tile-0's WMMA. This preserves
// (indeed sharpens) the early-PV-start that v135's monolithic rescale destroyed
// (round-19: monolithic = -1 TFLOPS dead end), while v128's per-tile rescale was
// confounded by its added V ring -- v137 isolates the per-tile rescale WITHOUT
// the ring, the clean experiment v128 could not be.
//
// ================= WHY IT IS BIT-EXACT ====================================
// Each of the eight O accumulators v_o[dt] is an INDEPENDENT fp32x8 fragment.
// v132 multiplies each fragment by the scalar `rescale` exactly once (in CHUNK=2
// groups); v137 multiplies each fragment by the SAME scalar `rescale` exactly
// once (per tile, in ascending dt order). Per fragment the operation is bit-
// identical: one `*= rescale` over the same 8 fp32 lanes. The two PV WMMA calls
// per tile keep the exact (v_v0 then v_v1) operand order and accumulate into the
// same v_o[dt]. No fp32 add/mul is reordered or reassociated anywhere; the QK^T
// ring, softmax cross-half folds, exp2 order, row_sum order, and bf16 RNE pack
// are byte-identical to v132. -> max_abs stays 0.0001, n_bad=0.
//
// ----- (original v132 design notes retained below) -----
// v132 -- v131's asymmetric wave-priority schedule, REFINED into a UNIFORM
// "every WMMA region at prio 2, every pure-VALU region at prio 0" policy, with
// the QK^T prio-2 raise HOISTED ABOVE the K preloads. ROUND-15 (lane B).
//
// ================= WHAT CHANGED vs v131 (the 92.75 champion) ===============
// v131 proved asymmetric s_setprio is real headroom (92.75 vs v126 91.53, all
// bit-exact). The round-14 reviewer named the two cheapest follow-on levers, and
// v132 applies BOTH in one coherent change:
//
//   (A) HOIST the QK^T prio raise ABOVE the K preloads (reviewer lever 3).
//       In v131 `s_setprio(2)` sat AFTER the two initial v_k0_next/v_k1_next
//       global loads. s_setprio only takes effect for instructions issued AFTER
//       it, so the FIRST QK^T WMMA pair -- the one most exposed to K-load latency
//       and the most valuable to win the shared matrix-pipe slot -- entered issue
//       arbitration at the DEFAULT priority, not prio 2. v132 issues s_setprio(2)
//       at the very top of the KV-loop body, so the K preloads AND every QK^T
//       WMMA from the first iteration onward run at prio 2.
//
//   (B) RAISE PV from prio 1 to prio 2 (reviewer lever 2: "PV may be under-
//       prioritized"). v131's theory was that PV's 8 independent O chains have
//       enough ILP to yield the slot, so it ran at the timid prio 1. But the
//       softmax island and epilogue ALREADY drop to prio 0, so within ONE wave
//       there is no contention to arbitrate between QK^T(2) and PV(1) -- they are
//       temporally disjoint. The only thing prio 1 does is make THIS wave's PV
//       WMMAs lose the issue slot to ANOTHER resident wave's QK^T (prio 2) AND to
//       its PV is now equal... actually at prio 1 a PV WMMA also loses to another
//       wave's prio-2 PV. Across 7-8 resident waves running the same kernel, the
//       steady state is many PV WMMAs ready at once; demoting them all to prio 1
//       just lengthens the PV critical path with no compensating QK^T win (QK^T
//       and PV of different waves overlap, but both regions are matrix-pipe work
//       of equal value -- there is no reason to systematically favor one wave's
//       QK^T over another's PV). Uniform prio 2 on ALL matrix work lets the HW
//       round-robin fairly while still keeping pure-VALU softmax/epilogue at
//       prio 0 so they never steal a slot from ANY ready WMMA. This is the
//       simplest policy that honors the proven lane-A rule "never lower priority
//       adjacent to a WMMA burst" -- prio drops to 0 only in the WMMA-free island.
//
// Net policy: prio 2 spans [K-preload .. end of QK^T] and [all of PV];
//            prio 0 spans [softmax island] and [epilogue store]. One fewer
//            distinct priority level than v131, and the raise covers the first
//            QK^T WMMA that v131 left at default priority.
//
// ================= WHY IT IS BIT-EXACT ====================================
// __builtin_amdgcn_s_setprio(N) emits a single `s_setprio N` SALU instruction
// that ONLY changes this wave's hardware issue-arbitration priority. It moves no
// data, performs no arithmetic, and does not reassociate any fp32 add/mul. Every
// v_o[dt][j], every row_max/row_sum fold, every exp2, every WMMA operand order
// is byte-identical to v131/v126. -> max_abs stays 0.0001. Only the PRIORITY
// VALUES and the PLACEMENT of the prio-2 raise changed; no arithmetic moved.
//
// ================= ROOT-CAUSE / WIRING NOTE (inherited from v126) ==========
// v132 MUST be in the host dVT (pre-transposed V, layout [B,H,D,N]) allow-list;
// the kernel indexes V with vt_stride_d = k.N. v111-v125 silently fell through to
// the non-transposed dV buffer (allow-list ended at v110) -> max_abs=0.0360. The
// kernel was always correct; only the wiring was wrong. Host registers v132 in
// BOTH the dispatch switch AND the dVT allow-list.
//
// ================= THE FAST SCHEDULE (inherited from v126/v111) ============
// v100's single `for dt: for j: v_o[dt][j] *= rescale` block touches ALL eight
// fp32x8 O accumulators (64 fp32/lane) in one VALU region. v131 splits the
// rescale into chunks of CHUNK=2 D-tiles, each chunk's rescale emitted
// immediately before that chunk's PV WMMAs, so only 2 O fragments are hot per
// VALU region. Same scalar `rescale`, same multiply order per fragment.
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast100(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v137_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v137_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v137(opus_attn_kargs k)
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
            v_q[kt][j] = bf16_fast100(bf16_to_f32(v_q[kt][j]) * qscale);
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
        row_max = v137_cross_half_max(row_max);

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

        #pragma unroll
        for (int j = 0; j < 8; ++j) v_s0[j] = __builtin_amdgcn_exp2f(v_s0[j] - new_m);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_s1[j] = __builtin_amdgcn_exp2f(v_s1[j] - new_m);

        fp32_t row_sum = v_s0[0];
        #pragma unroll
        for (int j = 1; j < 8; ++j) row_sum += v_s0[j];
        #pragma unroll
        for (int j = 0; j < 8; ++j) row_sum += v_s1[j];
        row_sum = v137_cross_half_sum(row_sum);
        l_row += row_sum;

        bf16x8_t v_p0, v_p1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p0[j] = bf16_fast100(v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p1[j] = bf16_fast100(v_s1[j]);

        // ---- PV: per-tile LOAD-BEFORE-RESCALE schedule (v137) ----
        // v132 used CHUNK=2 "rescale-group then WMMA-group". v137 collapses that to
        // a per-tile pipeline that issues each tile's V global LOAD first, then the
        // tile's O-accumulator rescale (8 pure-VALU FMAs that DO NOT depend on V),
        // then the two PV WMMAs that consume V. The 8 rescale FMAs cover the V-load
        // latency, so the V fetch is no longer exposed right in front of the first
        // PV WMMA -- WITHOUT holding any extra V fragment live (no ring/snapshot),
        // so the VGPR footprint == v132's (only v_v0/v_v1 of the CURRENT tile are
        // live at any instant). This is round-20 reviewer lever #3, the version of
        // the V-latency lever that adds zero liveness (v127/v133's rings regressed).
        //
        // Uniform prio 2 across the entire PV region (v132 policy / reviewer lever
        // #1: keep PV's prio-2 window uninterrupted, no per-chunk toggles). Raised
        // ONCE here and held -- no s_setprio inside the WMMA burst.
        __builtin_amdgcn_s_setprio(2);
        #pragma unroll
        for (int dt = 0; dt < DK; ++dt) {
            const bf16_t* vt_addr0 = VTp + (dt * W_K + col16) * vt_stride_d + n_base + row8;
            const bf16_t* vt_addr1 = VTp + (dt * W_K + col16) * vt_stride_d + n_base + 16 + row8;
            bf16x8_t v_v0 = *reinterpret_cast<const bf16x8_t*>(vt_addr0);
            bf16x8_t v_v1 = *reinterpret_cast<const bf16x8_t*>(vt_addr1);
            if (need_rescale) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) v_o[dt][j] *= rescale;
            }
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
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast100(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
