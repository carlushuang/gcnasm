// v148 -- v140 champion (93.69 TFLOPS) + the ONE untried PV-pressure REDUCTION:
// CHUNK = 1 (per-fragment deferred rescale interleave), the round-30 reviewer's
// explicit NEXT lever. ROUND-31 (lane B).
//
// ================= WHY THIS, AND WHY IT IS GENUINELY NEW =====================
// The round-30 review root-caused the v147 regression (93.63 -> 88.46): the PV
// region is the kernel's VGPR / issue-window PRESSURE PEAK. v_o[8], v_q[8],
// v_p0/v_p1, the transient V fragments, the rescale temporaries and loop/index
// state are all live around the PV WMMAs. EVERY structural probe that ADDED
// state into that region regressed, bit-exact:
//   v147 (carry next-tile K pair across PV)        -> 88.46
//   v133 (steady-state V ring, +1 V pair live)     -> regressed
//   v137 per-tile / v138 CHUNK=2 / v139 dt0 early-V -> 86.8 / 87.9 / 92.2
// The reviewer's conclusion: "the measured bottleneck is occupancy / issue-window
// pressure" and the correct direction is to REDUCE PV live pressure, not add to
// it. Explicit lever #3: "Test CHUNK=1 ... if performance recovers it confirms
// VGPR/issue-window pressure as the limiter."
//
// v140 (and all of v132..v146) defers the per-tile O rescale in CHUNK=2 groups:
// it multiplies TWO O fragments (16 FMAs) before issuing that 2-tile group's V
// loads + 4 PV WMMAs. CHUNK=2 keeps the just-rescaled fragment v_o[c0+1] live
// across its chunk-mate's two WMMAs before it is consumed -> a 2-fragment-wide
// hot rescale window. v148 sets CHUNK=1: each O fragment's 8 rescale FMAs are
// emitted IMMEDIATELY before that fragment's own two PV WMMAs, so the rescaled
// fragment is consumed by its WMMAs on the next issue slots instead of waiting
// for a chunk-mate. This is the SHORTEST post-rescale O-fragment live range and
// the finest-grained VALU-in-WMMA-shadow cadence (one 8-FMA group per WMMA pair,
// 8 uniform groups instead of 4 doubled ones) -> the smoothest issue stream and
// the LOWEST simultaneous rescale-temporary pressure in exactly the region the
// reviewer identified as the limiter.
//
// ================= WHY THIS IS NOT v113 =====================================
// v113 also set CHUNK=1, but on the v111 base which was mis-wired to the WRONG V
// buffer (dV not dVT), giving the spurious max_abs=0.036 the v126 round later
// root-caused as a WIRING bug, not an FMA-contraction effect. v148 sits on the
// v140 champion schedule (correct dVT wiring, split-priority s_setprio softmax
// island, hoisted prio2 raises) -- none of which existed at v113. CHUNK is an
// ORTHOGONAL knob to all the s_setprio work that lifted v132->v140; it has never
// been swept on top of the winning schedule. The PV loop is still fully
// #pragma unroll-ed, so matrix-pipe ILP (all 16 WMMAs visible) is UNCHANGED;
// CHUNK only controls where the 8 rescale FMA groups land in the WMMA stream.
//
// ================= WHY IT IS BIT-EXACT ======================================
// CHUNK only regroups the deferred rescale loop; it does NOT reorder any fp32
// reduction. Per O fragment dt the op sequence is IDENTICAL for any CHUNK:
//     v_o[dt][j] *= rescale;  (8 independent scalar multiplies)
//     v_o[dt] = wmma(v_v0, v_p0, v_o[dt]);
//     v_o[dt] = wmma(v_v1, v_p1, v_o[dt]);
// Fragments are independent accumulators; a scalar multiply cannot FMA-contract
// with the WMMA intrinsic (the intrinsic is an opaque matrix op), so no fp32 add
// or mul is reassociated relative to v140. Every QK^T, softmax max/sum fold,
// exp2, P-pack and PV operand order is byte-identical to v140 -> max_abs stays
// 0.0001, n_bad=0 (TRUE bit-exact, unlike the mis-wired v113).
//
// ================= WIRING NOTE (inherited from v140) ========================
// v148 MUST be in the host dVT (pre-transposed V, layout [B,H,D,N]) allow-list;
// the kernel indexes V with vt_stride_d = k.N. Host registers v148 in BOTH the
// dispatch switch AND the dVT allow-list. (Omitting it -> max_abs=0.0360.)
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast148(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v148_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v148_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v148(opus_attn_kargs k)
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
            v_q[kt][j] = bf16_fast148(bf16_to_f32(v_q[kt][j]) * qscale);
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
        // l_row rescale. Short, genuinely yieldable, and GATES the exp2 below ->
        // keep at prio 0 so other waves' ready matrix work wins these slots (this
        // wave is not yet on its exp2 critical path here). (No WMMA adjacent ->
        // honors "never lower priority next to a WMMA burst".)
        __builtin_amdgcn_s_setprio(0);
        fp32_t row_max = v_s0[0];
        #pragma unroll
        for (int j = 1; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s1[j]);
        row_max = v148_cross_half_max(row_max);

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
        // P-pack. These sit on the direct QK^T->PV critical path (PV needs v_p =
        // bf16(exp2(...))). exp2 issues to the TRANSCENDENTAL pipe, not the matrix
        // pipe, so it co-executes in the WMMA issue shadow (playbook lever 2/6).
        // v132 ran this at prio 0, where it loses issue arbitration to every
        // co-resident wave's prio-2 matrix work -- a pure arbitration loss that
        // stretches this wave's QK->PV gap with no compensating matrix benefit.
        // v148 raises it to the UNTRIED middle level 1: above other waves' prio-0
        // bookkeeping (so this wave's critical-path exp2 is no longer starved),
        // but still BELOW prio-2 matrix work (so it yields the matrix issue slot
        // to other waves' QK^T/PV -- avoiding v134's mistake of stealing matrix
        // slots with feeder VALU at prio 2). exp2 needs the transcendental pipe,
        // which those WMMAs do not contend for.
        __builtin_amdgcn_s_setprio(1);

        #pragma unroll
        for (int j = 0; j < 8; ++j) v_s0[j] = __builtin_amdgcn_exp2f(v_s0[j] - new_m);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_s1[j] = __builtin_amdgcn_exp2f(v_s1[j] - new_m);

        fp32_t row_sum = v_s0[0];
        #pragma unroll
        for (int j = 1; j < 8; ++j) row_sum += v_s0[j];
        #pragma unroll
        for (int j = 0; j < 8; ++j) row_sum += v_s1[j];
        row_sum = v148_cross_half_sum(row_sum);
        l_row += row_sum;

        bf16x8_t v_p0, v_p1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p0[j] = bf16_fast148(v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p1[j] = bf16_fast148(v_s1[j]);

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
        constexpr int CHUNK = 1;
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
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast148(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
