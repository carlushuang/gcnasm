// v143 -- v140 champion (93.38 TFLOPS) + ONE zero-liveness scheduling change:
// move the prio-1 window's START one step EARLIER than v140 -- from the bulk
// exp2 tail up to new_m / the conditional HEAD rescale exp2f -- while KEEPING
// the latency-bound row-max ds_bpermute reduction at prio 0. The asymmetric
// intermediate window between v140 (tail-only prio1) and v142 (whole-island
// prio1, REGRESSED to 92.9). ROUND-26 (lane B), per round-25 reviewer NEXT-FOCUS.
//
// ================= WHAT v140/v141/v142 ESTABLISHED ==========================
// v140 (93.38, champion): row-max reduce + new_m + head exp2f at prio0; TAIL
//   (16 bulk exp2 + row-sum + P-pack) raised to the middle prio1. WON.
// v141: narrowed prio1 to ONLY the 16 bulk exp2, dropped row-sum/P-pack back to
//   prio0 -> 93.1 REGRESSED. Lesson: the QK->PV feeder chain must NOT fall to
//   prio0 between the matrix bursts.
// v142: raised the ENTIRE island to prio1, including the row-max reduction
//   -> 92.9 REGRESSED. Round-25 reviewer root-cause: the row-max chain
//   (horizontal fmax + one ds_bpermute) is a tight latency-bound dependency
//   with no ILP; promoting it adds scheduler weight without exposing ready
//   work, and makes its feeders less yieldable during the inter-wave phase ->
//   net loss. "head feeders should remain prio0 OR be promoted only at a
//   narrower point; tail feeders need prio1."
//
// ================= v143's NARROW INTERMEDIATE WINDOW ========================
// The reviewer's explicit NEXT-FOCUS: restore v140, then move the prio1
// transition slightly EARLIER -- after the row-max reduction (kept prio0) and
// BEFORE the head rescale handling (new_m / need_rescale exp2f / l_row scale).
// This isolates the ONE head op that plausibly benefits: the conditional
// transcendental exp2f(m_row - new_m), which is on the SAME arbitration-starved
// transcendental pipe v140 targeted for the tail, AND gates new_m's consumer
// chain. v140 left it at prio0 where other waves' prio-2 matrix work can defer
// it before new_m feeds the tail exp2; v142 over-corrected by also promoting the
// useless-to-promote row-max reduction. v143 promotes from new_m onward only.
//
// Net policy: prio2 [K-preload..QK^T] and [PV]; prio0 [row-max reduction] and
// [epilogue]; prio1 [new_m / head rescale exp2f .. bulk exp2 .. row-sum ..
// P-pack] -- one contiguous window that starts after the row-max fold. No
// version has run exactly this boundary (v140 starts later, v142 starts earlier).
//
// ================= WHY IT IS BIT-EXACT ====================================
// __builtin_amdgcn_s_setprio(N) emits a single `s_setprio N` SALU instruction
// that ONLY changes this wave's issue-arbitration priority. It moves no data,
// does no arithmetic, reassociates no fp32 add/mul. Every exp2 input/output,
// every row_max/row_sum fold order, every bf16 pack, every WMMA operand order is
// byte-identical to v140/v132. -> max_abs stays 0.0001, n_bad=0. Only the
// POSITION of one s_setprio(1) moved (earlier by ~3 scalar ops); no instruction
// that produces a value moved relative to another.
//
// ================= WIRING NOTE (inherited from v126/v132/v140) =============
// v143 MUST be in the host dVT (pre-transposed V, layout [B,H,D,N]) allow-list;
// the kernel indexes V with vt_stride_d = k.N. Host registers v143 in BOTH the
// dispatch switch AND the dVT allow-list. (Omitting it -> max_abs=0.0360.)
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast143(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v143_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v143_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v143(opus_attn_kargs k)
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
            v_q[kt][j] = bf16_fast143(bf16_to_f32(v_q[kt][j]) * qscale);
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

        // Softmax ROW-MAX REDUCTION = horizontal fmax over the 16 scores +
        // cross-half ds_bpermute fold. THIS sub-region stays at prio 0. It is a
        // tight dependent chain (each fmax feeds the next) with NO ILP to expose,
        // it issues NO transcendental, and the v140 reviewer measured that raising
        // it (v142, whole-island prio1) REGRESSED to 92.9 -- promoting this short
        // unyieldable chain only makes its feeder ops less yieldable to other
        // resident waves' ready matrix/load work without reaching PV any sooner.
        // The single ds_bpermute here is genuinely latency-bound, not arbitration-
        // bound, so winning an early issue slot for it buys nothing. Keep prio0.
        __builtin_amdgcn_s_setprio(0);
        fp32_t row_max = v_s0[0];
        #pragma unroll
        for (int j = 1; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s1[j]);
        row_max = v143_cross_half_max(row_max);

        // ---- prio1 window STARTS HERE (one step earlier than v140) ----
        // v140 won by raising the TAIL (16 bulk exp2 + row-sum + P-pack) to the
        // middle prio 1; v141 proved the QK->PV feeder chain must not fall to
        // prio0; v142 raised the WHOLE island incl. the row-max reduction and
        // REGRESSED (92.9 < 93.38). The round-25 reviewer root-caused v142: the
        // row-max reduction is a tight latency-bound chain that gains nothing from
        // prio1, and over-claims issue slots. The untried INTERMEDIATE window is:
        // keep the row-max reduction at prio0 (above), but raise prio1 EARLIER
        // than v140 -- starting at new_m / the conditional HEAD rescale exp2f /
        // l_row scale, NOT just the bulk tail. Rationale: new_m gates every tail
        // exp2, and the head exp2f(m_row-new_m) is a TRANSCENDENTAL on the exact
        // arbitration-starved pipe v140 identified. At v140's prio0 the head
        // exp2f can be deferred by other waves' prio-2 matrix work BEFORE new_m
        // is consumed, stretching the gate the prio1 tail then waits on. Raising
        // from new_m onward (but NOT the row-max reduction) is the asymmetric
        // sweet spot: transcendental + dependent feeders prio1, latency-bound
        // reduction prio0. No version has run exactly this split.
        __builtin_amdgcn_s_setprio(1);
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
        // P-pack. Already running at prio 1 (raised once above, right after the
        // row-max reduction). v143 keeps a SINGLE contiguous prio-1 window from
        // new_m / head-rescale through the tail P-pack -- no setprio toggle is
        // needed here (v140 re-raised to 1 at this point because its head ran at
        // prio 0). The window ends only at the prio-2 PV burst below.
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_s0[j] = __builtin_amdgcn_exp2f(v_s0[j] - new_m);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_s1[j] = __builtin_amdgcn_exp2f(v_s1[j] - new_m);

        fp32_t row_sum = v_s0[0];
        #pragma unroll
        for (int j = 1; j < 8; ++j) row_sum += v_s0[j];
        #pragma unroll
        for (int j = 0; j < 8; ++j) row_sum += v_s1[j];
        row_sum = v143_cross_half_sum(row_sum);
        l_row += row_sum;

        bf16x8_t v_p0, v_p1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p0[j] = bf16_fast143(v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p1[j] = bf16_fast143(v_s1[j]);

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
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast143(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
