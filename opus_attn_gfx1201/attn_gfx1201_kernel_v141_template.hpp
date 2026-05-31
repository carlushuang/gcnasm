// v141 -- v140 champion (93.38 TFLOPS) NARROWED: tighten the prio-1 window to
// EXACTLY the 16 bulk exp2 transcendentals, then drop back to prio0 for the
// row-sum fold + bf16 P-pack. ROUND-24 (lane B), reviewer NEXT-FOCUS.
//
// ================= WHY THIS, AND WHY IT IS GENUINELY NEW =====================
// v140 won (93.38 > 92.78*1.005) by splitting v132's uniform-prio0 softmax island:
// it raised the WHOLE tail [exp2 loops .. row-sum fold .. P-pack] from prio0 to
// the middle prio1, leaving cross-lane reductions at prio0 and PV at prio2. The
// reviewer root-caused the win precisely: the QK^T->PV gap was an ISSUE-ARBITRA-
// TION loss on this wave's TRANSCENDENTAL work (the 16 exp2), not a matrix or
// memory scheduling problem. exp2 issues to the TRANSCENDENTAL pipe, which the
// co-resident prio2 WMMA waves do NOT contend for, so raising it to prio1 reached
// PV sooner without stealing matrix slots (avoiding v134's prio2-feeder mistake).
//
// BUT v140 over-scoped the raise. Only the exp2 needs the transcendental pipe and
// is the genuine critical-path item. The row-sum fold (+ ds_bpermute) and the
// bf16 P-pack are ORDINARY VALU/convert ops on the SAME VALU pipe other waves'
// prio0 bookkeeping uses. Holding them at prio1 makes THIS wave out-arbitrate
// other waves' useful VALU for slots it has no transcendental reason to win early
// -- a small reverse-starvation that likely gives back part of v140's headroom.
//
// THE FIX (surgical, minimal scope of v140's proven lever):
//   * Cross-lane row-max reduction + new_m + l_row rescale stay prio0 (gate exp2;
//     unchanged from v132/v140).
//   * Raise ONLY the two bulk exp2 loops to prio1 -- the exact transcendental-
//     pipe item the reviewer identified as the arbitration-starved critical path.
//   * DROP to prio0 immediately after the exp2 loops, for the row-sum fold and
//     P-pack: plain VALU, not transcendental -> no reason to keep out-arbitrating
//     other waves' VALU. (P-pack still gates the first PV WMMA, but it is short
//     and VALU-bound, not transcendental-bound; letting it run at prio0 returns
//     those VALU slots to other waves while exp2 -- the real long pole -- already
//     got its prio1 boost.)
//   * PV returns to prio2 (unchanged from v132/v140).
//
// Net policy: prio2 [K-preload..QK^T] and [PV]; prio1 [exp2 ONLY]; prio0 [cross-
// lane reductions], [row-sum fold + P-pack], and [epilogue]. This is v140 minus
// the over-extension of prio1 onto non-transcendental VALU -- the round-24
// reviewer's exact NEXT lever ("prio1 only over the bulk exp2 loops, then restore
// prio0 for row-sum/P-pack before the existing PV prio2"). No version tried it.
//
// ================= WHY IT IS BIT-EXACT ====================================
// __builtin_amdgcn_s_setprio(N) emits a single `s_setprio N` SALU instruction
// that ONLY changes this wave's issue-arbitration priority. It moves no data,
// does no arithmetic, reassociates no fp32 add/mul. Every exp2 input/output,
// every row_max/row_sum fold order, every bf16 pack, every WMMA operand order is
// byte-identical to v132. -> max_abs stays 0.0001, n_bad=0. Only priority VALUES
// changed; no instruction that produces a value moved relative to another.
//
// ================= WIRING NOTE (inherited from v126/v132) ==================
// v141 MUST be in the host dVT (pre-transposed V, layout [B,H,D,N]) allow-list;
// the kernel indexes V with vt_stride_d = k.N. Host registers v141 in BOTH the
// dispatch switch AND the dVT allow-list. (Omitting it -> max_abs=0.0360.)
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast141(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v141_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v141_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v141(opus_attn_kargs k)
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
            v_q[kt][j] = bf16_fast141(bf16_to_f32(v_q[kt][j]) * qscale);
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
        row_max = v141_cross_half_max(row_max);

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

        // Softmax TAIL, NARROWED (v141, round-24 reviewer NEXT-FOCUS):
        // v140 proved (93.38 TFLOPS, +0.6 over the 92.78 champion) that the
        // QK^T->PV gap is an ISSUE-ARBITRATION loss on this wave's TRANSCENDENTAL
        // work, not a matrix/memory scheduling problem: raising the whole softmax
        // tail [exp2..row-sum..P-pack] from v132's prio0 to the middle prio1
        // unstarved it WITHOUT stealing matrix slots from co-resident prio2 WMMAs.
        //
        // BUT v140 over-scoped the raise. The ONLY work that needs the transcen-
        // dental pipe -- the 16 bulk exp2 -- is the real critical-path item. The
        // row-sum fold (+ ds_bpermute) and bf16 P-pack are ORDINARY VALU/convert
        // ops on the SAME VALU pipe the other waves' prio0 bookkeeping uses. Held
        // at prio1 they out-arbitrate those other waves' useful VALU for slots
        // they don't actually need to win early -- a small reverse-starvation that
        // gives back part of v140's gain. v141 tightens the prio1 window to EXACTLY
        // the two exp2 loops, then DROPS to prio0 for row-sum + P-pack:
        //   prio1  -> only the 16 exp2 (the genuine transcendental-pipe item).
        //   prio0  -> row-sum fold + P-pack (plain VALU; let other waves' ready
        //             matrix/bookkeeping win those slots -- P-pack still gates the
        //             first PV WMMA but is short and not transcendental-bound).
        // This is the minimal, surgically-scoped form of v140's proven lever.
        __builtin_amdgcn_s_setprio(1);

        #pragma unroll
        for (int j = 0; j < 8; ++j) v_s0[j] = __builtin_amdgcn_exp2f(v_s0[j] - new_m);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_s1[j] = __builtin_amdgcn_exp2f(v_s1[j] - new_m);

        // Drop back to prio0: row-sum fold + P-pack are plain VALU, not transcen-
        // dental -- no reason to keep out-arbitrating other waves' VALU here.
        __builtin_amdgcn_s_setprio(0);

        fp32_t row_sum = v_s0[0];
        #pragma unroll
        for (int j = 1; j < 8; ++j) row_sum += v_s0[j];
        #pragma unroll
        for (int j = 0; j < 8; ++j) row_sum += v_s1[j];
        row_sum = v141_cross_half_sum(row_sum);
        l_row += row_sum;

        bf16x8_t v_p0, v_p1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p0[j] = bf16_fast141(v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p1[j] = bf16_fast141(v_s1[j]);

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
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast141(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
