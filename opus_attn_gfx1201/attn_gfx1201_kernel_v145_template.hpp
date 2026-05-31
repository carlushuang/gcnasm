// v145 -- v140 champion (93.38 TFLOPS) + the round-27 reviewer's exact NEXT
// FOCUS: split the prio1 tail at the UNTRIED interior boundary. Keep prio1 over
// [bulk exp2 + row-sum fold], but drop ONLY the bf16 P-pack back to prio0.
// ROUND-28 (lane B).
//
// ================= WHAT v140/v141 ESTABLISHED (the bracket) =================
// v140 (93.38, CHAMPION): raised the WHOLE softmax tail [16 bulk exp2 -> row-sum
//   fold -> bf16 P-pack] from v132's prio0 to the middle prio1. WON. The reviewer
//   root-caused it: the QK^T->PV gap was an issue-ARBITRATION loss on this wave's
//   TRANSCENDENTAL exp2 work; prio1 reaches PV sooner WITHOUT stealing matrix
//   slots from co-resident prio2 WMMAs (exp2 uses the transcendental pipe).
// v141: narrowed prio1 to ONLY the 16 exp2, dropping to prio0 BEFORE the row-sum
//   fold -> 93.1 REGRESSED. The lesson: the feeder chain must not fall to prio0
//   at the row-sum point; row-sum's fmadd chain + ds_bpermute is still on the hot
//   QK->PV path and benefits from staying unstarved.
//
// => v140 and v141 differ only in WHERE the prio1->prio0 drop lands. v140 keeps
// prio1 through P-pack (won); v141 drops it before row-sum (lost). The ONE
// interior split point NEITHER tried: drop AFTER row-sum, BEFORE the bf16 P-pack.
//
// ================= v145's HYPOTHESIS =======================================
// The bf16 P-pack (16 RNE convert ops: x += 0x7FFF + ((x>>16)&1); x>>16) is plain
// VALU/convert work on the SAME VALU pipe other resident waves' prio0 bookkeeping
// uses -- it is NOT transcendental and has no pipe other waves avoid. Holding it
// at prio1 (v140) makes THIS wave out-arbitrate other waves' useful VALU for the
// final pack slots; those waves may have ready matrix-feeder VALU. By contrast
// the exp2 (transcendental) and the row-sum fmadd reduction (the v141 lesson:
// dropping before it regressed) genuinely benefit from prio1. So the precise,
// minimal claim is: prio1 EARNS its keep through row-sum, but the P-pack tail is
// the v140 over-extension -- drop it to prio0 so other waves reclaim those VALU
// slots while THIS wave's real long poles (exp2, row-sum) already got the boost.
// P-pack still gates the first PV WMMA, but it is short (16 ops) and VALU-bound,
// not on a starved pipe; at prio0 it costs at most a few issue slots while
// returning VALU bandwidth to the WG. This is the SINGLE untried split between
// the v140-win and v141-loss brackets, and it isolates whether P-pack's prio1
// (the only difference from v141 that v141 lacked) helped or merely rode along.
//
// THE CHANGE (one added s_setprio(0), zero liveness):
//   * Raise to prio1 before the exp2 loops (v140 verbatim).
//   * KEEP prio1 across the 16 exp2 AND the row-sum fold + l_row += row_sum
//     (so the row-sum reduction stays unstarved -- the v141 regression boundary).
//   * DROP to prio0 immediately before the bf16 P-pack (v_p0/v_p1) -- the only
//     delta from v140.
//   * PV returns to prio2 (unchanged from v132/v140).
//
// Net policy: prio2 [K-preload..QK^T] and [PV]; prio1 [exp2 .. row-sum fold];
// prio0 [cross-lane row-max reduction], [bf16 P-pack], and [epilogue].
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
// v145 MUST be in the host dVT (pre-transposed V, layout [B,H,D,N]) allow-list;
// the kernel indexes V with vt_stride_d = k.N. Host registers v145 in BOTH the
// dispatch switch AND the dVT allow-list. (Omitting it -> max_abs=0.0360.)
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast145(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v145_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v145_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v145(opus_attn_kargs k)
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
            v_q[kt][j] = bf16_fast145(bf16_to_f32(v_q[kt][j]) * qscale);
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
        row_max = v145_cross_half_max(row_max);

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

        // Softmax tail, SPLIT at the v140/v141 interior boundary (v145).
        // The tail is [16 bulk exp2 transcendentals] -> [row-sum fmadd fold +
        // ds_bpermute] -> [bf16 P-pack]. All sit on the QK^T->PV critical path
        // (PV needs v_p = bf16(exp2(...))). v140 won by raising the WHOLE tail to
        // the middle prio1 (exp2 is on the transcendental pipe, which prio2 WMMA
        // waves do not contend for -> reaches PV sooner without stealing matrix
        // slots). v141 LOST by dropping to prio0 too early (before row-sum), so
        // the row-sum reduction fell back to starved prio0 -> 93.1.
        //
        // v145 keeps prio1 over EXACTLY [exp2 .. row-sum fold] (the part v141
        // proved must stay raised), then DROPS to prio0 for the bf16 P-pack only.
        // The P-pack is 16 plain RNE convert ops on the ordinary VALU pipe -- NOT
        // transcendental, no pipe other waves avoid. At prio1 (v140) it makes this
        // wave out-arbitrate other waves' useful matrix-feeder VALU for slots it
        // has no transcendental reason to win early. At prio0 those VALU slots
        // return to the WG while this wave's real long poles (exp2 + row-sum)
        // already got the boost. This is the SINGLE untried split between the
        // v140-win and v141-loss brackets.
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
        row_sum = v145_cross_half_sum(row_sum);
        l_row += row_sum;

        // DROP to prio0 for the bf16 P-pack: plain VALU/convert, not transcen-
        // dental -- the only delta from v140. Keeps prio1 ONLY where it earns its
        // keep (exp2 + row-sum, per the v141 lesson) and returns the final pack's
        // VALU slots to other resident waves.
        __builtin_amdgcn_s_setprio(0);

        bf16x8_t v_p0, v_p1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p0[j] = bf16_fast145(v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p1[j] = bf16_fast145(v_s1[j]);

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
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast145(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
