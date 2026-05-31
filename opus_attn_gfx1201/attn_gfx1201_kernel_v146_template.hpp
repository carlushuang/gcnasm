// v146 -- v145 champion (93.58 TFLOPS) + the round-28 reviewer's exact NEXT
// FOCUS: split the PV region so the deferred per-chunk v_o rescale runs at the
// middle prio1 (NOT prio0), then raise to prio2 only immediately before that
// chunk's V loads + PV WMMAs. ROUND-29 (lane B).
//
// ================= WHAT IS ESTABLISHED (the bracket) =======================
// Current PV region (v132..v145): ONE s_setprio(2) before the whole CHUNK loop,
// held across BOTH the deferred v_o rescale FMAs (pure feeder VALU) AND the PV
// WMMAs. So the rescale runs at prio2 -- it out-arbitrates co-resident waves'
// ready matrix-feeder VALU (and even their WMMAs) for slots it has no matrix
// reason to win early, while THIS wave's PV WMMAs are gated on those same rescale
// FMAs producing v_o anyway.
//
// v136 already tried per-chunk PV toggling but used PRIO0 for the rescale (demote
// to 0, raise to 2 around each chunk's FMAs) -> REGRESSED. The v136 lesson is the
// SAME one v141 taught on the softmax tail: dropping a dependent feeder all the
// way to prio0 STARVES it, and its starvation directly delays the dependent WMMA
// (PV needs the rescaled v_o). prio0 was too low.
//
// v140/v145 then PROVED the resolution on the softmax tail: the middle prio1 is
// the sweet spot for dependent feeder VALU -- unstarved enough to feed its WMMA
// on time, but NOT so high it steals matrix slots from other waves. That exact
// principle was never applied to the PV rescale: the only two points tried there
// are prio2 (current, over-prioritized) and prio0 (v136, starved). prio1 is the
// untried middle, and it is precisely the value v145 just confirmed wins.
//
// ================= v146's HYPOTHESIS =======================================
// The deferred v_o rescale (`v_o[dt][j] *= rescale`, CHUNK=2 -> 16 FMAs per chunk
// block) is pure feeder VALU that GATES this chunk's PV WMMAs. Running it at
// prio2 (current) makes this wave out-arbitrate other resident waves' equal- or
// higher-value matrix work for slots the rescale doesn't need to win that early;
// running it at prio0 (v136) starved it and pushed back the dependent PV WMMA.
// prio1 is the v140/v145-proven middle: the rescale stays unstarved (feeds v_o
// for its own PV in time) yet yields matrix-pipe arbitration to co-resident
// waves' ready WMMAs. Net: the rescale no longer over-competes, PV WMMAs still
// issue at prio2, and other waves reclaim the slots the prio2 rescale stole.
//
// THE CHANGE (per-chunk, ONLY when need_rescale; zero liveness, zero arithmetic):
//   * Enter the PV loop at prio2 (unchanged).
//   * Inside each CHUNK block, IF need_rescale: drop to prio1 around ONLY the
//     rescale FMAs, then raise back to prio2 immediately before this chunk's V
//     loads + PV WMMAs. The toggle pair straddles ONLY the pure-VALU rescale --
//     never punctuates a WMMA burst (lane A's v136 proved mid-burst toggling
//     regresses; here the WMMAs stay contiguously at prio2).
//   * When need_rescale is false NO toggle is emitted -- those (common) tiles are
//     byte- and schedule-identical to v145's PV region.
//   * This differs from v136 ONLY in the rescale priority value: 1 (v146) vs 0
//     (v136) -- the v140/v145-confirmed correction to v136's over-demotion.
//
// Net policy: prio2 [K-preload..QK^T] and [PV V-loads+WMMAs]; prio1 [softmax
// exp2..row-sum fold] AND [per-chunk PV v_o rescale]; prio0 [row-max reduce],
// [bf16 P-pack], [epilogue].
//
// ================= WHY IT IS BIT-EXACT ====================================
// __builtin_amdgcn_s_setprio(N) emits a single `s_setprio N` SALU instruction
// that ONLY changes this wave's issue-arbitration priority. It moves no data,
// does no arithmetic, reassociates no fp32 add/mul. Every exp2 input/output,
// every row_max/row_sum fold order, every v_o rescale FMA order, every bf16 pack,
// every WMMA operand order is byte-identical to v145/v132. -> max_abs stays
// 0.0001, n_bad=0. Only priority VALUES changed; no value-producing instruction
// moved relative to another.
//
// ================= WIRING NOTE (inherited from v126/v132) ==================
// v146 MUST be in the host dVT (pre-transposed V, layout [B,H,D,N]) allow-list;
// the kernel indexes V with vt_stride_d = k.N. Host registers v146 in BOTH the
// dispatch switch AND the dVT allow-list. (Omitting it -> max_abs=0.0360.)
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast146(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v146_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v146_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v146(opus_attn_kargs k)
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
            v_q[kt][j] = bf16_fast146(bf16_to_f32(v_q[kt][j]) * qscale);
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
        row_max = v146_cross_half_max(row_max);

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
        row_sum = v146_cross_half_sum(row_sum);
        l_row += row_sum;

        // DROP to prio0 for the bf16 P-pack: plain VALU/convert, not transcen-
        // dental -- the only delta from v140. Keeps prio1 ONLY where it earns its
        // keep (exp2 + row-sum, per the v141 lesson) and returns the final pack's
        // VALU slots to other resident waves.
        __builtin_amdgcn_s_setprio(0);

        bf16x8_t v_p0, v_p1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p0[j] = bf16_fast146(v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p1[j] = bf16_fast146(v_s1[j]);

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
                // v146: the deferred v_o rescale is pure feeder VALU that GATES
                // this chunk's PV WMMAs. Drop to the v140/v145-proven middle prio1
                // (NOT v136's prio0, which starved it) so it stays fed for its own
                // PV yet yields matrix arbitration to co-resident waves' WMMAs,
                // then raise back to prio2 before the V loads + WMMAs. Toggle pair
                // straddles ONLY the pure-VALU rescale -- never a WMMA burst.
                __builtin_amdgcn_s_setprio(1);
                #pragma unroll
                for (int dc = 0; dc < CHUNK; ++dc) {
                    const int dt = c0 + dc;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) v_o[dt][j] *= rescale;
                }
                __builtin_amdgcn_s_setprio(2);
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
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast146(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
