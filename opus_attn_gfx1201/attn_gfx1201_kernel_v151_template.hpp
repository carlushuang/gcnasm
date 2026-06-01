// v151 -- v140 champion (93.91 TFLOPS, CHUNK=2) + the ROUND-33 reviewer's exact
// NEXT-FOCUS: keep CHUNK=2 but restructure the chunk body to LOAD-THEN-COMPUTE --
// issue BOTH D-tiles' V load pairs first (4 b128 global_loads), THEN the four PV
// WMMAs. ROUND-34 (lane B). Identical v140 softmax-priority schedule retained.
//
// ================= WHY THIS, AND WHY IT IS GENUINELY NEW =====================
// Round 33's v150 swept CHUNK 2->4 and REGRESSED (89.82 vs v140's 93.91). The
// reviewer's root-cause: CHUNK=4 over-coarsened PV -- it doubled the contiguous
// O-rescale VALU island before the first WMMA (delaying matrix issue under the
// prio-2 setprio) and inflated live V-fragment/address state, while the monotone
// CHUNK trend (1<2) did NOT extend to 4. CHUNK=2 is the proven sweet spot.
//
// But v140's CHUNK=2 body interleaves load->WMMA->load->WMMA per D-tile, so at
// most ONE D-tile's V pair (2 b128 loads) is exposed ahead of its consuming WMMA
// pair before the next load is reached in program order. The actual limiter the
// round-32/33 reviewers named is V-load-to-WMMA latency in the PV region, NOT
// rescale granularity. v151 attacks that limiter directly WITHOUT v150's long
// rescale island:
//   * Same CHUNK=2 -> same short 2-fragment rescale block, same tight live range.
//   * Within the chunk, split the body: a LOAD phase issues all 4 b128 V loads
//     for both D-tiles into a small v_v0[2]/v_v1[2] register array, then a COMPUTE
//     phase issues the four PV WMMAs. dt=c0's two WMMAs now overlap dt=c0+1's
//     in-flight global loads (and the load phase exposes 4 V loads ahead of the
//     WMMA chain) -> matches CHUNK=4's memory-level parallelism (4 loads in flight)
//     over only 2 fragments of rescale + V liveness, avoiding v150's burst cost.
//
// This is distinct from the failed "early-V" family (v137/138/139): those moved V
// loads OUT of / before the PV loop entirely or only for dt0; v151 keeps every V
// load inside its own chunk (no cross-chunk hoist, no added cross-tile liveness)
// and only reorders load-vs-WMMA WITHIN the 2-tile chunk. Added liveness peak is
// 2 extra V pairs (the chunk's second tile loaded before the first tile's WMMAs
// retire) -- 4 transient bf16x8 regs, far under the 7-wave/197-VGPR budget v140
// sat at, nowhere near the BLOCK_N=64 spill cliff.
//
// ================= WHY IT IS BIT-EXACT ====================================
// Only the ORDER of independent global loads vs WMMAs within a chunk changes; the
// V bytes loaded are identical (vt_addr0+16 is the same address as v140's
// vt_addr1), each v_o[dt] is rescaled by the identical `rescale` scalar and
// accumulated by the identical pair of WMMAs in the identical order (dt ascending,
// v0 then v1). No fp32 add/mul is reassociated, no softmax reduction order moves,
// no bf16 pack changes, no WMMA operand order changes. Every per-fragment op
// sequence is byte-identical to v140 -> max_abs stays 0.0001, n_bad=0.
//
// ================= WIRING NOTE (inherited from v126/v132) ==================
// v151 MUST be in the host dVT (pre-transposed V, layout [B,H,D,N]) allow-list;
// the kernel indexes V with vt_stride_d = k.N. Host registers v151 in BOTH the
// dispatch switch AND the dVT allow-list. (Omitting it -> max_abs=0.0360.)
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast140(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v151_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v151_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v151(opus_attn_kargs k)
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
            v_q[kt][j] = bf16_fast140(bf16_to_f32(v_q[kt][j]) * qscale);
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
        row_max = v151_cross_half_max(row_max);

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
        // v151 raises it to the UNTRIED middle level 1: above other waves' prio-0
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
        row_sum = v151_cross_half_sum(row_sum);
        l_row += row_sum;

        bf16x8_t v_p0, v_p1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p0[j] = bf16_fast140(v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p1[j] = bf16_fast140(v_s1[j]);

        // ---- PV: CHUNK=2, load-then-compute V prefetch within each chunk (v151) -
        // ROUND-33 reviewer's NEXT-FOCUS: CHUNK=4 (v150, 89.82) over-coarsened the
        // PV region -- doubling the contiguous rescale-VALU block before the first
        // WMMA delayed matrix issue and inflated live V-fragment state, and the
        // monotone CHUNK trend (1<2) did NOT extend to 4. CHUNK=2 (v140) is the
        // sweet spot. v151 keeps CHUNK=2 but restructures the chunk body to target
        // V-load latency DIRECTLY without v150's long rescale island: issue BOTH
        // D-tiles' V load PAIRS first (4 b128 global_loads), THEN the four PV
        // WMMAs. The two WMMAs of dt=c0 now overlap the in-flight global loads of
        // dt=c0+1 (and vice versa) -- 4 V loads exposed ahead of the WMMA chain,
        // matching CHUNK=4's memory-level parallelism but over only 2 fragments of
        // O-rescale VALU and 2 fragments of V-register liveness (well under v150's
        // 4). Narrower live range than CHUNK=4; deeper load lookahead than v140's
        // interleaved load->WMMA->load->WMMA cadence.
        //
        // PV is matrix-pipe work like QK^T -> uniform prio 2 (v132 lesson; lane A's
        // v136 proved per-chunk toggling regresses). Raised ONCE, held across PV.
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
            // Prefetch this chunk's V load pairs FIRST, then issue the WMMAs.
            bf16x8_t v_v0[CHUNK], v_v1[CHUNK];
            #pragma unroll
            for (int dc = 0; dc < CHUNK; ++dc) {
                const int dt = c0 + dc;
                const bf16_t* vt_addr0 = VTp + (dt * W_K + col16) * vt_stride_d + n_base + row8;
                v_v0[dc] = *reinterpret_cast<const bf16x8_t*>(vt_addr0);
                v_v1[dc] = *reinterpret_cast<const bf16x8_t*>(vt_addr0 + 16);
            }
            #pragma unroll
            for (int dc = 0; dc < CHUNK; ++dc) {
                const int dt = c0 + dc;
                v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_v0[dc], v_p0, v_o[dt]);
                v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_v1[dc], v_p1, v_o[dt]);
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
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast140(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
