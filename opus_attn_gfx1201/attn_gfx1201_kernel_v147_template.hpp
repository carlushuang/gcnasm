// v147 -- v140 champion schedule (93.63 TFLOPS) + ONE STRUCTURAL change:
// CROSS-KV-TILE K PREFETCH. Carry the NEXT KV tile's kt=0 K pair across the PV
// region so the first QK^T WMMA of every tile no longer stalls on a cold global
// K load. ROUND-30 (lane B). Off the saturated s_setprio space (v140..v146).
//
// ================= WHY THIS, AND WHY IT IS GENUINELY NEW =====================
// The s_setprio placement space around the softmax/PV boundary is EXHAUSTED:
// v140 (tail prio1, 93.63), v141 (exp2-only, 93.1), v142 (whole island, 92.9),
// v143/v144 (head splits), v145 (drop P-pack, 93.58), v146 (PV-rescale prio1,
// 93.26 FAIL). All are the SAME kernel body with the prio boundary nudged; they
// cluster at 92.2-93.63 and round 29 REGRESSED. The director's note is explicit:
// the only path from ~93 to the 93-95 HIP ceiling is STRUCTURAL, not more prio.
//
// The one structural latency that NO version (v122/v126/v132/v140..v146) hides:
// the INTER-TILE K-load bubble. v63's K software-pipeline (lines below) only
// hides INTRA-tile D-tile latency -- it prefetches kt+1 within a tile. But at the
// TOP of every n_tile iteration, v_k0_next/v_k1_next are loaded FRESH from global
// (the old line 160/161), and the very next instruction is the first QK^T WMMA
// that consumes them. Between tiles, that load is issued right after the previous
// tile's PV WMMAs retire, so its ~global-memory latency is fully EXPOSED on the
// critical path -- paid once per KV tile (128 times at N=4096).
//
// THE FIX (KV-block-granularity SW pipeline, playbook lever #4):
//   * Before the n_tile loop, PRIME v_k0_next/v_k1_next with tile 0's kt=0 K.
//   * Inside the loop the QK^T head consumes the carried v_k0_next/v_k1_next
//     immediately (no top-of-loop load).
//   * After the bf16 P-pack -- the exact point where v_s0/v_s1 (16 fp32 VGPRs)
//     have just DIED into v_p0/v_p1 -- issue the NEXT tile's kt=0 K load into
//     v_k0_next/v_k1_next. The whole PV region (16 WMMAs, ~the longest matrix
//     burst in the loop) then hides that global-load latency before the next
//     tile's first QK^T WMMA consumes it. Per playbook lever #4: pipeline at the
//     KV-BLOCK granularity, NOT the instruction granularity (v4's heavy manual
//     multi-stage prefetch regressed 2x; this is a single light distance-1 carry,
//     exactly like the v63 intra-tile ring that DID help).
//
// ================= WHY IT IS ZERO-OCCUPANCY-COST ===========================
// The carry adds NO peak VGPR liveness. The next-tile K pair (v_k0_next/v_k1_next,
// 8 VGPRs) is issued at the P-pack point where v_s0/v_s1 (16 VGPRs of fp32x8
// scores) have just been consumed into the bf16 v_p0/v_p1 and are dead. The K
// carry simply reuses freed registers. No new live range spans the PV region that
// was not already there in v140's intra-tile ring (which kept a v_k0_next/v_k1_next
// pair live too). -> stays at v140's 7-wave footprint, no BLOCK_N=64-style spill.
//
// ================= WHY IT IS BIT-EXACT ====================================
// Moving WHERE a global load is issued changes only scheduling/latency, never a
// value. v_k0_next/v_k1_next hold byte-identical K data (same addresses, same
// order) whether loaded at the top of this iteration (v140) or at the tail of the
// previous one (v147). Every WMMA operand, every exp2 input/output, every
// row_max/row_sum fold order, every bf16 pack is byte-identical to v140.
// -> max_abs stays 0.0001, n_bad=0. No fp32 add/mul is reassociated.
//
// The v140 softmax-tail prio1 schedule (the proven 93.63 champion priority
// policy) is preserved VERBATIM -- v147 only relocates a K load.
//
// ================= WIRING NOTE (inherited from v126/v132) ==================
// v147 MUST be in the host dVT (pre-transposed V, layout [B,H,D,N]) allow-list;
// the kernel indexes V with vt_stride_d = k.N. Host registers v147 in BOTH the
// dispatch switch AND the dVT allow-list. (Omitting it -> max_abs=0.0360.)
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast147(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v147_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v147_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v147(opus_attn_kargs k)
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
            v_q[kt][j] = bf16_fast147(bf16_to_f32(v_q[kt][j]) * qscale);
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

    // ---- CROSS-KV-TILE K PREFETCH (v147 structural change) ----
    // Prime the kt=0 K pair for tile 0 BEFORE the loop. Inside the loop, the QK^T
    // head consumes the carried pair immediately and the NEXT tile's kt=0 K is
    // re-loaded at the P-pack point (where v_s0/v_s1 just died), so its global
    // latency is hidden under the entire PV WMMA burst -- eliminating the
    // per-tile inter-tile K-load bubble that v63's intra-tile ring leaves exposed.
    bf16x8_t v_k0_next = *reinterpret_cast<const bf16x8_t*>(Kp + (0 + col16) * stride_n + row8);
    bf16x8_t v_k1_next = *reinterpret_cast<const bf16x8_t*>(Kp + (0 + 16 + col16) * stride_n + row8);

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

        // ---- QKT with software-pipelined K loads (v63 intra-tile ring) ----
        // v147: the kt=0 K pair is ALREADY in v_k0_next/v_k1_next -- either primed
        // before the loop (tile 0) or carried from the previous tile's P-pack
        // point (tiles 1..). No cold top-of-loop K load. The kt+1 D-tile prefetch
        // below still hides intra-tile latency.
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
        row_max = v147_cross_half_max(row_max);

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
        // v140 raises it to the UNTRIED middle level 1: above other waves' prio-0
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
        row_sum = v147_cross_half_sum(row_sum);
        l_row += row_sum;

        bf16x8_t v_p0, v_p1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p0[j] = bf16_fast147(v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p1[j] = bf16_fast147(v_s1[j]);

        // ---- CROSS-KV-TILE K PREFETCH (v147) ----
        // v_s0/v_s1 (16 fp32 VGPRs) have just died into v_p0/v_p1. Reuse those
        // freed registers to load the NEXT tile's kt=0 K pair NOW, so the global
        // latency hides under the entire PV WMMA burst that follows. The next QK^T
        // head (top of the following iteration) then finds K already resident.
        // Same addresses/order as a top-of-loop load -> byte-identical, bit-exact.
        if (n_tile + 1 < num_kv_tiles) {
            const int n_next = (n_tile + 1) * BLOCK_N;
            v_k0_next = *reinterpret_cast<const bf16x8_t*>(Kp + (n_next + col16) * stride_n + row8);
            v_k1_next = *reinterpret_cast<const bf16x8_t*>(Kp + (n_next + 16 + col16) * stride_n + row8);
        }

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
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast147(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
