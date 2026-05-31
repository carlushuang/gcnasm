// v139 -- v132 champion (92.78 TFLOPS, uniform prio-2 s_setprio schedule) with
// the ROUND-21 reviewer's explicit, repeated NEXT-FOCUS lever applied in its
// strictly-minimal form: a SINGLE-TILE (dt0-only) V preload at the head of the
// PV loop, with EVERY OTHER tile's schedule byte-identical to v132. ROUND-22
// (lane B).
//
// ================= WHY THIS, AND WHY MINIMAL =================================
// v132 is pinned at the 7-wave occupancy cliff (~205 VGPR). The entire v133-v138
// family proved that ANY change adding steady-state V liveness drops occupancy to
// 6 waves and loses 5-9 TFLOPS:
//   v133 (single-buffer V ring):  regressed (1 extra V pair live ALL of PV).
//   v127 (CHUNK=2 double-buffer):  83.94 (4 extra V frags live).
//   v137 (per-TILE load-before-rescale, all 4 tiles):  86.76.
//   v138 (CHUNK=2 early-load, 4 frags hoisted per chunk):  87.87.
// The round-21 reviewer's diagnosis: v137/v138 confounded TWO things -- (a) does
// a head-of-PV V-load bubble actually exist that rescale FMAs could cover, and
// (b) the cost of carrying many V frags across the rescale VALU block. They both
// hoisted V for MULTIPLE tiles, so the added liveness/burst lead-in swamped any
// cover. The reviewer's mandated probe isolates (a) with the smallest possible
// (b):
//
//   "Try a narrower 'first tile only' head-of-chunk preload: load dt0 before
//    rescale, keep dt1 as v132. ... this tests it with only 2 V fragments live
//    and avoids the 4-fragment chunk live range plus full VMEM burst lead-in."
//   "NEXT FOCUS: Restore v132 PV ordering and test only a minimal first-tile
//    preload variant, not full chunk V hoisting."
//
// ================= EXACTLY WHAT CHANGED vs v132 =============================
// v132's PV loop, per CHUNK=2 group c0 in {0,2,4,6}:
//     if(need_rescale) rescale frags c0, c0+1
//     for dc in {0,1}: dt=c0+dc; load V[dt]; WMMA(v0); WMMA(v1)
//
// v139 issues ONLY tile 0's V pair (v_v0_0, v_v1_0) ONCE, at the very top of the
// PV region -- BEFORE chunk 0's rescale FMAs -- so its global-load latency hides
// under the 2-fragment rescale VALU block (and under the bf16 P-pack that just
// preceded it). Then in the loop, tile 0 consumes the preloaded pair; tiles
// 1..7 load their V exactly as v132 (inside the loop, immediately before their
// WMMAs). NOTHING else changes:
//   * Only ONE tile is hoisted (not 4 like v138, not all like v137).
//   * The hoisted pair is consumed at the FIRST chunk, so it is live only for the
//     short window [top-of-PV .. dt0 WMMAs] -- it does not span the whole PV loop
//     (unlike v133's ring) and does not coexist with later tiles' V frags
//     (unlike v127/v138). Expected register delta: ~0 net (2 frags briefly live,
//     freed immediately) -> stays at v132's 7-wave footprint, no occupancy cliff.
//   * Tiles 1..7 keep v132's exact load-then-WMMA cadence.
//
// If a head-of-PV bubble exists, this recovers ~1/4 of it (tile 0 only) at zero
// occupancy cost -> small primary gain. If no bubble exists, it is perf-neutral
// and bit-exact (a safe probe that conclusively closes the early-V-load line for
// lane B without risking the champion's occupancy).
//
// ================= WHY IT IS BIT-EXACT ====================================
// Reordering WHEN a global load is *issued* changes neither the value loaded nor
// any arithmetic. Tile 0's WMMA operands (v_v0_0, v_v1_0, v_p0, v_p1) are the
// SAME values in the SAME order; the v_o[0] accumulation chain is unchanged. The
// rescale order (per-fragment, scalar `rescale`), the QK^T chains, the softmax
// reductions, exp2, and bf16 packs are all byte-identical to v132. No fp32 add or
// mul is reassociated. -> max_abs stays 0.0001, n_bad=0.
//
// ================= WIRING NOTE (inherited from v126/v132) ==================
// v139 MUST be in the host dVT (pre-transposed V, layout [B,H,D,N]) allow-list;
// the kernel indexes V with vt_stride_d = k.N. Host registers v139 in BOTH the
// dispatch switch AND the dVT allow-list.
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast139(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v139_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v139_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v139(opus_attn_kargs k)
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
            v_q[kt][j] = bf16_fast139(bf16_to_f32(v_q[kt][j]) * qscale);
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

        // QK^T at prio 2 (raise hoisted above K preloads, as in v132).
        __builtin_amdgcn_s_setprio(2);

        // ---- QKT with software-pipelined K loads (v63) ----
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

        // Softmax island = pure VALU -> prio 0 (byte-identical to v132).
        __builtin_amdgcn_s_setprio(0);
        fp32_t row_max = v_s0[0];
        #pragma unroll
        for (int j = 1; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s1[j]);
        row_max = v139_cross_half_max(row_max);

        const fp32_t new_m = __builtin_fmaxf(m_row, row_max);
        fp32_t rescale = 1.0f;
        const bool need_rescale = (new_m != m_row);
        if (need_rescale) {
            rescale = __builtin_amdgcn_exp2f(m_row - new_m);
            l_row *= rescale;
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
        row_sum = v139_cross_half_sum(row_sum);
        l_row += row_sum;

        bf16x8_t v_p0, v_p1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p0[j] = bf16_fast139(v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p1[j] = bf16_fast139(v_s1[j]);

        // ---- PV: v132's chunked rescale+WMMA, with ONE minimal change ----
        // MINIMAL FIRST-TILE PRELOAD (round-21 reviewer NEXT-FOCUS): issue ONLY
        // tile 0's V pair here, at the top of the PV region, BEFORE chunk 0's
        // rescale FMAs. Its global-load latency hides under those 2-fragment
        // rescale FMAs (and under the bf16 P-pack just above). Tiles 1..7 keep
        // v132's exact in-loop load-then-WMMA cadence. The preloaded pair is
        // consumed at the first chunk, so it is live only briefly -> v132's
        // 7-wave footprint is preserved (no occupancy cliff). Bit-exact: only the
        // ISSUE TIME of tile 0's load moved; no value or arithmetic order changed.
        __builtin_amdgcn_s_setprio(2);
        bf16x8_t v_v0_0 = *reinterpret_cast<const bf16x8_t*>(VTp + (0 * W_K + col16) * vt_stride_d + n_base + row8);
        bf16x8_t v_v1_0 = *reinterpret_cast<const bf16x8_t*>(VTp + (0 * W_K + col16) * vt_stride_d + n_base + 16 + row8);
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
                bf16x8_t v_v0, v_v1;
                if (dt == 0) {
                    v_v0 = v_v0_0;
                    v_v1 = v_v1_0;
                } else {
                    v_v0 = *reinterpret_cast<const bf16x8_t*>(VTp + (dt * W_K + col16) * vt_stride_d + n_base + row8);
                    v_v1 = *reinterpret_cast<const bf16x8_t*>(VTp + (dt * W_K + col16) * vt_stride_d + n_base + 16 + row8);
                }
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
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast139(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
