// v138 -- v132 champion (92.78) + the ROUND-20 reviewer's NEXT-FOCUS lever:
// a CHUNK=2 EARLY-V-LOAD PV schedule. ROUND-21 (lane B).
//
// ================= WHAT CHANGED vs v132 (the 92.78 champion) ===============
// v132's PV loop, per CHUNK=2 group, does:
//     [rescale 2 O frags]  then  per dt: [load v0,v1] [WMMA v0] [WMMA v1]
// i.e. inside the chunk each tile interleaves its own V load right before its two
// WMMAs. The round-20 reviewer's NEXT-FOCUS (after v137's per-TILE load-before-
// rescale regressed to 86.76) was explicit:
//
//   "Test CHUNK=2 with early V loads hoisted only immediately before that chunk's
//    rescale, not per tile. Preserves v132's two-tile burst structure while giving
//    limited latency cover. Avoids eight separate VMEM->VALU->WMMA transitions."
//
// v138 implements EXACTLY that. Per CHUNK=2 group it now does:
//     [load all 4 V frags for the chunk]       <- VMEM burst, issued FIRST
//     [rescale this chunk's 2 O frags]          <- VALU cover for the loads
//     [WMMA v0,v1 dt0] [WMMA v0,v1 dt1]         <- compact 4-WMMA matrix burst
//
// WHY THIS IS DIFFERENT FROM v137 (which FAILED at 86.76):
//   * v137 hoisted the V load per-TILE (granularity 1): every one of the 8 tiles
//     became its own VMEM->VALU->WMMA transition, fragmenting the prio-2 matrix
//     issue window EIGHT times and stretching each V pair's live range across the
//     8 rescale FMAs (reviewer root-cause).
//   * v138 hoists per-CHUNK (granularity 2): only 4 transitions total. Within each
//     chunk the 4 V loads issue as one MLP burst, the 2 rescale FMA pairs act as a
//     SHORT latency cover, then the 4 PV WMMAs issue back-to-back as a COMPACT
//     uninterrupted matrix burst -- v132's two-tile burst structure is preserved,
//     not broken into singletons. The 4-fragment chunk V live set is exactly what
//     v123 (grouped loads-then-WMMAs) already proved compiles bit-exactly with NO
//     occupancy cliff (only one chunk's V live at a time).
//
// Hypothesis: the small head-of-chunk V-load latency that priority alone could not
// hide (v134/v135/v136 confirmed priority placement is saturated) is now covered
// by the chunk's own rescale FMAs, WITHOUT v137's window fragmentation, because the
// cover happens once per 2 tiles instead of once per tile.
//
// EXPECTED: primary 92.8-93.5 TFLOPS, max_abs 0.0001, n_bad=0. RISK: low-med.
// Worst case the rescale FMAs are too few to fully cover the load and it ties v132;
// the chunk live set is below v123's proven-safe footprint so no occupancy cliff.
//
// ================= UNIFORM prio-2 POLICY (inherited from v132) =============
// prio 2 spans [K-preload .. end of QK^T] and [all of PV]; prio 0 spans the
// softmax island and the epilogue store. The PV prio-2 raise is issued ONCE before
// the chunk loop and held across the whole region -- no toggle inside the burst
// (v136 proved per-chunk toggling regresses). The early V loads sit inside the
// prio-2 region, same as v132's in-tile loads.
//
// ================= WHY IT IS BIT-EXACT ====================================
// No fp32 add/mul is reordered. Per O accumulator v_o[dt], the rescale multiply
// still happens exactly ONCE before that tile's first PV WMMA, and the two PV
// WMMAs still accumulate v0 then v1 in the same order as v132/v126. Moving the V
// *loads* earlier moves no arithmetic (a load is not an fp op); moving the rescale
// to AFTER the chunk's loads but still BEFORE that chunk's WMMAs preserves the
// per-fragment "rescale then accumulate" order. exp2/row_max/row_sum/bf16-pack are
// byte-identical to v132. -> max_abs stays 0.0001.
//
// ================= ROOT-CAUSE / WIRING NOTE (inherited from v126) ==========
// v138 MUST be in the host dVT (pre-transposed V, layout [B,H,D,N]) allow-list;
// the kernel indexes V with vt_stride_d = k.N. Host registers v138 in BOTH the
// dispatch switch AND the dVT allow-list.
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast100(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v138_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v138_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v138(opus_attn_kargs k)
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
        // v138: raise BEFORE the K preloads (v131 raised it after), so the FIRST
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
        row_max = v138_cross_half_max(row_max);

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
        row_sum = v138_cross_half_sum(row_sum);
        l_row += row_sum;

        bf16x8_t v_p0, v_p1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p0[j] = bf16_fast100(v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p1[j] = bf16_fast100(v_s1[j]);

        // ---- PV: CHUNK=2 EARLY-V-LOAD schedule (v138, round-20 NEXT-FOCUS) ----
        // Per chunk of CHUNK D-tiles:
        //   (1) issue ALL of the chunk's V loads FIRST (a 4-load VMEM/MLP burst),
        //   (2) rescale just this chunk's O fragments (VALU) -- this short FMA work
        //       acts as latency cover for the V loads issued in step (1),
        //   (3) issue the chunk's PV WMMAs back-to-back as one COMPACT matrix burst.
        // vs v132 this only moves the V loads from step (3) up to step (1); the
        // rescale stays between the loads and the WMMAs. vs the FAILED v137 this is
        // per-CHUNK (2 tiles) not per-TILE (1 tile), so the WMMA window is broken
        // into 4 compact bursts not 8 singletons, and only one chunk's 4 V frags
        // are live at a time (v123's proven-safe footprint -> no occupancy cliff).
        // Uniform prio 2 raised ONCE here and held across the whole PV region -- no
        // toggle inside the WMMA burst (v136 proved per-chunk toggling regresses).
        __builtin_amdgcn_s_setprio(2);
        constexpr int CHUNK = 2;
        #pragma unroll
        for (int c0 = 0; c0 < DK; c0 += CHUNK) {
            // (1) early V-load burst for the whole chunk
            bf16x8_t vv0[CHUNK];
            bf16x8_t vv1[CHUNK];
            #pragma unroll
            for (int dc = 0; dc < CHUNK; ++dc) {
                const int dt = c0 + dc;
                vv0[dc] = *reinterpret_cast<const bf16x8_t*>(VTp + (dt * W_K + col16) * vt_stride_d + n_base + row8);
                vv1[dc] = *reinterpret_cast<const bf16x8_t*>(VTp + (dt * W_K + col16) * vt_stride_d + n_base + 16 + row8);
            }
            // (2) rescale this chunk's O fragments (VALU cover for the loads)
            if (need_rescale) {
                #pragma unroll
                for (int dc = 0; dc < CHUNK; ++dc) {
                    const int dt = c0 + dc;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) v_o[dt][j] *= rescale;
                }
            }
            // (3) compact PV WMMA burst (same v0-then-v1 accumulation order)
            #pragma unroll
            for (int dc = 0; dc < CHUNK; ++dc) {
                const int dt = c0 + dc;
                v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(vv0[dc], v_p0, v_o[dt]);
                v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(vv1[dc], v_p1, v_o[dt]);
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
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast100(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
