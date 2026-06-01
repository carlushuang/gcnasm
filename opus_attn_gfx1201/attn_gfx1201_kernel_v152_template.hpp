// v152 -- v140 champion (93.91 TFLOPS, CHUNK=2) + ROUND-35 reviewer NEXT-FOCUS:
// a MINIMAL distance-1 V prefetch ring in the PV loop that keeps only ONE extra
// V fragment pair live, instead of v151's whole-chunk load-then-compute burst.
// ROUND-35 (lane B).
//
// ================= WHY THIS, AND WHY IT IS GENUINELY NEW =====================
// v150 (CHUNK=4, 89.82) and v151 (CHUNK=2 chunk-load-then-compute, 92.26) both
// REGRESSED below v140's 93.91. The round-34 reviewer root-caused it precisely:
// front-loading 4 V fragments (v_v0[2]/v_v1[2]) ahead of the PV WMMAs builds a
// larger immediate VGPR/live-range + waitcnt island right before the matrix
// burst. The first PV WMMA still cannot consume v_v0[0] until the relevant vmcnt
// is satisfied, so the extra loads do NOT hide the oldest V-load latency; they
// only add transient V-fragment + address pressure in the tightest PV window.
// The verdict: v140's CHUNK=2 interleaved cadence (load->WMMA->load->WMMA) was
// better balanced because it kept V lifetimes SHORT and let the scheduler
// alternate memory-issue and matrix-issue.
//
// v152 takes the reviewer's #1 NEXT lever VERBATIM: keep v140's CHUNK=2 cadence
// and rescale grouping EXACTLY, but expose ONE D-tile of V-load latency by
// carrying a single distance-1 prefetch pair (v_v0_next/v_v1_next) across the
// whole PV loop -- the exact mirror of the v63 K double-buffer ring already
// proven in QK^T. While D-tile dt's two PV WMMAs execute, D-tile dt+1's V pair
// loads in the matrix-pipe shadow, removing the per-tile head-of-tile V-load
// s_waitcnt bubble that v140's pure interleave still leaves exposed on the FIRST
// load of each tile -- WITHOUT v151's 4-fragments-live burst.
//
// Liveness delta vs v140: EXACTLY one extra bf16x8 pair (v_v0_next/v_v1_next),
// identical to the K-ring footprint already resident in QK^T and well under
// v151's 2-fragment / v150's 4-fragment chunk arrays. This is the smallest
// possible V lookahead -- the reviewer's "keeps only one additional V fragment
// pair live" requirement, met literally.
//
// WHY THIS IS NOT v133 (the prior V-ring "dead end"): v133's ring sat on the
// OLDER v132 base (no v140 prio-1 softmax-tail split) AND used per-tile (CHUNK=1)
// rescale interleaved inside the ring. v152 keeps v140's WINNING schedule intact
// -- prio-1 softmax tail, CHUNK=2 deferred O-rescale grouping -- and only adds
// the one-pair ring. The combination (v140 schedule + 1-pair V ring + CHUNK=2
// rescale) has never been measured.
//
// Address-VGPR economy (inherited from v149, 93.83, bit-exact): the second V
// load of each tile is derived as (base + 16) so the +32B delta folds into the
// global_load immediate offset and reuses the first load's base-address VGPR
// pair -- keeps the prefetch's address lifetime minimal so the extra in-flight
// pair does not inflate the PV address-register peak.
//
// ================= WHY IT IS BIT-EXACT ====================================
// The ring only RELOCATES each V global_load earlier in program order; it loads
// the SAME bytes from the SAME addresses. Every O fragment is rescaled by the
// same scalar `rescale` in the same CHUNK=2 grouping as v140, and each v_o[dt]
// receives its two WMMAs (v_v0*v_p0 then v_v1*v_p1) in the identical operand
// order. No fp32 add/mul is reassociated; no softmax/reduction order changes.
// -> max_abs stays 0.0001, n_bad=0. Pure load-scheduling change.
//
// ================= WIRING NOTE (inherited from v126/v132/v140) =============
// v152 MUST be in the host dVT (pre-transposed V, layout [B,H,D,N]) allow-list;
// the kernel indexes V with vt_stride_d = k.N. Host registers v152 in BOTH the
// dispatch switch AND the dVT allow-list. (Omitting it -> max_abs=0.0360.)
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast152(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v152_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v152_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v152(opus_attn_kargs k)
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
            v_q[kt][j] = bf16_fast152(bf16_to_f32(v_q[kt][j]) * qscale);
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

        // QK^T highest priority (matrix-pipe arbitration win). Raise BEFORE the K
        // preloads so the FIRST, most-K-load-exposed QK^T WMMA also issues at prio2.
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

        // Softmax HEAD (cross-lane row-max reduce + new_m + l_row rescale) at prio0.
        __builtin_amdgcn_s_setprio(0);
        fp32_t row_max = v_s0[0];
        #pragma unroll
        for (int j = 1; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s1[j]);
        row_max = v152_cross_half_max(row_max);

        const fp32_t new_m = __builtin_fmaxf(m_row, row_max);
        fp32_t rescale = 1.0f;
        const bool need_rescale = (new_m != m_row);
        if (need_rescale) {
            rescale = __builtin_amdgcn_exp2f(m_row - new_m);
            l_row *= rescale;
        }
        m_row = new_m;

        // Softmax TAIL (16 bulk exp2 transcendentals + row-sum fold + bf16 P-pack)
        // at the v140-proven middle prio1: unstarves this wave's critical-path exp2
        // on the transcendental pipe without stealing matrix slots from other waves.
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
        row_sum = v152_cross_half_sum(row_sum);
        l_row += row_sum;

        bf16x8_t v_p0, v_p1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p0[j] = bf16_fast152(v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p1[j] = bf16_fast152(v_s1[j]);

        // ---- PV: v140 CHUNK=2 deferred-rescale cadence + MINIMAL distance-1 V
        //         prefetch ring (one extra V pair live) ----
        // Keep v140's grouping EXACTLY: per CHUNK=2 group, rescale this chunk's two
        // O fragments (VALU, in the matrix shadow), then issue per-tile WMMAs. The
        // ONLY change: carry a single distance-1 prefetch pair across the whole PV
        // loop -- while tile dt's two WMMAs execute, prefetch tile dt+1's V pair, so
        // the per-tile head-of-tile V-load bubble retires in the matrix-pipe shadow.
        // Exactly ONE extra bf16x8 pair live (mirror of the v63 K-ring), NOT v151's
        // 4-fragment burst. Second load derived as (base+16) -> +32B immediate offset
        // reuses the base-address VGPR (v149 trick), minimal address lifetime.
        __builtin_amdgcn_s_setprio(2);
        constexpr int CHUNK = 2;
        // Preload D-tile 0's V pair.
        const bf16_t* vt_pre = VTp + (0 * W_K + col16) * vt_stride_d + n_base + row8;
        bf16x8_t v_v0_next = *reinterpret_cast<const bf16x8_t*>(vt_pre);
        bf16x8_t v_v1_next = *reinterpret_cast<const bf16x8_t*>(vt_pre + 16);
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
                bf16x8_t v_v0 = v_v0_next;
                bf16x8_t v_v1 = v_v1_next;
                if (dt + 1 < DK) {
                    const bf16_t* vt_addr0 = VTp + ((dt + 1) * W_K + col16) * vt_stride_d + n_base + row8;
                    v_v0_next = *reinterpret_cast<const bf16x8_t*>(vt_addr0);
                    v_v1_next = *reinterpret_cast<const bf16x8_t*>(vt_addr0 + 16);
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
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast152(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
