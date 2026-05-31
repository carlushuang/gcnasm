// v110 -- v106 (frontier, ~90.2 TFLOPS) + MINIMAL TWO-FRAGMENT V PREFETCH.
//
// Base choice: v106 (the running best). QK^T, softmax, the prologue pair-0 rescale,
// and the one-pair-ahead rescale schedule are kept BYTE-FOR-BYTE identical to v106.
// The ONLY change is in the PV WMMA loop's V-load scheduling.
//
// Why v109 failed (round 10) and why this is different:
//   v109 carried a FULL 1-pair ring -- all FOUR next V fragments (cur_va0/va1/vb0/vb1
//   + nxt_*) live across the WMMA quartet. That +4 live bf16x8 V-fragment pressure on
//   top of v_o[8], v_q[DK], and the packed probabilities pushed the kernel over an
//   occupancy/allocation cliff (90.2 -> 85.12). v109 also left the final-iteration
//   `cur = nxt` assignment reading uninitialized `nxt_*` (dead, but UB-shaped codegen).
//
// v110's middle ground (reviewer-sanctioned next lever):
//   Carry ONLY the "a" chain's two fragments (cur_va0, cur_va1 -- the operands of the
//   v_o[dt] accumulator chain) ONE pair-iteration ahead. The "b" chain's fragments
//   (vb0, vb1, feeding v_o[dt1]) are loaded JUST-IN-TIME inside the iteration, exactly
//   as v106 loads them. Net steady-state liveness is +2 V fragments over v106 (the one
//   carried pair), HALF of v109's +4 -- well short of the cliff.
//
//   The carried cur_va0/cur_va1 are issued a full pair-iteration before they are
//   consumed: their load happens at the END of iteration dt-2 (the `nxt_va*` prefetch),
//   so it is in flight across iteration dt-2's rescale FMAs AND its 4 PV WMMAs before
//   iteration dt's first WMMA needs it. That is real load-use distance for the "a"
//   chain WITHOUT a 4-wide ring. The "b" chain still consumes JIT loads, but its WMMAs
//   are issued AFTER the "a" chain's two WMMAs, giving them natural in-quartet spacing.
//
//   The next-pair rescale (v106's schedule) is left exactly where v106 puts it -- in the
//   shadow of the in-flight prefetch loads -- so the rescale FMAs still overlap memory.
//
// Correctness: the WMMA operand order and rescale-once-per-accumulator semantics are
//   IDENTICAL to v106 (cur_va0==v106 va0, vb0==vb0, cur_va1==va1, vb1==vb1; same v_p0/
//   v_p1 pairing; same accumulator chain). -> bit-exact in bf16. The ring advance is
//   GUARDED by `if (dt + 2 < DK)` so the final iteration performs NO uninitialized
//   assignment (fixes the v109 UB-shaped tail). DK=8 even -> exact pairing, no remainder.
//
// RISK: low. If the compiler already scheduled v106's "a"-chain loads early, v110 ties
//   v106; the +2 liveness is small enough that an occupancy regression is unlikely.
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast110(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v110_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v110_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v110(opus_attn_kargs k)
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
            v_q[kt][j] = bf16_fast110(bf16_to_f32(v_q[kt][j]) * qscale);
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

        // ---- QK^T: BLOCK_N=32 -> two 16x16x16 WMMAs per D-tile.
        // TWO-AHEAD software-pipelined K loads (v101/v103/v106, byte-for-byte identical).
        fp32x8_t v_s0 = {0,0,0,0,0,0,0,0};
        fp32x8_t v_s1 = {0,0,0,0,0,0,0,0};

        const bf16_t* kbase0 = Kp + (n_base + col16) * stride_n + row8;
        const bf16_t* kbase1 = Kp + (n_base + 16 + col16) * stride_n + row8;
        bf16x8_t kr0[2];
        bf16x8_t kr1[2];
        kr0[0] = *reinterpret_cast<const bf16x8_t*>(kbase0);
        kr1[0] = *reinterpret_cast<const bf16x8_t*>(kbase1);
        if (DK > 1) {
            kr0[1] = *reinterpret_cast<const bf16x8_t*>(kbase0 + W_K);
            kr1[1] = *reinterpret_cast<const bf16x8_t*>(kbase1 + W_K);
        }
        #pragma unroll
        for (int kt = 0; kt < DK; ++kt) {
            const int cur = kt & 1;
            bf16x8_t v_k0 = kr0[cur];
            bf16x8_t v_k1 = kr1[cur];
            if (kt + 2 < DK) {
                kr0[cur] = *reinterpret_cast<const bf16x8_t*>(kbase0 + (kt+2) * W_K);
                kr1[cur] = *reinterpret_cast<const bf16x8_t*>(kbase1 + (kt+2) * W_K);
            }
            v_s0 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_k0, v_q[kt], v_s0);
            v_s1 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_k1, v_q[kt], v_s1);
        }

        // Softmax over 16 values (8 from s0, 8 from s1)
        fp32_t row_max = v_s0[0];
        #pragma unroll
        for (int j = 1; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s1[j]);
        row_max = v110_cross_half_max(row_max);

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
        row_sum = v110_cross_half_sum(row_sum);
        l_row += row_sum;

        bf16x8_t v_p0, v_p1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p0[j] = bf16_fast110(v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p1[j] = bf16_fast110(v_s1[j]);

        // ---- PV: v106's 2-D-tile grouped PV (two independent accumulator chains per
        // pair) + v106's one-pair-ahead rescale schedule, with a MINIMAL TWO-FRAGMENT
        // V prefetch on the "a" chain only. The "a" fragments (cur_va0/cur_va1, operands
        // of v_o[dt]) are carried one pair-iteration ahead; the "b" fragments (vb0/vb1,
        // operands of v_o[dt1]) are loaded just-in-time as in v106. +2 live V fragments
        // vs v106, half of v109's +4. WMMA operand order identical to v106 -> bit-exact.
        if (need_rescale) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_o[0][j] *= rescale;
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_o[1][j] *= rescale;
        }
        // Preload pair 0's "a" fragments (operands of the v_o[0] chain).
        bf16x8_t cur_va0 = *reinterpret_cast<const bf16x8_t*>(VTp + (0 * W_K + col16) * vt_stride_d + n_base + row8);
        bf16x8_t cur_va1 = *reinterpret_cast<const bf16x8_t*>(VTp + (0 * W_K + col16) * vt_stride_d + n_base + 16 + row8);
        #pragma unroll
        for (int dt = 0; dt < DK; dt += 2) {
            const int dt1 = dt + 1;
            // JIT-load the current pair's "b" fragments (operands of v_o[dt1]).
            bf16x8_t vb0 = *reinterpret_cast<const bf16x8_t*>(VTp + (dt1 * W_K + col16) * vt_stride_d + n_base + row8);
            bf16x8_t vb1 = *reinterpret_cast<const bf16x8_t*>(VTp + (dt1 * W_K + col16) * vt_stride_d + n_base + 16 + row8);
            // TWO-FRAGMENT PREFETCH: issue the NEXT pair's "a" fragments now, so they are
            // in flight across this iteration's rescale FMAs and 4 WMMAs.
            bf16x8_t nxt_va0, nxt_va1;
            if (dt + 2 < DK) {
                const int np = dt + 2;
                nxt_va0 = *reinterpret_cast<const bf16x8_t*>(VTp + (np * W_K + col16) * vt_stride_d + n_base + row8);
                nxt_va1 = *reinterpret_cast<const bf16x8_t*>(VTp + (np * W_K + col16) * vt_stride_d + n_base + 16 + row8);
            }
            // v106 rescale schedule, byte-for-byte: next pair's accumulators in the shadow
            // of the in-flight V loads. Independent regs -> independent of the WMMAs below.
            if (dt + 2 < DK && need_rescale) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) v_o[dt + 2][j] *= rescale;
                #pragma unroll
                for (int j = 0; j < 8; ++j) v_o[dt + 3][j] *= rescale;
            }
            // Current pair WMMAs. "a" operands prefetched a pair ago; "b" operands JIT.
            // Operand order identical to v106 (va0/p0, vb0/p0, va1/p1, vb1/p1).
            v_o[dt]  = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(cur_va0, v_p0, v_o[dt]);
            v_o[dt1] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(vb0,     v_p0, v_o[dt1]);
            v_o[dt]  = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(cur_va1, v_p1, v_o[dt]);
            v_o[dt1] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(vb1,     v_p1, v_o[dt1]);
            // Advance the 2-fragment ring ONLY when a next pair exists -> no dead
            // uninitialized assignment on the final iteration (fixes v109's UB tail).
            if (dt + 2 < DK) {
                cur_va0 = nxt_va0;
                cur_va1 = nxt_va1;
            }
        }
    }

    const fp32_t inv = (l_row > 0.0f) ? (1.0f / l_row) : 0.0f;
    bf16_t* o_row = Op + (q_m_base + col16) * stride_n;
    #pragma unroll
    for (int dt = 0; dt < DK; ++dt) {
        bf16x8_t o_pack;
        #pragma unroll
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast110(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
