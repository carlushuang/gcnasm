// v131 -- v122 champion (CHUNK=2 PV rescale) + DEPTH-2 A->B separation with
//          PHASED V loads (the reviewer's CHUNK=3 "A0 A1 A2 B0 B1 B2" idea,
//          but load-phased so peak live V fragments are LOWER than v130, not higher).
//
// PROVENANCE: v122 (91.68, this lane's champion) issues PV per D-tile as
//   rescale(o[dt]); A_dt = wmma(v0,p0,o[dt]); B_dt = wmma(v1,p1,o[dt]);
// The two WMMAs for the SAME accumulator are back-to-back -> B_dt RAW-stalls on
// A_dt's matrix-pipe accumulate result with NO independent WMMA between them.
//
// Round 21's v130 tried CHUNK=2 "A0 A1 B0 B1": exactly ONE independent WMMA (A1)
// between A0 and B0. The reviewer measured 90.03 (FAIL) and diagnosed TWO problems:
//   (1) one intervening WMMA is NOT enough to cover gfx12's WMMA accumulate latency
//       -- B0 still hits a RAW scoreboard stall;
//   (2) v130 bunched FOUR V loads (4 bf16x8 = 16 VGPR) live at once before the WMMA
//       group, raising transient VGPR + VMEM-scoreboard pressure in the hottest loop.
// NEXT FOCUS from the reviewer: try "A0 A1 A2 B0 B1 B2" (CHUNK=3 -> TWO independent
// WMMAs between each A and its B) but ONLY if VGPR/occupancy stays flat.
//
// THE CHANGE (PV phase only; QK^T/softmax/exp2/sum/pack/normalize byte-for-byte
// v122): walk the D-tiles in CHUNK=3 groups (DK=8 -> groups {0,1,2},{3,4,5},{6,7}).
// Within a group of size G we emit the WMMA stream  A0 A1 A2  B0 B1 B2, so between
// A_k (writes o[dt_k]) and B_k (reads o[dt_k]) sit TWO independent WMMAs on other
// accumulators (A_{k+1}, A_{k+2}) -- DOUBLE v130's separation, enough to cover the
// accumulate latency the reviewer says one WMMA missed.
//
// CRUCIAL DIFFERENCE FROM v130 (attacks failure-cause #2): we do NOT preload all of
// the group's V operands. We load only the G A-half V fragments (v0) up front, issue
// the A WMMAs, THEN load the G B-half V fragments (v1) and issue the B WMMAs. So the
// PEAK live V-fragment count is G=3 (12 VGPR) -- LESS than v130's 4 (16 VGPR) -- even
// though the WMMA separation is deeper. The B-half loads issue during the A WMMAs'
// matrix-pipe shadow, so their VMEM latency is hidden (the A operands already led the
// A WMMAs). Net: deeper RAW separation AND lower peak register/VMEM pressure than v130.
//
// BIT-EXACTNESS (non-negotiable, max_abs<=0.005): every accumulator v_o[dt] sees the
// IDENTICAL ops in the IDENTICAL order as v122/v100: `*= rescale` (only if
// need_rescale, same rescale) strictly before any WMMA on o[dt], then wmma(v0,p0,o[dt])
// then wmma(v1,p1,o[dt]). The grouping only changes WHEN B_k is issued relative to
// OTHER accumulators' WMMAs -- never the order of the two WMMAs touching the same
// o[dt], and independent accumulators never share an fp32 reduction. Associativity-
// neutral -> rounding-identical to v122/v100. row_max/row_sum/exp2/l_row untouched.
// -> n_bad=0, max_abs=0.0001 expected.
//
// VGPR: <= v130 by construction (peak 3 live V fragments vs v130's 4). No loop-carried
// V carry (unlike v129), no LDS, no barriers. Expect ~v122 occupancy (7 waves/SIMD).
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast131(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v131_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v131_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v131(opus_attn_kargs k)
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
            v_q[kt][j] = bf16_fast131(bf16_to_f32(v_q[kt][j]) * qscale);
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
        // Software-pipelined K loads: preload D-tile 0, then issue dt+1's K loads
        // before the WMMAs on dt so global VMEM overlaps the WMMA issue window.
        fp32x8_t v_s0 = {0,0,0,0,0,0,0,0};
        fp32x8_t v_s1 = {0,0,0,0,0,0,0,0};

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

        // Softmax over 16 values (8 from s0, 8 from s1)
        fp32_t row_max = v_s0[0];
        #pragma unroll
        for (int j = 1; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s1[j]);
        row_max = v131_cross_half_max(row_max);

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
        row_sum = v131_cross_half_sum(row_sum);
        l_row += row_sum;

        bf16x8_t v_p0, v_p1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p0[j] = bf16_fast131(v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p1[j] = bf16_fast131(v_s1[j]);

        // ---- PV: CHUNK=3 grouped online-rescale + DEPTH-2 A->B separation with
        // PHASED V loads. Per group of up to CHUNK D-tiles:
        //   1. rescale just this group's O fragments (short VALU region, <=3 hot);
        //   2. load the group's A-half V fragments (v0) and issue all A WMMAs;
        //   3. load the group's B-half V fragments (v1) and issue all B WMMAs.
        // The emitted WMMA stream within a group is  A0 A1 A2  B0 B1 B2, so between
        // A_k (writes o[dt_k]) and B_k (reads o[dt_k]) sit TWO independent WMMAs
        // (A_{k+1}, A_{k+2}) -- double v130's 1-WMMA separation -> covers the gfx12
        // accumulate latency. Peak live V fragments = CHUNK=3 (12 VGPR), LOWER than
        // v130's 4 (16 VGPR): we never hold A and B operands simultaneously. The B
        // loads issue in the A WMMAs' matrix-pipe shadow so their VMEM latency hides.
        // Per accumulator: `*= rescale` then wmma(v0,p0) then wmma(v1,p1), same order
        // as v122/v100; independent accumulators never share a reduction -> bit-exact.
        constexpr int CHUNK = 3;
        #pragma unroll
        for (int c0 = 0; c0 < DK; c0 += CHUNK) {
            const int G = (DK - c0 < CHUNK) ? (DK - c0) : CHUNK;
            if (need_rescale) {
                #pragma unroll
                for (int dc = 0; dc < CHUNK; ++dc) {
                    if (dc < G) {
                        const int dt = c0 + dc;
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) v_o[dt][j] *= rescale;
                    }
                }
            }
            // Phase A: load v0 halves, issue A WMMAs (A0 A1 A2).
            bf16x8_t v_v0[CHUNK];
            #pragma unroll
            for (int dc = 0; dc < CHUNK; ++dc) {
                if (dc < G) {
                    const int dt = c0 + dc;
                    v_v0[dc] = *reinterpret_cast<const bf16x8_t*>(
                        VTp + (dt * W_K + col16) * vt_stride_d + n_base + row8);
                }
            }
            #pragma unroll
            for (int dc = 0; dc < CHUNK; ++dc) {
                if (dc < G) {
                    const int dt = c0 + dc;
                    v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_v0[dc], v_p0, v_o[dt]);
                }
            }
            // Phase B: load v1 halves, issue B WMMAs (B0 B1 B2).
            bf16x8_t v_v1[CHUNK];
            #pragma unroll
            for (int dc = 0; dc < CHUNK; ++dc) {
                if (dc < G) {
                    const int dt = c0 + dc;
                    v_v1[dc] = *reinterpret_cast<const bf16x8_t*>(
                        VTp + (dt * W_K + col16) * vt_stride_d + n_base + 16 + row8);
                }
            }
            #pragma unroll
            for (int dc = 0; dc < CHUNK; ++dc) {
                if (dc < G) {
                    const int dt = c0 + dc;
                    v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_v1[dc], v_p1, v_o[dt]);
                }
            }
        }
    }

    const fp32_t inv = (l_row > 0.0f) ? (1.0f / l_row) : 0.0f;
    bf16_t* o_row = Op + (q_m_base + col16) * stride_n;
    #pragma unroll
    for (int dt = 0; dt < DK; ++dt) {
        bf16x8_t o_pack;
        #pragma unroll
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast131(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
