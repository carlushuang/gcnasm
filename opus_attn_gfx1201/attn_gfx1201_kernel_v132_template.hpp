// v132 -- v122 (best, 91.68) + CHUNK=2 PV with PHASED B-half V prefetch.
//
// PROVENANCE / why this round: the round-22 reviewer closed three dead-ends and
// gave one precise next lever. v122 (current best) issues each CHUNK=2 group as
// `A0 B0 A1 B1` (per-dt single chain), so within a tile A_dt (writes o[dt]) is
// immediately followed by B_dt (reads o[dt]) -> a back-to-back RAW scoreboard
// stall on the matrix pipe. v130 fixed the RAW by reordering to `A0 A1 B0 B1`,
// but to do so it loaded ALL FOUR V fragments (v0_a,v1_a,v0_b,v1_b) up front ->
// peak 4 live V fragments (16 VGPR) AND the B operands had no real VMEM lead
// time, so it regressed. v131 phased (all-A-then-all-B over CHUNK=3) but issued
// the B-half loads only AFTER all A WMMAs, too late to hide their VMEM latency,
// and also tied/regressed (87.09).
//
// THE CHANGE (reviewer's Lever 1, exactly): keep CHUNK=2 and v130's RAW-hiding
// WMMA order `A0 A1 B0 B1`, but PHASE the loads so each B operand gets WMMA-shadow
// lead time while peak live V fragments stays at 3 (12 VGPR), NOT 4:
//
//     load v0_a, v1_a          (tile dt0: both halves)
//     load v0_b                (tile dt1: A-half only)
//     A0 = wmma(v0_a, p0)      -> o[dt0]
//     A1 = wmma(v0_b, p0)      -> o[dt1]        (independent acc; hides A0 latency)
//     load v1_b                (tile dt1: B-half) -- issued in the A0/A1 shadow
//     B0 = wmma(v1_a, p1)      -> o[dt0]        (v1_a loaded long ago: no bubble)
//     B1 = wmma(v1_b, p1)      -> o[dt1]        (A1 sits between its load & use)
//
// Between A0 (writes o[dt0]) and B0 (reads o[dt0]) sits A1 -- one independent WMMA
// -- so A0's accumulate latency is hidden (the win v122 lacks). v1_b's load is
// emitted right after A1 issues, so its VMEM latency overlaps the A0+A1 matrix
// work (the lead time v130 lacked). At no point are all four V fragments live:
// v0_a is dead after A0, so the live set is {v1_a, v0_b, v1_b} = 3 (the cap the
// reviewer asked to hold), one fewer than v130's 4.
//
// BIT-EXACTNESS (non-negotiable, max_abs<=0.005): the arithmetic touching each
// accumulator v_o[dt] is UNCHANGED from v100/v122 -- for every dt it is still
// exactly `v_o[dt] *= rescale` (only when need_rescale, same `rescale`), then the
// same two WMMAs in the same order: wmma(v0,p0) then wmma(v1,p1). Only the
// INTERLEAVING of two INDEPENDENT accumulators (o[dt0] vs o[dt1], which never
// share an fp32 reduction) changes -> associativity-neutral, rounding-identical
// to v122/v100. row_max/row_sum/exp2/l_row untouched -> n_bad=0, max_abs=0.0001.
//
// VGPR: ~v122 (~205, 7 waves/SIMD). 3 live V fragments (vs v130's 4); no loop-
// carried V state, no LDS, no barriers.
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast132(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v132_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v132_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v132(opus_attn_kargs k)
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
            v_q[kt][j] = bf16_fast132(bf16_to_f32(v_q[kt][j]) * qscale);
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
        row_max = v132_cross_half_max(row_max);

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
        row_sum = v132_cross_half_sum(row_sum);
        l_row += row_sum;

        bf16x8_t v_p0, v_p1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p0[j] = bf16_fast132(v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p1[j] = bf16_fast132(v_s1[j]);

        // ---- PV: CHUNK=2 grouped online-rescale + RAW-hiding `A0 A1 B0 B1` WMMA
        // order with PHASED B-half V prefetch (reviewer Lever 1). Per group:
        //   1. rescale this group's 2 O fragments (short VALU region, <=2 hot);
        //   2. load v0_a, v1_a (tile dt0 both halves) and v0_b (tile dt1 A-half);
        //   3. A0 = wmma(v0_a,p0)->o[dt0];  A1 = wmma(v0_b,p0)->o[dt1];
        //   4. load v1_b (tile dt1 B-half) -- issues in the A0/A1 WMMA shadow;
        //   5. B0 = wmma(v1_a,p1)->o[dt0];  B1 = wmma(v1_b,p1)->o[dt1].
        // A1 sits between A0 and B0 (hides A0's accumulate latency: the RAW fix
        // v122 lacks). v1_b's load overlaps the A0+A1 matrix work (the lead time
        // v130 lacked). v0_a dies after A0, so live V set = {v1_a,v0_b,v1_b} = 3
        // (one fewer than v130's 4). Per accumulator the order is unchanged
        // (`*=rescale`, wmma(v0,p0), wmma(v1,p1)); independent accumulators never
        // share an fp32 reduction -> bit-exact vs v122/v100.
        constexpr int CHUNK = 2;
        #pragma unroll
        for (int c0 = 0; c0 < DK; c0 += CHUNK) {
            const int dt0 = c0;
            const int dt1 = c0 + 1;
            if (need_rescale) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) v_o[dt0][j] *= rescale;
                #pragma unroll
                for (int j = 0; j < 8; ++j) v_o[dt1][j] *= rescale;
            }
            // Phase 1: both halves of dt0 + A-half of dt1 (3 loads lead their use).
            bf16x8_t v_v0_a = *reinterpret_cast<const bf16x8_t*>(
                VTp + (dt0 * W_K + col16) * vt_stride_d + n_base + row8);
            bf16x8_t v_v1_a = *reinterpret_cast<const bf16x8_t*>(
                VTp + (dt0 * W_K + col16) * vt_stride_d + n_base + 16 + row8);
            bf16x8_t v_v0_b = *reinterpret_cast<const bf16x8_t*>(
                VTp + (dt1 * W_K + col16) * vt_stride_d + n_base + row8);
            // A WMMAs (v0_a dead after A0 -> live V set drops back to 3).
            v_o[dt0] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_v0_a, v_p0, v_o[dt0]); // A0
            v_o[dt1] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_v0_b, v_p0, v_o[dt1]); // A1
            // Phase 2: dt1 B-half load issues in the A0/A1 matrix-pipe shadow.
            bf16x8_t v_v1_b = *reinterpret_cast<const bf16x8_t*>(
                VTp + (dt1 * W_K + col16) * vt_stride_d + n_base + 16 + row8);
            v_o[dt0] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_v1_a, v_p1, v_o[dt0]); // B0
            v_o[dt1] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_v1_b, v_p1, v_o[dt1]); // B1
        }
    }

    const fp32_t inv = (l_row > 0.0f) ? (1.0f / l_row) : 0.0f;
    bf16_t* o_row = Op + (q_m_base + col16) * stride_n;
    #pragma unroll
    for (int dt = 0; dt < DK; ++dt) {
        bf16x8_t o_pack;
        #pragma unroll
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast132(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
