// v137 -- v135's SUSTAINED s_setprio matrix windows + batched pre-PV rescale.
// Combines the round-13 champion (v135, 92.42) with the v43 batched-rescale lever,
// implementing the round-13 reviewer's explicit NEXT. ZERO VGPR cost, bit-exact.
//
// PROVENANCE (the two relevant data points this round):
//   * v135 (CHAMPION, 92.42 TFLOPS): the v122 body + s_setprio(1) raised ONCE before
//     the QK^T WMMA loop and ONCE before the whole PV CHUNK loop, dropped to 0 for
//     the softmax/load stretches. The win (+1.2 over v122) comes from a *sustained*
//     high-priority PV WMMA window: a wave inside its WMMA region keeps winning the
//     SIMD round-robin issue slot over siblings mid-softmax.
//   * v136 (FAIL, 89.71, -2.71): tried to also exclude the per-CHUNK online-rescale
//     VALU from the high-prio window by toggling s_setprio per CHUNK (4x raise/drop
//     per KV tile). The round-13 reviewer's root cause: the rescale VALU is NOT the
//     dominant interference; the win is the CONTINUOUS PV WMMA window, and per-CHUNK
//     toggling SHATTERED it (added SALU churn + lost sustained WMMA arbitration).
//     Reviewer NEXT (lever #2): "keep ONE s_setprio(1) across all PV WMMAs, but move
//     the rescale block before the raise if structurally cheap."
//
// THE CHANGE (v135 -> v137), PV phase only; QK^T phase byte-for-byte v135:
//   v135 PV:  rescale interleaved per-CHUNK INSIDE the single sustained window:
//             setprio(1); { if(need) rescale 2 accs;  V-load; 2 WMMAs }x4; setprio(0)
//   v137 PV:  HOIST the rescale into one batched pass over all 8 O accumulators
//             BEFORE the raise, then ONE uninterrupted high-prio PV WMMA window:
//             if(need){ rescale all 8 accs }   // prio 0, pure VALU
//             setprio(1); { V-load; 2 WMMAs }x4;   setprio(0)
// This is the structurally-clean way to satisfy the reviewer's NEXT: it keeps EXACTLY
// the v135 sustained single-toggle PV window (one raise, one drop -- same SALU count
// as v135, HALF of v136), and additionally removes the rescale FMAs from that window
// without any per-group toggle. The batched pre-PV rescale is the v43 lever: a pure
// VALU block the compiler can overlap with the QK^T issue shadow / first V-load
// latency, while the matrix pipe in the PV window is fed ONLY by waves truly inside
// a WMMA burst. Hypothesis: recovers v135's sustained-window win AND lets rescale
// yield issue bandwidth, without the v136 churn that caused the regression.
//
// s_setprio is a SCALAR instruction: NO VGPR / NO SGPR live range. The rescale hoist
// only reorders independent per-accumulator multiplies (see BIT-EXACTNESS); it adds
// no buffers. v137 stays at v122/v135's footprint (~205 VGPR, 7 waves/SIMD) -- it does
// NOT touch the occupancy knee that killed every cross-iteration ILP attempt.
//
// BIT-EXACTNESS (non-negotiable, max_abs<=0.005): each accumulator v_o[dt] still
// computes, IN ORDER, `*= rescale` (only when need_rescale) then wmma(v0,p0) then
// wmma(v1,p1) -- byte-for-byte the v100/v122/v135 per-accumulator chain. The 8
// accumulators are independent and never share a reduction, so hoisting all the
// `*= rescale` multiplies ahead of all the WMMAs is associativity-neutral (it only
// reorders ops across independent accumulators, never within one). s_setprio cannot
// change any value or intra-wave program order. -> n_bad=0, max_abs=0.0001 expected.
//
// VGPR: identical to v122/v135 (~205, 7 waves/SIMD). No new buffers, no LDS, no barriers.
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast137(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v137_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v137_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v137(opus_attn_kargs k)
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
            v_q[kt][j] = bf16_fast137(bf16_to_f32(v_q[kt][j]) * qscale);
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
        // Raise issue priority for the QK^T matrix region so this wave wins the
        // SIMD issue slot over any sibling currently stalled in its softmax VALU
        // window. Scalar op: no VGPR/SGPR liveness, no arithmetic effect.
        __builtin_amdgcn_s_setprio(1);
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
        // Drop priority for the softmax VALU/exp2/cross-lane stretch: yield issue
        // bandwidth (which this wave cannot use for the matrix pipe anyway) to a
        // sibling wave that is in its high-prio QK^T / PV WMMA region.
        __builtin_amdgcn_s_setprio(0);

        // Softmax over 16 values (8 from s0, 8 from s1)
        fp32_t row_max = v_s0[0];
        #pragma unroll
        for (int j = 1; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s1[j]);
        row_max = v137_cross_half_max(row_max);

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
        row_sum = v137_cross_half_sum(row_sum);
        l_row += row_sum;

        bf16x8_t v_p0, v_p1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p0[j] = bf16_fast137(v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p1[j] = bf16_fast137(v_s1[j]);

        // ---- PV: SINGLE-TOGGLE s_setprio window + batched pre-PV rescale.
        //
        // v135 (champion, 92.42) raised priority ONCE before the whole CHUNK loop and
        // dropped it ONCE after -> one sustained high-prio PV WMMA window. v136 tried
        // to also exclude the rescale VALU by toggling prio per CHUNK (4x/tile) and
        // REGRESSED to 89.71: the round-13 reviewer's root cause was that the rescale
        // VALU is NOT the dominant interference -- the win comes from a *continuous*
        // high-priority PV WMMA window, and the per-CHUNK toggling shattered it.
        //
        // v137 = the reviewer's explicit NEXT (lever #2): keep ONE sustained
        // s_setprio(1) over ALL PV WMMAs (no per-group toggle), and remove the rescale
        // FMAs from that window the ONLY structurally-clean way -- hoist them into a
        // single batched pass over all 8 O accumulators BEFORE the raise (the v43
        // "batched pre-PV online rescale" lever). So the rescale runs at prio 0, the
        // PV WMMA window is one uninterrupted high-prio block (as in v135), and there
        // are exactly TWO scalar prio ops in the PV phase (same count as v135, half of
        // v136). The pre-PV rescale block is also a pure VALU region whose FMAs the
        // compiler can overlap with the QK^T issue shadow / V-load latency.
        //
        // BIT-EXACTNESS: each accumulator v_o[dt] still computes, in order,
        // `*= rescale` (only when need_rescale) then wmma(v0,p0) then wmma(v1,p1) --
        // byte-for-byte the v100/v122/v135 per-accumulator chain. Independent
        // accumulators never share a reduction, so hoisting all the `*= rescale`
        // multiplies ahead of all the WMMAs is associativity-neutral. -> n_bad=0.
        constexpr int CHUNK = 2;
        if (need_rescale) {
            #pragma unroll
            for (int dt = 0; dt < DK; ++dt) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) v_o[dt][j] *= rescale;
            }
        }
        // One sustained high-priority window across ALL PV WMMAs (mirrors v135).
        __builtin_amdgcn_s_setprio(1);
        #pragma unroll
        for (int c0 = 0; c0 < DK; c0 += CHUNK) {
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
        // Drop priority before looping back into the next tile's K loads + softmax.
        __builtin_amdgcn_s_setprio(0);
    }

    const fp32_t inv = (l_row > 0.0f) ? (1.0f / l_row) : 0.0f;
    bf16_t* o_row = Op + (q_m_base + col16) * stride_n;
    #pragma unroll
    for (int dt = 0; dt < DK; ++dt) {
        bf16x8_t o_pack;
        #pragma unroll
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast137(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
