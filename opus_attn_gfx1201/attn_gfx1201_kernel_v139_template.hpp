// v139 -- v138 champion body, push the asymmetric s_setprio gradient ONE step:
// 3-LEVEL scheme becomes (QK^T=3 > PV=1 > softmax/loads=0). ZERO VGPR cost, ZERO
// arithmetic change, byte-for-byte v138/v135/v122 dataflow.
//
// PROVENANCE / the measured positive gradient this extends:
//   * v135 {QK=1, PV=1, softmax=0} = 92.42 TFLOPS (PASS). Binary scheme: both WMMA
//     regions outrank softmax. The +1.2 over v122 came from matrix-phase waves
//     winning the SIMD issue arbiter over siblings mid-softmax.
//   * v138 {QK=2, PV=1, softmax=0} = 93.38 TFLOPS (PASS, current champion, +0.96).
//     Splitting QK^T above PV refined inter-matrix arbitration: the bubble-prone,
//     low-ILP QK^T wave wins the issue slot over a high-ILP PV sibling that coasts
//     on its own independent accumulators. The v138 reviewer confirmed the model
//     ("the gain is better scheduler arbitration among staggered resident waves")
//     and the gradient 1->2 on QK^T is a clean, monotone +0.96.
// v138's NEXT levers all kept QK at 2 and probed PV (PV=0, or a PV entry boost).
// This round instead pushes the ONE knob that has actually produced gain twice in a
// row -- the QK^T priority -- to the top of the s_setprio range.
//
// THE ASYMMETRY (why QK^T deserves the ABSOLUTE TOP priority):
//   * QK^T runs only TWO independent WMMA chains (v_s0, v_s1), depth DK=8. Low ILP,
//     and it is partly gated on the global K loads (even with the distance-1 K
//     double-buffer the first dt's K must arrive). It is the more BUBBLE-PRONE
//     matrix region -- the one that most needs to win the issue slot the instant it
//     has work ready, because it cannot self-fill the matrix pipe from its own ILP.
//   * PV runs EIGHT independent accumulator chains (v_o[0..7]), depth 2. High ILP;
//     it keeps the matrix pipe busy from its own independent WMMAs and tolerates
//     losing the occasional issue slot far better than QK^T does.
// v138 already proved QK should outrank PV (2 vs 1). The remaining headroom: at
// prio 2, QK^T can still TIE-then-lose to any non-FA sibling or scheduler state at
// prio 2; lifting it to 3 (the max immediate) makes the bubble-prone QK^T wave the
// unambiguous top of the SIMD arbiter whenever it has a ready WMMA, maximizing the
// exact 2-vs-1 effect that bought v138 its +0.96. PV stays at 1 (still above
// softmax=0, so v135's whole matrix-over-softmax yield is preserved verbatim), and
// the QK/PV separation only WIDENS (3 vs 1) -- it cannot shrink the v138 win.
//
// THE CHANGE (one s_setprio immediate; v138/v135/v122 dataflow byte-for-byte
// identical -- same QK^T + distance-1 K double-buffer, same softmax/exp2/cross-lane/
// pack, same CHUNK=2 PV with interleaved online-rescale, same normalize/store):
//   QK^T WMMA region : s_setprio(3)   (was 2 in v138)  -- ABSOLUTE top priority
//   PV   WMMA region : s_setprio(1)   (unchanged)      -- mid priority
//   softmax / loads  : s_setprio(0)   (unchanged)      -- yields the issue slot
// s_setprio takes a 2-bit immediate (0..3); 3 is in range (the maximum).
//
// BIT-EXACTNESS (non-negotiable, max_abs<=0.005): s_setprio is a SCALAR instruction
// that changes only this wave's hardware ISSUE PRIORITY (inter-wave arbitration). It
// consumes NO VGPR / NO SGPR live range and cannot change any computed value, any
// instruction, or any intra-wave program order of dependent ops. Every fp32
// reduction, exp2 arg, rescale fold, bf16 pack, and WMMA accumulate is byte-for-byte
// v138 / v135 / v122 / v100. -> n_bad=0, max_abs=0.0001 expected (identical to v138).
//
// VGPR: identical to v135/v122 (~205, 7 waves/SIMD). No new buffers, no LDS, no
// barriers, no occupancy change -- it does NOT touch the knee that killed every
// cross-iteration ILP attempt (v119/v121/v125/v134).
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

        // ---- QK^T: BLOCK_N=32 -> two 16x16x16 WMMAs per D-tile.
        // Software-pipelined K loads: preload D-tile 0, then issue dt+1's K loads
        // before the WMMAs on dt so global VMEM overlaps the WMMA issue window.
        fp32x8_t v_s0 = {0,0,0,0,0,0,0,0};
        fp32x8_t v_s1 = {0,0,0,0,0,0,0,0};

        bf16x8_t v_k0_next = *reinterpret_cast<const bf16x8_t*>(Kp + (n_base + col16) * stride_n + row8);
        bf16x8_t v_k1_next = *reinterpret_cast<const bf16x8_t*>(Kp + (n_base + 16 + col16) * stride_n + row8);
        // v139: raise issue priority to the ABSOLUTE TOP level (3) for the QK^T matrix
        // region (v138 used 2; this widens the QK>PV separation to 3>1). QK^T is the
        // bubble-prone, low-ILP, K-load-gated matrix region; give it the unambiguous
        // highest claim on the SIMD issue slot so it wins over BOTH softmax-phase
        // (prio 0) AND PV-phase (prio 1) sibling waves whenever it has a WMMA ready.
        // Scalar op: no VGPR/SGPR liveness, no arithmetic effect. (immediate range 0..3)
        __builtin_amdgcn_s_setprio(3);
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
        // Drop priority to 0 for the softmax VALU/exp2/cross-lane stretch: yield issue
        // bandwidth (which this wave cannot use for the matrix pipe anyway) to a
        // sibling wave that is in its high-prio QK^T (2) or PV (1) WMMA region.
        __builtin_amdgcn_s_setprio(0);

        // Softmax over 16 values (8 from s0, 8 from s1)
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

        // ---- PV: CHUNK=2 grouped online-rescale interleaved with the PV WMMAs.
        // Per group of CHUNK D-tiles: rescale just this group's O fragments (a short
        // VALU region, only 2 fp32x8 accumulators hot), then issue the group's V
        // loads and PV WMMAs. The next group's rescale FMAs overlap this group's
        // WMMAs in the matrix-pipe shadow. Each accumulator v_o[dt]'s arithmetic is
        // byte-for-byte v100: `*= rescale` (only if need_rescale), then wmma(v0,p0),
        // then wmma(v1,p1). Independent accumulators never share a reduction, so the
        // regrouping is associativity-neutral -> bit-exact vs v100.
        constexpr int CHUNK = 2;
        // v139: raise issue priority to the MID level (1) for the PV matrix region.
        // PV has 8 independent accumulator chains (high ILP) so it self-fills the
        // matrix pipe; it stays ABOVE softmax (prio 0) -- preserving v135's whole
        // win -- but BELOW QK^T (prio 2) so a bubble-prone QK-phase sibling wins the
        // contended slot while this high-ILP PV wave coasts on its own ILP.
        __builtin_amdgcn_s_setprio(1);
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
        // Drop priority before looping back into the next tile's K loads + softmax.
        __builtin_amdgcn_s_setprio(0);
    }

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
