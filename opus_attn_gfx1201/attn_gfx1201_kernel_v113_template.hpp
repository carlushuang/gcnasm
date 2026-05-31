// v113 -- v106 (MEASURED champion, 90.20 TFLOPS) + TWO bit-exact critical-path shorteners
//         on the exposed QK->PV softmax transition: (1) balanced-tree row_max and
//         (2) PACK-BEFORE-SUM reorder.
//
// Base choice: v106 (the running best, measured 90.20 @ b4h32n4096). v106 = v103's
// two-ahead QK K-prefetch ring + 2-D-tile grouped PV (4 live V fragments, two
// independent accumulator chains per pair) + one-pair-ahead software-pipelined rescale.
// PV restructuring is now SATURATED on this box: v104/v105 (4-chain) regressed hard,
// v107/v108 (rescale placement) regressed, v109/v110 (V-prefetch rings) dipped below
// v106, v102 (all-8 batched rescale) starved the matrix pipe. So v113 leaves the entire
// QK^T + PV WMMA schedule of v106 BYTE-FOR-BYTE untouched and attacks the OTHER
// component the round-7 reviewer named as the remaining mixed limit: the softmax cost
// wedged on the exposed QK->PV transition (where neither matrix pipe is running).
//
// CHANGE 1 -- balanced-tree row_max (depth 4 vs ~15 serial). v106 inherits v39/v100's
// reduction: a 15-deep serial fmax dependency chain sitting directly between the QK^T
// WMMAs (produce v_s0/v_s1) and the cross-half ds_bpermute + exp2 + PV. Shortening it
// to depth 4 collapses ~11 serial-dependent VALU latencies off the transition.
//
// CHANGE 2 -- PACK-BEFORE-SUM reorder. In v106 the ~15-deep SERIAL row_sum chain is
// emitted in program order BETWEEN exp2 and the bf16 pack that produces PV's B-operand
// (v_p0/v_p1). But l_row (the sum) is consumed only AFTER the whole KV loop -- it is OFF
// the QK->PV critical path. v113 emits the pack FIRST, then the sum. This shortens the
// exposed exp2->(PV operand) program-order distance so the matrix pipe can begin PV
// sooner, while the off-critical-path sum retires in the WMMA issue shadow. Pack and sum
// both only READ v_s and write disjoint outputs -> no data dependency -> pure reorder.
//
// Both changes target the SAME unsolved bottleneck (the softmax wedge) from two angles
// and are mutually orthogonal; together they should shorten the exposed transition more
// than the tree alone (v112). v111 applied only the tree to the slower v100 PV and merely
// tied (88.74) because v100's PV dominated; v106's faster PV makes the transition a larger
// fraction of the iteration, so shortening it can finally surface. Orthogonal to all PV
// levers tried.
//
// BIT-EXACTNESS: fmax is associative+commutative for finite values, so the balanced tree
// selects the SAME maximum element as the serial chain -> byte-identical row_max, hence
// byte-identical new_m / rescale / exp2 args / probs / O. (QK^T scores are always finite
// -- no masking.) The pack/sum reorder changes neither operand: v_p comes from the same
// bf16_fast113(v_s) values, l_row from the same serial-order sum -> byte-identical.
// VGPR: t[8] consumed immediately, pack/sum use the same live regs -> zero net new live
// state -> same 4-live-V-fragment footprint as v106 -> occupancy unchanged.
// RISK: low. If the compiler already schedules v106 this way, v113 ties v106.
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast113(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v113_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v113_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v113(opus_attn_kargs k)
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
            v_q[kt][j] = bf16_fast113(bf16_to_f32(v_q[kt][j]) * qscale);
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
        // TWO-AHEAD software-pipelined K loads (v101/v103, byte-for-byte identical).
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

        // Softmax row max over 16 values (8 from s0, 8 from s1).
        // BALANCED TREE (depth 4) instead of v106's serial ~15-deep fmax chain.
        // fmax is associative+commutative for finite values -> selects the same
        // maximum element -> byte-identical row_max as v106, but on a far shorter
        // dependency chain that no longer stalls the exposed QK->PV transition.
        // t[] is consumed immediately -> zero extra live registers.
        fp32_t t[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) t[j] = __builtin_fmaxf(v_s0[j], v_s1[j]); // depth 1
        #pragma unroll
        for (int j = 0; j < 4; ++j) t[j] = __builtin_fmaxf(t[j], t[j + 4]);   // depth 2
        t[0] = __builtin_fmaxf(t[0], t[2]);                                    // depth 3
        t[1] = __builtin_fmaxf(t[1], t[3]);
        fp32_t row_max = __builtin_fmaxf(t[0], t[1]);                          // depth 4
        row_max = v113_cross_half_max(row_max);

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

        // PACK-BEFORE-SUM reorder: emit the bf16 pack of the PV operands (v_p0/v_p1)
        // BEFORE the serial row_sum reduction. In v106/v112 the serial row_sum chain
        // sits in program order between exp2 and the pack that produces PV's B-operand
        // -- yet l_row (the sum) is consumed only AFTER the whole KV loop, so it is OFF
        // the QK->PV critical path. Hoisting the pack ahead of the sum shortens the
        // exposed program-order distance from exp2 (which produces v_s0/v_s1) to the
        // first PV WMMA operand, letting the matrix pipe start PV sooner while the
        // off-critical-path sum retires in the WMMA issue shadow. Pack reads v_s, sum
        // reads v_s, neither writes the other's input -> no data dependency between
        // them -> reordering is byte-identical (same v_p, same l_row).
        bf16x8_t v_p0, v_p1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p0[j] = bf16_fast113(v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p1[j] = bf16_fast113(v_s1[j]);

        fp32_t row_sum = v_s0[0];
        #pragma unroll
        for (int j = 1; j < 8; ++j) row_sum += v_s0[j];
        #pragma unroll
        for (int j = 0; j < 8; ++j) row_sum += v_s1[j];
        row_sum = v113_cross_half_sum(row_sum);
        l_row += row_sum;

        // ---- PV: v103's 2-D-tile grouped PV (4 live V fragments, two independent
        // accumulator chains per pair) with the online rescale SOFTWARE-PIPELINED BY
        // ONE PAIR. Prologue rescales pair 0; iteration dt then issues the current
        // pair's 4 V loads, interleaves the NEXT pair's rescale (independent regs)
        // between loads and WMMAs, and issues the current pair's 4 WMMAs. Each
        // accumulator is rescaled exactly once before its PV WMMA -> bit-exact.
        // DK=8 is even -> exact pairing, no remainder.
        if (need_rescale) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_o[0][j] *= rescale;
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_o[1][j] *= rescale;
        }
        #pragma unroll
        for (int dt = 0; dt < DK; dt += 2) {
            const int dt1 = dt + 1;
            const bf16_t* a0 = VTp + (dt  * W_K + col16) * vt_stride_d + n_base + row8;
            const bf16_t* a1 = VTp + (dt  * W_K + col16) * vt_stride_d + n_base + 16 + row8;
            const bf16_t* b0 = VTp + (dt1 * W_K + col16) * vt_stride_d + n_base + row8;
            const bf16_t* b1 = VTp + (dt1 * W_K + col16) * vt_stride_d + n_base + 16 + row8;
            bf16x8_t va0 = *reinterpret_cast<const bf16x8_t*>(a0);
            bf16x8_t va1 = *reinterpret_cast<const bf16x8_t*>(a1);
            bf16x8_t vb0 = *reinterpret_cast<const bf16x8_t*>(b0);
            bf16x8_t vb1 = *reinterpret_cast<const bf16x8_t*>(b1);
            // One-pair-ahead rescale: prepare the NEXT pair's accumulators while the
            // current pair's V loads are in flight. Independent of the WMMAs below.
            if (dt + 2 < DK && need_rescale) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) v_o[dt + 2][j] *= rescale;
                #pragma unroll
                for (int j = 0; j < 8; ++j) v_o[dt + 3][j] *= rescale;
            }
            v_o[dt]  = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(va0, v_p0, v_o[dt]);
            v_o[dt1] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(vb0, v_p0, v_o[dt1]);
            v_o[dt]  = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(va1, v_p1, v_o[dt]);
            v_o[dt1] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(vb1, v_p1, v_o[dt1]);
        }
    }

    const fp32_t inv = (l_row > 0.0f) ? (1.0f / l_row) : 0.0f;
    bf16_t* o_row = Op + (q_m_base + col16) * stride_n;
    #pragma unroll
    for (int dt = 0; dt < DK; ++dt) {
        bf16x8_t o_pack;
        #pragma unroll
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast113(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
