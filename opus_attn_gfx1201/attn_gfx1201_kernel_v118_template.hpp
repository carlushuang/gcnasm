// v118 -- v100 champion (90.45 TFLOPS) + TWO-SUBTILE-PASS PV to maximally defer the
// subtile-1 softmax tail out of the exposed QK->PV wedge.
//
// Base choice: v100 is the MEASURED running best. Every v101..v117 follow-up (deeper
// K prefetch, all PV grouping/prefetch/rescale-split variants, tree softmax, an
// unconditional-rescale VGPR cut to 9 waves/SIMD) FAILED to beat v100 at the canonical
// shape. That is strong evidence the kernel is (a) NOT occupancy-bound and (b) past the
// point where local per-D-tile / per-pair PV micro-reorders pay off. v118 therefore
// keeps v100's QK^T byte-for-byte and makes ONE structural change to the PV phase.
//
// The bottleneck v100..v117 all left standing: every one of them keeps p0 and p1
// INTERLEAVED inside each D-tile (or each pair) of PV -- the very first PV WMMA group
// consumes BOTH v_p0 AND v_p1. So the ENTIRE softmax wedge, including subtile-1's
// exp2(s1)+pack(p1) AND the full row_sum reduction, must retire before ANY PV matrix
// work can begin. That wedge runs with the WMMA (matrix) pipe fully idle -- it is the
// last big exposed serial region in the inner loop.
//
// v118 splits PV into TWO sequential passes over all D-tiles:
//   pass A:  for dt in 0..DK-1:  v_o[dt] = wmma(v_v0[dt], v_p0, v_o[dt] * rescale)
//   pass B:  for dt in 0..DK-1:  v_o[dt] = wmma(v_v1[dt], v_p1, v_o[dt])
// Each accumulator v_o[dt] still receives EXACTLY rescale, then the p0 contribution,
// then the p1 contribution, in that order -> BIT-EXACT vs v100 (same operands, same
// WMMA order per accumulator). This is NOT the v48 split-PV dead-end: v48 split into
// two INDEPENDENT accumulator chains that serialised on the single matrix pipe; v118
// has ONE accumulator chain traversed twice over the two N-subtiles -- no extra
// accumulator VGPR, exactly 1 live V fragment per pass.
//
// Why it should win: because pass A never touches v_p1, the producer of v_p1
// (exp2(s1) + pack) AND the off-critical-path row_sum reduction are free to schedule
// into pass A's 8-WMMA matrix-pipe shadow instead of in front of it. The exposed wedge
// before the first PV WMMA shrinks from {exp2 s0+s1, pack p0+p1, full sum} down to just
// {exp2 s0, pack p0}. v114 attempted a similar exp2/pack hoist but, keeping v106's
// grouped PV, could only hide it behind the FIRST 2 WMMAs (p1 needed at the 3rd WMMA of
// pair 0); v118 gives the s1 tail the full 8-WMMA pass-A shadow.
//
// VGPR: same fp32x8 O accumulators, V loaded JIT one fragment at a time (as v100's PV
// did, no prefetch ring) -> footprint == v100, occupancy unchanged (7 waves/SIMD). No
// LDS, no barriers. RISK: low; worst case the compiler already hoisted the s1 tail and
// v118 ties v100. The split doubles the dt-loop trip count but the WMMA count is
// identical (2*DK) so no arithmetic-throughput cost.
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast118(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v118_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v118_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v118(opus_attn_kargs k)
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
            v_q[kt][j] = bf16_fast118(bf16_to_f32(v_q[kt][j]) * qscale);
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
        // Software-pipelined K loads (v100, byte-for-byte identical).
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

        // Softmax over 16 values (8 from s0, 8 from s1). row_max must read BOTH subtiles
        // (unavoidable), but everything downstream of subtile 1 is pushed past pass A.
        fp32_t row_max = v_s0[0];
        #pragma unroll
        for (int j = 1; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s1[j]);
        row_max = v118_cross_half_max(row_max);

        const fp32_t new_m = __builtin_fmaxf(m_row, row_max);
        fp32_t rescale = 1.0f;
        const bool need_rescale = (new_m != m_row);
        if (need_rescale) {
            rescale = __builtin_amdgcn_exp2f(m_row - new_m);
            l_row *= rescale;
        }
        m_row = new_m;

        // --- Subtile-0 tail ONLY: exp2(s0) + pack(p0). This is the entire critical
        // path that must retire before PV pass A can start.
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_s0[j] = __builtin_amdgcn_exp2f(v_s0[j] - new_m);
        bf16x8_t v_p0;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p0[j] = bf16_fast118(v_s0[j]);

        // ---- PV pass A: accumulate the subtile-0 (v_p0) contribution into every
        // D-tile, applying the online rescale at each accumulator's head (v100's exact
        // rescale point and value). V loaded JIT, one fragment at a time (v100 PV
        // footprint -> no extra VGPR). Pass A touches NEITHER v_p1 NOR row_sum.
        #pragma unroll
        for (int dt = 0; dt < DK; ++dt) {
            if (need_rescale) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) v_o[dt][j] *= rescale;
            }
            const bf16_t* vt_addr0 = VTp + (dt * W_K + col16) * vt_stride_d + n_base + row8;
            bf16x8_t v_v0 = *reinterpret_cast<const bf16x8_t*>(vt_addr0);
            v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_v0, v_p0, v_o[dt]);
        }

        // --- Subtile-1 tail: exp2(s1) + pack(p1) + the FULL row_sum reduction. These
        // are emitted AFTER pass A so the compiler can sink them into pass A's 8-WMMA
        // matrix-pipe shadow (VALU + transcendental overlapping the matrix pipe). p1 is
        // first needed in pass B; l_row is consumed only after the whole KV loop, so the
        // sum is fully off the critical path.
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_s1[j] = __builtin_amdgcn_exp2f(v_s1[j] - new_m);
        bf16x8_t v_p1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p1[j] = bf16_fast118(v_s1[j]);

        fp32_t row_sum = v_s0[0];
        #pragma unroll
        for (int j = 1; j < 8; ++j) row_sum += v_s0[j];
        #pragma unroll
        for (int j = 0; j < 8; ++j) row_sum += v_s1[j];
        row_sum = v118_cross_half_sum(row_sum);
        l_row += row_sum;

        // ---- PV pass B: accumulate the subtile-1 (v_p1) contribution into every
        // D-tile. Each v_o[dt] now has had rescale, then the p0 WMMA (pass A), then this
        // p1 WMMA -> identical operand order to v100 -> bit-exact.
        #pragma unroll
        for (int dt = 0; dt < DK; ++dt) {
            const bf16_t* vt_addr1 = VTp + (dt * W_K + col16) * vt_stride_d + n_base + 16 + row8;
            bf16x8_t v_v1 = *reinterpret_cast<const bf16x8_t*>(vt_addr1);
            v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_v1, v_p1, v_o[dt]);
        }
    }

    const fp32_t inv = (l_row > 0.0f) ? (1.0f / l_row) : 0.0f;
    bf16_t* o_row = Op + (q_m_base + col16) * stride_n;
    #pragma unroll
    for (int dt = 0; dt < DK; ++dt) {
        bf16x8_t o_pack;
        #pragma unroll
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast118(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
