// v123 -- DIRECTOR DIRECTION (A) refined per round-13 reviewer:
//   "Restore v100's monolithic v_o rescale placement, but keep the CHUNK=2 PV
//    load/WMMA grouping."
//
// Round-13 (v122) confirmed the FALSIFICATION of the contraction theory:
// #pragma clang fp contract(off) did NOT recover bit-exactness (max_abs stayed
// 0.0360). v111/v122 drift therefore comes from the TEMPORAL RELOCATION of the
// rescale -- in v111/v122 the v_o *= rescale was deferred PAST the exp2/row_sum/
// bf16-pack work and interleaved into the PV loop. That relocation, not per-element
// multiply order, is what perturbs the accumulator rounding the compiler emits.
//
// v123 isolates the SPEED source the reviewer flagged (lever #2): keep the v_o
// rescale EXACTLY where v100 puts it -- monolithic, inside the need_rescale branch,
// BEFORE exp2/sum/bf16-pack -> byte-identical arithmetic to v100 -> bit-exact.
// THEN apply only the PV restructuring: process the 8 D-tiles in CHUNK=2 groups,
// and within each group issue ALL the chunk's V loads FIRST, then ALL the chunk's
// PV WMMAs. This gives the scheduler a clean "4 independent VMEM loads, then 4
// dependent matrix ops" window per chunk -> better V-load / WMMA overlap (memory-
// level parallelism) and tighter V-register live ranges (only one chunk's V regs
// live at a time) WITHOUT touching any fp accumulation order:
//   * v_o[dt] still receives v0-WMMA then v1-WMMA (same order as v100).
//   * different dt are independent accumulators (order among them is irrelevant).
//   * rescale is the same single monolithic pass v100 uses.
// Hypothesis: most of v111's +1.3 TFLOPS is PV V-load locality, not the rescale
// interleave -- so we can capture it bit-exactly.
// EXPECTED: primary ~90.5-91.5 TFLOPS, max_abs ~0.0001, n_bad=0.
// RISK: low. Worst case the compiler schedules identically to v100 -> ties 88.07
// bit-exactly. No arithmetic change can break the correctness gate.
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast123(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v123_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v123_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v123(opus_attn_kargs k)
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
            v_q[kt][j] = bf16_fast123(bf16_to_f32(v_q[kt][j]) * qscale);
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

        fp32_t row_max = v_s0[0];
        #pragma unroll
        for (int j = 1; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s1[j]);
        row_max = v123_cross_half_max(row_max);

        const fp32_t new_m = __builtin_fmaxf(m_row, row_max);
        fp32_t rescale = 1.0f;
        const bool need_rescale = (new_m != m_row);
        if (need_rescale) {
            rescale = __builtin_amdgcn_exp2f(m_row - new_m);
            l_row *= rescale;
            // ---- Batched pre-PV rescale (v43/v100), EXACT v100 placement ----
            // Monolithic single pass over all 8 O accumulators, BEFORE exp2/sum/
            // pack. This is byte-identical to v100 -> bit-exact. (v111/v122 lost
            // bit-exactness precisely by deferring this past exp2/sum into PV.)
            #pragma unroll
            for (int dt = 0; dt < DK; ++dt)
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                    v_o[dt][j] *= rescale;
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
        row_sum = v123_cross_half_sum(row_sum);
        l_row += row_sum;

        bf16x8_t v_p0, v_p1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p0[j] = bf16_fast123(v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p1[j] = bf16_fast123(v_s1[j]);

        // ---- PV: branch-free, CHUNK=2 with grouped loads-then-WMMAs (v123) ----
        // Rescale is ALREADY done (monolithic, above), so this loop is purely PV.
        // Each chunk of 2 D-tiles: (1) issue all 4 V loads, then (2) issue all 4
        // PV WMMAs. The grouped VMEM burst exposes memory-level parallelism (4
        // loads in flight) and the dependent WMMA burst keeps the matrix pipe fed,
        // while only one chunk's V registers are live at a time. The per-fragment
        // accumulation order (v0 then v1 into each v_o[dt]) is unchanged -> bit-exact.
        constexpr int CHUNK = 2;
        #pragma unroll
        for (int c0 = 0; c0 < DK; c0 += CHUNK) {
            bf16x8_t vv0[CHUNK];
            bf16x8_t vv1[CHUNK];
            #pragma unroll
            for (int dc = 0; dc < CHUNK; ++dc) {
                const int dt = c0 + dc;
                vv0[dc] = *reinterpret_cast<const bf16x8_t*>(VTp + (dt * W_K + col16) * vt_stride_d + n_base + row8);
                vv1[dc] = *reinterpret_cast<const bf16x8_t*>(VTp + (dt * W_K + col16) * vt_stride_d + n_base + 16 + row8);
            }
            #pragma unroll
            for (int dc = 0; dc < CHUNK; ++dc) {
                const int dt = c0 + dc;
                v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(vv0[dc], v_p0, v_o[dt]);
                v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(vv1[dc], v_p1, v_o[dt]);
            }
        }
    }

    const fp32_t inv = (l_row > 0.0f) ? (1.0f / l_row) : 0.0f;
    bf16_t* o_row = Op + (q_m_base + col16) * stride_n;
    #pragma unroll
    for (int dt = 0; dt < DK; ++dt) {
        bf16x8_t o_pack;
        #pragma unroll
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast123(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
