// v105 -- v103 (two-ahead K prefetch QK^T + hybrid 2-wide PV) PV phase replaced with
// a 4-CHAIN, LOW-LIVE, TWO-WAVE just-in-time V load (reviewer's explicit NEXT FOCUS #2).
//
// Base choice: v103 (the running best, 89.68 TFLOPS) = v101's QK^T (two-ahead K ring)
// + a hybrid 2-D-tile grouped PV rescale. QK^T is kept BYTE-FOR-BYTE identical to
// v101/v103. The ONLY change is the PV phase grouping/scheduling.
//
// History on the PV phase:
//   v101: per-dt fusion -> only ONE dependent WMMA chain live at a time
//         (v_o[dt] -> v_o[dt]); cannot hide the ~12.6-cyc WMMA result latency.
//   v102: hoist rescale of ALL 8 accumulators -> 64-deep VALU prefix stalls the
//         matrix pipe at PV entry. REGRESSED to 87.25.
//   v103: hybrid 2-D-tile grouping -> TWO independent chains + 16-FMA rescale prefix.
//         Best so far at 89.68. Two chains only put ONE independent WMMA between an
//         accumulator's producer and consumer -> partial latency cover.
//   v104: 4-D-tile group BUT made all EIGHT V fragments (va0/va1..vd0/vd1) live at
//         once, then 8 WMMAs. REGRESSED HARD to 82.23. Reviewer root cause: 8 live
//         bf16x8 V fragments on top of the 64-fp32/lane v_o accumulator inflated
//         VGPR live ranges around the WMMA window -> spills / waitcnt-constrained
//         scheduling. Lesson: 4 chains help, but 8 live fragments is the cliff.
//
// Hypothesis (reviewer NEXT FOCUS #2 -- "Keep 4 chains but load just-in-time in two
// waves: load/use p0 for four chains, then load/use p1"): process D-tiles in groups
// of 4 to expose FOUR independent accumulator chains, but NEVER hold more than 4 V
// fragments live. Per group:
//   1. rescale the 4 group accumulators (32-FMA bounded prefix, like v104, not v102).
//   2. WAVE p0: load va0,vb0,vc0,vd0 (the 4 p0-column V tiles), issue the 4 p0 WMMAs
//      across the four INDEPENDENT chains v_o[d0..d3]. 4 live fragments.
//   3. WAVE p1: load va1,vb1,vc1,vd1 (reusing the same 4 registers), issue the 4 p1
//      WMMAs. 4 live fragments.
// The second WMMA touching v_o[d0] (its p1) is separated from the first (its p0) by
// THREE independent p0 WMMAs PLUS the 4 p1 V loads -> the producer->consumer distance
// is at least as large as v104's, fully covering the result latency. But peak live V
// fragments is 4 (== v103, half of v104), so VGPR pressure stays at the v103 level
// that did NOT spill. This isolates the "4 independent chains" win from the "8 live
// fragments" regression that sank v104.
//
// Arithmetic is identical to v101/v103/v104: each v_o[dt] rescaled exactly once before
// its PV accumulation, same per-accumulator WMMA order (p0 before p1), same rescale
// value -> bit-exact in bf16. Reordering WMMAs only across DISTINCT (non-aliasing)
// accumulators is associativity-safe; each accumulator's own update sequence is intact.
// VGPR: peak 4 transient bf16x8 V fragments (== v103). DK=8 -> two groups of 4, no
// remainder. RISK: low; if the two-wave split serializes loads against WMMA it ties
// v103, and v103 remains the fallback. (Requires DK % 4 == 0; true for D=128.)
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast105(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v105_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v105_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v105(opus_attn_kargs k)
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
            v_q[kt][j] = bf16_fast105(bf16_to_f32(v_q[kt][j]) * qscale);
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
        // TWO-AHEAD software-pipelined K loads (v101): a 2-entry ring keeps kt and
        // kt+1's K fragments preloaded; iteration kt consumes its slot and immediately
        // issues kt+2's loads into the same slot before the WMMAs run.
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
        row_max = v105_cross_half_max(row_max);

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
        row_sum = v105_cross_half_sum(row_sum);
        l_row += row_sum;

        bf16x8_t v_p0, v_p1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p0[j] = bf16_fast105(v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p1[j] = bf16_fast105(v_s1[j]);

        // ---- PV: 4-CHAIN, LOW-LIVE, TWO-WAVE just-in-time V load (v105).
        // Process D-tiles in groups of 4 -> four INDEPENDENT accumulator chains
        // v_o[d0..d3] (never alias). Within each group, load V in two waves so at most
        // 4 bf16x8 V fragments are ever live (== v103), avoiding v104's 8-fragment
        // VGPR cliff. Wave p0: load the four p0-column tiles, issue the four p0 WMMAs.
        // Wave p1: reload into the same regs, issue the four p1 WMMAs. The producer of
        // v_o[d0] (its p0 WMMA) and its consumer (its p1 WMMA) are separated by three
        // independent p0 WMMAs + four p1 loads -> full result-latency cover.
        #pragma unroll
        for (int dg = 0; dg < DK; dg += 4) {
            const int d0 = dg, d1 = dg + 1, d2 = dg + 2, d3 = dg + 3;
            if (need_rescale) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) v_o[d0][j] *= rescale;
                #pragma unroll
                for (int j = 0; j < 8; ++j) v_o[d1][j] *= rescale;
                #pragma unroll
                for (int j = 0; j < 8; ++j) v_o[d2][j] *= rescale;
                #pragma unroll
                for (int j = 0; j < 8; ++j) v_o[d3][j] *= rescale;
            }
            const bf16_t* p0base = VTp + n_base + row8;
            const bf16_t* p1base = VTp + n_base + 16 + row8;

            // Wave p0: 4 live V fragments, 4 WMMAs over the four independent chains.
            {
                bf16x8_t va0 = *reinterpret_cast<const bf16x8_t*>(p0base + (d0 * W_K + col16) * vt_stride_d);
                bf16x8_t vb0 = *reinterpret_cast<const bf16x8_t*>(p0base + (d1 * W_K + col16) * vt_stride_d);
                bf16x8_t vc0 = *reinterpret_cast<const bf16x8_t*>(p0base + (d2 * W_K + col16) * vt_stride_d);
                bf16x8_t vd0 = *reinterpret_cast<const bf16x8_t*>(p0base + (d3 * W_K + col16) * vt_stride_d);
                v_o[d0] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(va0, v_p0, v_o[d0]);
                v_o[d1] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(vb0, v_p0, v_o[d1]);
                v_o[d2] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(vc0, v_p0, v_o[d2]);
                v_o[d3] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(vd0, v_p0, v_o[d3]);
            }
            // Wave p1: same 4 registers reused, 4 WMMAs over the four independent chains.
            {
                bf16x8_t va1 = *reinterpret_cast<const bf16x8_t*>(p1base + (d0 * W_K + col16) * vt_stride_d);
                bf16x8_t vb1 = *reinterpret_cast<const bf16x8_t*>(p1base + (d1 * W_K + col16) * vt_stride_d);
                bf16x8_t vc1 = *reinterpret_cast<const bf16x8_t*>(p1base + (d2 * W_K + col16) * vt_stride_d);
                bf16x8_t vd1 = *reinterpret_cast<const bf16x8_t*>(p1base + (d3 * W_K + col16) * vt_stride_d);
                v_o[d0] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(va1, v_p1, v_o[d0]);
                v_o[d1] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(vb1, v_p1, v_o[d1]);
                v_o[d2] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(vc1, v_p1, v_o[d2]);
                v_o[d3] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(vd1, v_p1, v_o[d3]);
            }
        }
    }

    const fp32_t inv = (l_row > 0.0f) ? (1.0f / l_row) : 0.0f;
    bf16_t* o_row = Op + (q_m_base + col16) * stride_n;
    #pragma unroll
    for (int dt = 0; dt < DK; ++dt) {
        bf16x8_t o_pack;
        #pragma unroll
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast105(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
