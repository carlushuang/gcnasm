// v103 -- v101 (two-ahead K prefetch in QK^T) + HYBRID 2-D-tile GROUPED PV rescale.
//
// Base choice: v101 (the running best, 89.15 TFLOPS) = v100 + two-ahead K ring in
// QK^T. QK^T is kept BYTE-FOR-BYTE identical to v101. The ONLY change is the PV phase.
//
// Round-3 result: v102 hoisted the rescale of ALL 8 O accumulators into one batched
// pass before PV (v43 style). It REGRESSED to 87.25 (vs v101 89.15). Root cause per
// reviewer: the all-8 rescale writes form one long VALU dependency block that must
// retire before ANY PV WMMA can accumulate, so the matrix pipe starves at the head of
// PV and WMMA issue cadence drops. v101's per-dt fusion (rescale one acc, load its V,
// 2 WMMAs) keeps short independent windows but exposes only ONE dependent WMMA chain
// at a time (v_o[dt] -> v_o[dt]) and re-evaluates the need_rescale branch 8x.
//
// Hypothesis (reviewer's explicit NEXT FOCUS): a HYBRID grouping of 2 D-tiles is the
// sweet spot. Process dt in pairs (dt, dt+1): rescale BOTH accumulators, issue all 4
// V loads for the pair, then the 4 WMMAs. This exposes TWO INDEPENDENT WMMA chains
// (v_o[dt] and v_o[dt+1] never alias) so the matrix pipe can interleave them and hide
// each WMMA's ~12.6-cyc result latency behind the other chain -- better issue cadence
// than v101's single-chain-per-region. The 4 concurrent V loads add memory-level
// parallelism. Crucially the rescale block is only 2 accumulators (16 FMAs) deep, not
// 8 (64 FMAs), so PV WMMA can start after a short prefix instead of v102's long stall.
//
// Arithmetic is identical to v101: each v_o[dt] is rescaled exactly once before its
// PV accumulation, same WMMA operand order, same rescale value -> bit-exact in bf16.
// VGPR: same accumulators, V buffers transient per pair (4 bf16x8 live briefly, same
// as v101's 2 + the next pair the compiler may overlap). No new persistent state vs
// v101 -> occupancy unchanged. RISK: low; if grouping helps nothing it ties v101.
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast103(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v103_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v103_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v103(opus_attn_kargs k)
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
            v_q[kt][j] = bf16_fast103(bf16_to_f32(v_q[kt][j]) * qscale);
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
        // issues kt+2's loads into the same slot before the WMMAs run. This doubles the
        // in-flight K fetch distance vs v100's single-stage prefetch to better hide
        // global VMEM latency behind the WMMA issue window.
        fp32x8_t v_s0 = {0,0,0,0,0,0,0,0};
        fp32x8_t v_s1 = {0,0,0,0,0,0,0,0};

        const bf16_t* kbase0 = Kp + (n_base + col16) * stride_n + row8;
        const bf16_t* kbase1 = Kp + (n_base + 16 + col16) * stride_n + row8;
        bf16x8_t kr0[2];
        bf16x8_t kr1[2];
        // Preload the first two D-tiles into the ring.
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
        row_max = v103_cross_half_max(row_max);

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
        row_sum = v103_cross_half_sum(row_sum);
        l_row += row_sum;

        bf16x8_t v_p0, v_p1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p0[j] = bf16_fast103(v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p1[j] = bf16_fast103(v_s1[j]);

        // ---- PV: HYBRID 2-D-tile grouped rescale (v103). Process D-tiles in pairs
        // (dt, dt+1): rescale BOTH O accumulators, issue all 4 V loads for the pair,
        // then 4 WMMAs across two INDEPENDENT accumulator chains (v_o[dt], v_o[dt+1]
        // never alias) so the matrix pipe can interleave them and hide each WMMA's
        // result latency behind the other chain. Rescale block is only 2 accumulators
        // deep (vs v102's 8) so PV WMMA starts after a short prefix, not a long stall.
        // DK=8 is even -> exact pairing, no remainder. V loads stay direct (a V
        // next-buffer was a measured regression on this box).
        #pragma unroll
        for (int dt = 0; dt < DK; dt += 2) {
            const int dt1 = dt + 1;
            if (need_rescale) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) v_o[dt][j]  *= rescale;
                #pragma unroll
                for (int j = 0; j < 8; ++j) v_o[dt1][j] *= rescale;
            }
            const bf16_t* a0 = VTp + (dt  * W_K + col16) * vt_stride_d + n_base + row8;
            const bf16_t* a1 = VTp + (dt  * W_K + col16) * vt_stride_d + n_base + 16 + row8;
            const bf16_t* b0 = VTp + (dt1 * W_K + col16) * vt_stride_d + n_base + row8;
            const bf16_t* b1 = VTp + (dt1 * W_K + col16) * vt_stride_d + n_base + 16 + row8;
            bf16x8_t va0 = *reinterpret_cast<const bf16x8_t*>(a0);
            bf16x8_t va1 = *reinterpret_cast<const bf16x8_t*>(a1);
            bf16x8_t vb0 = *reinterpret_cast<const bf16x8_t*>(b0);
            bf16x8_t vb1 = *reinterpret_cast<const bf16x8_t*>(b1);
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
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast103(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
