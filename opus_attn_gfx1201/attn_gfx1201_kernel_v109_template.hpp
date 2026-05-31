// v109 -- v106 (best, 90.2 TFLOPS) + ONE-PAIR-AHEAD SOFTWARE-PIPELINED PV V-PREFETCH.
//
// Base choice: v106 (the running best). QK^T is kept BYTE-FOR-BYTE identical to v106
// (two-ahead K prefetch ring), and v106's proven online-rescale schedule (prologue
// rescales pair 0; iteration dt rescales the NEXT pair's accumulators before its WMMAs)
// is preserved BYTE-FOR-BYTE. The PV phase retains v106's 2-D-tile grouped structure
// (4 V fragments live, two independent accumulator chains per pair).
//
// WHY the rescale-placement knob is exhausted: v106 (all 16 rescale FMAs BEFORE the
// contiguous 4-WMMA quartet) = 90.2; v108 (8 before + 8 between the two WMMA pairs) =
// 88.65. More pre-WMMA filler wins, and v106 already places ALL of it before a dense
// quartet. Redistributing those FMAs (the 12/4 split) only re-fragments matrix-pipe
// cadence -> lands between v108 and v106. So we stop tuning rescale placement.
//
// What changed vs v106 (the ONLY change): the PV phase now PREFETCHES the next pair's
// 4 V fragments one iteration ahead (a 1-pair V ring). v106 loads all 4 V frags at the
// head of each pair's iteration and immediately consumes them in the WMMA quartet ->
// load-use distance ~= 0. Those loads are only hidden when need_rescale==true (the
// rescale FMAs fill the shadow). But late in the KV loop the running max stabilizes and
// need_rescale becomes FALSE on most iterations -- then the PV V-loads are FULLY EXPOSED
// and the matrix pipe stalls on vmcnt at every pair head. v109 hoists the next pair's
// V loads to issue BEFORE the current pair's WMMAs, so each WMMA quartet consumes V
// fragments that were fetched a full pair ago: the global-load latency overlaps the
// previous quartet's matrix-pipe issue window regardless of need_rescale. This is
// playbook lever 3 ("V software-pipelining in the PV phase, mirroring the K pipeline,
// +1-3 TFLOPS") -- COMPLEMENTARY to v106's rescale schedule, not competing with it.
//
// Arithmetic identical to v106: V fragments are byte-for-byte the same values, just
// loaded earlier; each v_o[dt] is rescaled EXACTLY ONCE before its PV accumulation,
// same rescale value, same WMMA operand order -> bit-exact in bf16. VGPR: +4 live bf16x8
// fragments for the next-pair ring (the current AND next pair's V are live across the
// WMMAs) -- a modest bump; if it costs a wave it should still net positive since the
// exposed-load stall it removes is on the common-case path. RISK: low-med (the extra
// V-fragment liveness; watch the 256-VGPR / 7-wave ceiling).
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast106(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v109_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v109_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v109(opus_attn_kargs k)
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
            v_q[kt][j] = bf16_fast106(bf16_to_f32(v_q[kt][j]) * qscale);
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

        // Softmax over 16 values (8 from s0, 8 from s1)
        fp32_t row_max = v_s0[0];
        #pragma unroll
        for (int j = 1; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s1[j]);
        row_max = v109_cross_half_max(row_max);

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
        row_sum = v109_cross_half_sum(row_sum);
        l_row += row_sum;

        bf16x8_t v_p0, v_p1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p0[j] = bf16_fast106(v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p1[j] = bf16_fast106(v_s1[j]);

        // ---- PV: v106's 2-D-tile grouped PV (two independent accumulator chains per
        // pair) + v106's one-pair-ahead rescale schedule (byte-for-byte) + a NEW 1-pair
        // V-prefetch ring so each WMMA quartet consumes V fetched a full pair earlier.
        // Each accumulator is rescaled exactly once before its PV WMMA -> bit-exact.
        // DK=8 is even -> exact pairing, no remainder.
        if (need_rescale) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_o[0][j] *= rescale;
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_o[1][j] *= rescale;
        }
        // 1-PAIR V RING: preload pair 0's 4 fragments before the loop. The loop body then
        // prefetches the NEXT pair before issuing the CURRENT pair's WMMAs, so every WMMA
        // quartet consumes V fetched a full pair earlier -- the global-load latency hides
        // in the previous quartet's matrix-pipe issue window, independent of need_rescale.
        bf16x8_t cur_va0 = *reinterpret_cast<const bf16x8_t*>(VTp + (0 * W_K + col16) * vt_stride_d + n_base + row8);
        bf16x8_t cur_va1 = *reinterpret_cast<const bf16x8_t*>(VTp + (0 * W_K + col16) * vt_stride_d + n_base + 16 + row8);
        bf16x8_t cur_vb0 = *reinterpret_cast<const bf16x8_t*>(VTp + (1 * W_K + col16) * vt_stride_d + n_base + row8);
        bf16x8_t cur_vb1 = *reinterpret_cast<const bf16x8_t*>(VTp + (1 * W_K + col16) * vt_stride_d + n_base + 16 + row8);
        #pragma unroll
        for (int dt = 0; dt < DK; dt += 2) {
            const int dt1 = dt + 1;
            // Prefetch the NEXT pair's 4 V fragments (issue ahead of the current WMMAs).
            bf16x8_t nxt_va0, nxt_va1, nxt_vb0, nxt_vb1;
            if (dt + 2 < DK) {
                const int np  = dt + 2;
                const int np1 = dt + 3;
                nxt_va0 = *reinterpret_cast<const bf16x8_t*>(VTp + (np  * W_K + col16) * vt_stride_d + n_base + row8);
                nxt_va1 = *reinterpret_cast<const bf16x8_t*>(VTp + (np  * W_K + col16) * vt_stride_d + n_base + 16 + row8);
                nxt_vb0 = *reinterpret_cast<const bf16x8_t*>(VTp + (np1 * W_K + col16) * vt_stride_d + n_base + row8);
                nxt_vb1 = *reinterpret_cast<const bf16x8_t*>(VTp + (np1 * W_K + col16) * vt_stride_d + n_base + 16 + row8);
            }
            // v106 rescale schedule, byte-for-byte: prepare the NEXT pair's accumulators
            // while the next pair's V loads are in flight. Independent of the WMMAs below.
            if (dt + 2 < DK && need_rescale) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) v_o[dt + 2][j] *= rescale;
                #pragma unroll
                for (int j = 0; j < 8; ++j) v_o[dt + 3][j] *= rescale;
            }
            // Current pair WMMAs consume V fetched a pair ago (cur_* == v106's va0/va1/vb0/vb1).
            v_o[dt]  = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(cur_va0, v_p0, v_o[dt]);
            v_o[dt1] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(cur_vb0, v_p0, v_o[dt1]);
            v_o[dt]  = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(cur_va1, v_p1, v_o[dt]);
            v_o[dt1] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(cur_vb1, v_p1, v_o[dt1]);
            // Advance the ring (compile-time register rename under full unroll).
            cur_va0 = nxt_va0; cur_va1 = nxt_va1;
            cur_vb0 = nxt_vb0; cur_vb1 = nxt_vb1;
        }
    }

    const fp32_t inv = (l_row > 0.0f) ? (1.0f / l_row) : 0.0f;
    bf16_t* o_row = Op + (q_m_base + col16) * stride_n;
    #pragma unroll
    for (int dt = 0; dt < DK; ++dt) {
        bf16x8_t o_pack;
        #pragma unroll
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast106(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
