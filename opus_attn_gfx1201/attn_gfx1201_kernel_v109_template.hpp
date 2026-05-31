// v109 -- v100 (v63 K-prefetch + v43 batched pre-PV rescale) MINUS the QKT K
//         double-buffer. A REGISTER-PRESSURE / OCCUPANCY play, not a latency play.
//
// MOTIVATION (reviewer-guided, genuinely new direction vs v101-v108).
// Rounds 1..9 (v101..v108) all tried to ADD pipeline/prefetch state to v100 to hide
// a presumed memory-latency bubble (rolling V double-buffer, dt=0 V hoists/prologues,
// QKT accumulator split, distance-2 K prefetch, cross-tile head-K prefetch). EVERY one
// washed or regressed against v100's 88.07. The recurring failure mode the reviewer
// identified: each added live VGPRs and dropped occupancy (e.g. v108: 159->169 VGPR,
// 9->8 waves) -- "the bottleneck is occupancy and issue-window pressure, not raw K-load
// latency." v107 specifically proved deepening the K prefetch does NOT help.
//
// So this round inverts the lever: instead of adding state to hide latency, REMOVE the
// one piece of latency-hiding state that v107 already showed is not paying for itself.
// v100 inherits v63's QKT K double-buffer: it carries v_k0_next/v_k1_next (two live
// bf16x8 = 8 VGPR) across the whole DK=8 QKT loop just to prefetch one D-tile ahead.
// But the 8 D-tile K loads per n_tile are independent, contiguous, and the hardware OOO
// + s_waitcnt scoreboard already overlaps VMEM with the WMMAs without an explicit SW
// pipeline. Dropping the manual prefetch frees those live registers and shortens the
// QKT live ranges, which is exactly the pressure the reviewer says is capping us at 9
// waves. If the compiler still schedules the loads ahead (it has 8 independent loads to
// reorder freely), we keep the latency hiding for FREE and gain the occupancy headroom.
//
// Bit-exactness: arithmetic is byte-identical to v100 -- same load addresses, same WMMA
// order (v_s0 then v_s1 per D-tile), same softmax/rescale/PV. Only the *scheduling
// scaffold* (the _next temporaries) is removed; results are unchanged. The v43 batched
// pre-PV rescale is kept verbatim.
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast109(fp32_t f) {
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
            v_q[kt][j] = bf16_fast109(bf16_to_f32(v_q[kt][j]) * qscale);
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

        // ---- QKT: synchronous K loads, no manual SW pipeline (v109) ----
        // The DK=8 K loads are independent and contiguous; the HW OOO + s_waitcnt
        // scoreboard overlaps VMEM with the WMMAs on its own. Removing v100/v63's
        // v_k0_next/v_k1_next double-buffer frees 8 live VGPR and shortens QKT live
        // ranges (the occupancy/issue-window pressure the reviewer flagged), while the
        // compiler is still free to hoist all 8 loads ahead of the WMMA chain.
        #pragma unroll
        for (int kt = 0; kt < DK; ++kt) {
            bf16x8_t v_k0 = *reinterpret_cast<const bf16x8_t*>(Kp + (n_base + col16) * stride_n + kt * W_K + row8);
            bf16x8_t v_k1 = *reinterpret_cast<const bf16x8_t*>(Kp + (n_base + 16 + col16) * stride_n + kt * W_K + row8);
            v_s0 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_k0, v_q[kt], v_s0);
            v_s1 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_k1, v_q[kt], v_s1);
        }

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
            // ---- Batched pre-PV rescale (v43) ----
            // Rescale ALL O accumulators in one pass here so the PV loop below is
            // branch-free and its V loads overlap with these VALU-only FMAs.
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
        row_sum = v109_cross_half_sum(row_sum);
        l_row += row_sum;

        bf16x8_t v_p0, v_p1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p0[j] = bf16_fast109(v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p1[j] = bf16_fast109(v_s1[j]);

        // ---- PV: branch-free (rescale already applied above, v43) ----
        #pragma unroll
        for (int dt = 0; dt < DK; ++dt) {
            const bf16_t* vt_addr0 = VTp + (dt * W_K + col16) * vt_stride_d + n_base + row8;
            const bf16_t* vt_addr1 = VTp + (dt * W_K + col16) * vt_stride_d + n_base + 16 + row8;
            bf16x8_t v_v0 = *reinterpret_cast<const bf16x8_t*>(vt_addr0);
            bf16x8_t v_v1 = *reinterpret_cast<const bf16x8_t*>(vt_addr1);
            v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_v0, v_p0, v_o[dt]);
            v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_v1, v_p1, v_o[dt]);
        }
    }

    const fp32_t inv = (l_row > 0.0f) ? (1.0f / l_row) : 0.0f;
    bf16_t* o_row = Op + (q_m_base + col16) * stride_n;
    #pragma unroll
    for (int dt = 0; dt < DK; ++dt) {
        bf16x8_t o_pack;
        #pragma unroll
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast109(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
