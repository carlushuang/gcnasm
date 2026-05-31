// v108 -- v100 (v63 K double-buffer QKT + v43 batched pre-PV rescale) + CROSS-TILE
//          head-K prefetch across the PV->QKT boundary. Reviewer NEXT FOCUS for this
//          round (lever #3): "target the PV/QKT boundary scheduling instead of deeper
//          QKT prefetch -- v107 showed K-load latency is not worth extra QKT live state."
//
// DIAGNOSIS (the ONE exposure v100's pipeline structurally cannot cover).
// v100's QKT software-pipeline is distance-1: at the top of each n_tile it loads the
// dt=0 K head (v_k0_next/v_k1_next), then in the kt=0 iteration it consumes that load
// IMMEDIATELY (v_k0=v_k0_next) before any WMMA has run. For kt>=1 each K load overlaps
// the prior D-tile's 2 WMMAs, so those latencies are hidden. But the kt=0 HEAD load has
// nothing in front of it: the matrix pipe has just drained the previous tile's PV, then
// stalls on the dt=0 head K (re-streamed from L2, ~200 cyc) before the first QKT WMMA can
// issue. That is exactly ONE fully-exposed L2 latency per n_tile -- the single bubble the
// intra-QKT pipeline leaves on the table, and it sits precisely at the PV->QKT boundary.
//
// CHANGE. Hoist ONLY the dt=0 head-K loads out of the QKT-loop top and issue them one
// iteration ahead, during the tail of the PREVIOUS n_tile's PV loop (a region full of
// matrix-pipe WMMA latency to hide behind). A 1-iteration software pipeline at the
// KV-block granularity:
//   * prologue: prefetch n_tile=0's head K into hk0/hk1 before the loop.
//   * QKT now seeds v_k0_next/v_k1_next from hk0/hk1 (already resident) instead of
//     issuing the head load itself.
//   * right before tile i's PV loop, prefetch tile i+1's head K into hk0/hk1 (guarded by
//     n_tile+1<num) so the ~200-cyc L2 load overlaps the entire 16-WMMA PV loop. The PV
//     loop itself stays byte-identical to v100 (no in-loop branch -> unroll unperturbed).
//
// WHY THIS IS NOT v107. v107 deepened the prefetch to distance-2 INSIDE QKT, holding 4
// bf16x8 K fragments (kbuf0[2],kbuf1[2], ~16 VGPR) live across the whole QKT loop body,
// competing with the K-half WMMA schedule -> regressed to 86.59. v108 holds only 2 bf16x8
// (hk0,hk1, +8 VGPR) and their live range spans just the PV-tail -> next-QKT-head back
// edge -- it NEVER crosses the softmax window (max-reduce, bpermute, rescale, 16x exp2,
// sum-reduce, bf16 pack) and never overlaps the in-QKT K pipeline. Minimal added liveness,
// aimed at the one bubble v100 leaves, on the boundary the reviewer flagged.
//
// BIT-EXACTNESS. The dt=0 head K loads read the SAME global addresses and feed the SAME
// WMMAs in the SAME order; only their issue SITE moves earlier. v_s0/v_s1 accumulation
// order, online softmax, v43 batched rescale, and the entire PV loop are byte-for-byte
// identical to v100 -> n_bad==0 guaranteed.
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast108(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v108_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v108_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v108(opus_attn_kargs k)
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
            v_q[kt][j] = bf16_fast108(bf16_to_f32(v_q[kt][j]) * qscale);
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

    // ---- cross-tile head-K prefetch (v108) ----
    // Preload n_tile=0's dt=0 head K so the very first QKT does not stall on its head load.
    bf16x8_t hk0 = *reinterpret_cast<const bf16x8_t*>(Kp + (col16) * stride_n + row8);
    bf16x8_t hk1 = *reinterpret_cast<const bf16x8_t*>(Kp + (16 + col16) * stride_n + row8);

    for (int n_tile = 0; n_tile < num_kv_tiles; ++n_tile) {
        const int n_base = n_tile * BLOCK_N;

        fp32x8_t v_s0 = {0,0,0,0,0,0,0,0};
        fp32x8_t v_s1 = {0,0,0,0,0,0,0,0};

        // ---- QKT with software-pipelined K loads (v63) ----
        // dt=0 head K comes from the cross-tile prefetch (hk0/hk1, already resident).
        bf16x8_t v_k0_next = hk0;
        bf16x8_t v_k1_next = hk1;

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
        row_max = v108_cross_half_max(row_max);

        const fp32_t new_m = __builtin_fmaxf(m_row, row_max);
        fp32_t rescale = 1.0f;
        const bool need_rescale = (new_m != m_row);
        if (need_rescale) {
            rescale = __builtin_amdgcn_exp2f(m_row - new_m);
            l_row *= rescale;
            // ---- Batched pre-PV rescale (v43) ----
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
        row_sum = v108_cross_half_sum(row_sum);
        l_row += row_sum;

        bf16x8_t v_p0, v_p1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p0[j] = bf16_fast108(v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p1[j] = bf16_fast108(v_s1[j]);

        // ---- cross-tile head-K prefetch (v108) ----
        // Issue the NEXT n_tile's dt=0 head-K loads HERE, right before the PV loop. Their
        // ~200-cyc L2 latency then hides behind the whole PV loop (16 WMMAs) -- a hiding
        // window matched to the load latency -- and hk0/hk1 are consumed immediately at the
        // next iteration's kt=0, so their live range spans only PV + the loop back edge and
        // NEVER enters the next tile's softmax window. The PV loop below stays byte-for-byte
        // identical to v100 (no in-loop branch), so its unroll schedule is unperturbed.
        if (n_tile + 1 < num_kv_tiles) {
            const int n_next = n_base + BLOCK_N;
            hk0 = *reinterpret_cast<const bf16x8_t*>(Kp + (n_next + col16) * stride_n + row8);
            hk1 = *reinterpret_cast<const bf16x8_t*>(Kp + (n_next + 16 + col16) * stride_n + row8);
        }

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
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast108(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
