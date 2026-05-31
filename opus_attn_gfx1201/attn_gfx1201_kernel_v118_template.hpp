// v118 -- v100 (v63 distance-1 K-prefetch + v43 batched pre-PV rescale) with the QKT
//          K-prefetch turned into a DISTANCE-2 *LATE* schedule. Reviewer NEXT FOCUS:
//          "test a distance-2 late-prefetch variant where kt+2 K loads are issued AFTER
//           the current two WMMAs, preserving v100's register footprint as much as
//           possible."
//
// WHY v107 (distance-2 EARLY) FAILED (88.07 -> 86.59): it (a) copied each slot out into
// v_k0/v_k1 *and* held both rotating slots + the freshly issued load live at once
// (+VGPR / worse residency), and (b) issued the kt+2 global loads BEFORE the current
// WMMAs, so the load issue contended with the matrix issue window at the head of every
// D-tile instead of hiding behind it.
//
// v118 fixes both:
//   * NO copy-out registers: the WMMAs read K directly from the rotating slot
//     kbuf{0,1}[kt&1]. The old fragment is dead the instant its two WMMAs issue, so the
//     subsequent store into that same slot reuses the register -- live K state stays at
//     ~2 fragments per row-half, matching v100's footprint rather than v107's.
//   * LATE issue: the kt+2 loads are emitted AFTER both kt WMMAs, so they drop into the
//     memory pipe behind the matrix issue rather than ahead of it. The load still has a
//     full ~2-D-tile (4-WMMA) hiding window before it is consumed at iteration kt+2.
//
// QKT accumulation order (v_s0 then v_s1, D-tile ascending) is byte-for-byte identical
// to v100 -> bit-exact. Only K-load issue position/distance changes. v43 batched pre-PV
// rescale and the PV loop are untouched.
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

        fp32x8_t v_s0 = {0,0,0,0,0,0,0,0};
        fp32x8_t v_s1 = {0,0,0,0,0,0,0,0};

        // ---- QKT with DISTANCE-2 LATE software-pipelined K loads (v118) ----
        // Rotating 2-slot buffer holding D-tiles {kt, kt+1}. The WMMAs read K
        // directly from kbuf{0,1}[kt&1] (no copy-out), then -- AFTER both WMMAs --
        // the kt+2 load is issued into the now-dead slot. Each load thus has a
        // ~4-WMMA hiding window yet drops into the memory pipe behind the matrix
        // issue, and live K state matches v100 (slot recycled in place).
        const bf16_t* k0_row = Kp + (n_base + col16) * stride_n + row8;
        const bf16_t* k1_row = Kp + (n_base + 16 + col16) * stride_n + row8;
        bf16x8_t kbuf0[2];
        bf16x8_t kbuf1[2];
        kbuf0[0] = *reinterpret_cast<const bf16x8_t*>(k0_row);
        kbuf1[0] = *reinterpret_cast<const bf16x8_t*>(k1_row);
        if (DK > 1) {
            kbuf0[1] = *reinterpret_cast<const bf16x8_t*>(k0_row + W_K);
            kbuf1[1] = *reinterpret_cast<const bf16x8_t*>(k1_row + W_K);
        }

        #pragma unroll
        for (int kt = 0; kt < DK; ++kt) {
            const int slot = kt & 1;
            v_s0 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(kbuf0[slot], v_q[kt], v_s0);
            v_s1 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(kbuf1[slot], v_q[kt], v_s1);
            // LATE prefetch: refill the consumed slot for use at iteration kt+2.
            if (kt + 2 < DK) {
                kbuf0[slot] = *reinterpret_cast<const bf16x8_t*>(k0_row + (kt+2) * W_K);
                kbuf1[slot] = *reinterpret_cast<const bf16x8_t*>(k1_row + (kt+2) * W_K);
            }
        }

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
        row_sum = v118_cross_half_sum(row_sum);
        l_row += row_sum;

        bf16x8_t v_p0, v_p1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p0[j] = bf16_fast118(v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p1[j] = bf16_fast118(v_s1[j]);

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
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast118(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
