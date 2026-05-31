// v117 -- v100 (v63 K double-buffer QKT + v43 batched pre-PV rescale) + CROSS-TILE
//          head-K prefetch issued in the SOFTMAX VALU window. Reviewer NEXT FOCUS:
//          "K-prefetch distance/schedule variant that improves load-to-WMMA latency
//           hiding WITHOUT adding score-accumulator VGPRs" (v106's +16-VGPR split failed;
//           v107's distance-2 in-loop buffer regressed to 86.59).
//
// DIAGNOSIS (the one bubble v100's intra-QKT pipeline structurally cannot cover).
// v100's QKT software-pipeline is distance-1: at the top of each n_tile it loads the dt=0
// head K (v_k0_next/v_k1_next) and consumes it IMMEDIATELY at kt=0 before any WMMA runs.
// For kt>=1 the K loads overlap the prior D-tile's WMMAs, so they hide. But the kt=0 HEAD
// load has nothing in front of it -- the matrix pipe drains the previous tile's PV, then
// stalls on this dt=0 K (re-streamed from L2, ~200 cyc) before the first QKT WMMA. That is
// one fully-exposed L2 latency per n_tile, sitting at the PV->QKT boundary.
//
// WHY THE SOFTMAX WINDOW (the new lever, distinct from v108). The exposed head load needs
// a hiding region that is (a) long enough for ~200 cyc and (b) NOT already saturating the
// memory pipe. v108 issued the next tile's head-K right before the PV loop -- but PV then
// fires 16 V-fragment loads, so the head-K competes with V bandwidth in the busiest memory
// phase. The softmax phase is the opposite: after the QKT loop, K is dead, and the kernel
// runs a pure-VALU storm -- the row-max tree reduce, 16 v_exp2f transcendentals, the row-
// sum reduce, and 16 bf16 packs -- with the memory pipe COMPLETELY IDLE. Issuing the next
// tile's dt=0 head-K right after the QKT loop drops it into that idle slot; its L2 latency
// then hides behind the entire softmax + the whole PV loop (a much longer, uncontended
// window than v108's PV-only span) and hk0/hk1 are consumed at the next kt=0.
//
// COST. +2 bf16x8 live (hk0,hk1 ~ +8 VGPR), an order of magnitude cheaper than v106's +2
// fp32x8 score accumulators (+16 VGPR + a 16-add merge on the softmax critical path). The
// two QKT chains (v_s0,v_s1), WMMA issue order, and fp32 accumulation order are byte-for-
// byte identical to v100 -> bit-exact. Only the global head-K load is moved earlier.
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast117(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v117_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v117_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v117(opus_attn_kargs k)
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
            v_q[kt][j] = bf16_fast117(bf16_to_f32(v_q[kt][j]) * qscale);
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

    // ---- cross-tile head-K prefetch (v117) ----
    // Preload n_tile=0's dt=0 head K so the very first QKT does not stall on its head
    // load. Subsequent tiles' head K is prefetched inside the softmax window below.
    bf16x8_t hk0 = *reinterpret_cast<const bf16x8_t*>(Kp + (col16) * stride_n + row8);
    bf16x8_t hk1 = *reinterpret_cast<const bf16x8_t*>(Kp + (16 + col16) * stride_n + row8);

    for (int n_tile = 0; n_tile < num_kv_tiles; ++n_tile) {
        const int n_base = n_tile * BLOCK_N;

        fp32x8_t v_s0 = {0,0,0,0,0,0,0,0};
        fp32x8_t v_s1 = {0,0,0,0,0,0,0,0};

        // ---- QKT with software-pipelined K loads (v63) ----
        // dt=0 head K comes from the cross-tile prefetch (hk0/hk1, already resident);
        // for kt>=1 the next D-tile's K is prefetched while the current WMMA runs.
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

        // ---- cross-tile head-K prefetch (v117), issued in the SOFTMAX VALU window ----
        // K is now dead and the memory pipe is idle while the pure-VALU softmax storm
        // (row-max reduce, 16 v_exp2f, row-sum reduce, 16 bf16 packs) and the whole PV
        // loop run. Issue the NEXT tile's dt=0 head-K here so its ~200-cyc L2 latency
        // hides behind that long, uncontended window; hk0/hk1 are consumed at the next
        // iteration's kt=0. Unlike v108 (which issued this right before the PV loop, where
        // 16 V-fragment loads contend for memory bandwidth), this slot has zero competing
        // loads. The QKT/softmax/PV math below is byte-for-byte identical to v100.
        if (n_tile + 1 < num_kv_tiles) {
            const int n_next = n_base + BLOCK_N;
            hk0 = *reinterpret_cast<const bf16x8_t*>(Kp + (n_next + col16) * stride_n + row8);
            hk1 = *reinterpret_cast<const bf16x8_t*>(Kp + (n_next + 16 + col16) * stride_n + row8);
        }

        fp32_t row_max = v_s0[0];
        #pragma unroll
        for (int j = 1; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s1[j]);
        row_max = v117_cross_half_max(row_max);

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
        row_sum = v117_cross_half_sum(row_sum);
        l_row += row_sum;

        bf16x8_t v_p0, v_p1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p0[j] = bf16_fast117(v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p1[j] = bf16_fast117(v_s1[j]);

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
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast117(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
