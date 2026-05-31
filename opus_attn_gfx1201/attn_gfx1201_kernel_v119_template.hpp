// v119 -- v100 + TRUE KV-block software pipeline (n_pipe=2 at the matrix-pipe level).
//
// Base: v100 (measured champion, 90.45 TFLOPS) — the v39 fused-PV-rescale family with
// K-load software pipelining inside QK^T. v101..v118 (18 versions) all stayed inside
// v100's single-tile loop and only reshuffled PV / softmax / prefetched K, and NONE
// beat v100. That local-reorder class is exhausted.
//
// The untouched bottleneck is structural: in v100 each KV tile runs strictly
//     QK^T(t) -> softmax(t) -> PV(t)
// so the softmax wedge (exp2 + max/sum cross-half reduce + bf16 pack) executes with the
// WMMA *matrix* pipe completely IDLE. It is the last large exposed serial region.
//
// Lever (playbook #4, "n_pipe=2 at the KV-block granularity"): the QK^T of tile t+1 is
// data-independent of the softmax of tile t (it reads only Q and K[t+1]). So we hoist
// the next tile's QK^T WMMAs to run BEFORE PV(t):
//     softmax(t)  [VALU]   --->  QK^T(t+1) [MATRIX, overlaps softmax drain]  --->  PV(t)
// The independent matrix work of QK^T(t+1) now fills the softmax-VALU shadow that was
// previously a matrix-pipe bubble, and its K-load latency is hidden behind softmax+PV.
// This is distinct from v116 (which only PREFETCHED K across tiles but still issued the
// QK^T WMMAs inside the next iteration, after the v_o dependency chain blocked hoisting).
//
// Bit-exactness: per-tile arithmetic order is byte-for-byte v100. We only changed *when*
// the (independent) QK^T(t+1) WMMAs are emitted relative to softmax(t)/PV(t); the values
// they produce, the m_row/l_row update sequence, the rescale, the pack, and the v_o
// accumulation order are all unchanged. -> bit-exact in bf16.
//
// VGPR cost: one extra score pair (v_s0_next/v_s1_next = 16 fp32) live across PV, over
// v100's ~205. Should stay under 256 (perhaps dropping 7->6 waves/SIMD); the kernel is
// not occupancy-bound (v117 added 2 waves with no gain), so trading a wave for a busy
// matrix pipe during softmax is the intended bet.
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast119(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v119_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v119_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v119(opus_attn_kargs k)
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
            v_q[kt][j] = bf16_fast119(bf16_to_f32(v_q[kt][j]) * qscale);
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

    // ---- QK^T for a KV tile (K-load software pipelined, identical to v100's QK^T) ----
    // Emits 2*DK independent WMMAs accumulating into s0,s1. No dependence on softmax/PV
    // state -> safe to hoist ahead of the current tile's PV.
    auto qkt = [&](int n_base, fp32x8_t& s0, fp32x8_t& s1) {
        s0 = (fp32x8_t){0,0,0,0,0,0,0,0};
        s1 = (fp32x8_t){0,0,0,0,0,0,0,0};
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
            s0 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_k0, v_q[kt], s0);
            s1 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_k1, v_q[kt], s1);
        }
    };

    // Prologue: QK^T of tile 0.
    fp32x8_t v_s0, v_s1;
    qkt(0, v_s0, v_s1);

    for (int n_tile = 0; n_tile < num_kv_tiles; ++n_tile) {
        const int n_base = n_tile * BLOCK_N;

        // ---- softmax over the 16 current scores (8 from s0, 8 from s1) [VALU only] ----
        fp32_t row_max = v_s0[0];
        #pragma unroll
        for (int j = 1; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s1[j]);
        row_max = v119_cross_half_max(row_max);

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
        row_sum = v119_cross_half_sum(row_sum);
        l_row += row_sum;

        bf16x8_t v_p0, v_p1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p0[j] = bf16_fast119(v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p1[j] = bf16_fast119(v_s1[j]);

        // ---- KV pipeline: issue NEXT tile's QK^T (matrix) BEFORE PV(t). These WMMAs are
        // independent of the softmax above and of v_o, so they fill the matrix-pipe shadow
        // of the softmax VALU and prefetch+consume K[t+1] under PV(t). ----
        fp32x8_t v_s0_next, v_s1_next;
        const bool has_next = (n_tile + 1 < num_kv_tiles);
        if (has_next) {
            qkt((n_tile + 1) * BLOCK_N, v_s0_next, v_s1_next);
        }

        // ---- PV: two WMMAs per D-tile, rescale fused per D-tile (v100 structure) ----
        #pragma unroll
        for (int dt = 0; dt < DK; ++dt) {
            if (need_rescale) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) v_o[dt][j] *= rescale;
            }
            const bf16_t* vt_addr0 = VTp + (dt * W_K + col16) * vt_stride_d + n_base + row8;
            const bf16_t* vt_addr1 = VTp + (dt * W_K + col16) * vt_stride_d + n_base + 16 + row8;
            bf16x8_t v_v0 = *reinterpret_cast<const bf16x8_t*>(vt_addr0);
            bf16x8_t v_v1 = *reinterpret_cast<const bf16x8_t*>(vt_addr1);
            v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_v0, v_p0, v_o[dt]);
            v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_v1, v_p1, v_o[dt]);
        }

        // Roll the pipeline: next tile's scores become current.
        v_s0 = v_s0_next;
        v_s1 = v_s1_next;
    }

    const fp32_t inv = (l_row > 0.0f) ? (1.0f / l_row) : 0.0f;
    bf16_t* o_row = Op + (q_m_base + col16) * stride_n;
    #pragma unroll
    for (int dt = 0; dt < DK; ++dt) {
        bf16x8_t o_pack;
        #pragma unroll
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast119(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
