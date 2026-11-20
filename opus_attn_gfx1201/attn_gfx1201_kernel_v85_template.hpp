// v85 -- BLOCK_N=64: 4 KV columns, 4x score accumulators, 4x P packed
// Key: reuse score regs for exp2 output to stay within VGPR budget
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast85(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v85_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v85_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<int BLOCK_M_T, int BLOCK_N_T, int D_T>
struct opus_attn_traits_v85 {
    static constexpr int BLOCK_M    = BLOCK_M_T;
    static constexpr int BLOCK_N    = BLOCK_N_T;
    static constexpr int D          = D_T;
    static constexpr int W_K        = 16;
    static constexpr int D_TILES_K  = D_T / W_K;
    static constexpr int BLOCK_SIZE = (BLOCK_M_T / 16) * 32;
    static constexpr int N_COLS     = BLOCK_N_T / 16;
};

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v85(opus_attn_kargs k)
{
#if defined(__gfx1201__) || defined(__gfx1200__)
    constexpr int BLOCK_M = T::BLOCK_M;
    constexpr int BLOCK_N = T::BLOCK_N;
    constexpr int W_K     = T::W_K;
    constexpr int DK      = T::D_TILES_K;
    constexpr int N_COLS  = T::N_COLS;

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
            v_q[kt][j] = bf16_fast85(bf16_to_f32(v_q[kt][j]) * qscale);
    }

    fp32x8_t v_o[DK];
    #pragma unroll
    for (int kt = 0; kt < DK; ++kt) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_o[kt][j] = 0.0f;
    }
    fp32_t m_row = -3.4e38f;
    fp32_t l_row = 0.0f;

    const bf16_t* vt_base[DK];
    #pragma unroll
    for (int dt = 0; dt < DK; ++dt)
        vt_base[dt] = VTp + (dt * W_K + col16) * vt_stride_d + row8;

    const int num_kv_tiles = k.N / BLOCK_N;

    for (int n_tile = 0; n_tile < num_kv_tiles; ++n_tile) {
        const int n_base = n_tile * BLOCK_N;

        // QKT: N_COLS score accumulators
        fp32x8_t v_s[N_COLS];
        #pragma unroll
        for (int c = 0; c < N_COLS; ++c)
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_s[c][j] = 0.0f;

        // K double-buffer + QKT
        bf16x8_t v_k_next[N_COLS];
        #pragma unroll
        for (int c = 0; c < N_COLS; ++c)
            v_k_next[c] = *reinterpret_cast<const bf16x8_t*>(Kp + (n_base + c*16 + col16) * stride_n + row8);

        #pragma unroll
        for (int kt = 0; kt < DK; ++kt) {
            bf16x8_t v_k[N_COLS];
            #pragma unroll
            for (int c = 0; c < N_COLS; ++c)
                v_k[c] = v_k_next[c];
            if (kt + 1 < DK) {
                #pragma unroll
                for (int c = 0; c < N_COLS; ++c)
                    v_k_next[c] = *reinterpret_cast<const bf16x8_t*>(Kp + (n_base + c*16 + col16) * stride_n + (kt+1) * W_K + row8);
            }
            #pragma unroll
            for (int c = 0; c < N_COLS; ++c)
                v_s[c] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_k[c], v_q[kt], v_s[c]);
        }

        // Row max over all N_COLS × 8 scores
        fp32_t row_max = v_s[0][0];
        #pragma unroll
        for (int c = 0; c < N_COLS; ++c)
            #pragma unroll
            for (int j = (c == 0 ? 1 : 0); j < 8; ++j)
                row_max = __builtin_fmaxf(row_max, v_s[c][j]);
        row_max = v85_cross_half_max(row_max);

        const fp32_t new_m = __builtin_fmaxf(m_row, row_max);
        fp32_t rescale = 1.0f;
        const bool need_rescale = (new_m != m_row);
        if (need_rescale) {
            rescale = __builtin_amdgcn_exp2f(m_row - new_m);
            l_row *= rescale;
        }
        m_row = new_m;

        // exp2 in-place over scores
        #pragma unroll
        for (int c = 0; c < N_COLS; ++c)
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                v_s[c][j] = __builtin_amdgcn_exp2f(v_s[c][j] - new_m);

        // Row sum
        fp32_t row_sum = 0.0f;
        #pragma unroll
        for (int c = 0; c < N_COLS; ++c)
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                row_sum += v_s[c][j];
        row_sum = v85_cross_half_sum(row_sum);
        l_row += row_sum;

        // Convert P to bf16
        bf16x8_t v_p[N_COLS];
        #pragma unroll
        for (int c = 0; c < N_COLS; ++c)
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                v_p[c][j] = bf16_fast85(v_s[c][j]);

        // PV: accumulate output
        #pragma unroll
        for (int dt = 0; dt < DK; ++dt) {
            if (need_rescale) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) v_o[dt][j] *= rescale;
            }
            #pragma unroll
            for (int c = 0; c < N_COLS; ++c) {
                bf16x8_t v_v = *reinterpret_cast<const bf16x8_t*>(vt_base[dt] + n_base + c*16);
                v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_v, v_p[c], v_o[dt]);
            }
        }
    }

    const fp32_t inv = (l_row > 0.0f) ? (1.0f / l_row) : 0.0f;
    bf16_t* o_row = Op + (q_m_base + col16) * stride_n;
    #pragma unroll
    for (int dt = 0; dt < DK; ++dt) {
        bf16x8_t o_pack;
        #pragma unroll
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast85(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
