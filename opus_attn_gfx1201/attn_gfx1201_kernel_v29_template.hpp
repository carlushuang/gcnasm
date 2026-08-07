// v29 — v27 + double-buffered V transpose (overlap V[dt+1] LDS store with V[dt] WMMA)
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast29(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v29_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v29_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v29(opus_attn_kargs k)
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
    const bf16_t* __restrict__ Vp = reinterpret_cast<const bf16_t*>(k.ptr_v) + b * stride_b + h * stride_h;
    bf16_t*       __restrict__ Op = reinterpret_cast<bf16_t*>(k.ptr_o) + b * stride_b + h * stride_h;

    // Double-buffered V transpose: 2 slots of 16×16
    __shared__ bf16_t s_v[8][2][16 * 16];

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
            v_q[kt][j] = bf16_fast29(bf16_to_f32(v_q[kt][j]) * qscale);
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

        fp32x8_t v_s = {0,0,0,0,0,0,0,0};
        #pragma unroll
        for (int kt = 0; kt < DK; ++kt) {
            const bf16_t* k_addr = Kp + (n_base + col16) * stride_n + kt * W_K + row8;
            bf16x8_t v_k = *reinterpret_cast<const bf16x8_t*>(k_addr);
            v_s = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_k, v_q[kt], v_s);
        }

        fp32_t row_max = v_s[0];
        #pragma unroll
        for (int j = 1; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s[j]);
        row_max = v29_cross_half_max(row_max);

        const fp32_t new_m = __builtin_fmaxf(m_row, row_max);
        fp32_t rescale = 1.0f;
        const bool need_rescale = (new_m != m_row);
        if (need_rescale) {
            rescale = __builtin_amdgcn_exp2f(m_row - new_m);
            l_row *= rescale;
        }
        m_row = new_m;

        #pragma unroll
        for (int j = 0; j < 8; ++j) v_s[j] = __builtin_amdgcn_exp2f(v_s[j] - new_m);
        fp32_t row_sum = v_s[0];
        #pragma unroll
        for (int j = 1; j < 8; ++j) row_sum += v_s[j];
        row_sum = v29_cross_half_sum(row_sum);
        l_row += row_sum;

        bf16x8_t v_p;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p[j] = bf16_fast29(v_s[j]);

        // Double-buffered PV with fused rescale:
        // Prefetch V[dt=0] into buffer 0
        {
            const bf16_t* v_addr = Vp + (n_base + col16) * stride_n + row8;
            bf16x8_t v_load = *reinterpret_cast<const bf16x8_t*>(v_addr);
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                s_v[wave_id][0][(row8 + j) * 16 + col16] = v_load[j];
        }

        #pragma unroll
        for (int dt = 0; dt < DK; ++dt) {
            const int cur_buf = dt & 1;
            const int nxt_buf = (dt + 1) & 1;

            // Prefetch V[dt+1] into nxt_buf (if not last)
            bf16x8_t v_next_load;
            if (dt + 1 < DK) {
                const bf16_t* v_addr_next = Vp + (n_base + col16) * stride_n + (dt + 1) * W_K + row8;
                v_next_load = *reinterpret_cast<const bf16x8_t*>(v_addr_next);
            }

            // Wait for cur_buf store to complete, then read transposed
            __builtin_amdgcn_wave_barrier();
            bf16x8_t v_v = *reinterpret_cast<const bf16x8_t*>(
                &s_v[wave_id][cur_buf][col16 * 16 + row8]);

            // Store V[dt+1] while WMMA runs on V[dt]
            if (dt + 1 < DK) {
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                    s_v[wave_id][nxt_buf][(row8 + j) * 16 + col16] = v_next_load[j];
            }

            // Fused rescale before WMMA
            if (need_rescale) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) v_o[dt][j] *= rescale;
            }

            v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_v, v_p, v_o[dt]);
        }
    }

    const fp32_t inv = (l_row > 0.0f) ? (1.0f / l_row) : 0.0f;
    bf16_t* o_row = Op + (q_m_base + col16) * stride_n;
    #pragma unroll
    for (int dt = 0; dt < DK; ++dt) {
        bf16x8_t o_pack;
        #pragma unroll
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast29(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
