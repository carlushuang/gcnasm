// v32 -- unconditional rescale + pipelined V prefetch + batched WMMAs
//
// Changes from v27:
//   1. Unconditional rescale: always do v_o *= exp2(m_old - m_new).
//      When m is unchanged, exp2(0) = 1.0 so multiply is harmless.
//      Eliminates 8 s_and_saveexec/s_or_b32/s_wait_alu/s_cbranch_execz
//      branch sequences per PV phase (~32 instructions saved).
//   2. Inline asm for rescale prevents compiler from reintroducing branches.
//   3. V global_load prefetch: issue V[dt+1] load before wave_barrier/ds_load
//      for V[dt], overlapping ~300 cycle global memory latency with LDS ops.
//   4. Batch all 8 WMMAs at the end with staggered ds_load waits, matching
//      v27's pipeline but without the branch overhead.
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast32(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v32_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v32_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v32(opus_attn_kargs k)
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

    __shared__ bf16_t s_v[8][16 * 16];

    // Load and scale Q
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
            v_q[kt][j] = bf16_fast32(bf16_to_f32(v_q[kt][j]) * qscale);
    }

    // Initialize O accumulators
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

        // ==== QKT phase ====
        fp32x8_t v_s = {0,0,0,0,0,0,0,0};
        #pragma unroll
        for (int kt = 0; kt < DK; ++kt) {
            const bf16_t* k_addr = Kp + (n_base + col16) * stride_n + kt * W_K + row8;
            bf16x8_t v_k = *reinterpret_cast<const bf16x8_t*>(k_addr);
            v_s = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_k, v_q[kt], v_s);
        }

        // ==== Softmax ====
        fp32_t row_max = v_s[0];
        #pragma unroll
        for (int j = 1; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s[j]);
        row_max = v32_cross_half_max(row_max);

        const fp32_t new_m = __builtin_fmaxf(m_row, row_max);

        // Unconditional rescale: exp2(m_old - m_new).
        // When m_row == new_m, rescale = exp2(0) = 1.0 -- multiplies are no-ops.
        // Saves ~32 instructions (branch overhead) across 8 D-tiles vs v27.
        const fp32_t rescale = __builtin_amdgcn_exp2f(m_row - new_m);
        l_row *= rescale;
        m_row = new_m;

        #pragma unroll
        for (int j = 0; j < 8; ++j) v_s[j] = __builtin_amdgcn_exp2f(v_s[j] - new_m);
        fp32_t row_sum = v_s[0];
        #pragma unroll
        for (int j = 1; j < 8; ++j) row_sum += v_s[j];
        row_sum = v32_cross_half_sum(row_sum);
        l_row += row_sum;

        // Convert P to bf16 for WMMA
        bf16x8_t v_p;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p[j] = bf16_fast32(v_s[j]);

        // ==== PV phase ====
        // For each D-tile:
        //   1. Rescale v_o[dt] unconditionally (inline asm, no branch)
        //   2. Store current V tile to LDS for transpose
        //   3. Prefetch V[dt+1] from global memory (overlaps with LDS ops)
        //   4. Wave barrier + ds_load transposed V
        // Then batch all 8 WMMAs.

        const bf16_t* v_n_base = Vp + (n_base + col16) * stride_n;

        // Prefetch V for dt=0 (issued early to start hiding latency)
        bf16x8_t v_load = *reinterpret_cast<const bf16x8_t*>(v_n_base + row8);

        bf16x8_t v_v[DK]; // transposed V fragments for batched WMMAs

        #pragma unroll
        for (int dt = 0; dt < DK; ++dt) {
            // Unconditional rescale via inline asm.
            // Using asm volatile prevents the compiler from:
            //   a) Converting to conditional execution (s_and_saveexec pattern)
            //   b) Inserting unnecessary s_delay_alu between independent v_mul_f32
            // All 8 multiplies write different VGPRs -- no data dependency between them.
            __asm__ volatile(
                "v_mul_f32 %0, %8, %0\n"
                "v_mul_f32 %1, %8, %1\n"
                "v_mul_f32 %2, %8, %2\n"
                "v_mul_f32 %3, %8, %3\n"
                "v_mul_f32 %4, %8, %4\n"
                "v_mul_f32 %5, %8, %5\n"
                "v_mul_f32 %6, %8, %6\n"
                "v_mul_f32 %7, %8, %7\n"
                : "+v"(v_o[dt][0]), "+v"(v_o[dt][1]), "+v"(v_o[dt][2]), "+v"(v_o[dt][3]),
                  "+v"(v_o[dt][4]), "+v"(v_o[dt][5]), "+v"(v_o[dt][6]), "+v"(v_o[dt][7])
                : "v"(rescale)
            );

            // Store V to LDS for transpose (v_load has current dt's data)
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                s_v[wave_id][(row8 + j) * 16 + col16] = v_load[j];

            // Prefetch V for next D-tile BEFORE wave_barrier.
            // The global_load is issued now and will complete asynchronously
            // while we do the wave_barrier + ds_load below.
            if (dt + 1 < DK) {
                v_load = *reinterpret_cast<const bf16x8_t*>(v_n_base + (dt + 1) * W_K + row8);
            }

            __builtin_amdgcn_wave_barrier();

            // Read transposed V from LDS
            v_v[dt] = *reinterpret_cast<const bf16x8_t*>(
                &s_v[wave_id][col16 * 16 + row8]);
        }

        // Batch all WMMAs -- compiler will insert appropriate s_wait_dscnt
        // before each WMMA based on the ds_load issue order.
        #pragma unroll
        for (int dt = 0; dt < DK; ++dt) {
            v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_v[dt], v_p, v_o[dt]);
        }
    }

    // ==== Output normalization and store ====
    const fp32_t inv = (l_row > 0.0f) ? (1.0f / l_row) : 0.0f;
    bf16_t* o_row = Op + (q_m_base + col16) * stride_n;
    #pragma unroll
    for (int dt = 0; dt < DK; ++dt) {
        bf16x8_t o_pack;
        #pragma unroll
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast32(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
