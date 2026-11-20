// v33 — 2-wave cooperative D-tiling for 2 waves/SIMD occupancy
// Each wave pair shares the same 16 Q rows, splitting D=128 into two D=64 halves.
// Wave A (d_half=0) handles D[0:63], Wave B (d_half=1) handles D[64:127].
// Partial QKT sums exchanged via LDS to reconstruct full attention scores.
// Target: <=128 VGPRs per wave → 2 waves/SIMD on gfx1201 (256 VGPRs/SIMD).
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

// v33 traits: 16 waves, 512 threads, cooperative D-tiling
struct v33_traits {
    static constexpr int BLOCK_M    = 128;
    static constexpr int BLOCK_N    = 16;
    static constexpr int D          = 128;
    static constexpr int W_K        = 16;
    static constexpr int WARP_SIZE  = 32;
    static constexpr int WAVE_M     = 16;      // each wave pair does 16 M-rows
    static constexpr int NUM_PAIRS  = BLOCK_M / WAVE_M;  // 8 wave pairs
    static constexpr int NUM_WAVES  = NUM_PAIRS * 2;      // 16 waves total
    static constexpr int BLOCK_SIZE = NUM_WAVES * WARP_SIZE; // 512 threads
    static constexpr int DK         = D / W_K;             // 8 total D-tiles
    static constexpr int DK_HALF    = DK / 2;              // 4 D-tiles per wave
};

__device__ static inline bf16_t bf16_fast33(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

// Within-wave cross-half operations (lanes 0-15 <-> lanes 16-31)
__device__ static inline fp32_t v33_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v33_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

__launch_bounds__(v33_traits::BLOCK_SIZE, 2)
__global__ void opus_attn_gfx1201_kernel_v33(opus_attn_kargs k)
{
#if defined(__gfx1201__) || defined(__gfx1200__)
    constexpr int BLOCK_M  = v33_traits::BLOCK_M;
    constexpr int BLOCK_N  = v33_traits::BLOCK_N;
    constexpr int W_K      = v33_traits::W_K;
    constexpr int DK       = v33_traits::DK;
    constexpr int DK_HALF  = v33_traits::DK_HALF;

    const int tid     = static_cast<int>(threadIdx.x);
    const int wave_id = tid / 32;
    const int lane    = tid % 32;
    const int col16   = lane % 16;
    const int row_grp = lane / 16;
    const int row8    = row_grp * 8;

    // Wave pair decomposition
    const int wave_pair = wave_id / 2;  // which 16 Q-rows (0..7)
    const int d_half    = wave_id % 2;  // 0 = D[0:63], 1 = D[64:127]
    const int d_offset  = d_half * DK_HALF; // D-tile offset: 0 or 4

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

    // LDS layout:
    // [0] V transpose scratch: 16 waves × 16×16 bf16 = 16 × 256 × 2B = 8KB
    // [1] Partial QKT exchange: 8 pairs × 2 halves × 32 lanes × 8 values
    //     Layout [pair][d_half][elem][lane] for bank-conflict-free access:
    //     all 32 lanes hit distinct banks when writing the same elem index.
    //     Double-buffered (2 buffers) to eliminate the second __syncthreads().
    //     = 2 × 8 × 2 × 8 × 32 × 4B = 32KB
    // Total LDS: ~40KB (within 64KB limit)
    __shared__ bf16_t s_v[16][16 * 16];  // V transpose per wave
    // Partial S exchange: [buffer][wave_pair][d_half][elem][lane]
    __shared__ fp32_t s_partial[2][8][2][8][32];

    // Pre-load Q for this wave's D-half only (4 D-tiles instead of 8)
    const int q_m_base = q_tile_id * BLOCK_M + wave_pair * 16;
    const bf16_t* q_row = Qp + (q_m_base + col16) * stride_n;

    constexpr fp32_t LOG2_E = 1.44269504088896340736f;
    const fp32_t qscale = k.scale * LOG2_E;

    bf16x8_t v_q[DK_HALF];
    #pragma unroll
    for (int kt = 0; kt < DK_HALF; ++kt) {
        v_q[kt] = *reinterpret_cast<const bf16x8_t*>(&q_row[(d_offset + kt) * W_K + row8]);
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            v_q[kt][j] = bf16_fast33(bf16_to_f32(v_q[kt][j]) * qscale);
    }

    // Output accumulators: only 4 D-tiles (32 VGPRs instead of 64)
    fp32x8_t v_o[DK_HALF];
    #pragma unroll
    for (int kt = 0; kt < DK_HALF; ++kt) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_o[kt][j] = 0.0f;
    }

    fp32_t m_row = -3.4e38f;
    fp32_t l_row = 0.0f;

    const int num_kv_tiles = k.N / BLOCK_N;

    for (int n_tile = 0; n_tile < num_kv_tiles; ++n_tile) {
        const int n_base = n_tile * BLOCK_N;
        const int buf = n_tile & 1; // double-buffer index for s_partial

        // --- Phase 1: Compute partial QKT over this wave's D-half ---
        // Each wave only has 4 D-tiles of Q, so computes partial dot product.
        // K is loaded for the same 4 D-tiles (matching this wave's D-half).
        fp32x8_t v_s = {0, 0, 0, 0, 0, 0, 0, 0};
        #pragma unroll
        for (int kt = 0; kt < DK_HALF; ++kt) {
            const bf16_t* k_addr = Kp + (n_base + col16) * stride_n + (d_offset + kt) * W_K + row8;
            bf16x8_t v_k = *reinterpret_cast<const bf16x8_t*>(k_addr);
            v_s = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_k, v_q[kt], v_s);
        }

        // --- Phase 2: Exchange partial sums via LDS ---
        // Write this wave's partial S to shared memory (bank-conflict-free layout)
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            s_partial[buf][wave_pair][d_half][j][lane] = v_s[j];
        }

        __syncthreads();

        // Read partner wave's partial S and add to get full S
        const int partner_half = d_half ^ 1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            v_s[j] += s_partial[buf][wave_pair][partner_half][j][lane];
        }

        // --- Phase 3: Softmax (both waves compute independently, same result) ---
        fp32_t row_max = v_s[0];
        #pragma unroll
        for (int j = 1; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s[j]);
        row_max = v33_cross_half_max(row_max);

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
        row_sum = v33_cross_half_sum(row_sum);
        l_row += row_sum;

        bf16x8_t v_p;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p[j] = bf16_fast33(v_s[j]);

        // --- Phase 4: PV accumulation — only this wave's D-half ---
        // No second barrier needed: s_partial is double-buffered (next iteration
        // writes to buf^1), and s_v is per-wave (no cross-wave conflict).
        #pragma unroll
        for (int dt = 0; dt < DK_HALF; ++dt) {
            if (need_rescale) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) v_o[dt][j] *= rescale;
            }

            const bf16_t* v_addr = Vp + (n_base + col16) * stride_n + (d_offset + dt) * W_K + row8;
            bf16x8_t v_load = *reinterpret_cast<const bf16x8_t*>(v_addr);

            // Transpose V via per-wave LDS
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                s_v[wave_id][(row8 + j) * 16 + col16] = v_load[j];
            __builtin_amdgcn_wave_barrier();

            bf16x8_t v_v = *reinterpret_cast<const bf16x8_t*>(
                &s_v[wave_id][col16 * 16 + row8]);

            v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_v, v_p, v_o[dt]);
        }
    }

    // --- Phase 5: Write output — each wave writes its D-half ---
    const fp32_t inv = (l_row > 0.0f) ? (1.0f / l_row) : 0.0f;
    bf16_t* o_row = Op + (q_m_base + col16) * stride_n;
    #pragma unroll
    for (int dt = 0; dt < DK_HALF; ++dt) {
        bf16x8_t o_pack;
        #pragma unroll
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast33(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[(d_offset + dt) * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
