// SPDX-License-Identifier: MIT
// opus_attn_gfx1201 v1 — multi-wave per workgroup + cooperative smem K/V.
//
// Geometry (vs v0):
//   workgroup = 4 waves × 32 lanes = 128 threads
//   BLOCK_M   = 64        (each wave handles 16 of these)
//   BLOCK_N   = 16        (still 1 wmma N-tile per softmax — bumped in v2)
//   D         = 128
//
//   grid  = dim3(N / BLOCK_M, H, B)
//
// Per workgroup smem:
//   K tile: BLOCK_N × D × 2 = 16 × 128 × 2 = 4 KB
//   V tile: 4 KB
//   P scratch: 4 waves × 16 × 16 × 2 = 2 KB
//   Total: 10 KB
//
// Each thread cooperatively loads K and V from global into smem each KV
// iter (128 threads × 16 bytes = 2 KB per load → 2 loads each for K, V).
// This is the central win over v0: per (b, h, output_row), K and V are
// read from global 1/4 as often (4 waves share each load).
//
// Each wave still owns its own 16 M-rows and runs an independent softmax.
// No cross-wave communication needed beyond the smem K/V load barrier.
#include <hip/hip_runtime.h>
#include "attn_common.h"

using fp16x8_t = fp16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline fp32_t d_fmaxf(fp32_t a, fp32_t b) { return a > b ? a : b; }

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 2)
__global__ void opus_attn_gfx1201_kernel_v1(opus_attn_kargs k)
{
#if defined(__gfx1201__) || defined(__gfx1200__)
    constexpr int BLOCK_M    = T::BLOCK_M;       // 64
    constexpr int BLOCK_N    = T::BLOCK_N;       // 16
    constexpr int D          = T::D;             // 128
    constexpr int W_K        = T::W_K;           // 16
    constexpr int DK         = T::D_TILES_K;     // 8
    constexpr int NWAVES     = T::NUM_WAVES;     // 4
    constexpr int WAVE_M     = T::WAVE_M;        // 16
    constexpr int WARP_SIZE  = T::WARP_SIZE;     // 32

    const int tid     = static_cast<int>(threadIdx.x);
    const int wave_id = tid / WARP_SIZE;        // 0..3
    const int lane    = tid % WARP_SIZE;
    const int col16   = lane % 16;
    const int row_grp = lane / 16;
    const int row8    = row_grp * 8;

    const int q_block_id = blockIdx.x;          // along N dim (block stride = BLOCK_M)
    const int h          = blockIdx.y;
    const int b          = blockIdx.z;

    const int stride_n = k.D;
    const int stride_h = k.N * k.D;
    const int stride_b = k.H * k.N * k.D;

    const fp16_t* __restrict__ Qp = reinterpret_cast<const fp16_t*>(k.ptr_q) + b * stride_b + h * stride_h;
    const fp16_t* __restrict__ Kp = reinterpret_cast<const fp16_t*>(k.ptr_k) + b * stride_b + h * stride_h;
    const fp16_t* __restrict__ Vp = reinterpret_cast<const fp16_t*>(k.ptr_v) + b * stride_b + h * stride_h;
    fp16_t*       __restrict__ Op = reinterpret_cast<fp16_t*>      (k.ptr_o) + b * stride_b + h * stride_h;

    // ── Shared memory layout ─────────────────────────────────────────────
    // K and V tiles plus per-wave P scratch.
    constexpr int K_SMEM_ELEMS = BLOCK_N * D;
    constexpr int V_SMEM_ELEMS = BLOCK_N * D;
    constexpr int P_SMEM_ELEMS = NWAVES * WAVE_M * BLOCK_N;

    __shared__ fp16_t s_k[K_SMEM_ELEMS];
    __shared__ fp16_t s_v[V_SMEM_ELEMS];
    __shared__ fp16_t s_p[P_SMEM_ELEMS];

    // ── Q load (per wave, into registers, kept through KV loop) ──────────
    const int wave_m_base = q_block_id * BLOCK_M + wave_id * WAVE_M;
    fp16x8_t v_q[DK];
    {
        const fp16_t* q_row = Qp + (wave_m_base + col16) * stride_n;
        #pragma unroll
        for (int kt = 0; kt < DK; ++kt) {
            const int k_off = kt * W_K + row8;
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_q[kt][j] = q_row[k_off + j];
        }
    }

    // Pre-scale Q by 1/sqrt(D) * log2(e) so softmax uses exp2 directly.
    constexpr fp32_t LOG2_E = 1.44269504088896340736f;
    const fp32_t qscale = k.scale * LOG2_E;
    #pragma unroll
    for (int kt = 0; kt < DK; ++kt) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            v_q[kt][j] = static_cast<fp16_t>(static_cast<fp32_t>(v_q[kt][j]) * qscale);
        }
    }

    // ── Output accumulator (fp32), per wave covers M=16 rows × D=128 ─────
    fp32x8_t v_o[DK];
    #pragma unroll
    for (int kt = 0; kt < DK; ++kt) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_o[kt][j] = 0.0f;
    }

    fp32_t m_row[8], l_row[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        m_row[j] = -3.4e38f;
        l_row[j] = 0.0f;
    }

    const int num_kv_tiles = k.N / BLOCK_N;

    // ── KV loop ───────────────────────────────────────────────────────────
    for (int n_tile = 0; n_tile < num_kv_tiles; ++n_tile) {
        const int n_base = n_tile * BLOCK_N;

        // Cooperative K + V load: each thread loads 16 fp16 (= 32 bytes) from
        // each of K and V. The 128 threads × 16 fp16 = 2048 fp16 = K tile.
        constexpr int FP16_PER_THREAD = K_SMEM_ELEMS / T::BLOCK_SIZE;   // 16
        static_assert(FP16_PER_THREAD == V_SMEM_ELEMS / T::BLOCK_SIZE);
        // Linear K addressing: row = tid / (D / 16), col = (tid % (D / 16)) * 16
        const int row_per_load = tid / (D / FP16_PER_THREAD);
        const int col_per_load = (tid % (D / FP16_PER_THREAD)) * FP16_PER_THREAD;
        const fp16_t* k_src = Kp + (n_base + row_per_load) * stride_n + col_per_load;
        const fp16_t* v_src = Vp + (n_base + row_per_load) * stride_n + col_per_load;
        fp16_t* k_dst = s_k + row_per_load * D + col_per_load;
        fp16_t* v_dst = s_v + row_per_load * D + col_per_load;
        #pragma unroll
        for (int i = 0; i < FP16_PER_THREAD; ++i) {
            k_dst[i] = k_src[i];
            v_dst[i] = v_src[i];
        }
        __syncthreads();

        // 1) S = Q @ K^T, shape 16×16 per wave, fp32 acc
        //
        //    For wmma: A = Q (row-distributed), B = K^T (col-distributed).
        //    K^T lane mapping in smem: lane[i].B[j] = K^T[k = row8+j, n = col16]
        //                                            = K[n = col16, k = row8+j]
        //    K in smem is row-major [BLOCK_N=16, D=128], so K[n, k] = s_k[n * D + k].
        fp32x8_t v_s = {0,0,0,0,0,0,0,0};
        #pragma unroll
        for (int kt = 0; kt < DK; ++kt) {
            fp16x8_t v_k;
            const fp16_t* k_row = s_k + col16 * D + kt * W_K;
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_k[j] = k_row[row8 + j];
            v_s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v_q[kt], v_k, v_s);
        }

        // 2) Row max across 16 N-cols within this wave's lane group, per j ∈ [0,7]
        fp32_t s_max[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            fp32_t v = v_s[j];
            v = d_fmaxf(v, __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x041F)));
            v = d_fmaxf(v, __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x081F)));
            v = d_fmaxf(v, __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x101F)));
            v = d_fmaxf(v, __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x201F)));
            s_max[j] = v;
        }

        // 3) New m, rescale O, exp2(S - m), accumulate l
        fp32_t rescale[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const fp32_t new_m = d_fmaxf(m_row[j], s_max[j]);
            rescale[j] = __builtin_amdgcn_exp2f(m_row[j] - new_m);
            m_row[j]   = new_m;
        }
        fp32_t s_sum[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            v_s[j] = __builtin_amdgcn_exp2f(v_s[j] - m_row[j]);
            fp32_t v = v_s[j];
            v += __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x041F));
            v += __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x081F));
            v += __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x101F));
            v += __builtin_bit_cast(fp32_t, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x201F));
            s_sum[j] = v;
            l_row[j] = l_row[j] * rescale[j] + s_sum[j];
        }
        #pragma unroll
        for (int kt = 0; kt < DK; ++kt) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_o[kt][j] *= rescale[j];
        }

        // 4) Flip column-distributed S → row-distributed P via smem (per-wave
        //    region so 4 waves don't trample each other; tile is 16×16 fp16).
        fp16_t* p_wave = s_p + wave_id * (WAVE_M * BLOCK_N);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            p_wave[(row8 + j) * BLOCK_N + col16] = static_cast<fp16_t>(v_s[j]);
        }
        // No syncthreads needed — only this wave reads from its own p_wave.
        __builtin_amdgcn_wave_barrier();

        // 5) O += P @ V, P is (16×16) row-distributed, V is (16×128) col-distributed.
        //    V in smem is row-major [N=16, D=128], lane mapping for the B operand:
        //      lane[i].B[j] = V[k_dim_idx = row8+j, n_out_idx = dt*16 + col16]
        //                   = s_v[(row8+j) * D + dt*16 + col16]
        fp16x8_t v_p;
        {
            const int p_row = col16;
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_p[j] = p_wave[p_row * BLOCK_N + row8 + j];
        }
        __builtin_amdgcn_wave_barrier();

        #pragma unroll
        for (int dt = 0; dt < DK; ++dt) {
            const fp16_t* v_col = s_v + row8 * D + dt * W_K + col16;
            fp16x8_t v_v;
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_v[j] = v_col[j * D];
            v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v_p, v_v, v_o[dt]);
        }

        __syncthreads();   // before next K/V cooperative load
    }

    // ── Normalize + store O (column-distributed → row-major write) ────────
    fp32_t inv_l[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        inv_l[j] = (l_row[j] > 0.0f) ? (1.0f / l_row[j]) : 0.0f;
    }
    #pragma unroll
    for (int kt = 0; kt < DK; ++kt) {
        const int d_base = kt * W_K + col16;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const fp32_t out_val = v_o[kt][j] * inv_l[j];
            Op[(wave_m_base + row8 + j) * stride_n + d_base] = static_cast<fp16_t>(out_val);
        }
    }
#else
    (void)k;
#endif
}
