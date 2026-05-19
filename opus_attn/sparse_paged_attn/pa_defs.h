// Shared types and constants between device kernel and host code
#pragma once

using bf16_t = __bf16;
using fp16_t = __fp16;

// Kernel arguments for PA prefill attention
struct pa_kargs {
    const void* __restrict__ q_ptr;          // [N, H, D]
    const void* __restrict__ unified_kv_ptr; // [total_pages, D], prefix source
    const void* __restrict__ kv_ptr;         // [total_tokens, D], extend source
    const void* __restrict__ attn_sink_ptr;  // [H], softmax denominator sink
    void* __restrict__ out_ptr;              // [N, H, D]
    const int* __restrict__ kv_indptr_prefix;  // [N+1]
    const int* __restrict__ kv_indices_prefix; // [indices_prefix_sum_prefix]
    const int* __restrict__ kv_indptr_extend;  // [N+1]
    const int* __restrict__ kv_indices_extend; // [indices_prefix_sum_extend]
    int N;
    int H;
    int D;
    int total_pages;
    int total_tokens;
    int stride_qo_n;
    int stride_qo_h;
    int stride_kv_page;
    float softmax_scale;
};

// Configuration traits for PA kernel (tile sizes, data types, vector lengths, MFMA config).
template<int Q_TILE_SIZE_ = 16,
         int KV_TILE_SIZE_ = 32,
         int D_TILE_SIZE_ = 512,
         int NUM_WARPS_ = 8,
         typename D_ATTN_ = bf16_t>
struct pa_traits {
    static constexpr int Q_TILE_SIZE = Q_TILE_SIZE_;
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;
    static constexpr int D_TILE_SIZE = D_TILE_SIZE_;
    static constexpr int NUM_WARPS = NUM_WARPS_;

    static constexpr int WARP_SIZE = 64; // AMD wavefront size
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;

    // Data types: Q/K/V/O share one attention dtype; accumulation fp32
    using D_ATTN = D_ATTN_;
    using D_ACC  = float;

    // MFMA wave layout
    static constexpr int T_M = NUM_WARPS; // waves along M
    static constexpr int T_N = 1;         // waves along N
    static constexpr int T_K = 1;         // waves along K

    // MFMA base tile
    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 32;

    // D slicing: D=512 iterates D in SLICE_D=32 chunks
    static constexpr int SLICE_D = 32;
    static constexpr int NUM_D_SLICES = D_TILE_SIZE / SLICE_D;
    static_assert(D_TILE_SIZE % SLICE_D == 0);

    // GEMM0: S[Q_TILE x KV_TILE] = Q[Q_TILE x SLICE_D] @ K^T[SLICE_D x KV_TILE]
    static constexpr int GEMM0_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM0_E_N = KV_TILE_SIZE / W_N;
    static constexpr int GEMM0_E_K = SLICE_D / W_K;

    // GEMM1: O[Q_TILE x SLICE_D] = P[Q_TILE x KV_TILE] @ V[KV_TILE x SLICE_D]
    static constexpr int GEMM1_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM1_E_N = SLICE_D / W_N;
    static constexpr int GEMM1_E_K = KV_TILE_SIZE / W_K;

    // Vector lengths for global load/store
    static constexpr int VEC_Q    = 8;
    static constexpr int VEC_KV   = 8;
    static constexpr int VEC_TR_V = 4;
    static constexpr int VEC_O    = 4;

    // Minimal compact pixels for async copy for one wave
    static constexpr int D_128B_SIZE = 128 / sizeof(D_ATTN);
    static_assert(VEC_KV == 16 / sizeof(D_ATTN));
    static constexpr int smem_linear_wave = WARP_SIZE * 16 / sizeof(D_ATTN);
    static constexpr int smem_n_per_wave = smem_linear_wave / D_128B_SIZE;
    static constexpr int smem_n_rpt = KV_TILE_SIZE / smem_n_per_wave;
    static constexpr int smem_d_rpt = D_TILE_SIZE / D_128B_SIZE;
    static constexpr int smem_padding_32B = 32 / sizeof(D_ATTN);
    static constexpr int smem_kv_tile_elems = smem_n_rpt * smem_d_rpt * (smem_linear_wave + smem_padding_32B);

    static constexpr int kv_buffer_load_insts = (KV_TILE_SIZE * D_TILE_SIZE) / (BLOCK_SIZE * VEC_KV);
    static constexpr int k_ds_read_insts = (GEMM0_E_N * GEMM0_E_K * W_N * W_K) / (WARP_SIZE * VEC_KV);
    static constexpr int v_ds_read_insts = (GEMM1_E_N * GEMM1_E_K * W_N * W_K) / (WARP_SIZE * VEC_TR_V);

    static constexpr size_t smem_size_bytes() {
        return 4 * smem_kv_tile_elems * sizeof(D_ATTN);
    }
};

__host__ __device__ inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}
