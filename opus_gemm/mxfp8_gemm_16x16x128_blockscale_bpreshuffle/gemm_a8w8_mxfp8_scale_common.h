#pragma once

// gfx950 blockscale bpreshuffle: A is row-major FP8; B has the aiter
// shuffle_weight(layout=(16,16)) byte layout. SFA is E8M0 [K/128,M]
// column-major for logical [M,K/128]; SFB is E8M0 [N/128,K/128].
struct opus_gemm_scale_kargs {
    const void* __restrict__ ptr_a;
    const void* __restrict__ ptr_b;
    void* __restrict__ ptr_c;
    int m;
    int n;
    int k;
    int batch;
    int stride_a;
    int stride_b;
    int stride_c;
    int stride_a_batch;
    int stride_b_batch;
    int stride_c_batch;

    const void* __restrict__ ptr_sfa;
    const void* __restrict__ ptr_sfb;
    int stride_sfa;  // SFA K128-column stride in bytes (M for dense input).
    int stride_sfb;
    int stride_sfa_batch;
    int stride_sfb_batch;
};

// V_MFMA_SCALE_F32_16X16X128_F8F6F4.
template<
    int BLOCK_M_ = 256,
    int BLOCK_N_ = 256,
    int BLOCK_K_ = 128,
    int GROUP_M_ = 1,
    int GROUP_N_ = 128,
    int GROUP_K_ = 128,
    int OUTPUT_TILES_ = 4,
    bool OUTPUT_BF16_ = true>
struct gemm_a8w8_mxfp8_scale_traits {
    static constexpr int BLOCK_SIZE = 512;
    static constexpr int WARP_SIZE = 64;
    static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;

    static constexpr int B_M = BLOCK_M_;
    static constexpr int B_N = BLOCK_N_;
    static constexpr int B_K = BLOCK_K_;

    static constexpr int T_M = 4;
    static constexpr int T_N = 2;
    static constexpr int T_K = 1;

    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 128;

    static constexpr int HALF_B_M = B_M / 2;
    static constexpr int HALF_B_N = B_N / 2;

    static_assert(NUM_WAVES == T_M * T_N * T_K);
    static_assert(T_K == 1);
    static_assert(HALF_B_M % (W_M * T_M) == 0);
    static_assert(HALF_B_N % (W_N * T_N) == 0);
    static_assert(B_K % (W_K * T_K) == 0);

    static constexpr int E_M = HALF_B_M / (W_M * T_M);
    static constexpr int E_N = HALF_B_N / (W_N * T_N);
    static constexpr int E_K = B_K / (W_K * T_K);

    static constexpr int VEC_A = 16;
    static constexpr int VEC_B = 16;
    static constexpr int VEC_C = 4;
    static constexpr int VEC_LDS_SCALE = 4;

    // Process up to four adjacent M tiles with fixed N, allowing cache reuse
    // of B/SFB. The single-tile specialization assigns one M tile per workgroup.
    // The host selects the specialization in pick_output_tiles_per_wg().
    static constexpr int OUTPUT_TILES_PER_WG = OUTPUT_TILES_;
    static constexpr bool OUTPUT_BF16 = OUTPUT_BF16_;

    static constexpr int GROUP_M = GROUP_M_;
    static constexpr int GROUP_N = GROUP_N_;
    static constexpr int GROUP_K = GROUP_K_;
    static_assert(GROUP_M == 1 && GROUP_N == 128 && GROUP_K == 128);
    // Input quantization groups and the hardware K32 scale lanes are distinct.
    static constexpr int MFMA_SCALE_GROUP_K = 32;
    static constexpr int NUM_KGROUPS = B_K / MFMA_SCALE_GROUP_K;

    static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
    static constexpr int smem_sub = smem_linear_wave / B_K;
    static constexpr int smem_m_rep = HALF_B_M / smem_sub;
    static constexpr int smem_n_rep = HALF_B_N / smem_sub;
    static constexpr int smem_padding = 32;

    static constexpr int SCALE_M_CALLS = B_M / (T_M * W_M);
    static constexpr int SCALE_N_CALLS = E_N;
    static constexpr int SCALE_N_HALVES = B_N / HALF_B_N;
    static constexpr int packed_sfa_tile_elem = B_M * NUM_KGROUPS;
    static constexpr int packed_sfb_tile_elem = B_N * NUM_KGROUPS;

    static constexpr int b_ds_read_insts = E_N * W_N * W_K / (WARP_SIZE * VEC_B);
};

__host__ __device__ inline int ceil_div_scale(int a, int b) {
    return (a + b - 1) / b;
}
