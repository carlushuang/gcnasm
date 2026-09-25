#pragma once

#include <opus/opus.hpp>

// Device-side scale repack.
//
// The GEMM reads its scales pre-swizzled: each (tile, k-tile) block is a
// byte-for-byte image of the consumer-facing LDS tile, which is what lets the
// producer issue one fully coalesced buffer_load per k-tile.  The host packer
// in the benchmark builds that image on the CPU, which is fine when the host
// is also the one generating the scales -- but in a real pipeline the scales
// come out of a quantization kernel and are already in device memory.  This
// kernel does the same repack there.
//
// Both tensors reduce to the same shape.  Take SFA: the packed tile is
// [consumer_wave_m][r][q][m_call] = [T_M][W_M][NUM_KGROUPS][SCALE_M_CALLS],
// which for the default traits is [4][16][4][4] = 1024 bytes = 256 dwords.
// Let lane = consumer_wave_m * W_M + r, which covers exactly 0..63.  Then that
// lane owns output dwords 4*lane .. 4*lane+3 -- four consecutive dwords, one
// dwordx4 store, and 64 lanes cover the whole 1024-byte tile contiguously.
// Its inputs are one dword from each of four source rows, CALL_STRIDE apart.
// Output dword q takes byte j from input row j: a 4x4 byte transpose.
//
// SFB is the same with an extra outer half_n axis, so lane splits as
// half_n * (T_N * W_N) + consumer_wave_n * W_N + r and the row base picks up
// half_n * HALF_B_N.  Parameterising by (rows per half, call stride) covers
// both, exactly like the kernel's shared scale producer.

// out[q] = { in[0][q], in[1][q], in[2][q], in[3][q] }.  Written with shifts
// rather than __builtin_amdgcn_perm: it is byte-exact by construction, and
// clang lowers it to v_perm_b32 anyway.
__device__ inline void transpose_4x4_bytes(const unsigned int (&in)[4],
                                           unsigned int (&out)[4]) {
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        unsigned int v = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            v |= ((in[j] >> (8 * q)) & 0xffu) << (8 * j);
        }
        out[q] = v;
    }
}

// One wave per (batch, tile, k-tile) block.  ROWS_PER_HALF is B_M for SFA and
// HALF_B_N for SFB; NUM_HALVES is 1 and 2 respectively.
template<class Traits, int NUM_HALVES, int ROWS_PER_HALF, int CALL_STRIDE>
__global__ void gemm_a8w8_mxfp8_scale_repack_kernel(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    int rows,             // M for SFA, N for SFB
    int num_groups_k,     // K / GROUP_K, the source row stride in bytes
    int num_tiles,        // rows / B_M or rows / B_N
    int num_tiles_k) {
    constexpr int NUM_KGROUPS = Traits::NUM_KGROUPS;
    constexpr int CALLS = 4;                       // dwords gathered per lane
    constexpr int TILE_ROWS = NUM_HALVES * ROWS_PER_HALF;
    constexpr int TILE_ELEMS = TILE_ROWS * NUM_KGROUPS;
    static_assert(NUM_KGROUPS == 4, "the 4x4 byte transpose assumes 4 k-groups per tile");
    static_assert(ROWS_PER_HALF == CALLS * CALL_STRIDE, "lane mapping must cover one half exactly");

    const int wave_in_block = opus::thread_id_x() / 64;
    const int lane = opus::thread_id_x() % 64;
    const int wave_id = opus::block_id_x() * (opus::block_size_x() / 64) + wave_in_block;

    const int total_waves = num_tiles * num_tiles_k;
    if (wave_id >= total_waves) {
        return;
    }
    const int tile = wave_id / num_tiles_k;
    const int kt = wave_id % num_tiles_k;
    const int b = opus::block_id_y();

    // lane -> (half_n, consumer_wave, r); the halves stack along the row axis.
    const int half = NUM_HALVES == 1 ? 0 : lane / (64 / NUM_HALVES);
    const int lane_in_half = NUM_HALVES == 1 ? lane : lane % (64 / NUM_HALVES);

    const unsigned char* src_base =
        src + static_cast<size_t>(b) * rows * num_groups_k
            + (static_cast<size_t>(tile) * TILE_ROWS + half * ROWS_PER_HALF) * num_groups_k
            + kt * NUM_KGROUPS;

    unsigned int a[CALLS];
    #pragma unroll
    for (int j = 0; j < CALLS; ++j) {
        a[j] = *reinterpret_cast<const unsigned int*>(
            src_base + static_cast<size_t>(j * CALL_STRIDE + lane_in_half) * num_groups_k);
    }

    unsigned int o[CALLS];
    transpose_4x4_bytes(a, o);

    unsigned char* dst_tile =
        dst + (static_cast<size_t>(b) * num_tiles * num_tiles_k
               + static_cast<size_t>(tile) * num_tiles_k + kt) * TILE_ELEMS;
    unsigned int* out = reinterpret_cast<unsigned int*>(dst_tile + lane * 16);
    *reinterpret_cast<opus::vector_t<unsigned int, 4>*>(out) =
        opus::vector_t<unsigned int, 4>{o[0], o[1], o[2], o[3]};
}

// Repacks both scale tensors and returns the elapsed device time in ms.  This
// is a one-off preprocessing step and deliberately sits outside the GEMM
// timing loop -- and for SFB it is amortised over every GEMM that reuses those
// weights, not paid per call.
template<class Traits>
inline float launch_scale_repack(
    const unsigned char* dev_sfa_raw, unsigned char* dev_sfa,
    const unsigned char* dev_sfb_raw, unsigned char* dev_sfb,
    int batch, int m, int n, int k) {
    const int num_groups_k = k / Traits::GROUP_K;
    const int num_tiles_m = m / Traits::B_M;
    const int num_tiles_n = n / Traits::B_N;
    const int num_tiles_k = k / Traits::B_K;

    constexpr int WAVES_PER_BLOCK = 4;
    const int block = WAVES_PER_BLOCK * 64;
    const auto grid_for = [&](int num_tiles) {
        return dim3((num_tiles * num_tiles_k + WAVES_PER_BLOCK - 1) / WAVES_PER_BLOCK,
                    batch, 1);
    };

    // One untimed pass first: the first launch of a kernel pays a one-off code
    // load that is several hundred microseconds and has nothing to do with the
    // repack -- it shows up as a near-constant cost across problem sizes.
    gemm_a8w8_mxfp8_scale_repack_kernel<Traits, 1, Traits::B_M, Traits::T_M * Traits::W_M>
        <<<grid_for(num_tiles_m), block>>>(
            dev_sfa_raw, dev_sfa, m, num_groups_k, num_tiles_m, num_tiles_k);
    gemm_a8w8_mxfp8_scale_repack_kernel<Traits, 2, Traits::HALF_B_N, Traits::T_N * Traits::W_N>
        <<<grid_for(num_tiles_n), block>>>(
            dev_sfb_raw, dev_sfb, n, num_groups_k, num_tiles_n, num_tiles_k);

    hipEvent_t start;
    hipEvent_t stop;
    (void)hipEventCreate(&start);
    (void)hipEventCreate(&stop);
    (void)hipDeviceSynchronize();
    (void)hipEventRecord(start);

    gemm_a8w8_mxfp8_scale_repack_kernel<Traits, 1, Traits::B_M, Traits::T_M * Traits::W_M>
        <<<grid_for(num_tiles_m), block>>>(
            dev_sfa_raw, dev_sfa, m, num_groups_k, num_tiles_m, num_tiles_k);
    gemm_a8w8_mxfp8_scale_repack_kernel<Traits, 2, Traits::HALF_B_N, Traits::T_N * Traits::W_N>
        <<<grid_for(num_tiles_n), block>>>(
            dev_sfb_raw, dev_sfb, n, num_groups_k, num_tiles_n, num_tiles_k);

    (void)hipEventRecord(stop);
    (void)hipEventSynchronize(stop);
    float ms = 0.0f;
    (void)hipEventElapsedTime(&ms, start, stop);
    (void)hipEventDestroy(start);
    (void)hipEventDestroy(stop);
    return ms;
}
