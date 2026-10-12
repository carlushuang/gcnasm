// Shared types between host and GEMM kernel TUs.
// Intentionally has NO dependency on <opus/opus.hpp> so the host TU can
// instantiate opus_gemm_traits without pulling in opus containers.
// Per-kernel derived constants (WMMA decomposition, N-subtiles, TDM window
// types, ...) live inside the kernel template .hpp.
#pragma once

#include <type_traits>
#include <cstddef>

using bf16_t = __bf16;
using fp32_t = float;

__host__ __device__ constexpr inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

// Kernel arguments shared by all GEMM kernel variants.
struct opus_gemm_kargs {
    const void* __restrict__ ptr_a;   // A [batch, M, K]
    const void* __restrict__ ptr_b;   // B [batch, N, K]   (C = A * B^T)
    void* __restrict__ ptr_c;         // C [batch, M, N]
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
};

// User-facing GEMM configuration: block tile (B_M, B_N, B_K), LDS ring depth,
// data types. Plain int / type template parameters -- no opus dependency.
// Everything here is computable without opus, which is what lets the host TU
// size the workspace and the launch grid; the WMMA-level decomposition is
// derived in the kernel template's kernel_traits.
template<int BLOCK_SIZE_,
         int B_M_, int B_N_, int B_K_,
         int NUM_SLOTS_,
         typename D_A_, typename D_B_, typename D_C_, typename D_ACC_>
struct opus_gemm_traits {
    static constexpr int BLOCK_SIZE = BLOCK_SIZE_;   // 128 = 4 waves x 32

    static constexpr int B_M = B_M_;
    static constexpr int B_N = B_N_;
    static constexpr int B_K = B_K_;

    using D_A   = D_A_;
    using D_B   = D_B_;
    using D_C   = D_C_;
    using D_ACC = D_ACC_;
    static_assert(std::is_same<D_A, D_B>::value, "A/B dtype must match");

    static constexpr int VEC_A = 16 / (int)sizeof(D_A);   // 8 for bf16 (b128 ds_read)
    static constexpr int VEC_B = 16 / (int)sizeof(D_B);

    // LDS prefetch ring depth. The pipeline keeps 2 TDMs in flight and reuses
    // slot g%P three steps later, so P >= 3 keeps g, g+1, g+2 distinct.
    static constexpr int NUM_SLOTS = NUM_SLOTS_;
    static_assert(NUM_SLOTS == 3, "the unrolled K loop is written out for P == 3");

    // Cluster-launch multicast geometry: a CLUSTER_WG_M x CLUSTER_WG_N grid of
    // workgroups per cluster. A is multicast to the CLUSTER_WG_N peers sharing
    // an M row, B to the CLUSTER_WG_M peers sharing an N column.
    static constexpr int CLUSTER_WG_M = 4;
    static constexpr int CLUSTER_WG_N = 4;
    // TDM multicast fans out to at most 5 WGs per group, and the per-cluster
    // workgroup_mask is 16-bit.
    static_assert(CLUSTER_WG_M <= 5 && CLUSTER_WG_N <= 5 &&
                  CLUSTER_WG_M * CLUSTER_WG_N <= 16,
                  "cluster dims must be 1..5 per side and <= 16 WGs total");

    // TDM/LDS pad: +16B (one PAD_ELEMS group) per B_K row -> bank-conflict-free
    // b128 ds_read. Only the ELEMENT geometry lives here, because that is all the
    // host needs to size LDS; the D# pad_interval/pad_amount encoding is derived
    // by opus::tdm_traits::padding_auto, and the kernel template static_asserts that its
    // pitch matches SMEM_PITCH so the two can never drift.
    static_assert((B_K & (B_K - 1)) == 0, "B_K must be a power of 2 for a single pad per row");
    static constexpr int PAD_ELEMS    = 16 / (int)sizeof(D_A);   // 8 bf16 = +16B
    static constexpr int SMEM_PITCH   = B_K + PAD_ELEMS;

    // One LDS slot holds the full B_M x B_K (A) / B_N x B_K (B) tile.
    static constexpr int SLOT_BYTES_A = B_M * SMEM_PITCH * (int)sizeof(D_A);
    static constexpr int SLOT_BYTES_B = B_N * SMEM_PITCH * (int)sizeof(D_B);
    static constexpr int SEG_BYTES_A  = NUM_SLOTS * SLOT_BYTES_A;
    static constexpr int SEG_BYTES_B  = NUM_SLOTS * SLOT_BYTES_B;
    static constexpr int LDS_BYTES    = SEG_BYTES_A + SEG_BYTES_B;
    static_assert(LDS_BYTES <= 320 * 1024, "LDS exceeds the 320KB/CU budget");
};

// The benchmarked configuration. The stub TU explicitly instantiates the kernel
// for exactly this type, so the host and device passes must name the same
// alias rather than re-spelling the argument list.
using gemm_traits_128x256x128 =
    opus_gemm_traits<128, 128, 256, 128, 3, bf16_t, bf16_t, bf16_t, fp32_t>;
