// Shared types between host and GEMM kernel TUs.
// Intentionally has NO dependency on <opus/opus.hpp> so the host TU can
// instantiate opus_gemm_traits without pulling in opus containers.
// Per-kernel derived constants (HALF_B_M, E_M, smem_m_rep, etc.) live
// inside each kernel template .hpp.
#pragma once

#include <type_traits>
#include <cstddef>

using bf16_t = __bf16;

__host__ __device__ constexpr inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

// Minimal mirror of the leading fields of mori::cco::ccoWindowDevice, matching
// its layout (winBase, stride4G, lsaRank). Lets the kernel resolve an LSA peer
// pointer and read its own LSA rank without pulling in cco.hpp (host-only STL).
// Peer VA formula: peer = winBase + ((uint64_t)peerLsaRank * stride4G << 32)
// (identical to cco's __device__ ccoGetLsaPeerPtr).
struct cco_window_view {
    char* winBase;
    unsigned int stride4G;
    int lsaRank;
};

__device__ inline void* cco_lsa_peer_c(void* win_handle, int peer_lsa_rank) {
    auto* w = reinterpret_cast<const cco_window_view*>(win_handle);
    return w->winBase + ((static_cast<unsigned long long>(peer_lsa_rank) * w->stride4G) << 32);
}

__device__ inline int cco_lsa_rank(void* win_handle) {
    return reinterpret_cast<const cco_window_view*>(win_handle)->lsaRank;
}

// Kernel arguments shared by all GEMM kernel variants.
struct opus_gemm_kargs {
    const void* __restrict__ ptr_a;
    const void* __restrict__ ptr_b;
    void* __restrict__ ptr_c;
    unsigned int* tile_counter = nullptr;
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

    // Optional cco LSA peer-put target for the output C. When cco_c_win != nullptr
    // the kernel resolves the C base via ccoGetLsaPeerPtr(cco_c_win, peer_lsa_rank)
    // — i.e. it stores directly into the peer rank's window slot instead of ptr_c.
    // Stored as void* to keep this header free of any cco/opus dependency; the
    // kernel TU reinterprets it as ccoWindow_t. nullptr => use ptr_c (local).
    void* cco_c_win = nullptr;
    int peer_lsa_rank = 0;

    // Phase C (GEMM->A2A column scatter): when scatter_n_shard > 0, each block's
    // destination rank is chosen from its output column block: peer = col / n_shard.
    // Overrides peer_lsa_rank per-block. The output layout stays full-width [M,N]
    // (stride_c = N), so each destination rank receives its own column block filled
    // and the rest untouched. 0 => disabled (use peer_lsa_rank / local).
    int scatter_n_shard = 0;

    // Phase C-a2a (compact all-to-all): when a2a_n_shard > 0, every rank runs the
    // GEMM and scatters column block j into rank j's COMPACT window laid out as
    // [world*M, n_shard]. This rank's contribution lands at row-block my_lsa_rank,
    // i.e. dst rows [my_lsa_rank*M : (my_lsa_rank+1)*M], with row stride n_shard and
    // local column (col % n_shard). Requires cco_c_win set. 0 => disabled.
    int a2a_n_shard = 0;
    int a2a_M = 0;        // per-rank row count (M); receiver row-block offset = lsaRank * M
    int a2a_span = 0;     // scatter width AN: cols [0,AN) scatter, cols [AN,n) stay local
    int stride_c_full = 0;  // row stride of the local full-width [M,N] buffer (= N)
};

// Experimental persistent compute + comm pipeline args.
struct opus_persistent_comm_kargs {
    const void* __restrict__ ptr_a;
    const void* __restrict__ ptr_b;
    void* __restrict__ workspace;
    void* cco_c_win = nullptr;
    unsigned int* semaphores = nullptr;  // [sync_epochs * num_xcd]
    int* slot_tasks = nullptr;           // unused legacy scratch

    int m;
    int n;
    int k;
    int batch;
    int stride_a;
    int stride_b;
    int stride_a_batch;
    int stride_b_batch;

    int rank_count = 4;
    int rank_shard_n = 2560;
    int scatter_n = 10240;  // rank_count * rank_shard_n

    int num_tiles_m;
    int num_tiles_n_scatter;
    int workspace_slot_elems;
    int compute_wg_per_xcd = 30;
    int wg_per_xcd = 32;
    int num_xcd = 8;
    int mode = 0;       // 0=fused, 1=chunk compute-only, 2=chunk comm-only,
                        // 3=ablation compute-only local tail,
                        // 4=ablation compute+semaphore only,
                        // 5=ablation compute-only LSA tail
    int task_base = 0;  // used by non-fused chunked baseline
    int task_count = 0;
    int sync_epochs = 2;
};

// User-facing GEMM configuration: block tile (B_M, B_N, B_K), data types,
// global-memory vector widths, and workgroup size. Plain int / type template
// parameters — no opus dependency. Each kernel template hpp consumes one of
// these and computes its own derived constants (HALF_B_M, E_M, smem_*, ...).
template<int BLOCK_SIZE_,
         int B_M_, int B_N_, int B_K_,
         typename D_A_, typename D_B_, typename D_C_, typename D_ACC_>
struct opus_gemm_traits {
    static constexpr int BLOCK_SIZE = BLOCK_SIZE_;

    static constexpr int B_M = B_M_;
    static constexpr int B_N = B_N_;
    static constexpr int B_K = B_K_;

    using D_A   = D_A_;
    using D_B   = D_B_;
    using D_C   = D_C_;
    using D_ACC = D_ACC_;
    static_assert(std::is_same<D_A, D_B>::value);

    static constexpr int VEC_A = 8;
    static constexpr int VEC_B = 8;
    static constexpr int VEC_C = 8;
};
