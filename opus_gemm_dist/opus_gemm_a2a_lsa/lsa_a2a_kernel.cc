#include <hip/hip_runtime.h>

#include "gemm_defs.h"

using lsa_copy_vec_t = uint4;

// Retained optimization: split-LSA baseline uses 16-byte vector copies and
// partitions each destination slab across peer_block workgroups. The final
// release fence makes all remote stores visible before host-side completion.
__global__ void opus_lsa_a2a_copy_kernel(
    const bf16_t* staging,
    void* recv_win,
    bf16_t* recv_local,
    size_t elems_per_peer,
    int rank_count,
    int my_rank) {
    const int dst_rank = static_cast<int>(blockIdx.x) % rank_count;
    const int peer_block = static_cast<int>(blockIdx.x) / rank_count;
    const int peer_grid = static_cast<int>(gridDim.x) / rank_count;
    const size_t bytes_per_peer = elems_per_peer * sizeof(bf16_t);
    const size_t vecs_per_peer = bytes_per_peer / sizeof(lsa_copy_vec_t);

    const auto* src = reinterpret_cast<const lsa_copy_vec_t*>(
        staging + static_cast<size_t>(dst_rank) * elems_per_peer);
    char* dst_base = dst_rank == my_rank
        ? reinterpret_cast<char*>(recv_local)
        : static_cast<char*>(cco_lsa_peer_c(recv_win, dst_rank));
    auto* dst = reinterpret_cast<lsa_copy_vec_t*>(
        dst_base + static_cast<size_t>(my_rank) * bytes_per_peer);

    for (size_t i =
             static_cast<size_t>(peer_block * blockDim.x + threadIdx.x);
         i < vecs_per_peer;
         i += static_cast<size_t>(peer_grid * blockDim.x)) {
        dst[i] = src[i];
    }
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
}
