// Standalone all-to-all column-shard scatter kernel — apples-to-apples with the
// fused GEMM+A2A kernel: SAME persistent grid, SAME atomic tile scheduler, SAME
// TILE_ORDER peer round-robin and store stagger as gemm_a16w16_quad_subtile_kernel.
// The only difference vs the fused epilogue scatter is that each B_M x B_N tile
// is read from the already-materialized local C[M,N] in HBM instead of from the
// MMA output registers. Pairing this with the GEMM run in local-store mode is
// therefore a fair "split" (2 chained kernels) baseline for the fused kernel.
//
// Reuses cco_lsa_peer_c / cco_lsa_rank from gemm_defs.h.
#pragma once

#include "gemm_defs.h"

#ifndef OPUS_TILE_ORDER
#define OPUS_TILE_ORDER 1
#endif
#ifndef OPUS_STORE_STAGGER_PHASES
#define OPUS_STORE_STAGGER_PHASES 16
#endif
#ifndef OPUS_STORE_STAGGER_DELAY
#define OPUS_STORE_STAGGER_DELAY 4
#endif

struct a2a_scatter_kargs {
    const void* __restrict__ src_c;  // local C[M, stride_src], row-major, bf16
    void* cco_c_win;                 // ccoWindow_t (as void*), peer put target
    unsigned int* tile_counter;      // global atomic tile dispenser (reset to 0 per launch)
    int M;
    int n_shard;       // per-rank shard width (destination block width)
    int a2a_span;      // total scattered columns = world * n_shard
    int stride_src;    // row stride of src_c (= N)
    int B_M;           // tile height (= Traits::B_M)
    int B_N;           // tile width  (= Traits::B_N)
    int num_tiles_m;   // M / B_M
    int scatter_tiles; // num_tiles_m * (a2a_span / B_N)
};

// 16-byte vector = 8 bf16. n_shard % 8 == 0 and stride_src % 8 == 0 keep every
// access 16B-aligned and never straddling a shard boundary.
struct alignas(16) a2a_vec16 {
    unsigned int w[4];
};

// Persistent kernel: launch min(scatter_tiles, cu_count) workgroups; each grabs
// B_M x B_N tiles from tile_counter until the scatter region is exhausted —
// identical scheduling to the fused kernel.
__global__ void a2a_scatter_kernel(a2a_scatter_kargs k) {
    const int tpm            = k.num_tiles_m;
    const int tiles_per_peer = k.n_shard / k.B_N;      // column-tiles inside one shard
    const int num_peer_tiles = k.a2a_span / k.n_shard; // number of destination peers
    const int my             = cco_lsa_rank(k.cco_c_win);
    const int vcols          = k.B_N / 8;              // 16B vectors per tile row
    const int total_vec      = k.B_M * vcols;          // vectors per tile
    const int src_row_vecs   = k.stride_src / 8;
    const int dst_row_vecs   = k.n_shard / 8;
    const a2a_vec16* src     = reinterpret_cast<const a2a_vec16*>(k.src_c);

    __shared__ unsigned int next_tile;
    while (true) {
        if (__builtin_amdgcn_workitem_id_x() == 0)
            next_tile = __atomic_fetch_add(k.tile_counter, 1u, __ATOMIC_RELAXED);
        __builtin_amdgcn_s_barrier();
        const int seq = static_cast<int>(next_tile);
        if (seq >= k.scatter_tiles) break;

        // Same (m_tile, n_tile) mapping + TILE_ORDER peer round-robin as the fused kernel.
        const int m_tile = seq % tpm;
        const int n_seq  = seq / tpm;
        int n_tile = n_seq;
#if OPUS_TILE_ORDER == 1
        if (tiles_per_peer > 0 && num_peer_tiles > 0) {
            const int inner = n_seq / num_peer_tiles;
            const int peer  = n_seq - inner * num_peer_tiles;
            n_tile = peer * tiles_per_peer + inner;
        }
#endif
        const int row = m_tile * k.B_M;
        const int col = n_tile * k.B_N;
        const int dst = col / k.n_shard;
        const int local_col = col - dst * k.n_shard;

#if OPUS_STORE_STAGGER_PHASES > 0 && OPUS_STORE_STAGGER_DELAY > 0
        {
            const int spins = (seq % OPUS_STORE_STAGGER_PHASES) * OPUS_STORE_STAGGER_DELAY;
            for (int i = 0; i < spins; ++i) __builtin_amdgcn_s_sleep(1);
        }
#endif

        a2a_vec16* peer = reinterpret_cast<a2a_vec16*>(cco_lsa_peer_c(k.cco_c_win, dst));
        for (int t = __builtin_amdgcn_workitem_id_x(); t < total_vec;
             t += __builtin_amdgcn_workgroup_size_x()) {
            const int rr = t / vcols;
            const int vc = t % vcols;
            const long long si = static_cast<long long>(row + rr) * src_row_vecs + (col / 8) + vc;
            const long long di =
                static_cast<long long>(my * k.M + row + rr) * dst_row_vecs + (local_col / 8) + vc;
            peer[di] = src[si];
        }
    }
}
