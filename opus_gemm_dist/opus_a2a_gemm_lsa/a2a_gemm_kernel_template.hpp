#pragma once

#include "gemm_defs.h"
#include "../opus_gemm_a2a_lsa/gemm_a16w16_quad_subtile_kernel_template.hpp"

namespace a2a_gemm_lsa {

using opus::operator""_I;
static constexpr int CPOL_SC0_SC1_LOCAL = 17;
using copy_vec_t = unsigned int __attribute__((ext_vector_type(4)));

#define GETREG_IMMED(sz, os, reg) (((sz) << 11) | ((os) << 6) | (reg))
#define HWREG_XCC_ID 20
#define HWREG_HW_ID  4
#ifndef A2A_GEMM_RECORD_WG_HW
#define A2A_GEMM_RECORD_WG_HW 0
#endif
#ifndef A2A_GEMM_COMM_PEER_ORDER
#define A2A_GEMM_COMM_PEER_ORDER 1
#endif
#ifndef A2A_GEMM_COMM_WG_PLACEMENT
#define A2A_GEMM_COMM_WG_PLACEMENT 1
#endif
#ifndef A2A_GEMM_COMM_COPY_MODE
#define A2A_GEMM_COMM_COPY_MODE 2
#endif
#ifndef A2A_GEMM_TILE_READY
#define A2A_GEMM_TILE_READY 0
#endif
#ifndef A2A_GEMM_READY_AWARE_K
#define A2A_GEMM_READY_AWARE_K 0
#endif
#ifndef A2A_GEMM_C_STORE_MODE
#define A2A_GEMM_C_STORE_MODE 2
#endif
#ifndef A2A_GEMM_PRESTORE_BARRIER
#define A2A_GEMM_PRESTORE_BARRIER 1
#endif

__device__ inline unsigned current_xcc_id() {
    return __builtin_amdgcn_s_getreg(GETREG_IMMED(3, 0, HWREG_XCC_ID));
}

__device__ inline unsigned current_hw_id() {
    return __builtin_amdgcn_s_getreg(GETREG_IMMED(31, 0, HWREG_HW_ID));
}

__device__ inline unsigned flat_load_u32(const unsigned* ptr) {
    return *reinterpret_cast<const volatile unsigned*>(ptr);
}

template<typename T>
__device__ inline void warmup_input_shard(opus_a2a_gemm_kargs kargs, int k_part, int tid, int wave_id) {
    if (wave_id == 0) {
        return;
    }
    constexpr size_t kWarmupStrideBytes = 2ULL * 1024ULL * 1024ULL;
    const size_t shard_bytes = static_cast<size_t>(kargs.m) * kargs.k_shard * sizeof(typename T::D_A);
    const char* shard_base = static_cast<const char*>(kargs.recv_a_local) +
                             static_cast<size_t>(k_part) * shard_bytes;
    const int warmup_thread = tid - opus::get_warp_size();
    const size_t offset = static_cast<size_t>(warmup_thread) * kWarmupStrideBytes;
    if (offset + sizeof(unsigned) <= shard_bytes) {
        const unsigned value = flat_load_u32(reinterpret_cast<const unsigned*>(shard_base + offset));
        asm volatile("" : : "v"(value));
    }
}

template<typename T>
__device__ inline void copy_local_a_to_peer(opus_a2a_gemm_kargs kargs, int comm_slot, int comm_slots) {
    const int tid = opus::thread_id_x();
    const int src_rank = kargs.my_rank;
    const int step_count = kargs.rank_count - 1;
    const int rank_mask = kargs.rank_count - 1;
    const size_t bytes = static_cast<size_t>(kargs.m) * kargs.k_shard * sizeof(typename T::D_A);
    const size_t vec_count = bytes / sizeof(copy_vec_t);

    for (int step = 1; step <= step_count; ++step) {
        int peer_step = step;
#if A2A_GEMM_COMM_PEER_ORDER == 1
        // Phase-shift each communication WG so concurrent WGs target different
        // peers instead of bursting all stores to one peer at a time.
        peer_step = 1 + ((step - 1 + comm_slot) % step_count);
#endif
        const int dst_rank = (src_rank - peer_step + kargs.rank_count) & rank_mask;
        const size_t src_chunk = kargs.input_mode == OPUS_A2A_INPUT_GENERIC
                                     ? static_cast<size_t>(dst_rank)
                                     : 0;
        const copy_vec_t* src = reinterpret_cast<const copy_vec_t*>(kargs.local_a) +
                                src_chunk * vec_count;
        char* peer_a = static_cast<char*>(cco_lsa_peer_c(kargs.recv_a_win, dst_rank));
        copy_vec_t* dst = reinterpret_cast<copy_vec_t*>(peer_a + static_cast<size_t>(src_rank) * bytes);
#if A2A_GEMM_COMM_COPY_MODE == 2
        auto src_mem = opus::make_gmem(src, static_cast<unsigned int>(bytes));
        auto dst_mem = opus::make_gmem(dst, static_cast<unsigned int>(bytes));
#endif
        auto load_vec = [&](size_t idx) -> copy_vec_t {
#if A2A_GEMM_COMM_COPY_MODE == 2
            return src_mem.template load<1>(static_cast<int>(idx), 0,
                                            opus::number<CPOL_SC0_SC1_LOCAL>{});
#else
            return src[idx];
#endif
        };
        auto store_vec = [&](size_t idx, const copy_vec_t& value) {
#if A2A_GEMM_COMM_COPY_MODE == 2
            dst_mem.template store<1>(value, static_cast<int>(idx), 0,
                                      opus::number<CPOL_SC0_SC1_LOCAL>{});
#else
            dst[idx] = value;
#endif
        };

        const size_t stride = static_cast<size_t>(comm_slots) * T::BLOCK_SIZE;
#if A2A_GEMM_TILE_READY
        const int ready_tile_count = kargs.num_m_tiles;
        const size_t vecs_per_ready_tile =
            static_cast<size_t>(T::B_M) * kargs.k_shard * sizeof(typename T::D_A) /
            sizeof(copy_vec_t);
#else
        const int ready_tile_count = 1;
        const size_t vecs_per_ready_tile = vec_count;
#endif
        for (int ready_tile = 0; ready_tile < ready_tile_count; ++ready_tile) {
        const size_t tile_begin = static_cast<size_t>(ready_tile) * vecs_per_ready_tile;
        const size_t tile_end = tile_begin + vecs_per_ready_tile;
        for (size_t i = tile_begin + static_cast<size_t>(comm_slot * T::BLOCK_SIZE + tid);
             i < tile_end;
             i += stride * 8) {
            const size_t i0 = i;
            const size_t i1 = i + stride;
            const size_t i2 = i + stride * 2;
            const size_t i3 = i + stride * 3;
            const size_t i4 = i + stride * 4;
            const size_t i5 = i + stride * 5;
            const size_t i6 = i + stride * 6;
            const size_t i7 = i + stride * 7;
            copy_vec_t v0, v1, v2, v3, v4, v5, v6, v7;
            const bool p0 = i0 < tile_end;
            const bool p1 = i1 < tile_end;
            const bool p2 = i2 < tile_end;
            const bool p3 = i3 < tile_end;
            const bool p4 = i4 < tile_end;
            const bool p5 = i5 < tile_end;
            const bool p6 = i6 < tile_end;
            const bool p7 = i7 < tile_end;
            if (p0) v0 = load_vec(i0);
            if (p1) v1 = load_vec(i1);
            if (p2) v2 = load_vec(i2);
            if (p3) v3 = load_vec(i3);
            if (p4) v4 = load_vec(i4);
            if (p5) v5 = load_vec(i5);
            if (p6) v6 = load_vec(i6);
            if (p7) v7 = load_vec(i7);
#if A2A_GEMM_COMM_COPY_MODE == 0
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#endif
            if (p0) store_vec(i0, v0);
            if (p1) store_vec(i1, v1);
            if (p2) store_vec(i2, v2);
            if (p3) store_vec(i3, v3);
            if (p4) store_vec(i4, v4);
            if (p5) store_vec(i5, v5);
            if (p6) store_vec(i6, v6);
            if (p7) store_vec(i7, v7);
        }

        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "");
        __builtin_amdgcn_s_barrier();
        if (tid == 0) {
            unsigned int* peer_ready = static_cast<unsigned int*>(cco_lsa_peer_c(kargs.ready_win, dst_rank));
#if A2A_GEMM_TILE_READY
            const int ready_index = src_rank * kargs.num_m_tiles + ready_tile;
#else
            const int ready_index = src_rank;
#endif
            __atomic_fetch_add(peer_ready + ready_index, 1u, __ATOMIC_RELAXED);
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "");
        }
        }
    }
}

template<typename T>
__device__ inline void wait_input_ready(opus_a2a_gemm_kargs kargs, int k_part, int m_tile, int tid, int wave_id) {
    if (tid == 0 && k_part != kargs.my_rank) {
        unsigned int* local_ready = static_cast<unsigned int*>(kargs.ready_local);
#if A2A_GEMM_TILE_READY
        const int ready_index = k_part * kargs.num_m_tiles + m_tile;
#else
        const int ready_index = k_part;
#endif
        while (flat_load_u32(local_ready + ready_index) < static_cast<unsigned int>(kargs.comm_wgs)) {
        }
    } else {
        warmup_input_shard<T>(kargs, k_part, tid, wave_id);
    }
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __builtin_amdgcn_s_barrier();
}

}  // namespace a2a_gemm_lsa

template<typename UserTraits>
__global__ __launch_bounds__(UserTraits::BLOCK_SIZE, 2)
void a2a_lsa_comm_kernel(opus_a2a_gemm_kargs kargs) {
    using T = gemm_quad_subtile::kernel_traits<opus::remove_cvref_t<UserTraits>>;
    const int comm_slot = opus::block_id_x();
    if (comm_slot < kargs.comm_wgs) {
        a2a_gemm_lsa::copy_local_a_to_peer<T>(kargs, comm_slot, kargs.comm_wgs);
    }
}

template<typename UserTraits, int Mode, bool Persistent = false>
__global__ __launch_bounds__(UserTraits::BLOCK_SIZE, 2)
void a2a_gemm_lsa_kernel(opus_a2a_gemm_kargs kargs) {
    static_assert(Mode == 0 || Mode == 1 || Mode == 2);
    static_assert(!Persistent || Mode == 0, "persistent scheduling is only enabled for fused mode");
    using namespace opus;
    using namespace gemm_quad_subtile;
    using namespace a2a_gemm_lsa;
    using opus::operator""_I;
    using T = kernel_traits<opus::remove_cvref_t<UserTraits>>;
    using D_A = typename T::D_A;
    using D_B = typename T::D_B;
    using D_C = typename T::D_C;
    using D_ACC = typename T::D_ACC;

    const int bx = opus::block_id_x();
    const int tid = opus::thread_id_x();
    const int wave_id = __builtin_amdgcn_readfirstlane(tid / get_warp_size());
    const int lane_id = tid % get_warp_size();

#if A2A_GEMM_COMM_WG_PLACEMENT == 1
    const bool is_comm_wg = Mode == 0 && bx < kargs.comm_wgs;
    const int comm_slot = bx;
#else
    constexpr int kXccStride = 8;
    const int comm_span = kargs.comm_wgs * kXccStride;
    const bool is_comm_wg = Mode == 0 &&
                            bx < comm_span &&
                            ((bx & (kXccStride - 1)) == 0);
    const int comm_slot = bx >> 3;
#endif
#if A2A_GEMM_RECORD_WG_HW
    if (tid == 0 && kargs.record_wg_hw && kargs.wg_hw_records != nullptr && bx < kargs.wg_hw_record_count) {
        const unsigned hw_id = current_hw_id();
        unsigned* rec = kargs.wg_hw_records + static_cast<size_t>(bx) * 6;
        rec[0] = static_cast<unsigned>(bx);
        rec[1] = current_xcc_id();
        rec[2] = (hw_id >> 13) & 0x3;
        rec[3] = (hw_id >> 12) & 0x1;
        rec[4] = (hw_id >> 8) & 0xf;
        rec[5] = is_comm_wg ? 1u : 0u;
    }
#endif
    if (is_comm_wg) {
        copy_local_a_to_peer<T>(kargs, comm_slot, kargs.comm_wgs);
        return;
    }

    int comm_before = 0;
    if constexpr (Mode == 0) {
#if A2A_GEMM_COMM_WG_PLACEMENT == 1
        comm_before = kargs.comm_wgs;
#else
        comm_before = bx < comm_span ? ((bx + kXccStride - 1) >> 3) : kargs.comm_wgs;
#endif
    }
    const int compute_worker = bx - comm_before;
    constexpr int kComputeNWorkers = 28;

    const int compute_tile_count = Mode == 1 ? (kargs.num_m_tiles * kComputeNWorkers)
                                             : (kargs.num_m_tiles * kargs.num_n_tiles);
    __shared__ unsigned int next_compute_task;
    bool first_compute_task = true;

    while (true) {
    int compute_task = compute_worker;
    if constexpr (Persistent) {
        if (first_compute_task) {
            first_compute_task = false;
        } else {
            if (tid == 0) {
                next_compute_task = __atomic_fetch_add(kargs.tile_counter, 1u, __ATOMIC_RELAXED);
            }
            __builtin_amdgcn_s_barrier();
            compute_task = static_cast<int>(next_compute_task);
        }
    }
    if (compute_task >= compute_tile_count) {
        return;
    }

    int m_tile = 0;
    int n_worker = 0;
    int n_tile_initial = 0;
    if constexpr (Mode == 1) {
        m_tile = compute_task / kComputeNWorkers;
        n_worker = compute_task - m_tile * kComputeNWorkers;
        n_tile_initial = n_worker;
    } else {
        if (kargs.num_n_tiles == 32) {
            m_tile = compute_task >> 5;
            n_tile_initial = compute_task & 31;
        } else {
            m_tile = compute_task / 31;
            n_tile_initial = compute_task - m_tile * 31;
        }
    }

    const int row = m_tile * T::B_M;
    auto compute_one_n_tile = [&](int n_tile) {
    int col = n_tile * T::B_N;
    int k_base = 0;

    const D_A* a_base = nullptr;
    if constexpr (Mode != 1) {
        a_base = reinterpret_cast<const D_A*>(kargs.recv_a_local);
    } else {
        a_base = reinterpret_cast<const D_A*>(kargs.local_a);
    }

    unsigned int a_bytes = 0;
    if constexpr (Mode != 1) {
        a_bytes = kargs.recv_a_bytes - static_cast<unsigned int>(row * kargs.stride_a * sizeof(D_A));
    } else {
        a_bytes = static_cast<unsigned int>((kargs.m - row) * kargs.stride_a * sizeof(D_A));
    }
    auto g_a = make_gmem(a_base + row * kargs.stride_a, a_bytes);
    auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + col * kargs.stride_b + k_base,
                         static_cast<unsigned int>((kargs.n - col) * kargs.stride_b * sizeof(D_B)));
    auto g_c = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + row * kargs.stride_c,
                         kargs.output_bytes - static_cast<unsigned int>(row * kargs.stride_c * sizeof(D_C)));

    int wave_id_m = wave_id / T::T_N;
    int wave_id_n = wave_id % T::T_N;

    auto u_ga = make_layout_ga<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
    auto u_sa = make_layout_sa<T>(lane_id, wave_id_m, wave_id_n);
    auto u_ra = make_layout_ra<T>(lane_id, wave_id_m);
    auto u_gb = make_layout_gb<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
    auto u_sb = make_layout_sb<T>(lane_id, wave_id_m, wave_id_n);
    auto u_rb = make_layout_rb<T>(lane_id, wave_id_n);

    constexpr int smem_a_byte = T::smem_m_rep * (T::smem_linear_wave + T::smem_padding) * sizeof(D_A);
    __shared__ char smem_a[smem_a_byte * 4];
    constexpr int smem_b_byte = T::smem_n_rep * (T::smem_linear_wave + T::smem_padding) * sizeof(D_B);
    __shared__ char smem_b[smem_b_byte * 4];
#define A_TILE(slot, half) make_smem(reinterpret_cast<D_A*>(smem_a + (((slot) << 1) + (half)) * smem_a_byte))
#define B_TILE(slot, half) make_smem(reinterpret_cast<D_B*>(smem_b + (((slot) << 1) + (half)) * smem_b_byte))
#define LOAD_A_TILE(slot, half) ([&]() { auto mem = A_TILE(slot, half); return load<T::VEC_A>(mem, u_ra); }())
#define LOAD_B_TILE(slot, half) ([&]() { auto mem = B_TILE(slot, half); return load<T::VEC_B>(mem, u_rb); }())

    auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
        seq<T::E_M, T::E_N, T::E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});

    typename decltype(mma)::vtype_a v_a;
    typename decltype(mma)::vtype_b v_b[2];
    typename decltype(mma)::vtype_c v_c[2][2];
    clear(v_c[0][0]);
    clear(v_c[0][1]);
    clear(v_c[1][0]);
    clear(v_c[1][1]);
    constexpr int kFullLocalShardTiles = 1024 / T::B_K;
    static_assert((kFullLocalShardTiles & (kFullLocalShardTiles - 1)) == 0);
#if A2A_GEMM_READY_AWARE_K
    [[maybe_unused]] int active_part = kargs.my_rank;
    [[maybe_unused]] unsigned int processed_parts = 1u << kargs.my_rank;
    __shared__ int selected_ready_part;
#endif
#if !A2A_GEMM_READY_AWARE_K
    auto ordered_part = [&](int tile_k) {
        return (kargs.my_rank + (tile_k >> 4)) & (kargs.rank_count - 1);
    };
#endif
    auto a_offset = [&](int half_tile_m, int tile_k) {
        const int shard_tile_k = tile_k & (kFullLocalShardTiles - 1);
#if A2A_GEMM_READY_AWARE_K
        const int part = Mode != 1 ? active_part : 0;
#else
        const int part = Mode != 1 ? ordered_part(tile_k) : 0;
#endif
        return part * kargs.m * kargs.k_shard +
               half_tile_m * T::HALF_B_M * kargs.stride_a + shard_tile_k * T::B_K;
    };
    auto b_offset = [&](int half_tile_n, int tile_k) {
        if constexpr (Mode != 1) {
            const int shard_tile_k = tile_k & (kFullLocalShardTiles - 1);
            return half_tile_n * T::HALF_B_N * kargs.stride_b +
#if A2A_GEMM_READY_AWARE_K
                   active_part * kargs.k_shard + shard_tile_k * T::B_K;
#else
                   ordered_part(tile_k) * kargs.k_shard + shard_tile_k * T::B_K;
#endif
        }
        return half_tile_n * T::HALF_B_N * kargs.stride_b + tile_k * T::B_K;
    };

    constexpr int kFullLocalLoops = 8192 / T::B_K;
    const int shard_outer_loops = Mode != 1 ? kargs.rank_count : 1;
    const int shard_loops = Mode != 1 ? kFullLocalShardTiles : kFullLocalLoops;
    int tic = 0, toc = 1;

    for (int shard_iter = 0; shard_iter < shard_outer_loops; ++shard_iter) {
#if A2A_GEMM_READY_AWARE_K
    const int shard_base_tile = 0;
#else
    const int shard_base_tile = Mode != 1 ? shard_iter * kFullLocalShardTiles : 0;
#endif
    if constexpr (Mode != 1) {
        if (shard_iter != 0) {
            s_waitcnt_vmcnt(0_I);
            if (wave_id_m == 1) __builtin_amdgcn_s_barrier();
        }
    }
    if constexpr (Mode != 1) {
#if A2A_GEMM_READY_AWARE_K
        if (shard_iter != 0) {
            if (tid == 0) {
                unsigned int* local_ready = static_cast<unsigned int*>(kargs.ready_local);
                int found = -1;
                while (found < 0) {
                    for (int delta = 1; delta < kargs.rank_count; ++delta) {
                        const int candidate = (kargs.my_rank + delta) & (kargs.rank_count - 1);
                        if (processed_parts & (1u << candidate)) continue;
#if A2A_GEMM_TILE_READY
                        const int ready_index = candidate * kargs.num_m_tiles + m_tile;
#else
                        const int ready_index = candidate;
#endif
                        if (flat_load_u32(local_ready + ready_index) >=
                            static_cast<unsigned int>(kargs.comm_wgs)) {
                            found = candidate;
                            break;
                        }
                    }
                }
                selected_ready_part = found;
            }
            __builtin_amdgcn_s_barrier();
            active_part = selected_ready_part;
            processed_parts |= 1u << active_part;
        }
        const int part = active_part;
#else
        const int part = ordered_part(shard_base_tile);
#endif
        if constexpr (Mode == 0) {
            wait_input_ready<T>(kargs, part, m_tile, tid, wave_id);
        }
    }

    async_load<T::VEC_B>(g_b, B_TILE(tic, 0).ptr, u_gb, u_sb, b_offset(0, shard_base_tile + 0));
    async_load<T::VEC_A>(g_a, A_TILE(tic, 0).ptr, u_ga, u_sa, a_offset(0, shard_base_tile + 0));
    async_load<T::VEC_B>(g_b, B_TILE(tic, 1).ptr, u_gb, u_sb, b_offset(1, shard_base_tile + 0));
    async_load<T::VEC_A>(g_a, A_TILE(tic, 1).ptr, u_ga, u_sa, a_offset(1, shard_base_tile + 0));

    if (shard_iter == 0 && wave_id_m == 1) __builtin_amdgcn_s_barrier();

    s_waitcnt_vmcnt(number<T::a_buffer_load_insts + T::b_buffer_load_insts>{});
    __builtin_amdgcn_s_barrier();

    async_load<T::VEC_B>(g_b, B_TILE(toc, 0).ptr, u_gb, u_sb, b_offset(0, shard_base_tile + 1));
    async_load<T::VEC_A>(g_a, A_TILE(toc, 0).ptr, u_ga, u_sa, a_offset(0, shard_base_tile + 1));
    async_load<T::VEC_B>(g_b, B_TILE(toc, 1).ptr, u_gb, u_sb, b_offset(1, shard_base_tile + 1));

    s_waitcnt_vmcnt(number<T::a_buffer_load_insts + 2 * T::b_buffer_load_insts>{});
    __builtin_amdgcn_s_barrier();

    v_b[0] = LOAD_B_TILE(tic, 0);
    __builtin_amdgcn_s_barrier();

    for(int tile = 0; tile < shard_loops - 2; tile += 2) {
        v_a = LOAD_A_TILE(tic, 0);
        async_load<T::VEC_A>(g_a, A_TILE(toc, 1).ptr, u_ga, u_sa, a_offset(1, shard_base_tile + tile + 1));
        s_waitcnt_lgkmcnt(number<T::a_ds_read_insts>{});
        __builtin_amdgcn_s_barrier();

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[0][0] = mma(v_a, v_b[0], v_c[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_b[1] = LOAD_B_TILE(tic, 1);
        async_load<T::VEC_B>(g_b, B_TILE(tic, 0).ptr, u_gb, u_sb, b_offset(0, shard_base_tile + tile + 2));
        __builtin_amdgcn_s_barrier();

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[0][1] = mma(v_a, v_b[1], v_c[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        v_a = LOAD_A_TILE(tic, 1);
        async_load<T::VEC_A>(g_a, A_TILE(tic, 0).ptr, u_ga, u_sa, a_offset(0, shard_base_tile + tile + 2));
        __builtin_amdgcn_s_barrier();

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[1][0] = mma(v_a, v_b[0], v_c[1][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_b[0] = LOAD_B_TILE(toc, 0);
        async_load<T::VEC_B>(g_b, B_TILE(tic, 1).ptr, u_gb, u_sb, b_offset(1, shard_base_tile + tile + 2));
        s_waitcnt_vmcnt(number<T::a_buffer_load_insts + 2 * T::b_buffer_load_insts>{});
        __builtin_amdgcn_s_barrier();

        __builtin_amdgcn_s_setprio(1);
        v_c[1][1] = mma(v_a, v_b[1], v_c[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        v_a = LOAD_A_TILE(toc, 0);
        async_load<T::VEC_A>(g_a, A_TILE(tic, 1).ptr, u_ga, u_sa, a_offset(1, shard_base_tile + tile + 2));
        s_waitcnt_lgkmcnt(number<T::a_ds_read_insts>{});
        __builtin_amdgcn_s_barrier();

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[0][0] = mma(v_a, v_b[0], v_c[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_b[1] = LOAD_B_TILE(toc, 1);
        async_load<T::VEC_B>(g_b, B_TILE(toc, 0).ptr, u_gb, u_sb, b_offset(0, shard_base_tile + tile + 3));
        __builtin_amdgcn_s_barrier();

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[0][1] = mma(v_a, v_b[1], v_c[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        v_a = LOAD_A_TILE(toc, 1);
        async_load<T::VEC_A>(g_a, A_TILE(toc, 0).ptr, u_ga, u_sa, a_offset(0, shard_base_tile + tile + 3));
        __builtin_amdgcn_s_barrier();

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[1][0] = mma(v_a, v_b[0], v_c[1][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_b[0] = LOAD_B_TILE(tic, 0);
        async_load<T::VEC_B>(g_b, B_TILE(toc, 1).ptr, u_gb, u_sb, b_offset(1, shard_base_tile + tile + 3));
        s_waitcnt_vmcnt(number<T::a_buffer_load_insts + 2 * T::b_buffer_load_insts>{});
        __builtin_amdgcn_s_barrier();

        __builtin_amdgcn_s_setprio(1);
        v_c[1][1] = mma(v_a, v_b[1], v_c[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    {
        int tile = shard_loops - 2;
        v_a = LOAD_A_TILE(tic, 0);
        async_load<T::VEC_A>(g_a, A_TILE(toc, 1).ptr, u_ga, u_sa, a_offset(1, shard_base_tile + tile + 1));
        __builtin_amdgcn_s_barrier();
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[0][0] = mma(v_a, v_b[0], v_c[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        v_b[1] = LOAD_B_TILE(tic, 1);
        __builtin_amdgcn_s_barrier();
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[0][1] = mma(v_a, v_b[1], v_c[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        v_a = LOAD_A_TILE(tic, 1);
        s_waitcnt_vmcnt(number<T::a_buffer_load_insts + T::b_buffer_load_insts>{});
        __builtin_amdgcn_s_barrier();
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[1][0] = mma(v_a, v_b[0], v_c[1][0]);
        v_c[1][1] = mma(v_a, v_b[1], v_c[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    {
        tic ^= 1;
        toc ^= 1;
    }

    {
        v_b[0] = LOAD_B_TILE(tic, 0);
        v_a = LOAD_A_TILE(tic, 0);
        s_waitcnt_vmcnt(number<T::a_buffer_load_insts>{});
        __builtin_amdgcn_s_barrier();
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[0][0] = mma(v_a, v_b[0], v_c[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        v_b[1] = LOAD_B_TILE(tic, 1);
        s_waitcnt_vmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[0][1] = mma(v_a, v_b[1], v_c[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        v_a = LOAD_A_TILE(tic, 1);
        __builtin_amdgcn_s_barrier();
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[1][0] = mma(v_a, v_b[0], v_c[1][0]);
        v_c[1][1] = mma(v_a, v_b[1], v_c[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }
    if constexpr (Mode != 1) {
        if (shard_iter + 1 < shard_outer_loops) {
            s_waitcnt_vmcnt(0_I);
            if (wave_id_m == 0) __builtin_amdgcn_s_barrier();
        }
    }
    }

#if A2A_GEMM_PRESTORE_BARRIER
    if (wave_id_m == 0) __builtin_amdgcn_s_barrier();
#endif

    [[maybe_unused]] auto u_gc = make_layout_gc<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_c);
    auto c_offset = [&](int half_tile_m, int half_tile_n) {
        return half_tile_m * T::HALF_B_M * kargs.stride_c + half_tile_n * T::HALF_B_N + col;
    };

    auto direct_store = [&](auto& v_c_in, int half_tile_m, int half_tile_n) {
        auto v_c_f16 = cast<D_C>(v_c_in);
        static_assert(sizeof(D_C) * 8 % sizeof(u32_t) == 0);
        constexpr int u32_per_chunk = sizeof(D_C) * 8 / sizeof(u32_t);
        constexpr int num_chunks = sizeof(v_c_f16) / (sizeof(u32_t) * u32_per_chunk);
        auto* p_u32 = reinterpret_cast<u32_t*>(&v_c_f16);
        static_for<num_chunks>([&](auto c) {
            auto* p = p_u32 + c.value * u32_per_chunk;
            auto r0 = __builtin_amdgcn_permlane16_swap(p[0], p[2], false, true);
            auto r1 = __builtin_amdgcn_permlane16_swap(p[1], p[3], false, true);
            p[0] = r0[0]; p[2] = r0[1];
            p[1] = r1[0]; p[3] = r1[1];
        });
#if A2A_GEMM_C_STORE_MODE == 2
        static_for<num_chunks>([&](auto c) {
            auto* p = p_u32 + c.value * u32_per_chunk;
            const int half_col_block = lane_id / 32;
            const int lane_in_half = lane_id - half_col_block * 32;
            const int row_in_16 = lane_in_half / 2;
            const int vec_pair = lane_in_half - row_in_16 * 2;
            const int src_lane = row_in_16 + half_col_block * 16 + vec_pair * 32;

            i32x4_t raw;
            raw[0] = __builtin_bit_cast(i32_t, shfl(p[0], src_lane));
            raw[1] = __builtin_bit_cast(i32_t, shfl(p[1], src_lane));
            raw[2] = __builtin_bit_cast(i32_t, shfl(p[2], src_lane));
            raw[3] = __builtin_bit_cast(i32_t, shfl(p[3], src_lane));
            auto v = __builtin_bit_cast(
                typename decltype(g_c)::template vector_type<T::VEC_C>, raw);

            const int row_in_half =
                c.value * (T::T_M * T::W_M) + wave_id_m * T::W_M + row_in_16;
            const int col_vec = half_col_block * 8 + wave_id_n * 2 + vec_pair;
            store<T::VEC_C>(
                g_c, v, c_offset(half_tile_m, half_tile_n) +
                            row_in_half * kargs.stride_c + col_vec * T::VEC_C);
        });
#else
        g_c.template store<T::VEC_C>(v_c_f16, u_gc, c_offset(half_tile_m, half_tile_n));
#endif
    };

    direct_store(v_c[0][0], 0, 0);
    direct_store(v_c[0][1], 0, 1);
    direct_store(v_c[1][0], 1, 0);
    direct_store(v_c[1][1], 1, 1);
    };

    compute_one_n_tile(n_tile_initial);
    if constexpr (Mode == 1) {
        if (n_worker + kComputeNWorkers < kargs.num_n_tiles) {
            compute_one_n_tile(n_worker + kComputeNWorkers);
        }
    }
    if constexpr (!Persistent) {
        return;
    }
    }
}
