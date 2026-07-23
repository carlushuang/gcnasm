#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

#include <omp.h>
#include <hip/hip_runtime.h>
#include "mori/cco/cco.hpp"

#include "gemm_defs.h"

#ifndef A2A_GEMM_COMM_WG_PLACEMENT
#define A2A_GEMM_COMM_WG_PLACEMENT 1
#endif
#ifndef A2A_GEMM_TILE_READY
#define A2A_GEMM_TILE_READY 0
#endif

using namespace mori::cco;

using ncclResult_t = int;
using ncclComm_t = void*;
static constexpr ncclResult_t ncclSuccess = 0;
static constexpr int ncclBfloat16 = 9;
struct ncclUniqueId {
    char internal[128];
};
extern "C" ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId);
extern "C" ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);
extern "C" ncclResult_t ncclCommDestroy(ncclComm_t comm);
extern "C" const char* ncclGetErrorString(ncclResult_t result);
extern "C" ncclResult_t ncclGroupStart();
extern "C" ncclResult_t ncclGroupEnd();
extern "C" ncclResult_t ncclSend(const void* sendbuff, size_t count, int datatype, int peer, ncclComm_t comm, hipStream_t stream);
extern "C" ncclResult_t ncclRecv(void* recvbuff, size_t count, int datatype, int peer, ncclComm_t comm, hipStream_t stream);

#define CHECK_HIP(call)                                                                                   \
    do {                                                                                                  \
        hipError_t status_ = call;                                                                        \
        if (status_ != hipSuccess) {                                                                      \
            fprintf(stderr, "HIP error (%s:%d): %s\n", __FILE__, __LINE__, hipGetErrorString(status_));   \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                                 \
        }                                                                                                 \
    } while (0)

#define CHECK_CCO(call)                                                                    \
    do {                                                                                   \
        int cco_status_ = (call);                                                          \
        if (cco_status_ != 0) {                                                            \
            fprintf(stderr, "cco error %d (%s:%d): %s\n", cco_status_, __FILE__, __LINE__, \
                    #call);                                                                \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                  \
        }                                                                                  \
    } while (0)

#define CHECK_NCCL(call)                                                                                   \
    do {                                                                                                   \
        ncclResult_t status_ = call;                                                                       \
        if (status_ != ncclSuccess) {                                                                      \
            fprintf(stderr, "NCCL/RCCL error (%s:%d): %s\n", __FILE__, __LINE__, ncclGetErrorString(status_)); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                                  \
        }                                                                                                  \
    } while (0)

template<typename Traits, int Mode, bool Persistent>
__global__ void a2a_gemm_lsa_kernel(opus_a2a_gemm_kargs kargs);
template<typename Traits>
__global__ void a2a_lsa_comm_kernel(opus_a2a_gemm_kargs kargs);
template<typename Traits>
__global__ void gemm_a16w16_quad_subtile_kernel(opus_gemm_kargs kargs);

__global__ void pack_a_shards_kernel(const bf16_t* __restrict__ recv,
                                     bf16_t* __restrict__ full,
                                     int m,
                                     int k_shard,
                                     int ranks) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = m * k_shard * ranks;
    const int grid_threads = gridDim.x * blockDim.x;
    for (int idx = tid; idx < total; idx += grid_threads) {
        const int kk = idx % k_shard;
        const int t0 = idx / k_shard;
        const int part = t0 % ranks;
        const int row = t0 / ranks;
        full[static_cast<size_t>(row) * k_shard * ranks + part * k_shard + kk] =
            recv[(static_cast<size_t>(part) * m + row) * k_shard + kk];
    }
}

static constexpr size_t kPerRankVmm = 512ULL * 1024 * 1024;

static float a_value(int src_rank, int dst_rank, int row, int k_local) {
    return 0.001f * float(src_rank + 1) + 0.0003f * float(dst_rank) +
           0.0002f * float((row % 17) - 8) +
           0.0001f * float((k_local % 29) - 14);
}

static float b_value(int dst_rank, int col, int global_k) {
    return 0.0002f * float(dst_rank) + 0.0003f * float((col % 23) - 11) +
           0.0001f * float((global_k % 31) - 15);
}

static void fill_a(bf16_t* a, int rank, int rank_count, int m, int k_shard, int input_mode) {
    const int chunks = input_mode == OPUS_A2A_INPUT_GENERIC ? rank_count : 1;
#pragma omp parallel for collapse(3)
    for (int dst = 0; dst < chunks; ++dst) {
        for (int r = 0; r < m; ++r) {
            for (int k = 0; k < k_shard; ++k) {
                a[(static_cast<size_t>(dst) * m + r) * k_shard + k] =
                    static_cast<bf16_t>(a_value(rank, dst, r, k));
            }
        }
    }
}

static void fill_b(bf16_t* b, int rank, int n, int k, int input_mode) {
    const int dst = input_mode == OPUS_A2A_INPUT_GENERIC ? rank : 0;
#pragma omp parallel for collapse(2)
    for (int c = 0; c < n; ++c) {
        for (int kk = 0; kk < k; ++kk) {
            b[c * k + kk] = static_cast<bf16_t>(b_value(dst, c, kk));
        }
    }
}

static float sample_ref(int dst_rank, int row, int col, int k_shard, int ranks, int input_mode) {
    const int a_dst = input_mode == OPUS_A2A_INPUT_GENERIC ? dst_rank : 0;
    const int b_dst = input_mode == OPUS_A2A_INPUT_GENERIC ? dst_rank : 0;
    float acc = 0.0f;
    for (int src = 0; src < ranks; ++src) {
        for (int kk = 0; kk < k_shard; ++kk) {
            const int global_k = src * k_shard + kk;
            const float av = static_cast<float>(static_cast<bf16_t>(a_value(src, a_dst, row, kk)));
            const float bv = static_cast<float>(static_cast<bf16_t>(b_value(b_dst, col, global_k)));
            acc += av * bv;
        }
    }
    return static_cast<float>(static_cast<bf16_t>(acc));
}

static float sample_ref_local_repeat(int rank, int row, int col, int k_shard, int ranks) {
    float acc = 0.0f;
    for (int part = 0; part < ranks; ++part) {
        for (int kk = 0; kk < k_shard; ++kk) {
            const int global_k = part * k_shard + kk;
            const float av = static_cast<float>(static_cast<bf16_t>(a_value(rank, 0, row, kk)));
            const float bv = static_cast<float>(static_cast<bf16_t>(b_value(0, col, global_k)));
            acc += av * bv;
        }
    }
    return static_cast<float>(static_cast<bf16_t>(acc));
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    int M = 2048;
    int N = 8192;
    int K_SHARD = 1024;
    int warmup = 0;
    int iters = 1;
    int mode = 0;
    int comm_wgs = 16;
    int persistent = 0;
    int compute_wgs_arg = 0;
    int input_mode = OPUS_A2A_INPUT_BROADCAST;
    bool record_wg_hw = false;
    bool validate = true;
    for (int i = 1; i < argc; ++i) {
        if ((std::strcmp(argv[i], "-m") == 0 || std::strcmp(argv[i], "--m") == 0) && i + 1 < argc) M = std::atoi(argv[++i]);
        else if ((std::strcmp(argv[i], "-n") == 0 || std::strcmp(argv[i], "--n") == 0) && i + 1 < argc) N = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--k-shard") == 0 && i + 1 < argc) K_SHARD = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) warmup = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) mode = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--comm-wgs") == 0 && i + 1 < argc) comm_wgs = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--persistent") == 0 && i + 1 < argc) persistent = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--compute-wgs") == 0 && i + 1 < argc) compute_wgs_arg = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--input-mode") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            if (std::strcmp(value, "broadcast") == 0) input_mode = OPUS_A2A_INPUT_BROADCAST;
            else if (std::strcmp(value, "generic_a2a") == 0) input_mode = OPUS_A2A_INPUT_GENERIC;
            else input_mode = -1;
        }
        else if (std::strcmp(argv[i], "--record-wg-hw") == 0) record_wg_hw = true;
        else if (std::strcmp(argv[i], "--no-validate") == 0) validate = false;
    }

    if (nranks != 4 && nranks != 8) {
        if (rank == 0) fprintf(stderr, "requires exactly 4 or 8 ranks\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    using Traits = opus_gemm_traits<512, 256, 256, 64, bf16_t, bf16_t, bf16_t, float>;
    const int K = K_SHARD * nranks;
    if (M % Traits::B_M != 0 || N % Traits::B_N != 0 || K_SHARD % Traits::B_K != 0 ||
        K_SHARD < 2 * Traits::B_K || ((K_SHARD / Traits::B_K) % 2) != 0) {
        if (rank == 0) fprintf(stderr, "unsupported shape for first correctness kernel\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (comm_wgs <= 0) {
        if (rank == 0) fprintf(stderr, "unsupported comm config: comm_wgs must be positive\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (mode < 0 || mode > 3) {
        if (rank == 0) fprintf(stderr, "unsupported mode: 0=fused, 1=compute-only, 2=RCCL baseline, 3=split LSA\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if ((persistent != 0 && persistent != 1) || (persistent && mode != 0)) {
        if (rank == 0) fprintf(stderr, "--persistent must be 0 or 1 and is only supported with --mode 0\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (compute_wgs_arg < 0) {
        if (rank == 0) fprintf(stderr, "--compute-wgs must be non-negative\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (input_mode != OPUS_A2A_INPUT_BROADCAST && input_mode != OPUS_A2A_INPUT_GENERIC) {
        if (rank == 0) fprintf(stderr, "--input-mode must be broadcast or generic_a2a\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (input_mode == OPUS_A2A_INPUT_GENERIC && mode == 1) {
        if (rank == 0) fprintf(stderr, "generic_a2a input is not supported with compute-only mode 1\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const char* input_mode_name =
        input_mode == OPUS_A2A_INPUT_GENERIC ? "generic_a2a" : "broadcast";

    int ndev = 0;
    CHECK_HIP(hipGetDeviceCount(&ndev));
    CHECK_HIP(hipSetDevice(rank % ndev));
    int cu_count = 0;
    CHECK_HIP(hipDeviceGetAttribute(&cu_count, hipDeviceAttributeMultiprocessorCount, rank % ndev));

    ccoUniqueId uid;
    if (rank == 0) CHECK_CCO(ccoGetUniqueId(&uid));
    MPI_Bcast(&uid, sizeof(uid), MPI_BYTE, 0, MPI_COMM_WORLD);
    ccoComm* comm = nullptr;
    CHECK_CCO(ccoCommCreate(uid, nranks, rank, kPerRankVmm, &comm));

    ncclComm_t nccl_comm = nullptr;

    const size_t a_chunk_elems = static_cast<size_t>(M) * K_SHARD;
    const int local_a_chunks = input_mode == OPUS_A2A_INPUT_GENERIC ? nranks : 1;
    const size_t local_a_elems = static_cast<size_t>(local_a_chunks) * a_chunk_elems;
    const size_t b_elems = static_cast<size_t>(N) * K;
    const size_t c_elems = static_cast<size_t>(M) * N;
    const size_t a_full_elems = static_cast<size_t>(M) * K;
    const size_t recv_elems = static_cast<size_t>(nranks) * a_chunk_elems;
    const int num_m_tiles = ceil_div(M, Traits::B_M);
    const int num_n_tiles = ceil_div(N, Traits::B_N);
    const size_t ready_elems = static_cast<size_t>(nranks) *
                               (A2A_GEMM_TILE_READY ? num_m_tiles : 1);
    static constexpr int kComputeNWorkers = 28;
    const bool n_outer_mode = mode == 1;
    const int compute_tasks = n_outer_mode ? (num_m_tiles * kComputeNWorkers)
                                           : (num_m_tiles * num_n_tiles);
    int compute_wgs = compute_tasks;
    if (persistent) {
        const int auto_compute_wgs = cu_count - comm_wgs;
        compute_wgs = compute_wgs_arg > 0 ? compute_wgs_arg : auto_compute_wgs;
        if (compute_wgs > compute_tasks) compute_wgs = compute_tasks;
        const int min_compute_wgs_for_interleaved_comm =
            A2A_GEMM_COMM_WG_PLACEMENT == 0 ? 7 * (comm_wgs - 1) : 1;
        if (compute_wgs <= 0 || compute_wgs < min_compute_wgs_for_interleaved_comm) {
            if (rank == 0) {
                fprintf(stderr,
                        "persistent compute WG count %d is too small for %d interleaved comm WGs\n",
                        compute_wgs, comm_wgs);
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    const int grid_wgs = n_outer_mode ? compute_tasks
                                      : ((persistent ? compute_wgs : compute_tasks) + comm_wgs);
    static constexpr int kHwRecordWidth = 6;
    const size_t hw_record_elems = static_cast<size_t>(grid_wgs) * kHwRecordWidth;

    auto h_a = std::make_unique<bf16_t[]>(local_a_elems);
    auto h_b = std::make_unique<bf16_t[]>(b_elems);
    fill_a(h_a.get(), rank, nranks, M, K_SHARD, input_mode);
    fill_b(h_b.get(), rank, N, K, input_mode);

    bf16_t* d_a = nullptr;
    bf16_t* d_a_full = nullptr;
    bf16_t* d_a_recv = nullptr;
    bf16_t* d_b = nullptr;
    bf16_t* d_c = nullptr;
    unsigned int* d_wg_hw_records = nullptr;
    unsigned int* d_tile_counter = nullptr;
    CHECK_HIP(hipMalloc(&d_a, local_a_elems * sizeof(bf16_t)));
    CHECK_HIP(hipMalloc(&d_a_full, a_full_elems * sizeof(bf16_t)));
    CHECK_HIP(hipMalloc(&d_a_recv, recv_elems * sizeof(bf16_t)));
    CHECK_HIP(hipMalloc(&d_b, b_elems * sizeof(bf16_t)));
    CHECK_HIP(hipMalloc(&d_c, c_elems * sizeof(bf16_t)));
    CHECK_HIP(hipMalloc(&d_wg_hw_records, hw_record_elems * sizeof(unsigned int)));
    CHECK_HIP(hipMalloc(&d_tile_counter, sizeof(unsigned int)));
    CHECK_HIP(hipMemcpy(d_a, h_a.get(), local_a_elems * sizeof(bf16_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_b, h_b.get(), b_elems * sizeof(bf16_t), hipMemcpyHostToDevice));

    ccoWindow_t recv_win = nullptr;
    ccoWindow_t ready_win = nullptr;
    void* recv_local = nullptr;
    void* ready_local = nullptr;
    CHECK_CCO(ccoWindowRegister(comm, recv_elems * sizeof(bf16_t), &recv_win, &recv_local));
    CHECK_CCO(ccoWindowRegister(comm, ready_elems * sizeof(unsigned int), &ready_win, &ready_local));

    if (mode == 2) {
        ncclUniqueId nccl_uid;
        if (rank == 0) CHECK_NCCL(ncclGetUniqueId(&nccl_uid));
        MPI_Bcast(&nccl_uid, sizeof(nccl_uid), MPI_BYTE, 0, MPI_COMM_WORLD);
        CHECK_NCCL(ncclCommInitRank(&nccl_comm, nranks, nccl_uid, rank));
    }

    opus_a2a_gemm_kargs kargs{};
    kargs.local_a = d_a;
    kargs.ptr_b = d_b;
    kargs.ptr_c = d_c;
    kargs.recv_a_win = recv_win;
    kargs.ready_win = ready_win;
    kargs.recv_a_local = recv_local;
    kargs.ready_local = ready_local;
    kargs.wg_hw_records = d_wg_hw_records;
    kargs.tile_counter = d_tile_counter;
    kargs.m = M;
    kargs.n = N;
    kargs.k = K;
    kargs.k_shard = K_SHARD;
    kargs.rank_count = nranks;
    kargs.my_rank = rank;
    kargs.input_mode = input_mode;
    kargs.stride_a = K_SHARD;
    kargs.stride_b = K;
    kargs.stride_c = N;
    kargs.stride_ws = N;
    kargs.recv_a_bytes = static_cast<unsigned int>(recv_elems * sizeof(bf16_t));
    kargs.ready_bytes = static_cast<unsigned int>(ready_elems * sizeof(unsigned int));
    kargs.output_bytes = static_cast<unsigned int>(c_elems * sizeof(bf16_t));
    kargs.comm_wgs = comm_wgs;
    kargs.wg_hw_record_count = grid_wgs;
    kargs.record_wg_hw = record_wg_hw ? 1 : 0;
    kargs.num_m_tiles = num_m_tiles;
    kargs.num_n_tiles = num_n_tiles;
    kargs.mode = mode;

    opus_gemm_kargs gemm_args{};
    gemm_args.ptr_a = d_a_full;
    gemm_args.ptr_b = d_b;
    gemm_args.ptr_c = d_c;
    gemm_args.m = M;
    gemm_args.n = N;
    gemm_args.k = K;
    gemm_args.batch = 1;
    gemm_args.stride_a = K;
    gemm_args.stride_b = K;
    gemm_args.stride_c = N;
    gemm_args.stride_a_batch = static_cast<int>(a_full_elems);
    gemm_args.stride_b_batch = static_cast<int>(b_elems);
    gemm_args.stride_c_batch = static_cast<int>(c_elems);

    dim3 grid(grid_wgs, 1, 1);
    dim3 block(Traits::BLOCK_SIZE, 1, 1);
    dim3 gemm_grid(num_m_tiles * num_n_tiles, 1, 1);
    dim3 pack_grid(ceil_div(static_cast<int>(a_full_elems), 256), 1, 1);
    dim3 pack_block(256, 1, 1);
    const unsigned int tile_counter_start = persistent ? static_cast<unsigned int>(compute_wgs) : 0u;

    auto clear_for_launch = [&]() {
        CHECK_HIP(hipMemset(recv_local, 0, recv_elems * sizeof(bf16_t)));
        CHECK_HIP(hipMemset(ready_local, 0, ready_elems * sizeof(unsigned int)));
        CHECK_HIP(hipMemset(d_a_full, 0, a_full_elems * sizeof(bf16_t)));
        CHECK_HIP(hipMemset(d_a_recv, 0, recv_elems * sizeof(bf16_t)));
        CHECK_HIP(hipMemset(d_c, 0, c_elems * sizeof(bf16_t)));
        CHECK_HIP(hipMemset(d_wg_hw_records, 0xff, hw_record_elems * sizeof(unsigned int)));
        CHECK_HIP(hipMemcpy(d_tile_counter, &tile_counter_start, sizeof(tile_counter_start),
                            hipMemcpyHostToDevice));
        const size_t self_src_chunk =
            input_mode == OPUS_A2A_INPUT_GENERIC ? static_cast<size_t>(rank) : 0;
        CHECK_HIP(hipMemcpy(static_cast<char*>(recv_local) +
                                static_cast<size_t>(rank) * a_chunk_elems * sizeof(bf16_t),
                            d_a + self_src_chunk * a_chunk_elems,
                            a_chunk_elems * sizeof(bf16_t), hipMemcpyDeviceToDevice));
        CHECK_HIP(hipDeviceSynchronize());
        CHECK_CCO(ccoBarrierAll(comm));
    };

    auto launch_nonfused_once = [&]() {
        CHECK_NCCL(ncclGroupStart());
        for (int peer = 0; peer < nranks; ++peer) {
            const size_t send_chunk =
                input_mode == OPUS_A2A_INPUT_GENERIC ? static_cast<size_t>(peer) : 0;
            CHECK_NCCL(ncclSend(d_a + send_chunk * a_chunk_elems,
                                a_chunk_elems, ncclBfloat16, peer, nccl_comm, nullptr));
            CHECK_NCCL(ncclRecv(d_a_recv + static_cast<size_t>(peer) * a_chunk_elems,
                                a_chunk_elems, ncclBfloat16, peer, nccl_comm, nullptr));
        }
        CHECK_NCCL(ncclGroupEnd());
        pack_a_shards_kernel<<<pack_grid, pack_block>>>(d_a_recv, d_a_full, M, K_SHARD, nranks);
        CHECK_HIP(hipGetLastError());
        gemm_a16w16_quad_subtile_kernel<Traits><<<gemm_grid, block>>>(gemm_args);
        CHECK_HIP(hipGetLastError());
    };

    auto launch_once = [&]() {
        if (mode == 0) {
            if (persistent) {
                a2a_gemm_lsa_kernel<Traits, 0, true><<<grid, block>>>(kargs);
            } else {
                a2a_gemm_lsa_kernel<Traits, 0, false><<<grid, block>>>(kargs);
            }
        } else if (mode == 1) {
            a2a_gemm_lsa_kernel<Traits, 1, false><<<grid, block>>>(kargs);
        } else if (mode == 2) {
            launch_nonfused_once();
        }
        CHECK_HIP(hipGetLastError());
    };

    auto launch_split_comm = [&]() {
        a2a_lsa_comm_kernel<Traits><<<dim3(comm_wgs), block>>>(kargs);
        CHECK_HIP(hipGetLastError());
    };
    auto launch_split_compute = [&]() {
        a2a_gemm_lsa_kernel<Traits, 2, false><<<gemm_grid, block>>>(kargs);
        CHECK_HIP(hipGetLastError());
    };

    hipEvent_t start, comm_stop, compute_start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&comm_stop));
    CHECK_HIP(hipEventCreate(&compute_start));
    CHECK_HIP(hipEventCreate(&stop));
    float total_ms = 0.0f, comm_total_ms = 0.0f, compute_total_ms = 0.0f;
    if (mode == 3) {
        for (int i = 0; i < warmup; ++i) {
            clear_for_launch();
            launch_split_comm();
            CHECK_HIP(hipDeviceSynchronize());
            CHECK_CCO(ccoBarrierAll(comm));
            launch_split_compute();
            CHECK_HIP(hipDeviceSynchronize());
            CHECK_CCO(ccoBarrierAll(comm));
        }
        for (int i = 0; i < iters; ++i) {
            clear_for_launch();
            CHECK_HIP(hipEventRecord(start));
            launch_split_comm();
            CHECK_HIP(hipEventRecord(comm_stop));
            CHECK_HIP(hipEventSynchronize(comm_stop));
            CHECK_CCO(ccoBarrierAll(comm));
            CHECK_HIP(hipEventRecord(compute_start));
            launch_split_compute();
            CHECK_HIP(hipEventRecord(stop));
            CHECK_HIP(hipEventSynchronize(stop));
            float comm_ms = 0.0f, compute_ms = 0.0f, split_ms = 0.0f;
            CHECK_HIP(hipEventElapsedTime(&comm_ms, start, comm_stop));
            CHECK_HIP(hipEventElapsedTime(&compute_ms, compute_start, stop));
            CHECK_HIP(hipEventElapsedTime(&split_ms, start, stop));
            comm_total_ms += comm_ms;
            compute_total_ms += compute_ms;
            total_ms += split_ms;
            CHECK_CCO(ccoBarrierAll(comm));
        }
    } else {
        for (int i = 0; i < warmup; ++i) {
            clear_for_launch();
            launch_once();
            CHECK_HIP(hipDeviceSynchronize());
            CHECK_CCO(ccoBarrierAll(comm));
        }
        for (int i = 0; i < iters; ++i) {
            clear_for_launch();
            CHECK_HIP(hipEventRecord(start));
            launch_once();
            CHECK_HIP(hipEventRecord(stop));
            CHECK_HIP(hipEventSynchronize(stop));
            float ms = 0.0f;
            CHECK_HIP(hipEventElapsedTime(&ms, start, stop));
            total_ms += ms;
            CHECK_CCO(ccoBarrierAll(comm));
        }
    }
    const double local_ms = static_cast<double>(total_ms) / iters;
    const double local_comm_ms = static_cast<double>(comm_total_ms) / iters;
    const double local_compute_ms = static_cast<double>(compute_total_ms) / iters;
    double max_ms = 0.0, max_comm_ms = 0.0, max_compute_ms = 0.0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_comm_ms, &max_comm_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_compute_ms, &max_compute_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0 && record_wg_hw) {
        auto h_hw_records = std::make_unique<unsigned int[]>(hw_record_elems);
        CHECK_HIP(hipMemcpy(h_hw_records.get(), d_wg_hw_records,
                            hw_record_elems * sizeof(unsigned int), hipMemcpyDeviceToHost));
        FILE* trace = std::fopen("build/cu_trace_rank0.csv", "w");
        if (trace) {
            std::fprintf(trace, "wg,xcc,se,sh,cu,is_comm\n");
            unsigned xcc_total[8] = {};
            unsigned xcc_comm[8] = {};
            for (int wg = 0; wg < grid_wgs; ++wg) {
                const unsigned* rec = h_hw_records.get() + static_cast<size_t>(wg) * kHwRecordWidth;
                if (rec[0] == 0xffffffffu) continue;
                std::fprintf(trace, "%u,%u,%u,%u,%u,%u\n", rec[0], rec[1], rec[2], rec[3], rec[4], rec[5]);
                if (rec[1] < 8) {
                    ++xcc_total[rec[1]];
                    xcc_comm[rec[1]] += rec[5] ? 1u : 0u;
                }
            }
            std::fclose(trace);
            std::printf("rank0_cu_trace_csv=build/cu_trace_rank0.csv");
            for (int x = 0; x < 8; ++x) {
                if (xcc_total[x] != 0) std::printf(" xcc%d_total=%u xcc%d_comm=%u", x, xcc_total[x], x, xcc_comm[x]);
            }
            std::printf("\n");
        } else {
            std::fprintf(stderr, "failed to open build/cu_trace_rank0.csv for writing\n");
        }
    }

    int mism = 0;
    if (validate) {
        auto h_c = std::make_unique<bf16_t[]>(c_elems);
        CHECK_HIP(hipMemcpy(h_c.get(), d_c, c_elems * sizeof(bf16_t), hipMemcpyDeviceToHost));

        const int sample_rows[] = {0, 17, 255, 511, M / 2, M - 1};
        const int sample_cols[] = {0, 127, 255, 1024, 4096, 8191};
        for (int r : sample_rows) {
            if (r >= M) continue;
            for (int c : sample_cols) {
                if (c >= N) continue;
                const float got = static_cast<float>(h_c[static_cast<size_t>(r) * N + c]);
                float ref = sample_ref(rank, r, c, K_SHARD, nranks, input_mode);
                if (mode == 1) ref = sample_ref_local_repeat(rank, r, c, K_SHARD, nranks);
                const float diff = std::fabs(got - ref);
                if (diff > 0.1f) {
                    if (mism < 8) {
                        printf("[rank %d] mismatch row=%d col=%d got=%f ref=%f diff=%f\n",
                               rank, r, c, got, ref, diff);
                    }
                    ++mism;
                }
            }
        }
    }

    int total_mism = 0;
    MPI_Reduce(&mism, &total_mism, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const double flops = 2.0 * double(M) * double(N) * double(K);
        if (mode == 3) {
            printf("a2a_gemm_split M=%d N=%d K=%d ranks=%d input_mode=%s comm_wgs=%d comm_ms=%.4f compute_ms=%.4f comm_plus_compute=%.4f split_total_ms=%.4f %.2f TFLOP/s %s\n",
                   M, N, K, nranks, input_mode_name, comm_wgs,
                   max_comm_ms, max_compute_ms, max_comm_ms + max_compute_ms, max_ms,
                   flops / (max_ms * 1.0e9),
                   (!validate || total_mism == 0) ? "SUCCESS" : "FAILED");
        } else {
            printf("a2a_gemm_lsa M=%d N=%d K=%d ranks=%d mode=%d input_mode=%s persistent=%d comm_wgs=%d compute_wgs=%d grid_wgs=%d max_rank_time=%.4f ms %.2f TFLOP/s %s\n",
                   M, N, K, nranks, mode, input_mode_name, persistent, comm_wgs,
                   persistent ? compute_wgs : compute_tasks, grid_wgs,
                   max_ms, flops / (max_ms * 1.0e9),
                   (!validate || total_mism == 0) ? "SUCCESS" : "FAILED");
        }
    }

    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(comm_stop));
    CHECK_HIP(hipEventDestroy(compute_start));
    CHECK_HIP(hipEventDestroy(stop));
    CHECK_CCO(ccoWindowDeregister(comm, recv_win));
    CHECK_CCO(ccoWindowDeregister(comm, ready_win));
    if (nccl_comm) CHECK_NCCL(ncclCommDestroy(nccl_comm));
    CHECK_CCO(ccoCommDestroy(comm));
    CHECK_HIP(hipFree(d_a));
    CHECK_HIP(hipFree(d_a_full));
    CHECK_HIP(hipFree(d_a_recv));
    CHECK_HIP(hipFree(d_b));
    CHECK_HIP(hipFree(d_c));
    CHECK_HIP(hipFree(d_wg_hw_records));
    CHECK_HIP(hipFree(d_tile_counter));
    MPI_Finalize();
    return (!validate || total_mism == 0) ? 0 : 1;
}
