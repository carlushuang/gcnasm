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

template<typename Traits, int Mode, bool Persistent,
         typename Kargs = opus_a2a_gemm_kargs>
__global__ void a2a_gemm_lsa_kernel(Kargs kargs);
template<typename Traits>
__global__ void a2a_lsa_comm_kernel(opus_a2a_gemm_kargs kargs);
template<typename Traits,
         bool LocalStaging = false,
         bool ChunkFused = false,
         bool DirectStriped = false>
__global__ void gemm_a16w16_quad_subtile_kernel(opus_gemm_kargs kargs);
__global__ void a2a_sdma_post_kernel(
    ccoWindow_t, ccoWindow_t, ccoDevComm,
    size_t, size_t, size_t, int);
__global__ void a2a_sdma_quiet_notify_kernel(
    ccoWindow_t, ccoDevComm, size_t);
__global__ void a2a_sdma_wait_ready_kernel(
    const uint64_t*, size_t, uint64_t, int, int);

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

enum class SdmaSchedule {
    Serial,
    Parallel,
    Intra,
    Fused,
};

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

static void fill_a_epoch_one(
    bf16_t* a, int rank, int rank_count, int m, int k_shard, int input_mode) {
    const int chunks = input_mode == OPUS_A2A_INPUT_GENERIC ? rank_count : 1;
#pragma omp parallel for collapse(3)
    for (int dst = 0; dst < chunks; ++dst) {
        for (int r = 0; r < m; ++r) {
            for (int k = 0; k < k_shard; ++k) {
                a[(static_cast<size_t>(dst) * m + r) * k_shard + k] =
                    static_cast<bf16_t>(1.5f * a_value(rank, dst, r, k));
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

static float sample_ref(
    int dst_rank, int row, int col, int k_shard, int ranks, int input_mode,
    int epoch_variant = 0) {
    const int a_dst = input_mode == OPUS_A2A_INPUT_GENERIC ? dst_rank : 0;
    const int b_dst = input_mode == OPUS_A2A_INPUT_GENERIC ? dst_rank : 0;
    float acc = 0.0f;
    for (int src = 0; src < ranks; ++src) {
        for (int kk = 0; kk < k_shard; ++kk) {
            const int global_k = src * k_shard + kk;
            const float source_a =
                (epoch_variant == 0 ? 1.0f : 1.5f) *
                a_value(src, a_dst, row, kk);
            const float av =
                static_cast<float>(static_cast<bf16_t>(source_a));
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
    bool comm_backend_sdma = false;
    SdmaSchedule comm_schedule = SdmaSchedule::Parallel;
    bool record_wg_hw = false;
    bool strict_timing = true;
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
        else if (std::strcmp(argv[i], "--comm-schedule") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            if (std::strcmp(value, "parallel") == 0 ||
                std::strcmp(value, "auto") == 0) {
                comm_schedule = SdmaSchedule::Parallel;
            } else if (std::strcmp(value, "serial") == 0) {
                comm_schedule = SdmaSchedule::Serial;
            } else if (std::strcmp(value, "intra") == 0) {
                comm_schedule = SdmaSchedule::Intra;
            } else if (std::strcmp(value, "fused") == 0) {
                comm_schedule = SdmaSchedule::Fused;
            }
            else {
                if (rank == 0) fprintf(stderr, "--comm-schedule must be serial, parallel, intra, fused, or auto\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        else if (std::strcmp(argv[i], "--comm-backend") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            if (std::strcmp(value, "lsa") == 0) comm_backend_sdma = false;
            else if (std::strcmp(value, "sdma") == 0) comm_backend_sdma = true;
            else {
                if (rank == 0) fprintf(stderr, "--comm-backend must be lsa or sdma\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        else if (std::strcmp(argv[i], "--record-wg-hw") == 0) record_wg_hw = true;
        else if (std::strcmp(argv[i], "--strict-timing") == 0 &&
                 i + 1 < argc) {
            const char* value = argv[++i];
            if (std::strcmp(value, "0") == 0) strict_timing = false;
            else if (std::strcmp(value, "1") == 0) strict_timing = true;
            else {
                if (rank == 0) fprintf(stderr, "--strict-timing must be 0 or 1\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
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
    if (comm_backend_sdma) {
        if (mode != 0 && mode != 4) {
            if (rank == 0) fprintf(stderr, "--comm-backend sdma cannot be combined with --mode %d\n", mode);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        mode = 4;
    }
    if (mode < 0 || mode > 4) {
        if (rank == 0) fprintf(stderr, "unsupported mode: 0=fused, 1=compute-only, 2=RCCL, 3=split LSA, 4=CCO SDMA pipeline\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if ((comm_schedule == SdmaSchedule::Intra ||
         comm_schedule == SdmaSchedule::Fused) &&
        mode != 4) {
        if (rank == 0) fprintf(stderr, "--comm-schedule intra/fused requires --comm-backend sdma or --mode 4\n");
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
    if (warmup < 0 || iters <= 0) {
        if (rank == 0) fprintf(stderr, "--warmup must be non-negative and --iters must be positive\n");
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
    const char* sdma_schedule_name =
        comm_schedule == SdmaSchedule::Serial
            ? "serial"
            : (comm_schedule == SdmaSchedule::Intra
                   ? "intra"
                   : (comm_schedule == SdmaSchedule::Fused
                          ? "fused"
                          : "parallel"));
    if (mode == 4) {
        const char* value = std::getenv("MORI_ENABLE_SDMA");
        if (!value || std::strcmp(value, "1") != 0) {
            if (rank == 0) fprintf(stderr, "mode 4 requires MORI_ENABLE_SDMA=1\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

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
    const size_t ready_elems = static_cast<size_t>(nranks);
    static constexpr int kComputeNWorkers = 28;
    const bool n_outer_mode = mode == 1;
    const int compute_tasks = n_outer_mode ? (num_m_tiles * kComputeNWorkers)
                                           : (num_m_tiles * num_n_tiles);
    int compute_wgs = compute_tasks;
    if (persistent) {
        const int auto_compute_wgs = cu_count - comm_wgs;
        compute_wgs = compute_wgs_arg > 0 ? compute_wgs_arg : auto_compute_wgs;
        if (compute_wgs > compute_tasks) compute_wgs = compute_tasks;
        if (compute_wgs <= 0) {
            if (rank == 0) {
                fprintf(stderr,
                        "persistent compute WG count %d is invalid for %d communication WGs\n",
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
    std::unique_ptr<bf16_t[]> h_a_epoch_one;
    auto h_b = std::make_unique<bf16_t[]>(b_elems);
    fill_a(h_a.get(), rank, nranks, M, K_SHARD, input_mode);
    if (mode == 4) {
        h_a_epoch_one = std::make_unique<bf16_t[]>(local_a_elems);
        fill_a_epoch_one(
            h_a_epoch_one.get(), rank, nranks, M, K_SHARD, input_mode);
    }
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

    ccoWindow_t sdma_send_win = nullptr, sdma_recv_win = nullptr, sdma_ready_win = nullptr;
    void* sdma_send_local = nullptr;
    void* sdma_recv_local = nullptr;
    void* sdma_ready_local = nullptr;
    ccoDevComm sdma_dev_comm{};
    ccoDevComm* sdma_fused_dev_comm = nullptr;
    opus_a2a_gemm_sdma_fused_comm_state* sdma_fused_states = nullptr;
    bool sdma_dev_comm_created = false;
    const bool sdma_shared_stream =
        mode == 4 &&
        (comm_schedule == SdmaSchedule::Serial ||
         comm_schedule == SdmaSchedule::Fused);
    hipStream_t sdma_comm_stream = nullptr, sdma_compute_stream = nullptr;
    hipEvent_t sdma_stage_ready[2] = {nullptr, nullptr};
    hipEvent_t sdma_comm_ready[2] = {nullptr, nullptr};
    hipEvent_t sdma_recv_free[2] = {nullptr, nullptr};
    if (mode == 4) {
        CHECK_CCO(ccoWindowRegister(
            comm, 2 * local_a_elems * sizeof(bf16_t),
            &sdma_send_win, &sdma_send_local));
        CHECK_CCO(ccoWindowRegister(
            comm, 2 * recv_elems * sizeof(bf16_t),
            &sdma_recv_win, &sdma_recv_local));
        CHECK_CCO(ccoWindowRegister(
            comm, 2 * static_cast<size_t>(nranks) * sizeof(uint64_t),
            &sdma_ready_win, &sdma_ready_local));
        CHECK_HIP(hipMemcpy(
            sdma_send_local, d_a, local_a_elems * sizeof(bf16_t),
            hipMemcpyDeviceToDevice));
        CHECK_HIP(hipMemcpy(
            static_cast<char*>(sdma_send_local) +
                local_a_elems * sizeof(bf16_t),
            h_a_epoch_one.get(), local_a_elems * sizeof(bf16_t),
            hipMemcpyHostToDevice));
        CHECK_HIP(hipMemset(
            sdma_recv_local, 0, 2 * recv_elems * sizeof(bf16_t)));
        CHECK_HIP(hipMemset(
            sdma_ready_local, 0,
            2 * static_cast<size_t>(nranks) * sizeof(uint64_t)));
        ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
        reqs.gdaConnectionType = CCO_GDA_CONNECTION_NONE;
        reqs.gdaSignalCount = 0;
        reqs.gdaCounterCount = 0;
        reqs.sdmaQueueCount = 1;
        CHECK_CCO(ccoDevCommCreate(comm, &reqs, &sdma_dev_comm));
        const hipError_t cco_hip_status = hipGetLastError();
        if (cco_hip_status != hipSuccess &&
            cco_hip_status != hipErrorPeerAccessAlreadyEnabled) {
            CHECK_HIP(cco_hip_status);
        }
        sdma_dev_comm_created = true;
        if (comm_schedule == SdmaSchedule::Fused) {
            sdma_fused_dev_comm = ccoDevCommCopyToDevice(&sdma_dev_comm);
        }
        CHECK_HIP(hipStreamCreateWithFlags(
            &sdma_comm_stream, hipStreamNonBlocking));
        if (sdma_shared_stream) {
            sdma_compute_stream = sdma_comm_stream;
        } else {
            CHECK_HIP(hipStreamCreateWithFlags(
                &sdma_compute_stream, hipStreamNonBlocking));
        }
        for (int slot = 0; slot < 2; ++slot) {
            CHECK_HIP(hipEventCreateWithFlags(
                &sdma_stage_ready[slot], hipEventDisableTiming));
            CHECK_HIP(hipEventCreateWithFlags(
                &sdma_comm_ready[slot], hipEventDisableTiming));
            CHECK_HIP(hipEventCreateWithFlags(
                &sdma_recv_free[slot], hipEventDisableTiming));
        }
    }

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
    const size_t sdma_bytes_per_peer = a_chunk_elems * sizeof(bf16_t);
    const size_t sdma_send_slot_bytes = local_a_elems * sizeof(bf16_t);
    const size_t sdma_recv_slot_bytes = recv_elems * sizeof(bf16_t);
    const size_t sdma_ready_slot_bytes = static_cast<size_t>(nranks) * sizeof(uint64_t);
    const dim3 sdma_block(static_cast<unsigned>(nranks * 64));
    uint64_t sdma_ready_target[2] = {0, 0};
    if (mode == 4 && comm_schedule == SdmaSchedule::Fused) {
        opus_a2a_gemm_sdma_fused_comm_state host_states[2]{};
        for (int slot = 0; slot < 2; ++slot) {
            host_states[slot].sdma_send_win = sdma_send_win;
            host_states[slot].sdma_recv_win = sdma_recv_win;
            host_states[slot].sdma_ready_win = sdma_ready_win;
            host_states[slot].sdma_dev_comm = sdma_fused_dev_comm;
            host_states[slot].sdma_send_slot_offset =
                static_cast<uint64_t>(slot) * sdma_send_slot_bytes;
            host_states[slot].sdma_recv_slot_offset =
                static_cast<uint64_t>(slot) * sdma_recv_slot_bytes;
            host_states[slot].sdma_ready_slot_offset =
                static_cast<uint64_t>(slot) * sdma_ready_slot_bytes;
            host_states[slot].sdma_bytes_per_peer = sdma_bytes_per_peer;
        }
        CHECK_HIP(hipMalloc(
            &sdma_fused_states,
            sizeof(host_states)));
        CHECK_HIP(hipMemcpy(
            sdma_fused_states,
            host_states,
            sizeof(host_states),
            hipMemcpyHostToDevice));
    }
    auto launch_sdma_post_and_self = [&](int slot) {
        const size_t send_slot_offset = static_cast<size_t>(slot) * sdma_send_slot_bytes;
        const size_t recv_slot_offset = static_cast<size_t>(slot) * sdma_recv_slot_bytes;
        a2a_sdma_post_kernel<<<1, sdma_block, 0, sdma_comm_stream>>>(
            sdma_send_win, sdma_recv_win, sdma_dev_comm,
            send_slot_offset, recv_slot_offset, sdma_bytes_per_peer, input_mode);
        CHECK_HIP(hipGetLastError());
        const size_t self_src_chunk =
            input_mode == OPUS_A2A_INPUT_GENERIC ? static_cast<size_t>(rank) : 0;
        CHECK_HIP(hipMemcpyAsync(
            static_cast<char*>(sdma_recv_local) + recv_slot_offset +
                static_cast<size_t>(rank) * sdma_bytes_per_peer,
            static_cast<char*>(sdma_send_local) + send_slot_offset +
                self_src_chunk * sdma_bytes_per_peer,
            sdma_bytes_per_peer, hipMemcpyDeviceToDevice, sdma_comm_stream));
    };
    auto launch_sdma_quiet_notify = [&](int slot) {
        const size_t ready_slot_offset =
            static_cast<size_t>(slot) * sdma_ready_slot_bytes;
        a2a_sdma_quiet_notify_kernel<<<1, sdma_block, 0, sdma_comm_stream>>>(
            sdma_ready_win, sdma_dev_comm, ready_slot_offset);
        CHECK_HIP(hipGetLastError());
    };
    auto launch_sdma_comm = [&](int slot) {
        launch_sdma_post_and_self(slot);
        launch_sdma_quiet_notify(slot);
        const uint64_t ready_target = ++sdma_ready_target[slot];
        const size_t ready_slot_offset =
            static_cast<size_t>(slot) * sdma_ready_slot_bytes;
        a2a_sdma_wait_ready_kernel<<<1, 64, 0, sdma_comm_stream>>>(
            static_cast<const uint64_t*>(sdma_ready_local),
            ready_slot_offset, ready_target, nranks, rank);
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipEventRecord(sdma_comm_ready[slot], sdma_comm_stream));
    };
    auto launch_sdma_intra_comm = [&](int slot) {
        launch_sdma_post_and_self(slot);
        ++sdma_ready_target[slot];
        CHECK_HIP(hipEventRecord(sdma_stage_ready[slot], sdma_comm_stream));
        launch_sdma_quiet_notify(slot);
    };
    auto wait_sdma_compute = [&](int slot) {
        if (!sdma_shared_stream) {
            CHECK_HIP(hipStreamWaitEvent(
                sdma_compute_stream, sdma_comm_ready[slot], 0));
        }
    };
    auto launch_sdma_compute_kernel = [&](int slot) {
        opus_a2a_gemm_kargs slot_args = kargs;
        slot_args.recv_a_local =
            static_cast<char*>(sdma_recv_local) +
            static_cast<size_t>(slot) * sdma_recv_slot_bytes;
        slot_args.recv_a_win = sdma_recv_win;
        slot_args.recv_a_bytes = static_cast<unsigned int>(sdma_recv_slot_bytes);
        a2a_gemm_lsa_kernel<Traits, 2, false>
            <<<gemm_grid, block, 0, sdma_compute_stream>>>(slot_args);
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipEventRecord(sdma_recv_free[slot], sdma_compute_stream));
    };
    auto launch_sdma_compute = [&](int slot) {
        wait_sdma_compute(slot);
        launch_sdma_compute_kernel(slot);
    };
    auto wait_sdma_intra_compute = [&](int slot) {
        CHECK_HIP(hipStreamWaitEvent(
            sdma_compute_stream, sdma_stage_ready[slot], 0));
    };
    auto launch_sdma_intra_compute_kernel = [&](int slot) {
        opus_a2a_gemm_kargs slot_args = kargs;
        slot_args.recv_a_local =
            static_cast<char*>(sdma_recv_local) +
            static_cast<size_t>(slot) * sdma_recv_slot_bytes;
        slot_args.recv_a_win = sdma_recv_win;
        slot_args.recv_a_bytes = static_cast<unsigned int>(sdma_recv_slot_bytes);
        slot_args.sdma_ready_local =
            static_cast<const uint64_t*>(sdma_ready_local) +
            static_cast<size_t>(slot) * nranks;
        slot_args.sdma_ready_target = sdma_ready_target[slot];
        a2a_gemm_lsa_kernel<Traits, 3, false>
            <<<gemm_grid, block, 0, sdma_compute_stream>>>(slot_args);
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipEventRecord(sdma_recv_free[slot], sdma_compute_stream));
    };
    auto launch_sdma_intra_compute = [&](int slot) {
        wait_sdma_intra_compute(slot);
        launch_sdma_intra_compute_kernel(slot);
    };
    const bool fused_persistent = false;
    const int fused_compute_wgs = compute_tasks;
    const unsigned int fused_counter_start =
        static_cast<unsigned int>(fused_compute_wgs);
    auto prepare_literal_fused = [&](int slot) {
        const size_t send_slot_offset =
            static_cast<size_t>(slot) * sdma_send_slot_bytes;
        const size_t recv_slot_offset =
            static_cast<size_t>(slot) * sdma_recv_slot_bytes;
        const size_t self_src_chunk =
            input_mode == OPUS_A2A_INPUT_GENERIC
                ? static_cast<size_t>(rank)
                : 0;
        CHECK_HIP(hipMemcpyAsync(
            static_cast<char*>(sdma_recv_local) + recv_slot_offset +
                static_cast<size_t>(rank) * sdma_bytes_per_peer,
            static_cast<char*>(sdma_send_local) + send_slot_offset +
                self_src_chunk * sdma_bytes_per_peer,
            sdma_bytes_per_peer,
            hipMemcpyDeviceToDevice,
            sdma_compute_stream));
        if (fused_persistent) {
            CHECK_HIP(hipMemcpyAsync(
                d_tile_counter,
                &fused_counter_start,
                sizeof(fused_counter_start),
                hipMemcpyHostToDevice,
                sdma_compute_stream));
        }

        opus_a2a_gemm_sdma_fused_kargs fused_args{};
        static_cast<opus_a2a_gemm_kargs&>(fused_args) = kargs;
        fused_args.recv_a_local =
            static_cast<char*>(sdma_recv_local) + recv_slot_offset;
        fused_args.recv_a_win = sdma_recv_win;
        fused_args.recv_a_bytes =
            static_cast<unsigned int>(sdma_recv_slot_bytes);
        fused_args.sdma_ready_local =
            static_cast<const uint64_t*>(sdma_ready_local) +
            static_cast<size_t>(slot) * nranks;
        fused_args.sdma_ready_target = ++sdma_ready_target[slot];
        fused_args.sdma_comm_state = sdma_fused_states + slot;
        return fused_args;
    };
    auto launch_literal_fused =
        [&](opus_a2a_gemm_sdma_fused_kargs& fused_args) {
            if (fused_persistent) {
                void* kernel_args[] = {&fused_args};
                CHECK_HIP(hipLaunchCooperativeKernel(
                    reinterpret_cast<const void*>(
                        a2a_gemm_lsa_kernel<
                            Traits, 4, true,
                            opus_a2a_gemm_sdma_fused_kargs>),
                    dim3(fused_compute_wgs), block,
                    kernel_args, 0, sdma_compute_stream));
            } else {
                a2a_gemm_lsa_kernel<
                    Traits, 4, false,
                    opus_a2a_gemm_sdma_fused_kargs>
                    <<<dim3(fused_compute_wgs), block, 0, sdma_compute_stream>>>(
                        fused_args);
                CHECK_HIP(hipGetLastError());
            }
        };

    int mism = 0;
    auto validate_output_samples = [&](int epoch_variant, const char* label) {
        auto h_c = std::make_unique<bf16_t[]>(c_elems);
        CHECK_HIP(hipMemcpy(
            h_c.get(), d_c, c_elems * sizeof(bf16_t),
            hipMemcpyDeviceToHost));
        const int sample_rows[] = {0, 17, 255, 511, M / 2, M - 1};
        const int sample_cols[] = {0, 127, 255, 1024, 4096, 8191};
        for (int r : sample_rows) {
            if (r >= M) continue;
            for (int c : sample_cols) {
                if (c >= N) continue;
                const float got =
                    static_cast<float>(h_c[static_cast<size_t>(r) * N + c]);
                float ref = sample_ref(
                    rank, r, c, K_SHARD, nranks, input_mode, epoch_variant);
                if (mode == 1) {
                    ref = sample_ref_local_repeat(
                        rank, r, c, K_SHARD, nranks);
                }
                const float diff = std::fabs(got - ref);
                if (diff > 0.1f) {
                    if (mism < 8) {
                        printf("[%s rank %d] mismatch row=%d col=%d got=%f ref=%f diff=%f\n",
                               label, rank, r, c, got, ref, diff);
                    }
                    ++mism;
                }
            }
        }
    };
    auto validate_sdma_ready_counts =
        [&](int pipeline_epochs, int common_epochs, int fused_extra_epochs,
            const char* label) {
            auto h_ready = std::make_unique<uint64_t[]>(
                2 * static_cast<size_t>(nranks));
            CHECK_HIP(hipMemcpy(
                h_ready.get(), sdma_ready_local,
                2 * static_cast<size_t>(nranks) * sizeof(uint64_t),
                hipMemcpyDeviceToHost));
            for (int slot = 0; slot < 2; ++slot) {
                const uint64_t expected =
                    static_cast<uint64_t>(
                        (pipeline_epochs + (slot == 0 ? 1 : 0)) / 2) +
                    static_cast<uint64_t>(
                        (common_epochs + (slot == 0 ? 1 : 0)) / 2) +
                    static_cast<uint64_t>(
                        (fused_extra_epochs + (slot == 0 ? 1 : 0)) / 2);
                for (int source = 0; source < nranks; ++source) {
                    const uint64_t got =
                        h_ready[static_cast<size_t>(slot) * nranks + source];
                    const uint64_t source_expected =
                        source == rank ? 0 : expected;
                    if (got != source_expected) {
                        if (mism < 8) {
                            printf("[%s rank %d] ready mismatch slot=%d source=%d got=%llu expected=%llu\n",
                                   label, rank, slot, source,
                                   static_cast<unsigned long long>(got),
                                   static_cast<unsigned long long>(source_expected));
                        }
                        ++mism;
                    }
                }
            }
        };

    hipEvent_t start, comm_stop, compute_start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&comm_stop));
    CHECK_HIP(hipEventCreate(&compute_start));
    CHECK_HIP(hipEventCreate(&stop));
    float total_ms = 0.0f, comm_total_ms = 0.0f, compute_total_ms = 0.0f;
    float barrier_idle_total_ms = 0.0f;
    float fused_kernel_total_ms = 0.0f;
    float strict_total_ms = 0.0f;
    float strict_comm_total_ms = 0.0f;
    float strict_compute_total_ms = 0.0f;
    float strict_barrier_idle_total_ms = 0.0f;
    float strict_overlap_total_ms = 0.0f;
    if (mode == 4) {
        for (int i = 0; i < warmup; ++i) {
            const int slot = i & 1;
            if (!sdma_shared_stream && i >= 2) {
                CHECK_HIP(hipStreamWaitEvent(
                    sdma_comm_stream, sdma_recv_free[slot], 0));
            }
            launch_sdma_comm(slot);
            launch_sdma_compute(slot);
        }
        CHECK_HIP(hipStreamSynchronize(sdma_comm_stream));
        if (!sdma_shared_stream) {
            CHECK_HIP(hipStreamSynchronize(sdma_compute_stream));
        }
        CHECK_CCO(ccoBarrierAll(comm));
        CHECK_HIP(hipEventRecord(start, sdma_comm_stream));
        for (int i = 0; i < iters; ++i) {
            launch_sdma_comm(i & 1);
        }
        CHECK_HIP(hipEventRecord(comm_stop, sdma_comm_stream));
        CHECK_HIP(hipEventSynchronize(comm_stop));
        CHECK_HIP(hipEventElapsedTime(
            &comm_total_ms, start, comm_stop));

        CHECK_CCO(ccoBarrierAll(comm));
        CHECK_HIP(hipEventRecord(compute_start, sdma_compute_stream));
        for (int i = 0; i < iters; ++i) {
            launch_sdma_compute(i & 1);
        }
        CHECK_HIP(hipEventRecord(stop, sdma_compute_stream));
        CHECK_HIP(hipEventSynchronize(stop));
        CHECK_HIP(hipEventElapsedTime(
            &compute_total_ms, compute_start, stop));
        CHECK_CCO(ccoBarrierAll(comm));
    }
    if (mode == 4 && comm_schedule == SdmaSchedule::Fused) {
        uint64_t epoch = 0;
        for (int i = 0; i < warmup; ++i) {
            auto fused_args = prepare_literal_fused(static_cast<int>(epoch & 1));
            launch_literal_fused(fused_args);
            ++epoch;
        }
        CHECK_HIP(hipStreamSynchronize(sdma_compute_stream));
        CHECK_CCO(ccoBarrierAll(comm));

        if (!strict_timing) {
            CHECK_HIP(hipEventRecord(start, sdma_compute_stream));
            for (int i = 0; i < iters; ++i) {
                auto fused_args =
                    prepare_literal_fused(static_cast<int>(epoch & 1));
                launch_literal_fused(fused_args);
                ++epoch;
            }
            CHECK_HIP(hipEventRecord(stop, sdma_compute_stream));
            CHECK_HIP(hipEventSynchronize(stop));
            CHECK_HIP(hipEventElapsedTime(&total_ms, start, stop));
        } else {
            auto phase_events =
                std::make_unique<hipEvent_t[]>(4 * static_cast<size_t>(iters));
            for (int i = 0; i < 4 * iters; ++i) {
                CHECK_HIP(hipEventCreate(&phase_events[i]));
            }
            for (int i = 0; i < iters; ++i) {
                const int event_base = 4 * i;
                CHECK_HIP(hipEventRecord(
                    phase_events[event_base], sdma_compute_stream));
                auto fused_args =
                    prepare_literal_fused(static_cast<int>(epoch & 1));
                CHECK_HIP(hipEventRecord(
                    phase_events[event_base + 1], sdma_compute_stream));
                CHECK_HIP(hipEventRecord(
                    phase_events[event_base + 2], sdma_compute_stream));
                launch_literal_fused(fused_args);
                CHECK_HIP(hipEventRecord(
                    phase_events[event_base + 3], sdma_compute_stream));
                ++epoch;
            }
            CHECK_HIP(hipEventSynchronize(phase_events[4 * iters - 1]));
            CHECK_HIP(hipEventElapsedTime(
                &total_ms, phase_events[0], phase_events[4 * iters - 1]));
            for (int i = 0; i < iters; ++i) {
                const int event_base = 4 * i;
                float self_copy_ms = 0.0f, fused_compute_ms = 0.0f;
                CHECK_HIP(hipEventElapsedTime(
                    &self_copy_ms,
                    phase_events[event_base], phase_events[event_base + 1]));
                CHECK_HIP(hipEventElapsedTime(
                    &fused_compute_ms,
                    phase_events[event_base + 2],
                    phase_events[event_base + 3]));
                strict_comm_total_ms += self_copy_ms;
                strict_compute_total_ms += fused_compute_ms;
            }
            strict_total_ms = total_ms;
            strict_barrier_idle_total_ms =
                strict_total_ms - strict_comm_total_ms -
                strict_compute_total_ms;
            for (int i = 0; i < 4 * iters; ++i) {
                CHECK_HIP(hipEventDestroy(phase_events[i]));
            }
        }
        if (validate) {
            validate_output_samples(
                (warmup + iters - 1) & 1, "literal_fused");
            validate_sdma_ready_counts(
                warmup + iters, warmup + iters, 0, "literal_fused");
        }

        for (int i = 0; i < iters; ++i) {
            const int slot = i & 1;
            auto fused_args = prepare_literal_fused(slot);
            CHECK_HIP(hipStreamSynchronize(sdma_compute_stream));
            CHECK_CCO(ccoBarrierAll(comm));
            CHECK_HIP(hipEventRecord(start, sdma_compute_stream));
            launch_literal_fused(fused_args);
            CHECK_HIP(hipEventRecord(stop, sdma_compute_stream));
            CHECK_HIP(hipEventSynchronize(stop));
            float kernel_ms = 0.0f;
            CHECK_HIP(hipEventElapsedTime(&kernel_ms, start, stop));
            fused_kernel_total_ms += kernel_ms;
            CHECK_CCO(ccoBarrierAll(comm));
        }
    } else if (mode == 4) {
        uint64_t epoch = 0;
        auto launch_sdma_epoch = [&]() {
            const int slot = static_cast<int>(epoch & 1);
            if (comm_schedule != SdmaSchedule::Serial && epoch >= 2) {
                CHECK_HIP(hipStreamWaitEvent(
                    sdma_comm_stream, sdma_recv_free[slot], 0));
            }
            if (comm_schedule == SdmaSchedule::Intra) {
                launch_sdma_intra_comm(slot);
                launch_sdma_intra_compute(slot);
            } else {
                launch_sdma_comm(slot);
                launch_sdma_compute(slot);
            }
            ++epoch;
        };
        for (int i = 0; i < warmup; ++i) launch_sdma_epoch();
        CHECK_HIP(hipStreamSynchronize(sdma_comm_stream));
        if (!sdma_shared_stream) {
            CHECK_HIP(hipStreamSynchronize(sdma_compute_stream));
        }
        CHECK_CCO(ccoBarrierAll(comm));
        if (!strict_timing) {
            CHECK_HIP(hipEventRecord(start, sdma_comm_stream));
            for (int i = 0; i < iters; ++i) launch_sdma_epoch();
            CHECK_HIP(hipEventRecord(stop, sdma_compute_stream));
            CHECK_HIP(hipEventSynchronize(stop));
            CHECK_HIP(hipEventElapsedTime(&total_ms, start, stop));
        } else if (comm_schedule == SdmaSchedule::Serial) {
            auto phase_events =
                std::make_unique<hipEvent_t[]>(4 * static_cast<size_t>(iters));
            for (int i = 0; i < 4 * iters; ++i) {
                CHECK_HIP(hipEventCreate(&phase_events[i]));
            }
            comm_total_ms = 0.0f;
            compute_total_ms = 0.0f;
            for (int i = 0; i < iters; ++i) {
                const int event_base = 4 * i;
                const int slot = static_cast<int>(epoch & 1);
                CHECK_HIP(hipEventRecord(
                    phase_events[event_base], sdma_comm_stream));
                launch_sdma_comm(slot);
                CHECK_HIP(hipEventRecord(
                    phase_events[event_base + 1], sdma_comm_stream));
                CHECK_HIP(hipEventRecord(
                    phase_events[event_base + 2], sdma_compute_stream));
                launch_sdma_compute(slot);
                CHECK_HIP(hipEventRecord(
                    phase_events[event_base + 3], sdma_compute_stream));
                ++epoch;
            }
            CHECK_HIP(hipEventSynchronize(phase_events[4 * iters - 1]));
            CHECK_HIP(hipEventElapsedTime(
                &total_ms, phase_events[0], phase_events[4 * iters - 1]));
            for (int i = 0; i < iters; ++i) {
                const int event_base = 4 * i;
                float comm_ms = 0.0f, compute_ms = 0.0f;
                CHECK_HIP(hipEventElapsedTime(
                    &comm_ms,
                    phase_events[event_base], phase_events[event_base + 1]));
                CHECK_HIP(hipEventElapsedTime(
                    &compute_ms,
                    phase_events[event_base + 2],
                    phase_events[event_base + 3]));
                comm_total_ms += comm_ms;
                compute_total_ms += compute_ms;
            }
            barrier_idle_total_ms =
                total_ms - comm_total_ms - compute_total_ms;
            for (int i = 0; i < 4 * iters; ++i) {
                CHECK_HIP(hipEventDestroy(phase_events[i]));
            }
        } else {
            auto phase_events =
                std::make_unique<hipEvent_t[]>(4 * static_cast<size_t>(iters));
            for (int i = 0; i < 4 * iters; ++i) {
                CHECK_HIP(hipEventCreate(&phase_events[i]));
            }
            hipEvent_t strict_start, comm_done, compute_done, strict_stop;
            hipStream_t timing_stream;
            CHECK_HIP(hipEventCreate(&strict_start));
            CHECK_HIP(hipEventCreate(&comm_done));
            CHECK_HIP(hipEventCreate(&compute_done));
            CHECK_HIP(hipEventCreate(&strict_stop));
            CHECK_HIP(hipStreamCreateWithFlags(
                &timing_stream, hipStreamNonBlocking));
            CHECK_HIP(hipEventRecord(strict_start, sdma_comm_stream));
            CHECK_HIP(hipStreamWaitEvent(
                sdma_compute_stream, strict_start, 0));
            for (int i = 0; i < iters; ++i) {
                const int event_base = 4 * i;
                const int slot = static_cast<int>(epoch & 1);
                if (epoch >= 2) {
                    CHECK_HIP(hipStreamWaitEvent(
                        sdma_comm_stream, sdma_recv_free[slot], 0));
                }
                CHECK_HIP(hipEventRecord(
                    phase_events[event_base], sdma_comm_stream));
                if (comm_schedule == SdmaSchedule::Intra) {
                    launch_sdma_intra_comm(slot);
                } else {
                    launch_sdma_comm(slot);
                }
                CHECK_HIP(hipEventRecord(
                    phase_events[event_base + 1], sdma_comm_stream));
                if (comm_schedule == SdmaSchedule::Intra) {
                    wait_sdma_intra_compute(slot);
                } else {
                    wait_sdma_compute(slot);
                }
                CHECK_HIP(hipEventRecord(
                    phase_events[event_base + 2], sdma_compute_stream));
                if (comm_schedule == SdmaSchedule::Intra) {
                    launch_sdma_intra_compute_kernel(slot);
                } else {
                    launch_sdma_compute_kernel(slot);
                }
                CHECK_HIP(hipEventRecord(
                    phase_events[event_base + 3], sdma_compute_stream));
                ++epoch;
            }
            CHECK_HIP(hipEventRecord(comm_done, sdma_comm_stream));
            CHECK_HIP(hipEventRecord(compute_done, sdma_compute_stream));
            CHECK_HIP(hipStreamWaitEvent(timing_stream, comm_done, 0));
            CHECK_HIP(hipStreamWaitEvent(timing_stream, compute_done, 0));
            CHECK_HIP(hipEventRecord(strict_stop, timing_stream));
            CHECK_HIP(hipEventSynchronize(strict_stop));
            CHECK_HIP(hipEventElapsedTime(
                &strict_total_ms, strict_start, strict_stop));

            auto comm_intervals =
                std::make_unique<float[]>(2 * static_cast<size_t>(iters));
            auto compute_intervals =
                std::make_unique<float[]>(2 * static_cast<size_t>(iters));
            for (int i = 0; i < iters; ++i) {
                const int event_base = 4 * i;
                float comm_start_ms = 0.0f, comm_end_ms = 0.0f;
                float compute_start_ms = 0.0f, compute_end_ms = 0.0f;
                CHECK_HIP(hipEventElapsedTime(
                    &comm_start_ms, strict_start, phase_events[event_base]));
                CHECK_HIP(hipEventElapsedTime(
                    &comm_end_ms, strict_start, phase_events[event_base + 1]));
                CHECK_HIP(hipEventElapsedTime(
                    &compute_start_ms, strict_start, phase_events[event_base + 2]));
                CHECK_HIP(hipEventElapsedTime(
                    &compute_end_ms, strict_start, phase_events[event_base + 3]));
                comm_intervals[2 * i] = comm_start_ms;
                comm_intervals[2 * i + 1] = comm_end_ms;
                compute_intervals[2 * i] = compute_start_ms;
                compute_intervals[2 * i + 1] = compute_end_ms;
                strict_comm_total_ms += comm_end_ms - comm_start_ms;
                strict_compute_total_ms += compute_end_ms - compute_start_ms;
            }
            for (int comm_index = 0; comm_index < iters; ++comm_index) {
                for (int compute_index = 0; compute_index < iters;
                     ++compute_index) {
                    const float overlap_begin =
                        comm_intervals[2 * comm_index] >
                                compute_intervals[2 * compute_index]
                            ? comm_intervals[2 * comm_index]
                            : compute_intervals[2 * compute_index];
                    const float overlap_end =
                        comm_intervals[2 * comm_index + 1] <
                                compute_intervals[2 * compute_index + 1]
                            ? comm_intervals[2 * comm_index + 1]
                            : compute_intervals[2 * compute_index + 1];
                    if (overlap_end > overlap_begin) {
                        strict_overlap_total_ms += overlap_end - overlap_begin;
                    }
                }
            }
            strict_barrier_idle_total_ms =
                strict_total_ms -
                (strict_comm_total_ms + strict_compute_total_ms -
                 strict_overlap_total_ms);
            total_ms = strict_total_ms;
            for (int i = 0; i < 4 * iters; ++i) {
                CHECK_HIP(hipEventDestroy(phase_events[i]));
            }
            CHECK_HIP(hipEventDestroy(strict_start));
            CHECK_HIP(hipEventDestroy(comm_done));
            CHECK_HIP(hipEventDestroy(compute_done));
            CHECK_HIP(hipEventDestroy(strict_stop));
            CHECK_HIP(hipStreamDestroy(timing_stream));
        }
        if (validate && comm_schedule == SdmaSchedule::Intra) {
            validate_output_samples(
                (warmup + iters - 1) & 1, "intra_pipeline");
            validate_sdma_ready_counts(
                warmup + iters, warmup + iters, 0, "intra_pipeline");
        }
    } else if (mode == 3) {
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
            barrier_idle_total_ms += split_ms - comm_ms - compute_ms;
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
    if (mode == 3 ||
        (mode == 4 && comm_schedule == SdmaSchedule::Serial)) {
        strict_total_ms = total_ms;
        strict_comm_total_ms = comm_total_ms;
        strict_compute_total_ms = compute_total_ms;
        strict_barrier_idle_total_ms = barrier_idle_total_ms;
    }
    const double local_ms = static_cast<double>(total_ms) / iters;
    const double local_comm_ms = static_cast<double>(comm_total_ms) / iters;
    const double local_compute_ms = static_cast<double>(compute_total_ms) / iters;
    const double local_strict_total_ms =
        static_cast<double>(strict_total_ms) / iters;
    const double local_strict_comm_ms =
        static_cast<double>(strict_comm_total_ms) / iters;
    const double local_strict_compute_ms =
        static_cast<double>(strict_compute_total_ms) / iters;
    const double local_strict_barrier_idle_ms =
        static_cast<double>(strict_barrier_idle_total_ms) / iters;
    const double local_strict_overlap_ms =
        static_cast<double>(strict_overlap_total_ms) / iters;
    const double local_fused_kernel_ms =
        static_cast<double>(fused_kernel_total_ms) / iters;
    double max_ms = 0.0, max_comm_ms = 0.0, max_compute_ms = 0.0;
    double max_fused_kernel_ms = 0.0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_comm_ms, &max_comm_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_compute_ms, &max_compute_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(
        &local_fused_kernel_ms, &max_fused_kernel_ms,
        1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    const double local_strict_metrics[5] = {
        local_strict_total_ms,
        local_strict_comm_ms,
        local_strict_compute_ms,
        local_strict_barrier_idle_ms,
        local_strict_overlap_ms};
    std::unique_ptr<double[]> gathered_strict_metrics;
    if (rank == 0) {
        gathered_strict_metrics =
            std::make_unique<double[]>(static_cast<size_t>(nranks) * 5);
    }
    MPI_Gather(
        local_strict_metrics, 5, MPI_DOUBLE,
        rank == 0 ? gathered_strict_metrics.get() : nullptr,
        5, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int critical_rank = 0;
    double critical_total_ms = 0.0;
    double critical_comm_ms = 0.0;
    double critical_compute_ms = 0.0;
    double critical_barrier_idle_ms = 0.0;
    double critical_overlap_ms = 0.0;
    if (rank == 0) {
        for (int candidate = 0; candidate < nranks; ++candidate) {
            const double* metrics =
                gathered_strict_metrics.get() + static_cast<size_t>(candidate) * 5;
            if (candidate == 0 || metrics[0] > critical_total_ms) {
                critical_rank = candidate;
                critical_total_ms = metrics[0];
                critical_comm_ms = metrics[1];
                critical_compute_ms = metrics[2];
                critical_barrier_idle_ms = metrics[3];
                critical_overlap_ms = metrics[4];
            }
        }
    }

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

    if (validate) {
        if (mode == 4) {
            validate_sdma_ready_counts(
                warmup + iters, warmup + iters,
                comm_schedule == SdmaSchedule::Fused ? iters : 0,
                "final");
        }
        validate_output_samples(
            mode == 4 ? ((iters - 1) & 1) : 0, "final");
    }

    int total_mism = 0;
    MPI_Reduce(&mism, &total_mism, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const double flops = 2.0 * double(M) * double(N) * double(K);
        if (mode == 4 && comm_schedule == SdmaSchedule::Fused) {
            const double split_sum = max_comm_ms + max_compute_ms;
            const double fusion_win =
                split_sum > 0.0
                    ? 100.0 * (split_sum - max_fused_kernel_ms) / split_sum
                    : 0.0;
            const double critical_phase_sum =
                critical_comm_ms + critical_compute_ms +
                critical_barrier_idle_ms;
            if (!strict_timing) {
                printf("a2a_gemm_sdma M=%d N=%d K=%d ranks=%d input_mode=%s schedule=fused comm_ms=%.4f compute_ms=%.4f split_sum_ms=%.4f fused_kernel_ms=%.4f fused_e2e_ms=%.4f fusion_win=%.2f%% strict_timing=0 %.2f TFLOP/s %s\n",
                       M, N, K, nranks, input_mode_name,
                       max_comm_ms, max_compute_ms, split_sum,
                       max_fused_kernel_ms, max_ms, fusion_win,
                       flops / (max_ms * 1.0e9),
                       (!validate || total_mism == 0) ? "SUCCESS" : "FAILED");
            } else {
                printf("a2a_gemm_sdma M=%d N=%d K=%d ranks=%d input_mode=%s schedule=fused comm_ms=%.4f compute_ms=%.4f split_sum_ms=%.4f fused_kernel_ms=%.4f fused_e2e_ms=%.4f fusion_win=%.2f%% strict_timing=1 critical_rank=%d critical_self_copy_ms=%.4f critical_fused_kernel_ms=%.4f barrier_idle_ms=%.4f critical_phase_sum=%.4f critical_e2e_ms=%.4f %.2f TFLOP/s %s\n",
                       M, N, K, nranks, input_mode_name,
                       max_comm_ms, max_compute_ms, split_sum,
                       max_fused_kernel_ms, max_ms, fusion_win,
                       critical_rank, critical_comm_ms, critical_compute_ms,
                       critical_barrier_idle_ms, critical_phase_sum,
                       critical_total_ms,
                       flops / (max_ms * 1.0e9),
                       (!validate || total_mism == 0) ? "SUCCESS" : "FAILED");
            }
        } else if (mode == 4) {
            if (!strict_timing) {
                printf("a2a_gemm_sdma M=%d N=%d K=%d ranks=%d input_mode=%s schedule=%s comm_ms=%.4f compute_ms=%.4f comm_plus_compute=%.4f pipeline_total_ms=%.4f strict_timing=0 %.2f TFLOP/s %s\n",
                       M, N, K, nranks, input_mode_name,
                       sdma_schedule_name,
                       max_comm_ms, max_compute_ms,
                       max_comm_ms + max_compute_ms, max_ms,
                       flops / (max_ms * 1.0e9),
                       (!validate || total_mism == 0) ? "SUCCESS" : "FAILED");
            } else if (comm_schedule == SdmaSchedule::Serial) {
                const double critical_phase_sum =
                    critical_comm_ms + critical_compute_ms +
                    critical_barrier_idle_ms;
                printf("a2a_gemm_sdma M=%d N=%d K=%d ranks=%d input_mode=%s schedule=serial comm_ms=%.4f compute_ms=%.4f comm_plus_compute=%.4f pipeline_total_ms=%.4f critical_rank=%d critical_comm_ms=%.4f critical_compute_ms=%.4f overlap_ms=0.0000 exposed_comm_ms=%.4f fill_drain_idle_ms=%.4f critical_phase_sum=%.4f critical_e2e_ms=%.4f %.2f TFLOP/s %s\n",
                       M, N, K, nranks, input_mode_name,
                       max_comm_ms, max_compute_ms,
                       max_comm_ms + max_compute_ms, max_ms,
                       critical_rank, critical_comm_ms, critical_compute_ms,
                       critical_comm_ms, critical_barrier_idle_ms, critical_phase_sum,
                       critical_total_ms,
                       flops / (max_ms * 1.0e9),
                       (!validate || total_mism == 0) ? "SUCCESS" : "FAILED");
            } else {
                const double exposed_comm_ms =
                    critical_comm_ms - critical_overlap_ms;
                printf("a2a_gemm_sdma M=%d N=%d K=%d ranks=%d input_mode=%s schedule=%s comm_ms=%.4f compute_ms=%.4f comm_plus_compute=%.4f pipeline_total_ms=%.4f critical_rank=%d critical_comm_ms=%.4f critical_compute_ms=%.4f overlap_ms=%.4f exposed_comm_ms=%.4f fill_drain_idle_ms=%.4f critical_e2e_ms=%.4f %.2f TFLOP/s %s\n",
                       M, N, K, nranks, input_mode_name,
                       sdma_schedule_name,
                       max_comm_ms, max_compute_ms,
                       max_comm_ms + max_compute_ms, max_ms,
                       critical_rank, critical_comm_ms, critical_compute_ms,
                       critical_overlap_ms, exposed_comm_ms,
                       critical_barrier_idle_ms, critical_total_ms,
                       flops / (max_ms * 1.0e9),
                       (!validate || total_mism == 0) ? "SUCCESS" : "FAILED");
            }
        } else if (mode == 3) {
            const double critical_phase_sum =
                critical_comm_ms + critical_compute_ms +
                critical_barrier_idle_ms;
            printf("a2a_gemm_split M=%d N=%d K=%d ranks=%d input_mode=%s comm_wgs=%d comm_ms=%.4f compute_ms=%.4f comm_plus_compute=%.4f split_total_ms=%.4f critical_rank=%d critical_comm_ms=%.4f critical_compute_ms=%.4f barrier_idle_ms=%.4f critical_phase_sum=%.4f critical_e2e_ms=%.4f %.2f TFLOP/s %s\n",
                   M, N, K, nranks, input_mode_name, comm_wgs,
                   max_comm_ms, max_compute_ms, max_comm_ms + max_compute_ms, max_ms,
                   critical_rank, critical_comm_ms, critical_compute_ms,
                   critical_barrier_idle_ms, critical_phase_sum,
                   critical_total_ms,
                   flops / (max_ms * 1.0e9),
                   (!validate || total_mism == 0) ? "SUCCESS" : "FAILED");
        } else {
            printf("a2a_gemm_lsa M=%d N=%d K=%d ranks=%d mode=%d input_mode=%s schedule=%s persistent=%d comm_wgs=%d compute_wgs=%d grid_wgs=%d max_rank_time=%.4f ms %.2f TFLOP/s %s\n",
                   M, N, K, nranks, mode, input_mode_name,
                   mode == 4 ? sdma_schedule_name : "n/a",
                   persistent, comm_wgs,
                   persistent ? compute_wgs : compute_tasks, grid_wgs,
                   max_ms, flops / (max_ms * 1.0e9),
                   (!validate || total_mism == 0) ? "SUCCESS" : "FAILED");
        }
    }

    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(comm_stop));
    CHECK_HIP(hipEventDestroy(compute_start));
    CHECK_HIP(hipEventDestroy(stop));
    if (mode == 4) {
        for (int slot = 0; slot < 2; ++slot) {
            CHECK_HIP(hipEventDestroy(sdma_stage_ready[slot]));
            CHECK_HIP(hipEventDestroy(sdma_comm_ready[slot]));
            CHECK_HIP(hipEventDestroy(sdma_recv_free[slot]));
        }
        CHECK_HIP(hipStreamDestroy(sdma_comm_stream));
        if (!sdma_shared_stream) {
            CHECK_HIP(hipStreamDestroy(sdma_compute_stream));
        }
        if (sdma_fused_states) {
            CHECK_HIP(hipFree(sdma_fused_states));
        }
        if (sdma_fused_dev_comm) {
            ccoDevCommFreeDeviceCopy(sdma_fused_dev_comm);
        }
        if (sdma_dev_comm_created) {
            CHECK_CCO(ccoDevCommDestroy(comm, &sdma_dev_comm));
        }
        CHECK_CCO(ccoWindowDeregister(comm, sdma_ready_win));
        CHECK_CCO(ccoWindowDeregister(comm, sdma_recv_win));
        CHECK_CCO(ccoWindowDeregister(comm, sdma_send_win));
    }
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
