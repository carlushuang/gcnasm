#include <hip/hip_runtime.h>
#include <opus/hip_minimal.hpp>

#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include <mpi.h>
#include "mori/cco/cco.hpp"

#include "gemm_defs.h"

using namespace mori::cco;

#ifndef OPUS_PERSISTENT
#define OPUS_PERSISTENT 1
#endif

#if !defined(HIP_INCLUDE_HIP_HIP_RUNTIME_API_H)
extern "C" hipError_t hipGetDeviceCount(int* count);
extern "C" hipError_t hipSetDevice(int deviceId);
enum hipDeviceAttribute_t {
    hipDeviceAttributeMultiprocessorCount = 63,
};
extern "C" hipError_t hipDeviceGetAttribute(int* pi, hipDeviceAttribute_t attr, int deviceId);
#endif

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

template<typename Traits,
         bool LocalStaging,
         bool ChunkFused = false,
         bool DirectStriped = false>
__global__ void gemm_a16w16_quad_subtile_kernel(opus_gemm_kargs kargs);
__global__ void opus_sdma_a2a_post_kernel(
    ccoWindow_t, ccoWindow_t, ccoDevComm, size_t, size_t);
__global__ void opus_sdma_a2a_chunked_post_kernel(
    ccoWindow_t, ccoWindow_t, ccoDevComm, size_t, size_t, size_t);
__global__ void opus_sdma_a2a_quiet_notify_kernel(ccoWindow_t, ccoDevComm);
__global__ void opus_lsa_a2a_copy_kernel(
    const bf16_t*, void*, bf16_t*, size_t, int, int);

static constexpr size_t PER_RANK_VMM = 1024ULL * 1024 * 1024;
static constexpr int WAVE_SIZE = 64;

enum class OutputMode {
    Direct,
    SplitLsa,
    Local,
    Sdma,
    ChunkSdma,
};

enum class CommSchedule {
    Auto,
    Serial,
    Parallel,
};

static float a_value(int src_rank, int row, int k, int variant = 0) {
    const float base =
        0.001f * float(src_rank + 1) +
        0.01f * float((row % 17) - 8) +
        0.002f * float((k % 29) - 14);
    return variant ? 1.5f * base : base;
}

static float b_value(int col, int k) {
    return 0.003f * float((col % 23) - 11) + 0.001f * float((k % 31) - 15);
}

static void fill_a(bf16_t* a, int rank, int m, int k, int variant = 0) {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < m; ++i) {
        for (int kk = 0; kk < k; ++kk) {
            a[i * k + kk] =
                static_cast<bf16_t>(a_value(rank, i, kk, variant));
        }
    }
}

static void fill_b(bf16_t* b, int n, int k) {
#pragma omp parallel for collapse(2)
    for (int j = 0; j < n; ++j) {
        for (int kk = 0; kk < k; ++kk) {
            b[j * k + kk] = static_cast<bf16_t>(b_value(j, kk));
        }
    }
}

static float sample_ref(int src_rank, int row, int col, int k, int variant = 0) {
    float acc = 0.0f;
    for (int kk = 0; kk < k; ++kk) {
        const float av = static_cast<float>(
            static_cast<bf16_t>(a_value(src_rank, row, kk, variant)));
        const float bv = static_cast<float>(static_cast<bf16_t>(b_value(col, kk)));
        acc += av * bv;
    }
    return static_cast<float>(static_cast<bf16_t>(acc));
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    int M = 2048;
    int N = 18432;
    int K = 8192;
    int shard_n = 2560;
    int warmup = 3;
    int iters = 20;
    int chunk_m_tiles_per_put = 1;
    bool chunk_m_tiles_explicit = false;
    int sdma_post_m_tiles_per_put = 0;
    int fused_lsa_stripe_override = -1;
    bool strict_timing = true;
    OutputMode output_mode = OutputMode::Direct;
    CommSchedule comm_schedule = CommSchedule::Auto;
    for (int i = 1; i < argc; ++i) {
        if ((std::strcmp(argv[i], "-m") == 0 || std::strcmp(argv[i], "--m") == 0) && i + 1 < argc) M = std::atoi(argv[++i]);
        else if ((std::strcmp(argv[i], "-n") == 0 || std::strcmp(argv[i], "--n") == 0) && i + 1 < argc) N = std::atoi(argv[++i]);
        else if ((std::strcmp(argv[i], "-k") == 0 || std::strcmp(argv[i], "--k") == 0) && i + 1 < argc) K = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--shard-n") == 0 && i + 1 < argc) shard_n = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) warmup = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--chunk-m-tiles") == 0 && i + 1 < argc) {
            chunk_m_tiles_per_put = std::atoi(argv[++i]);
            chunk_m_tiles_explicit = true;
        }
        else if (std::strcmp(argv[i], "--sdma-post-m-tiles") == 0 &&
                 i + 1 < argc) {
            const char* value = argv[++i];
            sdma_post_m_tiles_per_put =
                std::strcmp(value, "auto") == 0 ? -1 : std::atoi(value);
        }
        else if (std::strcmp(argv[i], "--fused-lsa-stripe") == 0 &&
                 i + 1 < argc) {
            const char* value = argv[++i];
            if (std::strcmp(value, "auto") == 0) {
                fused_lsa_stripe_override = -1;
            } else if (std::strcmp(value, "0") == 0) {
                fused_lsa_stripe_override = 0;
            } else if (std::strcmp(value, "1") == 0) {
                fused_lsa_stripe_override = 1;
            } else {
                if (rank == 0) {
                    fprintf(stderr, "--fused-lsa-stripe must be auto, 0, or 1\n");
                }
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        else if (std::strcmp(argv[i], "--strict-timing") == 0 &&
                 i + 1 < argc) {
            const char* value = argv[++i];
            if (std::strcmp(value, "0") == 0) {
                strict_timing = false;
            } else if (std::strcmp(value, "1") == 0) {
                strict_timing = true;
            } else {
                if (rank == 0) {
                    fprintf(stderr, "--strict-timing must be 0 or 1\n");
                }
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        else if (std::strcmp(argv[i], "--output-mode") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            if (std::strcmp(value, "direct") == 0) output_mode = OutputMode::Direct;
            else if (std::strcmp(value, "split-lsa") == 0) output_mode = OutputMode::SplitLsa;
            else if (std::strcmp(value, "local") == 0) output_mode = OutputMode::Local;
            else if (std::strcmp(value, "sdma") == 0) output_mode = OutputMode::Sdma;
            else if (std::strcmp(value, "chunk-sdma") == 0) output_mode = OutputMode::ChunkSdma;
            else {
                if (rank == 0) fprintf(stderr, "--output-mode must be direct, split-lsa, local, sdma, or chunk-sdma\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        else if (std::strcmp(argv[i], "--comm-schedule") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            if (std::strcmp(value, "serial") == 0) comm_schedule = CommSchedule::Serial;
            else if (std::strcmp(value, "parallel") == 0) comm_schedule = CommSchedule::Parallel;
            else if (std::strcmp(value, "auto") == 0) comm_schedule = CommSchedule::Auto;
            else {
                if (rank == 0) fprintf(stderr, "--comm-schedule must be auto, serial, or parallel\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    if (nranks != 4 && nranks != 8) {
        if (rank == 0) fprintf(stderr, "requires exactly 4 or 8 ranks\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const bool local_staging = output_mode == OutputMode::Local;
    const bool split_lsa = output_mode == OutputMode::SplitLsa;
    const bool staging_compute = local_staging || split_lsa;
    const bool sdma_pipeline = output_mode == OutputMode::Sdma;
    const bool chunk_fused = output_mode == OutputMode::ChunkSdma;
    const bool uses_sdma = sdma_pipeline || chunk_fused;
    if (fused_lsa_stripe_override >= 0 && output_mode != OutputMode::Direct) {
        if (rank == 0) {
            fprintf(stderr, "--fused-lsa-stripe requires --output-mode direct\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const bool fused_lsa_stripe =
        output_mode == OutputMode::Direct &&
        (fused_lsa_stripe_override > 0 ||
         (fused_lsa_stripe_override < 0 && nranks >= 8 && M >= 8192));
    if (sdma_post_m_tiles_per_put != 0 && !sdma_pipeline) {
        if (rank == 0) {
            fprintf(stderr, "--sdma-post-m-tiles requires --output-mode sdma\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const bool overlap_epochs =
        comm_schedule == CommSchedule::Parallel ||
        (comm_schedule == CommSchedule::Auto && sdma_pipeline);
    if (uses_sdma) {
        const char* value = std::getenv("MORI_ENABLE_SDMA");
        if (!value || std::strcmp(value, "1") != 0) {
            if (rank == 0) fprintf(stderr, "SDMA mode requires MORI_ENABLE_SDMA=1 before launch\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    using Traits = opus_gemm_traits<512, 256, 256, 64, bf16_t, bf16_t, bf16_t, float>;
    const int scatter_n = shard_n * nranks;
    if (M % Traits::B_M != 0 || shard_n % Traits::B_N != 0 || scatter_n > N || K % Traits::B_K != 0 || ((K / Traits::B_K) % 2) != 0) {
        if (rank == 0) fprintf(stderr, "unsupported shape\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int ndev = 0;
    CHECK_HIP(hipGetDeviceCount(&ndev));
    CHECK_HIP(hipSetDevice(rank % ndev));

    ccoUniqueId uid;
    if (rank == 0) CHECK_CCO(ccoGetUniqueId(&uid));
    MPI_Bcast(&uid, sizeof(uid), MPI_BYTE, 0, MPI_COMM_WORLD);
    ccoComm* comm = nullptr;
    CHECK_CCO(ccoCommCreate(uid, nranks, rank, PER_RANK_VMM, &comm));

    const size_t a_elems = static_cast<size_t>(M) * K;
    const size_t b_elems = static_cast<size_t>(N) * K;
    const size_t recv_elems = static_cast<size_t>(nranks) * M * shard_n;
    const size_t staging_elems = static_cast<size_t>(nranks) * M * shard_n;
    const size_t local_c_elems = static_cast<size_t>(M) * N;

    auto h_a = std::make_unique<bf16_t[]>(a_elems);
    std::unique_ptr<bf16_t[]> h_a_alt;
    auto h_b = std::make_unique<bf16_t[]>(b_elems);
    fill_a(h_a.get(), rank, M, K);
    if (uses_sdma) {
        h_a_alt = std::make_unique<bf16_t[]>(a_elems);
        fill_a(h_a_alt.get(), rank, M, K, 1);
    }
    fill_b(h_b.get(), N, K);

    bf16_t* d_a = nullptr;
    bf16_t* d_a_alt = nullptr;
    bf16_t* d_b = nullptr;
    bf16_t* d_tail = nullptr;
    unsigned int* d_tile_counter = nullptr;
    CHECK_HIP(hipMalloc(&d_a, a_elems * sizeof(bf16_t)));
    if (uses_sdma) {
        CHECK_HIP(hipMalloc(&d_a_alt, a_elems * sizeof(bf16_t)));
    }
    CHECK_HIP(hipMalloc(&d_b, b_elems * sizeof(bf16_t)));
    CHECK_HIP(hipMalloc(&d_tail, local_c_elems * sizeof(bf16_t)));
#if OPUS_PERSISTENT
    CHECK_HIP(hipMalloc(&d_tile_counter, sizeof(unsigned int)));
#endif
    CHECK_HIP(hipMemcpy(d_a, h_a.get(), a_elems * sizeof(bf16_t), hipMemcpyHostToDevice));
    if (uses_sdma) {
        CHECK_HIP(hipMemcpy(
            d_a_alt, h_a_alt.get(), a_elems * sizeof(bf16_t),
            hipMemcpyHostToDevice));
    }
    CHECK_HIP(hipMemcpy(d_b, h_b.get(), b_elems * sizeof(bf16_t), hipMemcpyHostToDevice));

    ccoWindow_t win = nullptr;
    void* win_local = nullptr;
    CHECK_CCO(ccoWindowRegister(comm, recv_elems * sizeof(bf16_t), &win, &win_local));
    ccoWindow_t staging_win = nullptr;
    void* staging_local = nullptr;
    if (staging_compute) {
        CHECK_CCO(ccoWindowRegister(
            comm, staging_elems * sizeof(bf16_t), &staging_win, &staging_local));
    }
    ccoWindow_t sdma_ready_win = nullptr;
    void* sdma_ready_local = nullptr;
    if (uses_sdma) {
        CHECK_CCO(ccoWindowRegister(
            comm, static_cast<size_t>(nranks) * sizeof(uint64_t),
            &sdma_ready_win, &sdma_ready_local));
    }
    ccoWindow_t sdma_staging_win = nullptr, sdma_recv_win = nullptr;
    void* sdma_staging_local = nullptr;
    void* sdma_recv_local = nullptr;
    if (uses_sdma) {
        CHECK_CCO(ccoWindowRegister(
            comm, 2 * staging_elems * sizeof(bf16_t),
            &sdma_staging_win, &sdma_staging_local));
        CHECK_CCO(ccoWindowRegister(
            comm, recv_elems * sizeof(bf16_t),
            &sdma_recv_win, &sdma_recv_local));
    }

    ccoDevComm dev_comm{};
    bool dev_comm_created = false;
    bf16_t* sdma_staging[2] = {nullptr, nullptr};
    bf16_t* sdma_recv = nullptr;
    hipStream_t compute_stream = nullptr, comm_stream = nullptr;
    hipEvent_t stage_ready[2] = {nullptr, nullptr};
    hipEvent_t slot_free[2] = {nullptr, nullptr};
    ccoDevComm* chunk_dev_comm = nullptr;
    unsigned int* chunk_done = nullptr;
    unsigned int* chunk_peer_lock = nullptr;
    if (uses_sdma) {
        ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
        reqs.gdaConnectionType = CCO_GDA_CONNECTION_NONE;
        reqs.gdaSignalCount = 0;
        reqs.gdaCounterCount = 0;
        reqs.sdmaQueueCount = 1;
        CHECK_CCO(ccoDevCommCreate(comm, &reqs, &dev_comm));
        const hipError_t cco_hip_status = hipGetLastError();
        if (cco_hip_status != hipSuccess &&
            cco_hip_status != hipErrorPeerAccessAlreadyEnabled) {
            CHECK_HIP(cco_hip_status);
        }
        dev_comm_created = true;
        if (dev_comm.sdma.sdmaNumQueue == 0 || !dev_comm.sdma.deviceHandles) {
            if (rank == 0) fprintf(stderr, "MORI did not materialize SDMA queues\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        sdma_staging[0] = static_cast<bf16_t*>(sdma_staging_local);
        sdma_staging[1] = sdma_staging[0] + staging_elems;
        sdma_recv = static_cast<bf16_t*>(sdma_recv_local);

        CHECK_HIP(hipStreamCreateWithFlags(&compute_stream, hipStreamNonBlocking));
        CHECK_HIP(hipStreamCreateWithFlags(&comm_stream, hipStreamNonBlocking));
        for (int slot = 0; slot < 2; ++slot) {
            CHECK_HIP(hipEventCreateWithFlags(
                &stage_ready[slot], hipEventDisableTiming));
            CHECK_HIP(hipEventCreateWithFlags(
                &slot_free[slot], hipEventDisableTiming));
        }
    }

    opus_gemm_kargs kargs{};
    kargs.ptr_a = d_a;
    kargs.ptr_b = d_b;
    kargs.ptr_c = d_tail;        // non-scattered tail columns land here
    kargs.tile_counter = d_tile_counter;
    // Compile-time output backends interpret this as either a CCO window
    // handle (direct) or the raw local compact staging pointer (local).
    kargs.cco_c_win = staging_compute
        ? staging_local
        : (uses_sdma ? static_cast<void*>(sdma_staging[0])
                         : static_cast<void*>(win));
    kargs.peer_lsa_rank = rank;
    kargs.a2a_n_shard = shard_n;
    kargs.a2a_M = M;
    kargs.a2a_span = scatter_n;
    kargs.stride_c_full = N;
    kargs.m = M;
    kargs.n = N;
    kargs.k = K;
    kargs.batch = 1;
    kargs.stride_a = K;
    kargs.stride_b = K;
    kargs.stride_c = shard_n;
    kargs.stride_a_batch = M * K;
    kargs.stride_b_batch = N * K;
    kargs.stride_c_batch = M * N;

    const int num_tiles_m = ceil_div(M, Traits::B_M);
    const int num_tiles_n = ceil_div(N, Traits::B_N);
    const int total_tiles = num_tiles_m * num_tiles_n * kargs.batch;
    if (sdma_post_m_tiles_per_put < 0) {
        sdma_post_m_tiles_per_put = num_tiles_m < 8 ? num_tiles_m : 8;
        while (num_tiles_m % sdma_post_m_tiles_per_put != 0) {
            --sdma_post_m_tiles_per_put;
        }
    }
    if (sdma_post_m_tiles_per_put > 0 &&
        num_tiles_m % sdma_post_m_tiles_per_put != 0) {
        if (rank == 0) {
            fprintf(
                stderr,
                "--sdma-post-m-tiles must be positive and divide M/B_M\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const size_t sdma_post_chunk_bytes =
        sdma_post_m_tiles_per_put > 0
            ? static_cast<size_t>(Traits::B_M) *
                  sdma_post_m_tiles_per_put * shard_n * sizeof(bf16_t)
            : 0;
    if (chunk_fused) {
        if (!chunk_m_tiles_explicit) {
            chunk_m_tiles_per_put = num_tiles_m < 8 ? num_tiles_m : 8;
            while (num_tiles_m % chunk_m_tiles_per_put != 0) {
                --chunk_m_tiles_per_put;
            }
        }
        if (chunk_m_tiles_per_put <= 0 ||
            num_tiles_m % chunk_m_tiles_per_put != 0) {
            if (rank == 0) {
                fprintf(stderr, "--chunk-m-tiles must be positive and divide M/B_M\n");
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        const int chunk_groups = num_tiles_m / chunk_m_tiles_per_put;
        chunk_dev_comm = ccoDevCommCopyToDevice(&dev_comm);
        CHECK_HIP(hipMalloc(
            &chunk_done,
            static_cast<size_t>(nranks) * chunk_groups * sizeof(unsigned int)));
        CHECK_HIP(hipMalloc(
            &chunk_peer_lock, static_cast<size_t>(nranks) * sizeof(unsigned int)));
        kargs.chunk_dev_comm = chunk_dev_comm;
        kargs.chunk_staging_win = sdma_staging_win;
        kargs.chunk_recv_win = sdma_recv_win;
        kargs.chunk_done = chunk_done;
        kargs.chunk_peer_lock = chunk_peer_lock;
        kargs.chunk_tiles_per_peer = shard_n / Traits::B_N;
        kargs.chunk_num_m_tiles = chunk_groups;
        kargs.chunk_m_tiles_per_put = chunk_m_tiles_per_put;
    }
    int cu_count = 0;
    CHECK_HIP(hipDeviceGetAttribute(&cu_count, hipDeviceAttributeMultiprocessorCount, rank % ndev));
#if OPUS_PERSISTENT
    const int persistent_wgs = total_tiles < cu_count ? total_tiles : cu_count;
    dim3 grid(persistent_wgs, 1, 1);
#else
    dim3 grid(total_tiles, 1, 1);
#endif
    dim3 block(Traits::BLOCK_SIZE);
    const int lsa_copy_wgs = (cu_count / nranks) * nranks;
    dim3 lsa_copy_grid(lsa_copy_wgs > 0 ? lsa_copy_wgs : nranks);
    dim3 lsa_copy_block(256);

    auto clear_buffers = [&]() {
        CHECK_HIP(hipMemset(win_local, 0, recv_elems * sizeof(bf16_t)));
        if (staging_local) {
            CHECK_HIP(hipMemset(staging_local, 0, staging_elems * sizeof(bf16_t)));
        }
        if (uses_sdma) {
            CHECK_HIP(hipMemset(
                sdma_staging[0], 0, staging_elems * sizeof(bf16_t)));
            CHECK_HIP(hipMemset(
                sdma_staging[1], 0, staging_elems * sizeof(bf16_t)));
            CHECK_HIP(hipMemset(
                sdma_recv, 0, recv_elems * sizeof(bf16_t)));
            CHECK_HIP(hipMemset(
                sdma_ready_local, 0,
                static_cast<size_t>(nranks) * sizeof(uint64_t)));
        }
        CHECK_HIP(hipMemset(d_tail, 0, local_c_elems * sizeof(bf16_t)));
        CHECK_HIP(hipDeviceSynchronize());
        CHECK_CCO(ccoBarrierAll(comm));
    };

    auto launch = [&]() {
#if OPUS_PERSISTENT
        CHECK_HIP(hipMemset(d_tile_counter, 0, sizeof(unsigned int)));
#endif
        if (staging_compute) {
            gemm_a16w16_quad_subtile_kernel<Traits, true><<<grid, block>>>(kargs);
        } else if (fused_lsa_stripe) {
            gemm_a16w16_quad_subtile_kernel<Traits, false, false, true>
                <<<grid, block>>>(kargs);
        } else {
            gemm_a16w16_quad_subtile_kernel<Traits, false><<<grid, block>>>(kargs);
        }
        CHECK_HIP(hipGetLastError());
    };
    auto launch_split_lsa_comm = [&]() {
        opus_lsa_a2a_copy_kernel<<<lsa_copy_grid, lsa_copy_block>>>(
            static_cast<const bf16_t*>(staging_local),
            win,
            static_cast<bf16_t*>(win_local),
            static_cast<size_t>(M) * shard_n,
            nranks,
            rank);
        CHECK_HIP(hipGetLastError());
    };
    auto launch_split_lsa = [&]() {
        launch();
        launch_split_lsa_comm();
    };

    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));
    float total_ms = 0.0f;
    float strict_compute_total_ms = 0.0f;
    float strict_comm_total_ms = 0.0f;
    float strict_barrier_idle_total_ms = 0.0f;
    bool used_strict_measurement = false;
    auto measure_strict_phases =
        [&](hipStream_t stream, auto&& launch_compute_phase,
            auto&& launch_comm_phase) {
            auto phase_events =
                std::make_unique<hipEvent_t[]>(4 * static_cast<size_t>(iters));
            for (int i = 0; i < 4 * iters; ++i) {
                CHECK_HIP(hipEventCreate(&phase_events[i]));
            }
            for (int i = 0; i < iters; ++i) {
                const int event_base = 4 * i;
                CHECK_HIP(hipEventRecord(phase_events[event_base], stream));
                launch_compute_phase(i);
                CHECK_HIP(hipEventRecord(phase_events[event_base + 1], stream));
                CHECK_HIP(hipEventRecord(phase_events[event_base + 2], stream));
                launch_comm_phase(i);
                CHECK_HIP(hipEventRecord(phase_events[event_base + 3], stream));
            }
            CHECK_HIP(hipEventSynchronize(phase_events[4 * iters - 1]));
            CHECK_HIP(hipEventElapsedTime(
                &total_ms, phase_events[0], phase_events[4 * iters - 1]));
            for (int i = 0; i < iters; ++i) {
                const int event_base = 4 * i;
                float compute_ms = 0.0f, comm_ms = 0.0f;
                CHECK_HIP(hipEventElapsedTime(
                    &compute_ms,
                    phase_events[event_base], phase_events[event_base + 1]));
                CHECK_HIP(hipEventElapsedTime(
                    &comm_ms,
                    phase_events[event_base + 2], phase_events[event_base + 3]));
                strict_compute_total_ms += compute_ms;
                strict_comm_total_ms += comm_ms;
            }
            strict_barrier_idle_total_ms =
                total_ms - strict_compute_total_ms - strict_comm_total_ms;
            for (int i = 0; i < 4 * iters; ++i) {
                CHECK_HIP(hipEventDestroy(phase_events[i]));
            }
            used_strict_measurement = true;
        };
    clear_buffers();
    if (split_lsa) {
        for (int i = 0; i < warmup; ++i) launch_split_lsa();
        if (strict_timing) {
            measure_strict_phases(
                nullptr,
                [&](int) { launch(); },
                [&](int) { launch_split_lsa_comm(); });
        } else {
            CHECK_HIP(hipEventRecord(start));
            for (int i = 0; i < iters; ++i) launch_split_lsa();
            CHECK_HIP(hipEventRecord(stop));
        }
    } else if (!uses_sdma) {
        for (int i = 0; i < warmup; ++i) launch();
        if (output_mode == OutputMode::Direct && strict_timing) {
            measure_strict_phases(
                nullptr,
                [&](int) { launch(); },
                [&](int) {});
        } else {
            CHECK_HIP(hipEventRecord(start));
            for (int i = 0; i < iters; ++i) launch();
            CHECK_HIP(hipEventRecord(stop));
        }
    } else if (sdma_pipeline) {
        uint64_t pipeline_epoch = 0;
        const size_t bytes_per_peer =
            static_cast<size_t>(M) * shard_n * sizeof(bf16_t);
        const size_t elems_per_peer =
            static_cast<size_t>(M) * shard_n;
        const size_t staging_bytes = staging_elems * sizeof(bf16_t);
        const dim3 sdma_block(static_cast<unsigned>(nranks * WAVE_SIZE));
        hipStream_t transfer_stream =
            overlap_epochs ? comm_stream : compute_stream;

        auto launch_sdma_compute_phase = [&](int slot) {
            if (overlap_epochs && pipeline_epoch >= 2) {
                CHECK_HIP(hipStreamWaitEvent(
                    compute_stream, slot_free[slot], 0));
            }
#if OPUS_PERSISTENT
            CHECK_HIP(hipMemsetAsync(
                d_tile_counter, 0, sizeof(unsigned int), compute_stream));
#endif
            kargs.ptr_a = (pipeline_epoch & 1) ? d_a_alt : d_a;
            kargs.cco_c_win = sdma_staging[slot];
            gemm_a16w16_quad_subtile_kernel<Traits, true>
                <<<grid, block, 0, compute_stream>>>(kargs);
            CHECK_HIP(hipGetLastError());
            if (overlap_epochs) {
                CHECK_HIP(hipEventRecord(stage_ready[slot], compute_stream));
            }
        };
        auto launch_sdma_comm_phase = [&](int slot) {
            if (overlap_epochs) {
                CHECK_HIP(hipStreamWaitEvent(
                    transfer_stream, stage_ready[slot], 0));
            }
            if (sdma_post_chunk_bytes != 0) {
                opus_sdma_a2a_chunked_post_kernel
                    <<<1, sdma_block, 0, transfer_stream>>>(
                        sdma_staging_win, sdma_recv_win, dev_comm,
                        static_cast<size_t>(slot) * staging_bytes,
                        bytes_per_peer, sdma_post_chunk_bytes);
            } else {
                opus_sdma_a2a_post_kernel
                    <<<1, sdma_block, 0, transfer_stream>>>(
                        sdma_staging_win, sdma_recv_win, dev_comm,
                        static_cast<size_t>(slot) * staging_bytes,
                        bytes_per_peer);
            }
            CHECK_HIP(hipGetLastError());
            CHECK_HIP(hipMemcpyAsync(
                sdma_recv + static_cast<size_t>(rank) * elems_per_peer,
                sdma_staging[slot] + static_cast<size_t>(rank) * elems_per_peer,
                bytes_per_peer, hipMemcpyDeviceToDevice, transfer_stream));
            opus_sdma_a2a_quiet_notify_kernel<<<1, sdma_block, 0, transfer_stream>>>(
                sdma_ready_win, dev_comm);
            CHECK_HIP(hipGetLastError());
            if (overlap_epochs) {
                CHECK_HIP(hipEventRecord(slot_free[slot], transfer_stream));
            }
        };
        auto launch_sdma_epoch = [&]() {
            const int slot = static_cast<int>(pipeline_epoch & 1);
            launch_sdma_compute_phase(slot);
            launch_sdma_comm_phase(slot);
            ++pipeline_epoch;
        };

        for (int i = 0; i < warmup; ++i) launch_sdma_epoch();
        CHECK_HIP(hipStreamSynchronize(compute_stream));
        CHECK_HIP(hipStreamSynchronize(comm_stream));
        CHECK_CCO(ccoBarrierAll(comm));

        if (!overlap_epochs) {
            if (strict_timing) {
                measure_strict_phases(
                    compute_stream,
                    [&](int) {
                        const int slot = static_cast<int>(pipeline_epoch & 1);
                        launch_sdma_compute_phase(slot);
                    },
                    [&](int) {
                        const int slot = static_cast<int>(pipeline_epoch & 1);
                        launch_sdma_comm_phase(slot);
                        ++pipeline_epoch;
                    });
            } else {
                CHECK_HIP(hipEventRecord(start, compute_stream));
                for (int i = 0; i < iters; ++i) launch_sdma_epoch();
                CHECK_HIP(hipEventRecord(stop, transfer_stream));
            }
        } else {
            CHECK_HIP(hipEventRecord(start, compute_stream));
            for (int i = 0; i < iters; ++i) launch_sdma_epoch();
            CHECK_HIP(hipEventRecord(stop, transfer_stream));
        }
    } else {
        uint64_t round = 0;
        const size_t bytes_per_peer =
            static_cast<size_t>(M) * shard_n * sizeof(bf16_t);
        const size_t elems_per_peer =
            static_cast<size_t>(M) * shard_n;
        const size_t staging_bytes = staging_elems * sizeof(bf16_t);
        const dim3 sdma_block(static_cast<unsigned>(nranks * WAVE_SIZE));
        hipStream_t transfer_stream =
            overlap_epochs ? comm_stream : compute_stream;
        auto launch_chunk_compute_phase = [&](int slot) {
            if (overlap_epochs && round >= 2) {
                CHECK_HIP(hipStreamWaitEvent(compute_stream, slot_free[slot], 0));
            }
#if OPUS_PERSISTENT
            CHECK_HIP(hipMemsetAsync(
                d_tile_counter, 0, sizeof(unsigned int), compute_stream));
#endif
            CHECK_HIP(hipMemsetAsync(
                chunk_done, 0,
                static_cast<size_t>(nranks) *
                    kargs.chunk_num_m_tiles * sizeof(unsigned int),
                compute_stream));
            CHECK_HIP(hipMemsetAsync(
                chunk_peer_lock, 0,
                static_cast<size_t>(nranks) * sizeof(unsigned int), compute_stream));
            kargs.ptr_a = (round & 1) ? d_a_alt : d_a;
            kargs.cco_c_win = sdma_staging[slot];
            kargs.chunk_staging_slot_offset =
                static_cast<unsigned long long>(slot) * staging_bytes;
            gemm_a16w16_quad_subtile_kernel<Traits, true, true>
                <<<grid, block, 0, compute_stream>>>(kargs);
            CHECK_HIP(hipGetLastError());
            if (overlap_epochs) {
                CHECK_HIP(hipEventRecord(stage_ready[slot], compute_stream));
            }
        };
        auto launch_chunk_comm_phase = [&](int slot) {
            if (overlap_epochs) {
                CHECK_HIP(hipStreamWaitEvent(
                    transfer_stream, stage_ready[slot], 0));
            }
            CHECK_HIP(hipMemcpyAsync(
                sdma_recv + static_cast<size_t>(rank) * elems_per_peer,
                sdma_staging[slot] + static_cast<size_t>(rank) * elems_per_peer,
                bytes_per_peer, hipMemcpyDeviceToDevice, transfer_stream));
            opus_sdma_a2a_quiet_notify_kernel<<<1, sdma_block, 0, transfer_stream>>>(
                sdma_ready_win, dev_comm);
            CHECK_HIP(hipGetLastError());
            if (overlap_epochs) {
                CHECK_HIP(hipEventRecord(slot_free[slot], transfer_stream));
            }
        };
        auto launch_chunk_round = [&]() {
            const int slot = overlap_epochs ? static_cast<int>(round & 1) : 0;
            launch_chunk_compute_phase(slot);
            launch_chunk_comm_phase(slot);
            ++round;
        };
        for (int i = 0; i < warmup; ++i) launch_chunk_round();
        CHECK_HIP(hipStreamSynchronize(compute_stream));
        CHECK_HIP(hipStreamSynchronize(comm_stream));
        CHECK_CCO(ccoBarrierAll(comm));
        if (!overlap_epochs) {
            if (strict_timing) {
                measure_strict_phases(
                    compute_stream,
                    [&](int) {
                        const int slot = 0;
                        launch_chunk_compute_phase(slot);
                    },
                    [&](int) {
                        const int slot = 0;
                        launch_chunk_comm_phase(slot);
                        ++round;
                    });
            } else {
                CHECK_HIP(hipEventRecord(start, compute_stream));
                for (int i = 0; i < iters; ++i) launch_chunk_round();
                CHECK_HIP(hipEventRecord(stop, transfer_stream));
            }
        } else {
            CHECK_HIP(hipEventRecord(start, compute_stream));
            for (int i = 0; i < iters; ++i) launch_chunk_round();
            CHECK_HIP(hipEventRecord(stop, transfer_stream));
        }
    }
    if (!used_strict_measurement) {
        CHECK_HIP(hipEventSynchronize(stop));
        CHECK_HIP(hipEventElapsedTime(&total_ms, start, stop));
    }
    const double local_ms = static_cast<double>(total_ms) / iters;
    double avg_ms = 0.0, max_ms = 0.0;
    MPI_Allreduce(&local_ms, &avg_ms, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_ms, &max_ms, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    avg_ms /= nranks;
    const double local_strict_metrics[4] = {
        local_ms,
        static_cast<double>(strict_compute_total_ms) / iters,
        static_cast<double>(strict_comm_total_ms) / iters,
        static_cast<double>(strict_barrier_idle_total_ms) / iters};
    std::unique_ptr<double[]> gathered_strict_metrics;
    if (rank == 0) {
        gathered_strict_metrics =
            std::make_unique<double[]>(static_cast<size_t>(nranks) * 4);
    }
    MPI_Gather(
        local_strict_metrics, 4, MPI_DOUBLE,
        rank == 0 ? gathered_strict_metrics.get() : nullptr,
        4, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int critical_rank = 0;
    double critical_total_ms = 0.0;
    double critical_compute_ms = 0.0;
    double critical_comm_ms = 0.0;
    double critical_barrier_idle_ms = 0.0;
    if (rank == 0) {
        for (int candidate = 0; candidate < nranks; ++candidate) {
            const double* metrics =
                gathered_strict_metrics.get() + static_cast<size_t>(candidate) * 4;
            if (candidate == 0 || metrics[0] > critical_total_ms) {
                critical_rank = candidate;
                critical_total_ms = metrics[0];
                critical_compute_ms = metrics[1];
                critical_comm_ms = metrics[2];
                critical_barrier_idle_ms = metrics[3];
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    void* output_device = local_staging
        ? staging_local
        : (uses_sdma ? static_cast<void*>(sdma_recv) : win_local);
    auto h_output = std::make_unique<bf16_t[]>(
        local_staging ? staging_elems : recv_elems);
    CHECK_HIP(hipMemcpy(
        h_output.get(), output_device,
        (local_staging ? staging_elems : recv_elems) * sizeof(bf16_t),
        hipMemcpyDeviceToHost));
    auto h_tail = std::make_unique<bf16_t[]>(local_c_elems);
    CHECK_HIP(hipMemcpy(
        h_tail.get(), d_tail, local_c_elems * sizeof(bf16_t), hipMemcpyDeviceToHost));
    std::vector<uint64_t> h_sdma_ready;
    if (uses_sdma) {
        h_sdma_ready.resize(static_cast<size_t>(nranks));
        CHECK_HIP(hipMemcpy(
            h_sdma_ready.data(), sdma_ready_local,
            static_cast<size_t>(nranks) * sizeof(uint64_t),
            hipMemcpyDeviceToHost));
    }
    int mism = 0;
    const int output_variant =
        uses_sdma ? ((warmup + iters - 1) & 1) : 0;
    const int sample_rows[] = {0, 17, 511, 1023, 2047};
    const int sample_cols[] = {0, 255, 1024, 2559};
    for (int peer = 0; peer < nranks; ++peer) {
        for (int r : sample_rows) {
            if (r >= M) continue;
            for (int c : sample_cols) {
                if (c >= shard_n) continue;
                const int src = local_staging ? rank : peer;
                const int dst = local_staging ? peer : rank;
                const int global_col = dst * shard_n + c;
                const float ref =
                    sample_ref(src, r, global_col, K, output_variant);
                const size_t index = local_staging
                    ? (static_cast<size_t>(dst) * M + r) * shard_n + c
                    : (static_cast<size_t>(src) * M + r) * shard_n + c;
                const float got = static_cast<float>(h_output[index]);
                if (std::fabs(ref - got) > 2.0f) {
                    if (mism < 8) printf("[rank %d] mismatch src=%d dst=%d row=%d col=%d got=%f ref=%f\n", rank, src, dst, r, global_col, got, ref);
                    ++mism;
                }
            }
        }
    }
    if (uses_sdma) {
        const uint64_t expected_epoch =
            static_cast<uint64_t>(warmup + iters);
        for (int src = 0; src < nranks; ++src) {
            if (src == rank || h_sdma_ready[src] == expected_epoch) continue;
            if (mism < 8) {
                printf("[rank %d] SDMA ready mismatch src=%d got=%llu expected=%llu\n",
                       rank, src,
                       static_cast<unsigned long long>(h_sdma_ready[src]),
                       static_cast<unsigned long long>(expected_epoch));
            }
            ++mism;
        }
    }
    const int tail_cols[] = {scatter_n, N - 1};
    for (int r : sample_rows) {
        if (r >= M) continue;
        for (int c : tail_cols) {
            if (c < scatter_n || c >= N) continue;
            const float ref =
                sample_ref(rank, r, c, K, output_variant);
            const float got = static_cast<float>(h_tail[static_cast<size_t>(r) * N + c]);
            if (std::fabs(ref - got) > 2.0f) {
                if (mism < 8) printf("[rank %d] tail mismatch row=%d col=%d got=%f ref=%f\n", rank, r, c, got, ref);
                ++mism;
            }
        }
    }
    int total_mism = 0;
    MPI_Allreduce(&mism, &total_mism, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (rank == 0) {
        const double flops = 2.0 * double(M) * double(N) * double(K) * double(nranks);
        const char* output_name = output_mode == OutputMode::Direct
            ? "direct"
            : (output_mode == OutputMode::SplitLsa
                   ? "split-lsa"
                   : (output_mode == OutputMode::Local
                          ? "local"
                          : (output_mode == OutputMode::Sdma ? "sdma" : "chunk-sdma")));
        const char* schedule_name = overlap_epochs ? "parallel" : "serial";
        if (used_strict_measurement) {
            printf("quad_gemm_a2a M=%d output=%s schedule=%s shard_n=%d chunk_m_tiles=%d sdma_post_m_tiles=%d fused_lsa_stripe=%d strict_timing=%d %s grid=%u avg_rank_time=%.4f ms max_rank_time=%.4f ms critical_rank=%d critical_compute_ms=%.4f critical_comm_ms=%.4f comm_plus_compute=%.4f barrier_idle_residual_ms=%.4f critical_e2e_ms=%.4f aggregate=%.2f TFLOP/s %s\n",
                   M,
                   output_name,
                   schedule_name,
                   shard_n,
                   chunk_m_tiles_per_put,
                   sdma_post_m_tiles_per_put,
                   fused_lsa_stripe ? 1 : 0,
                   strict_timing ? 1 : 0,
                   OPUS_PERSISTENT ? "persistent" : "non-persistent", grid.x,
                   avg_ms, max_ms,
                   critical_rank, critical_compute_ms, critical_comm_ms,
                   critical_compute_ms + critical_comm_ms,
                   critical_barrier_idle_ms, critical_total_ms,
                   flops / (max_ms * 1.0e9),
                   total_mism == 0 ? "SUCCESS" : "FAILED");
        } else {
            printf("quad_gemm_a2a M=%d output=%s schedule=%s shard_n=%d chunk_m_tiles=%d sdma_post_m_tiles=%d fused_lsa_stripe=%d strict_timing=%d %s grid=%u avg_rank_time=%.4f ms max_rank_time=%.4f ms aggregate=%.2f TFLOP/s %s\n",
                   M,
                   output_name,
                   schedule_name,
                   shard_n,
                   chunk_m_tiles_per_put,
                   sdma_post_m_tiles_per_put,
                   fused_lsa_stripe ? 1 : 0,
                   strict_timing ? 1 : 0,
                   OPUS_PERSISTENT ? "persistent" : "non-persistent", grid.x,
                   avg_ms, max_ms, flops / (max_ms * 1.0e9),
                   total_mism == 0 ? "SUCCESS" : "FAILED");
        }
    }

    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));
    if (uses_sdma) {
        for (int slot = 0; slot < 2; ++slot) {
            CHECK_HIP(hipEventDestroy(stage_ready[slot]));
            CHECK_HIP(hipEventDestroy(slot_free[slot]));
        }
        CHECK_HIP(hipStreamDestroy(compute_stream));
        CHECK_HIP(hipStreamDestroy(comm_stream));
    }
    if (chunk_dev_comm) ccoDevCommFreeDeviceCopy(chunk_dev_comm);
    if (chunk_done) CHECK_HIP(hipFree(chunk_done));
    if (chunk_peer_lock) CHECK_HIP(hipFree(chunk_peer_lock));
    if (dev_comm_created) CHECK_CCO(ccoDevCommDestroy(comm, &dev_comm));
    if (sdma_recv_win) CHECK_CCO(ccoWindowDeregister(comm, sdma_recv_win));
    if (sdma_staging_win) CHECK_CCO(ccoWindowDeregister(comm, sdma_staging_win));
    if (sdma_ready_win) CHECK_CCO(ccoWindowDeregister(comm, sdma_ready_win));
    if (staging_win) CHECK_CCO(ccoWindowDeregister(comm, staging_win));
    CHECK_CCO(ccoWindowDeregister(comm, win));
    CHECK_CCO(ccoCommDestroy(comm));
    CHECK_HIP(hipFree(d_a));
    if (d_a_alt) CHECK_HIP(hipFree(d_a_alt));
    CHECK_HIP(hipFree(d_b));
    CHECK_HIP(hipFree(d_tail));
#if OPUS_PERSISTENT
    CHECK_HIP(hipFree(d_tile_counter));
#endif
    MPI_Finalize();
    return total_mism == 0 ? 0 : 1;
}
