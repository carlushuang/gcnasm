#include <opus/hip_minimal.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

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

template<typename Traits>
__global__ void gemm_a16w16_quad_subtile_kernel(opus_gemm_kargs kargs);

static constexpr size_t PER_RANK_VMM = 512ULL * 1024 * 1024;
static constexpr int RANKS = 4;

static float a_value(int src_rank, int row, int k) {
    return 0.001f * float(src_rank + 1) + 0.01f * float((row % 17) - 8) + 0.002f * float((k % 29) - 14);
}

static float b_value(int col, int k) {
    return 0.003f * float((col % 23) - 11) + 0.001f * float((k % 31) - 15);
}

static void fill_a(bf16_t* a, int rank, int m, int k) {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < m; ++i) {
        for (int kk = 0; kk < k; ++kk) {
            a[i * k + kk] = static_cast<bf16_t>(a_value(rank, i, kk));
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

static float sample_ref(int src_rank, int row, int col, int k) {
    float acc = 0.0f;
    for (int kk = 0; kk < k; ++kk) {
        const float av = static_cast<float>(static_cast<bf16_t>(a_value(src_rank, row, kk)));
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
    for (int i = 1; i < argc; ++i) {
        if ((std::strcmp(argv[i], "-m") == 0 || std::strcmp(argv[i], "--m") == 0) && i + 1 < argc) M = std::atoi(argv[++i]);
        else if ((std::strcmp(argv[i], "-n") == 0 || std::strcmp(argv[i], "--n") == 0) && i + 1 < argc) N = std::atoi(argv[++i]);
        else if ((std::strcmp(argv[i], "-k") == 0 || std::strcmp(argv[i], "--k") == 0) && i + 1 < argc) K = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--shard-n") == 0 && i + 1 < argc) shard_n = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) warmup = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) iters = std::atoi(argv[++i]);
    }

    if (nranks != RANKS) {
        if (rank == 0) fprintf(stderr, "requires exactly %d ranks\n", RANKS);
        MPI_Abort(MPI_COMM_WORLD, 1);
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
    const size_t local_c_elems = static_cast<size_t>(M) * N;

    auto h_a = std::make_unique<bf16_t[]>(a_elems);
    auto h_b = std::make_unique<bf16_t[]>(b_elems);
    fill_a(h_a.get(), rank, M, K);
    fill_b(h_b.get(), N, K);

    bf16_t* d_a = nullptr;
    bf16_t* d_b = nullptr;
    bf16_t* d_tail = nullptr;
    unsigned int* d_tile_counter = nullptr;
    CHECK_HIP(hipMalloc(&d_a, a_elems * sizeof(bf16_t)));
    CHECK_HIP(hipMalloc(&d_b, b_elems * sizeof(bf16_t)));
    CHECK_HIP(hipMalloc(&d_tail, local_c_elems * sizeof(bf16_t)));
#if OPUS_PERSISTENT
    CHECK_HIP(hipMalloc(&d_tile_counter, sizeof(unsigned int)));
#endif
    CHECK_HIP(hipMemcpy(d_a, h_a.get(), a_elems * sizeof(bf16_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_b, h_b.get(), b_elems * sizeof(bf16_t), hipMemcpyHostToDevice));

    ccoWindow_t win = nullptr;
    void* win_local = nullptr;
    CHECK_CCO(ccoWindowRegister(comm, recv_elems * sizeof(bf16_t), &win, &win_local));

    opus_gemm_kargs kargs{};
    kargs.ptr_a = d_a;
    kargs.ptr_b = d_b;
    kargs.ptr_c = d_tail;        // non-scattered tail columns land here
    kargs.tile_counter = d_tile_counter;
    kargs.cco_c_win = win;       // scattered shard columns land directly in peer LSA
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
#if OPUS_PERSISTENT
    int cu_count = 0;
    CHECK_HIP(hipDeviceGetAttribute(&cu_count, hipDeviceAttributeMultiprocessorCount, rank % ndev));
    const int persistent_wgs = total_tiles < cu_count ? total_tiles : cu_count;
    dim3 grid(persistent_wgs, 1, 1);
#else
    dim3 grid(total_tiles, 1, 1);
#endif
    dim3 block(Traits::BLOCK_SIZE);

    auto clear_buffers = [&]() {
        CHECK_HIP(hipMemset(win_local, 0, recv_elems * sizeof(bf16_t)));
        CHECK_HIP(hipMemset(d_tail, 0, local_c_elems * sizeof(bf16_t)));
        CHECK_HIP(hipDeviceSynchronize());
        CHECK_CCO(ccoBarrierAll(comm));
    };

    auto launch = [&]() {
#if OPUS_PERSISTENT
        CHECK_HIP(hipMemset(d_tile_counter, 0, sizeof(unsigned int)));
#endif
        gemm_a16w16_quad_subtile_kernel<Traits><<<grid, block>>>(kargs);
        CHECK_HIP(hipGetLastError());
    };

    clear_buffers();
    for (int i = 0; i < warmup; ++i) launch();

    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));
    CHECK_HIP(hipEventRecord(start));
    for (int i = 0; i < iters; ++i) launch();
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_HIP(hipEventElapsedTime(&total_ms, start, stop));
    const double local_ms = static_cast<double>(total_ms) / iters;
    double avg_ms = 0.0;
    MPI_Allreduce(&local_ms, &avg_ms, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    avg_ms /= nranks;

    MPI_Barrier(MPI_COMM_WORLD);
    auto h_recv = std::make_unique<bf16_t[]>(recv_elems);
    CHECK_HIP(hipMemcpy(h_recv.get(), win_local, recv_elems * sizeof(bf16_t), hipMemcpyDeviceToHost));
    int mism = 0;
    const int sample_rows[] = {0, 17, 511, 1023, 2047};
    const int sample_cols[] = {0, 255, 1024, 2559};
    for (int src = 0; src < nranks; ++src) {
        for (int r : sample_rows) {
            if (r >= M) continue;
            for (int c : sample_cols) {
                const int global_col = rank * shard_n + c;
                const float ref = sample_ref(src, r, global_col, K);
                const float got = static_cast<float>(h_recv[(static_cast<size_t>(src) * M + r) * shard_n + c]);
                if (std::fabs(ref - got) > 2.0f) {
                    if (mism < 8) printf("[rank %d] mismatch src=%d row=%d col=%d got=%f ref=%f\n", rank, src, r, global_col, got, ref);
                    ++mism;
                }
            }
        }
    }
    int total_mism = 0;
    MPI_Allreduce(&mism, &total_mism, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (rank == 0) {
        const double flops = 2.0 * double(M) * double(N) * double(K) * double(nranks);
        printf("quad_lsa_direct %s grid=%u avg_rank_time: %.4f ms, aggregate %.2f TFLOP/s, %s\n",
               OPUS_PERSISTENT ? "persistent" : "non-persistent", grid.x,
               avg_ms, flops / (avg_ms * 1.0e9), total_mism == 0 ? "SUCCESS" : "FAILED");
    }

    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));
    CHECK_CCO(ccoWindowDeregister(comm, win));
    CHECK_CCO(ccoCommDestroy(comm));
    CHECK_HIP(hipFree(d_a));
    CHECK_HIP(hipFree(d_b));
    CHECK_HIP(hipFree(d_tail));
#if OPUS_PERSISTENT
    CHECK_HIP(hipFree(d_tile_counter));
#endif
    MPI_Finalize();
    return total_mism == 0 ? 0 : 1;
}
