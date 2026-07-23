#include <opus/hip_minimal.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include <mpi.h>
#include "mori/cco/cco.hpp"

#include "gemm_defs.h"
#include "a2a_scatter_kernel.hpp"

using namespace mori::cco;

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

static float a_value(int src_rank, int row, int k) {
    return 0.001f * float(src_rank + 1) + 0.01f * float((row % 17) - 8) + 0.002f * float((k % 29) - 14);
}

static float b_value(int col, int k) {
    return 0.003f * float((col % 23) - 11) + 0.001f * float((k % 31) - 15);
}

static void fill_a(bf16_t* a, int rank, int m, int k) {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < m; ++i)
        for (int kk = 0; kk < k; ++kk) a[i * k + kk] = static_cast<bf16_t>(a_value(rank, i, kk));
}

static void fill_b(bf16_t* b, int n, int k) {
#pragma omp parallel for collapse(2)
    for (int j = 0; j < n; ++j)
        for (int kk = 0; kk < k; ++kk) b[j * k + kk] = static_cast<bf16_t>(b_value(j, kk));
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    int M = 2048, N = 18432, K = 8192, shard_n = 2560, warmup = 3, iters = 20;
    const char* only = "all";  // all | fused | gemm | a2a
    bool do_verify = true;
    for (int i = 1; i < argc; ++i) {
        if ((std::strcmp(argv[i], "-m") == 0 || std::strcmp(argv[i], "--m") == 0) && i + 1 < argc) M = std::atoi(argv[++i]);
        else if ((std::strcmp(argv[i], "-n") == 0 || std::strcmp(argv[i], "--n") == 0) && i + 1 < argc) N = std::atoi(argv[++i]);
        else if ((std::strcmp(argv[i], "-k") == 0 || std::strcmp(argv[i], "--k") == 0) && i + 1 < argc) K = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--shard-n") == 0 && i + 1 < argc) shard_n = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) warmup = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--only") == 0 && i + 1 < argc) only = argv[++i];
        else if (std::strcmp(argv[i], "--noverify") == 0) do_verify = false;
    }
    const bool run_fused = !std::strcmp(only, "all") || !std::strcmp(only, "fused");
    const bool run_gemm  = !std::strcmp(only, "all") || !std::strcmp(only, "gemm") || !std::strcmp(only, "a2a");
    const bool run_a2a   = !std::strcmp(only, "all") || !std::strcmp(only, "a2a");

    if (nranks < 2) {
        if (rank == 0) fprintf(stderr, "requires >= 2 ranks\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    using Traits = opus_gemm_traits<512, 256, 256, 64, bf16_t, bf16_t, bf16_t, float>;
    const int scatter_n = shard_n * nranks;
    if (M % Traits::B_M != 0 || shard_n % Traits::B_N != 0 || scatter_n > N || K % Traits::B_K != 0 ||
        ((K / Traits::B_K) % 2) != 0 || (shard_n % 8) != 0 || (N % 8) != 0) {
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
    bf16_t* d_tail = nullptr;    // fused path: non-scattered tail columns
    bf16_t* d_c_full = nullptr;  // split path: full local GEMM output [M,N]
    unsigned int* d_tile_counter = nullptr;
    CHECK_HIP(hipMalloc(&d_a, a_elems * sizeof(bf16_t)));
    CHECK_HIP(hipMalloc(&d_b, b_elems * sizeof(bf16_t)));
    CHECK_HIP(hipMalloc(&d_tail, local_c_elems * sizeof(bf16_t)));
    CHECK_HIP(hipMalloc(&d_c_full, local_c_elems * sizeof(bf16_t)));
    CHECK_HIP(hipMalloc(&d_tile_counter, sizeof(unsigned int)));
    CHECK_HIP(hipMemcpy(d_a, h_a.get(), a_elems * sizeof(bf16_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_b, h_b.get(), b_elems * sizeof(bf16_t), hipMemcpyHostToDevice));

    ccoWindow_t win = nullptr;
    void* win_local = nullptr;
    CHECK_CCO(ccoWindowRegister(comm, recv_elems * sizeof(bf16_t), &win, &win_local));

    // ---- GEMM kargs (shared config) ----
    opus_gemm_kargs base{};
    base.ptr_a = d_a;
    base.ptr_b = d_b;
    base.tile_counter = d_tile_counter;
    base.m = M; base.n = N; base.k = K; base.batch = 1;
    base.stride_a = K; base.stride_b = K;
    base.stride_a_batch = M * K; base.stride_b_batch = N * K; base.stride_c_batch = M * N;

    // Fused (PR#40 optimized): scattered shards -> peer LSA, tail stays local in d_tail.
    opus_gemm_kargs kargs_fused = base;
    kargs_fused.ptr_c = d_tail;
    kargs_fused.cco_c_win = win;
    kargs_fused.a2a_n_shard = shard_n;
    kargs_fused.a2a_M = M;
    kargs_fused.a2a_span = scatter_n;
    kargs_fused.stride_c_full = N;
    kargs_fused.stride_c = shard_n;

    // Standalone GEMM: SAME PR#40 kernel, but store full C[M,N] locally (a2a off).
    opus_gemm_kargs kargs_gemm = base;
    kargs_gemm.ptr_c = d_c_full;
    kargs_gemm.cco_c_win = nullptr;
    kargs_gemm.a2a_n_shard = 0;
    kargs_gemm.scatter_n_shard = 0;
    kargs_gemm.stride_c = N;

    // Persistent grid (PR#40): one workgroup per CU, tiles pulled via atomic counter.
    int cu_count = 0;
    CHECK_HIP(hipDeviceGetAttribute(&cu_count, hipDeviceAttributeMultiprocessorCount, rank % ndev));
    const int num_tiles_m = ceil_div(M, Traits::B_M);
    const int num_tiles_n = ceil_div(N, Traits::B_N);
    const int total_tiles = num_tiles_m * num_tiles_n;
    const int persistent_wgs = total_tiles < cu_count ? total_tiles : cu_count;
    dim3 gemm_grid(persistent_wgs, 1, 1);
    dim3 gemm_block(Traits::BLOCK_SIZE);

    // Standalone A2A: SAME persistent grid + tile scheduler as fused, sourced from HBM.
    const int scatter_tiles = num_tiles_m * (scatter_n / Traits::B_N);
    a2a_scatter_kargs akargs{};
    akargs.src_c = d_c_full;
    akargs.cco_c_win = win;
    akargs.tile_counter = d_tile_counter;
    akargs.M = M; akargs.n_shard = shard_n; akargs.a2a_span = scatter_n; akargs.stride_src = N;
    akargs.B_M = Traits::B_M; akargs.B_N = Traits::B_N;
    akargs.num_tiles_m = num_tiles_m; akargs.scatter_tiles = scatter_tiles;
    const int a2a_wgs = scatter_tiles < cu_count ? scatter_tiles : cu_count;
    dim3 a2a_grid(a2a_wgs, 1, 1);
    dim3 a2a_block(Traits::BLOCK_SIZE);

    auto launch_fused = [&]() {
        CHECK_HIP(hipMemset(d_tile_counter, 0, sizeof(unsigned int)));
        gemm_a16w16_quad_subtile_kernel<Traits><<<gemm_grid, gemm_block>>>(kargs_fused);
        CHECK_HIP(hipGetLastError());
    };
    auto launch_gemm = [&]() {
        CHECK_HIP(hipMemset(d_tile_counter, 0, sizeof(unsigned int)));
        gemm_a16w16_quad_subtile_kernel<Traits><<<gemm_grid, gemm_block>>>(kargs_gemm);
        CHECK_HIP(hipGetLastError());
    };
    auto launch_a2a = [&]() {
        CHECK_HIP(hipMemset(d_tile_counter, 0, sizeof(unsigned int)));
        a2a_scatter_kernel<<<a2a_grid, a2a_block>>>(akargs);
        CHECK_HIP(hipGetLastError());
    };

    auto clear_win = [&]() {
        CHECK_HIP(hipMemset(win_local, 0, recv_elems * sizeof(bf16_t)));
        CHECK_HIP(hipDeviceSynchronize());
        CHECK_CCO(ccoBarrierAll(comm));
    };
    auto clear_tail = [&]() {
        CHECK_HIP(hipMemset(d_tail, 0, local_c_elems * sizeof(bf16_t)));
        CHECK_HIP(hipDeviceSynchronize());
    };

    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));

    auto time_ms = [&](auto&& launch) -> double {
        for (int i = 0; i < warmup; ++i) launch();
        CHECK_HIP(hipDeviceSynchronize());
        MPI_Barrier(MPI_COMM_WORLD);
        CHECK_HIP(hipEventRecord(start));
        for (int i = 0; i < iters; ++i) launch();
        CHECK_HIP(hipEventRecord(stop));
        CHECK_HIP(hipEventSynchronize(stop));
        float ms = 0.0f;
        CHECK_HIP(hipEventElapsedTime(&ms, start, stop));
        double local = static_cast<double>(ms) / iters, avg = 0.0;
        MPI_Allreduce(&local, &avg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return avg / nranks;
    };

    // ---- full-elementwise verification vs independent fp32 CPU reference ----
    const float RTOL = 2.0e-2f, ATOL = 1.5e-1f;
    auto h_recv = std::make_unique<bf16_t[]>(recv_elems);
    auto h_tail = std::make_unique<bf16_t[]>(local_c_elems);
    const bf16_t* pa = h_a.get();
    const bf16_t* pb = h_b.get();

    auto verify = [&](void* recv_dev, void* tail_dev, int tail_stride, const char* label) -> long long {
        CHECK_HIP(hipMemcpy(h_recv.get(), recv_dev, recv_elems * sizeof(bf16_t), hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(h_tail.get(), tail_dev, local_c_elems * sizeof(bf16_t), hipMemcpyDeviceToHost));
        const bf16_t* pr = h_recv.get();
        const bf16_t* pt = h_tail.get();
        long long sc = 0, sm = 0; double smaxa = 0.0, smaxr = 0.0;
#pragma omp parallel for collapse(2) schedule(dynamic) reduction(+ : sc, sm) reduction(max : smaxa, smaxr)
        for (int src = 0; src < nranks; ++src) {
            for (int r = 0; r < M; ++r) {
                std::vector<float> arow(K);
                for (int kk = 0; kk < K; ++kk) arow[kk] = static_cast<float>(static_cast<bf16_t>(a_value(src, r, kk)));
                for (int c = 0; c < shard_n; ++c) {
                    const int gc = rank * shard_n + c;
                    const bf16_t* bcol = pb + static_cast<size_t>(gc) * K;
                    float acc = 0.0f;
                    for (int kk = 0; kk < K; ++kk) acc += arow[kk] * static_cast<float>(bcol[kk]);
                    const float ref = static_cast<float>(static_cast<bf16_t>(acc));
                    const float got = static_cast<float>(pr[(static_cast<size_t>(src) * M + r) * shard_n + c]);
                    const double abserr = std::fabs(ref - got), relerr = abserr / (std::fabs(ref) + 1e-6);
                    ++sc;
                    if (abserr > ATOL + RTOL * std::fabs(ref)) ++sm;
                    if (abserr > smaxa) smaxa = abserr;
                    if (relerr > smaxr) smaxr = relerr;
                }
            }
        }
        long long tc = 0, tm = 0; double tmaxa = 0.0, tmaxr = 0.0;
#pragma omp parallel for schedule(dynamic) reduction(+ : tc, tm) reduction(max : tmaxa, tmaxr)
        for (int r = 0; r < M; ++r) {
            const bf16_t* arow = pa + static_cast<size_t>(r) * K;
            for (int col = scatter_n; col < N; ++col) {
                const bf16_t* bcol = pb + static_cast<size_t>(col) * K;
                float acc = 0.0f;
                for (int kk = 0; kk < K; ++kk) acc += static_cast<float>(arow[kk]) * static_cast<float>(bcol[kk]);
                const float ref = static_cast<float>(static_cast<bf16_t>(acc));
                const float got = static_cast<float>(pt[static_cast<size_t>(r) * tail_stride + col]);
                const double abserr = std::fabs(ref - got), relerr = abserr / (std::fabs(ref) + 1e-6);
                ++tc;
                if (abserr > ATOL + RTOL * std::fabs(ref)) ++tm;
                if (abserr > tmaxa) tmaxa = abserr;
                if (relerr > tmaxr) tmaxr = relerr;
            }
        }
        long long g_sc = 0, g_sm = 0, g_tc = 0, g_tm = 0;
        double g_smaxa = 0, g_smaxr = 0, g_tmaxa = 0, g_tmaxr = 0;
        MPI_Allreduce(&sc, &g_sc, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&sm, &g_sm, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&tc, &g_tc, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&tm, &g_tm, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&smaxa, &g_smaxa, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&smaxr, &g_smaxr, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&tmaxa, &g_tmaxa, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&tmaxr, &g_tmaxr, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        const long long total = g_sm + g_tm;
        if (rank == 0)
            printf("verify[%-7s]: %s  scatter(chk=%lld mism=%lld max_abs=%.4g max_rel=%.4g)  "
                   "tail(chk=%lld mism=%lld max_abs=%.4g max_rel=%.4g)\n",
                   label, total == 0 ? "SUCCESS" : "FAILED", g_sc, g_sm, g_smaxa, g_smaxr,
                   g_tc, g_tm, g_tmaxa, g_tmaxr);
        return total;
    };

    double t_fused = 0, t_gemm = 0, t_a2a = 0;
    long long mism_fused = 0, mism_split = 0;

    if (run_fused) {
        clear_win(); clear_tail();
        t_fused = time_ms(launch_fused);
        CHECK_HIP(hipDeviceSynchronize()); MPI_Barrier(MPI_COMM_WORLD);
        if (do_verify) mism_fused = verify(win_local, d_tail, N, "fused");
    }
    if (run_gemm) {
        CHECK_HIP(hipDeviceSynchronize()); MPI_Barrier(MPI_COMM_WORLD);
        t_gemm = time_ms(launch_gemm);  // leaves full C in d_c_full
    }
    if (run_a2a) {
        clear_win();
        t_a2a = time_ms(launch_a2a);
        CHECK_HIP(hipDeviceSynchronize()); MPI_Barrier(MPI_COMM_WORLD);
        if (do_verify) mism_split = verify(win_local, d_c_full, N, "split");
    }

    if (rank == 0) {
        const double flops_agg = 2.0 * double(M) * double(N) * double(K) * double(nranks);
        const double a2a_bytes = 2.0 * double(M) * double(scatter_n) * sizeof(bf16_t);  // rd+wr per rank
        const double t_split = t_gemm + t_a2a;
        printf("\n==== PR#40 fused vs split (standalone GEMM + standalone A2A)  [M=%d N=%d K=%d shard_n=%d ranks=%d] ====\n",
               M, N, K, shard_n, nranks);
        printf("[fused] gemm+a2a : %8.4f ms   %8.1f TFLOP/s (agg)\n", t_fused, flops_agg / (t_fused * 1e9));
        printf("[split] gemm     : %8.4f ms   %8.1f TFLOP/s (agg)\n", t_gemm, flops_agg / (t_gemm * 1e9));
        printf("[split] a2a      : %8.4f ms   %8.1f GB/s (rd+wr/rank)\n", t_a2a, a2a_bytes / (t_a2a * 1e6));
        printf("[split] total    : %8.4f ms\n", t_split);
        printf("speedup split/fused : %.3fx   (fusion %+.1f%% vs split)\n",
               t_split / t_fused, 100.0 * (t_split - t_fused) / t_split);
        printf("verify tolerance: |ref-got| <= %.3g + %.3g*|ref|\n", ATOL, RTOL);
    }

    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));
    CHECK_CCO(ccoWindowDeregister(comm, win));
    CHECK_CCO(ccoCommDestroy(comm));
    CHECK_HIP(hipFree(d_a));
    CHECK_HIP(hipFree(d_b));
    CHECK_HIP(hipFree(d_tail));
    CHECK_HIP(hipFree(d_c_full));
    CHECK_HIP(hipFree(d_tile_counter));
    MPI_Finalize();
    return (mism_fused == 0 && mism_split == 0) ? 0 : 1;
}
