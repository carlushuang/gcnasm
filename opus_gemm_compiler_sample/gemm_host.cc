// Host driver for the gfx1250 4wave_compute BF16 GEMM: random init, CPU
// reference, validation and benchmark. C = A * B^T with A [M, K] row-major and
// B [N, K] row-major (i.e. B is K-major, same as the device input).
#include <opus/hip_minimal.hpp>
#include <random>
#include <iostream>
#include <memory>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <omp.h>

#include "gemm_defs.h"

// Device-stub declaration resolved by linking against the kernel TU.
template<typename Traits>
__global__ void gemm_a16w16_4wave_compute_kernel(opus_gemm_kargs kargs);

#define CHECK_HIP(call)                                                                                   \
    do {                                                                                                  \
        hipError_t status_ = call;                                                                        \
        if (status_ != hipSuccess) {                                                                      \
            fprintf(stderr, "HIP error (%s:%d): %s\n", __FILE__, __LINE__, hipGetErrorString(status_));   \
            exit(1);                                                                                      \
        }                                                                                                 \
    } while(0)

#define CHECK_HIP_KERNEL_LAUNCH() CHECK_HIP(hipGetLastError())

template<typename T>
void rand_vector_2d(T* ptr, int m, int n, int ld, float min_val, float max_val) {
    #pragma omp parallel
    {
        std::mt19937 gen(1234 + omp_get_thread_num());
        std::uniform_real_distribution<float> dis(min_val, max_val);
        #pragma omp for collapse(2)
        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) {
                ptr[i * ld + j] = static_cast<T>(dis(gen));
            }
        }
    }
}

// Constant fill. Feeding every lane the same bits leaves the WMMA datapath
// with almost nothing to toggle, which takes data-dependent power draw (and
// therefore clock throttling) out of a measurement.
template<typename T>
void const_vector(T* ptr, size_t n, float val) {
    #pragma omp parallel for schedule(static)
    for(size_t i = 0; i < n; i++) ptr[i] = static_cast<T>(val);
}

// CPU reference GEMM (row-major A, K-major B, fp32 accumulation).
void gemm_ref(const bf16_t* a, const bf16_t* b, bf16_t* c, int m, int n, int k, int lda, int ldb, int ldc) {
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            float sum = 0.0f;
            for(int p = 0; p < k; p++) {
                sum += static_cast<float>(a[i * lda + p]) * static_cast<float>(b[j * ldb + p]);
            }
            c[i * ldc + j] = static_cast<bf16_t>(sum);
        }
    }
}

struct validation { double max_abs; double max_rel; int bad; };

// A value counts as bad only when BOTH the absolute and the relative deviation
// are out of tolerance: bf16 has ~3 decimal digits, so large accumulations are
// coarse in absolute terms while still being correct.
validation valid_vector(const bf16_t* ref, const bf16_t* result, size_t n,
                        float abs_tol = 0.5f, float rel_tol = 0.05f) {
    double max_abs = 0.0, max_rel = 0.0;
    int bad = 0;
    #pragma omp parallel for reduction(max : max_abs, max_rel) reduction(+ : bad) schedule(static)
    for(size_t i = 0; i < n; i++) {
        const float r = static_cast<float>(ref[i]);
        const float g = static_cast<float>(result[i]);
        const float ad = std::fabs(g - r);
        const float rd = ad / (std::fabs(r) + 1e-6f);
        if (ad > max_abs) max_abs = ad;
        if (rd > max_rel) max_rel = rd;
        if (ad > abs_tol && rd > rel_tol) ++bad;
    }
    return {max_abs, max_rel, bad};
}

struct bench_result { float avg_ms; float tflops; };

template<typename Launch>
bench_result benchmark_kernel(Launch&& launch, const opus_gemm_kargs& kargs,
                              int warmup = 25, int iterations = 200) {
    for (int i = 0; i < warmup; ++i) {
        launch();
        CHECK_HIP_KERNEL_LAUNCH();
    }

    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));

    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipEventRecord(start));

    for (int i = 0; i < iterations; ++i) {
        launch();
        CHECK_HIP_KERNEL_LAUNCH();
    }

    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));

    float total_time = 0;
    CHECK_HIP(hipEventElapsedTime(&total_time, start, stop));

    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));

    const float avg_ms = total_time / iterations;
    const std::size_t flop = std::size_t(2) * kargs.m * kargs.n * kargs.k * kargs.batch;
    return {avg_ms, static_cast<float>(flop) / 1.0e9f / avg_ms};
}

int main(int argc, char** argv) {
    int M = 256;
    int N = 5120;
    int K = 2880;
    int batch = 1;
    int iterations = 200;
    float const_val = 0.0f;
    bool const_init = false;
    bool skip_verify = false;

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if ((std::strcmp(arg, "-m") == 0 || std::strcmp(arg, "--m") == 0) && i + 1 < argc) {
            M = std::atoi(argv[++i]);
        } else if ((std::strcmp(arg, "-n") == 0 || std::strcmp(arg, "--n") == 0) && i + 1 < argc) {
            N = std::atoi(argv[++i]);
        } else if ((std::strcmp(arg, "-k") == 0 || std::strcmp(arg, "--k") == 0) && i + 1 < argc) {
            K = std::atoi(argv[++i]);
        } else if ((std::strcmp(arg, "-b") == 0 || std::strcmp(arg, "--b") == 0) && i + 1 < argc) {
            batch = std::atoi(argv[++i]);
        } else if ((std::strcmp(arg, "-i") == 0 || std::strcmp(arg, "--iters") == 0) && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        } else if ((std::strcmp(arg, "-c") == 0 || std::strcmp(arg, "--const") == 0) && i + 1 < argc) {
            const_val = static_cast<float>(std::atof(argv[++i]));
            const_init = true;
        } else if (std::strcmp(arg, "--no-verify") == 0) {
            skip_verify = true;
        } else {
            std::cerr << "usage: " << argv[0]
                      << " [-m M] [-n N] [-k K] [-b batch] [-i iters]"
                         " [-c CONST] [--no-verify]\n";
            return 1;
        }
    }

    if (M <= 0 || N <= 0 || K <= 0 || batch <= 0 || iterations <= 0) {
        std::cerr << "Invalid problem size: M, N, K, batch and iters must be positive.\n";
        return 1;
    }

    using Traits = gemm_traits_128x256x128;

    printf("BF16 GEMM (gfx1250 4wave_compute): M=%d N=%d K=%d batch=%d\n", M, N, K, batch);
    printf("  tile   : %dx%dx%d  slots=%d  block=%d  cluster=%dx%d\n",
           Traits::B_M, Traits::B_N, Traits::B_K, Traits::NUM_SLOTS,
           Traits::BLOCK_SIZE, Traits::CLUSTER_WG_M, Traits::CLUSTER_WG_N);
    printf("  LDS    : %.1fKB (A seg %.1fKB, B seg %.1fKB)\n",
           Traits::LDS_BYTES / 1024.0, Traits::SEG_BYTES_A / 1024.0,
           Traits::SEG_BYTES_B / 1024.0);

    const size_t elems_a = (size_t)batch * M * K;
    const size_t elems_b = (size_t)batch * N * K;
    const size_t elems_c = (size_t)batch * M * N;

    auto host_a     = std::make_unique<bf16_t[]>(elems_a);
    auto host_b     = std::make_unique<bf16_t[]>(elems_b);
    auto host_c_ref = std::make_unique<bf16_t[]>(elems_c);
    auto host_c_out = std::make_unique<bf16_t[]>(elems_c);

    printf("  init   : %s\n", const_init ? "constant" : "random [-1, 1]");
    for(int b = 0; b < batch; b++) {
        if (const_init) {
            const_vector(host_a.get() + (size_t)b * M * K, (size_t)M * K, const_val);
            const_vector(host_b.get() + (size_t)b * N * K, (size_t)N * K, const_val);
        } else {
            rand_vector_2d(host_a.get() + (size_t)b * M * K, M, K, K, -1.0f, 1.0f);
            rand_vector_2d(host_b.get() + (size_t)b * N * K, N, K, K, -1.0f, 1.0f);
        }
    }

    bf16_t *dev_a, *dev_b, *dev_c;
    CHECK_HIP(hipMalloc(&dev_a, elems_a * sizeof(bf16_t)));
    CHECK_HIP(hipMalloc(&dev_b, elems_b * sizeof(bf16_t)));
    CHECK_HIP(hipMalloc(&dev_c, elems_c * sizeof(bf16_t)));

    CHECK_HIP(hipMemcpy(dev_a, host_a.get(), elems_a * sizeof(bf16_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dev_b, host_b.get(), elems_b * sizeof(bf16_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemset(dev_c, 0, elems_c * sizeof(bf16_t)));

    opus_gemm_kargs kargs{};
    kargs.ptr_a = dev_a;
    kargs.ptr_b = dev_b;
    kargs.ptr_c = dev_c;
    kargs.m = M;
    kargs.n = N;
    kargs.k = K;
    kargs.batch = batch;
    kargs.stride_a = K;
    kargs.stride_b = K;
    kargs.stride_c = N;
    kargs.stride_a_batch = M * K;
    kargs.stride_b_batch = N * K;
    kargs.stride_c_batch = M * N;

    // Round the tile grid up to whole clusters. Not a choice: the runtime
    // rejects a cluster launch whose grid is not a multiple of the cluster dims
    // (rocclr LaunchParams::CheckClusterDivisibility), so a partially populated
    // cluster does not exist and the surplus workgroups are always dispatched.
    // The kernel gives them nothing to do -- they fail the tile bound check and
    // return as soon as they have reported to the cluster barrier their peers
    // count on, before touching LDS or the TDM.
    const int grid_m = ceil_div(ceil_div(M, Traits::B_M), Traits::CLUSTER_WG_M) * Traits::CLUSTER_WG_M;
    const int grid_n = ceil_div(ceil_div(N, Traits::B_N), Traits::CLUSTER_WG_N) * Traits::CLUSTER_WG_N;
    dim3 grid(grid_m, grid_n, batch);
    dim3 block(Traits::BLOCK_SIZE);
    printf("  grid   : %ux%ux%u  block=%u\n", grid.x, grid.y, grid.z, block.x);

    auto launch = [&] { gemm_a16w16_4wave_compute_kernel<Traits><<<grid, block>>>(kargs); };

    validation v{0.0, 0.0, 0};
    if (skip_verify) {
        printf("  verify : SKIPPED\n");
    } else {
        printf("Computing CPU reference ...\n");
        for(int b = 0; b < batch; b++) {
            gemm_ref(host_a.get() + (size_t)b * M * K,
                     host_b.get() + (size_t)b * N * K,
                     host_c_ref.get() + (size_t)b * M * N,
                     M, N, K, K, K, N);
        }

        launch();
        CHECK_HIP_KERNEL_LAUNCH();
        CHECK_HIP(hipDeviceSynchronize());
        CHECK_HIP(hipMemcpy(host_c_out.get(), dev_c, elems_c * sizeof(bf16_t), hipMemcpyDeviceToHost));

        v = valid_vector(host_c_ref.get(), host_c_out.get(), elems_c);
        printf("  verify : %s  max_abs=%.4f max_rel=%.4f bad=%d/%zu\n",
               v.bad == 0 ? "PASS" : "FAIL", v.max_abs, v.max_rel, v.bad, elems_c);
    }

    const auto bench = benchmark_kernel(launch, kargs, 25, iterations);
    printf("  perf   : %.3f us/iter  %8.2f TFlops\n", bench.avg_ms * 1000.0f, bench.tflops);

    CHECK_HIP(hipFree(dev_a));
    CHECK_HIP(hipFree(dev_b));
    CHECK_HIP(hipFree(dev_c));

    return v.bad == 0 ? 0 : 1;
}
