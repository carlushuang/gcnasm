#include <opus/hip_minimal.hpp>
#include <hip/hip_fp8.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>

#include <omp.h>

#include "gemm_a8w8_blockscale_common.h"

template<class Traits>
__global__ void gemm_a8w8_blockscale_kernel(opus_gemm_kargs kargs);

#define CHECK_HIP(call)                                                                                  \
    do {                                                                                                 \
        hipError_t status_ = call;                                                                       \
        if (status_ != hipSuccess) {                                                                     \
            std::fprintf(stderr, "HIP error (%s:%d): %s\n", __FILE__, __LINE__, hipGetErrorString(status_)); \
            std::exit(1);                                                                                \
        }                                                                                                \
    } while (0)

#define CHECK_HIP_KERNEL_LAUNCH() CHECK_HIP(hipGetLastError())

using host_fp8_t = __hip_fp8_e4m3;
using fp32_t = float;
using GemmTraits = gemm_a8w8_blockscale_traits<>;

template<typename T>
void rand_vector(T* ptr, std::size_t size, fp32_t min_val = 0.0f, fp32_t max_val = 1.0f) {
    #pragma omp parallel
    {
        std::random_device rd;
        std::mt19937 gen(rd() + omp_get_thread_num());
        std::uniform_real_distribution<fp32_t> dis(min_val, max_val);
        #pragma omp for
        for (std::size_t i = 0; i < size; ++i) {
            ptr[i] = static_cast<T>(dis(gen));
        }
    }
}

template<typename T>
bool valid_vector(const T* ref, const T* result, int n, fp32_t threshold = 1e-3f) {
    int errors = 0;
    for (int i = 0; i < n; ++i) {
        const fp32_t diff = std::abs(static_cast<fp32_t>(ref[i]) - static_cast<fp32_t>(result[i]));
        if (diff > threshold) {
            if (errors < 10) {
                std::printf("Error at %d: ref=%.6f, result=%.6f, diff=%.6f\n",
                            i, static_cast<fp32_t>(ref[i]), static_cast<fp32_t>(result[i]), diff);
            }
            ++errors;
            if (errors >= 10) {
                break;
            }
        }
    }
    return errors == 0;
}

// CPU reference GEMM: fp8 inputs, fp32 output, with grouped scale factors.
void gemm_ref(const host_fp8_t* a, const host_fp8_t* b, const fp32_t* sfa, const fp32_t* sfb, fp32_t* c,
              int m, int n, int k, int lda, int ldb, int ldc, int stride_sfa, int stride_sfb,
              int group_m, int group_n, int group_k) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            const host_fp8_t* a_row = a + i * lda;
            const host_fp8_t* b_row = b + j * ldb;
            const int m_group = i / group_m;
            const int n_group = j / group_n;
            fp32_t sum = 0.0f;
            for (int k_group_idx = 0; k_group_idx < k / group_k; ++k_group_idx) {
                const fp32_t scale_a = sfa[k_group_idx * stride_sfa + m_group];
                const fp32_t scale_b = sfb[n_group * stride_sfb + k_group_idx];
                const fp32_t scale = scale_a * scale_b;
                const int p_begin = k_group_idx * group_k;
                const int p_end = p_begin + group_k;
                for (int p = p_begin; p < p_end; ++p) {
                    sum += static_cast<fp32_t>(a_row[p]) * static_cast<fp32_t>(b_row[p]) * scale;
                }
            }
            c[i * ldc + j] = sum;
        }
    }
}

template<class Traits>
void benchmark_kernel(const opus_gemm_kargs& kargs, dim3 grid, dim3 block, int warmup = 200, int iterations = 100) {
    for (int i = 0; i < warmup; ++i) {
        gemm_a8w8_blockscale_kernel<Traits><<<grid, block>>>(kargs);
        CHECK_HIP_KERNEL_LAUNCH();
    }

    hipEvent_t start;
    hipEvent_t stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));

    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipEventRecord(start));

    for (int i = 0; i < iterations; ++i) {
        gemm_a8w8_blockscale_kernel<Traits><<<grid, block>>>(kargs);
        CHECK_HIP_KERNEL_LAUNCH();
    }

    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));

    fp32_t total_time = 0.0f;
    CHECK_HIP(hipEventElapsedTime(&total_time, start, stop));

    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));

    const fp32_t avg_time = total_time / iterations;
    const std::size_t flop = static_cast<std::size_t>(2) * kargs.m * kargs.n * kargs.k * kargs.batch;
    const fp32_t tflops = static_cast<fp32_t>(flop) / 1.0e9f / avg_time;

    std::printf("Kernel Performance: avg_time=%.4f ms, %.2f TFlops\n", avg_time, tflops);
}

int main(int argc, char** argv) {
    constexpr int BLOCK_M = GemmTraits::B_M;
    constexpr int BLOCK_N = GemmTraits::B_N;
    constexpr int BLOCK_K = GemmTraits::B_K;
    constexpr int BLOCK_SIZE = GemmTraits::BLOCK_SIZE;

    int M = 256;
    int N = 512;
    int K = 256;
    int batch = 8;

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
        }
    }

    if (M <= 0 || N <= 0 || K <= 0 || batch <= 0) {
        std::cerr << "Invalid problem size: M, N, K and batch must be positive.\n";
        return 1;
    }

    constexpr int GROUP_M = GemmTraits::GROUP_M;
    constexpr int GROUP_N = GemmTraits::GROUP_N;
    constexpr int GROUP_K = GemmTraits::GROUP_K;
    if (M % GROUP_M != 0 || N % GROUP_N != 0 || K % GROUP_K != 0) {
        std::cerr << "M/N/K must be multiples of GROUP_M/GROUP_N/GROUP_K ("
                  << GROUP_M << "," << GROUP_N << "," << GROUP_K << ") for scale factors.\n";
        return 1;
    }

    const int num_groups_m = M / GROUP_M;
    const int num_groups_n = N / GROUP_N;
    const int num_groups_k = K / GROUP_K;

    auto host_a = std::make_unique<host_fp8_t[]>(static_cast<std::size_t>(batch) * M * K);
    auto host_b = std::make_unique<host_fp8_t[]>(static_cast<std::size_t>(batch) * N * K);
    auto host_c = std::make_unique<fp32_t[]>(static_cast<std::size_t>(batch) * M * N);
    auto host_c_out = std::make_unique<fp32_t[]>(static_cast<std::size_t>(batch) * M * N);

    const std::size_t sfa_count = static_cast<std::size_t>(batch) * num_groups_m * num_groups_k;
    const std::size_t sfb_count = static_cast<std::size_t>(batch) * num_groups_n * num_groups_k;
    auto host_sfa = std::make_unique<fp32_t[]>(sfa_count);
    auto host_sfb = std::make_unique<fp32_t[]>(sfb_count);

    rand_vector(host_a.get(), static_cast<std::size_t>(batch) * M * K, 0.0f, 1.0f);
    rand_vector(host_b.get(), static_cast<std::size_t>(batch) * N * K, -0.5f, 0.5f);
    rand_vector(host_sfa.get(), sfa_count, 0.8f, 1.2f);
    rand_vector(host_sfb.get(), sfb_count, 0.8f, 1.2f);

    void* dev_a = nullptr;
    void* dev_b = nullptr;
    void* dev_sfa = nullptr;
    void* dev_sfb = nullptr;
    fp32_t* dev_c = nullptr;
    CHECK_HIP(hipMalloc(&dev_a, static_cast<std::size_t>(batch) * M * K * sizeof(host_fp8_t)));
    CHECK_HIP(hipMalloc(&dev_b, static_cast<std::size_t>(batch) * N * K * sizeof(host_fp8_t)));
    CHECK_HIP(hipMalloc(&dev_c, static_cast<std::size_t>(batch) * M * N * sizeof(fp32_t)));
    CHECK_HIP(hipMalloc(&dev_sfa, sfa_count * sizeof(fp32_t)));
    CHECK_HIP(hipMalloc(&dev_sfb, sfb_count * sizeof(fp32_t)));

    CHECK_HIP(hipMemcpy(dev_a, host_a.get(), static_cast<std::size_t>(batch) * M * K * sizeof(host_fp8_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dev_b, host_b.get(), static_cast<std::size_t>(batch) * N * K * sizeof(host_fp8_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dev_sfa, host_sfa.get(), sfa_count * sizeof(fp32_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dev_sfb, host_sfb.get(), sfb_count * sizeof(fp32_t), hipMemcpyHostToDevice));

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
    kargs.ptr_sfa = dev_sfa;
    kargs.ptr_sfb = dev_sfb;
    kargs.stride_sfa = num_groups_m;
    kargs.stride_sfb = num_groups_k;
    kargs.stride_sfa_batch = num_groups_m * num_groups_k;
    kargs.stride_sfb_batch = num_groups_n * num_groups_k;

    const int num_tiles_m = ceil_div(M, BLOCK_M);
    const int num_tiles_n = ceil_div(N, BLOCK_N);
    dim3 grid(num_tiles_n, num_tiles_m, batch);
    dim3 block(BLOCK_SIZE);

    std::printf("Launching GEMM kernel: M=%d, N=%d, K=%d, grid=(%u,%u,%u), block=%d\n",
                M, N, K, grid.x, grid.y, grid.z, BLOCK_SIZE);

    gemm_a8w8_blockscale_kernel<GemmTraits><<<grid, block>>>(kargs);
    CHECK_HIP_KERNEL_LAUNCH();

    CHECK_HIP(hipMemcpy(host_c_out.get(), dev_c, static_cast<std::size_t>(batch) * M * N * sizeof(fp32_t),
                        hipMemcpyDeviceToHost));

    bool all_valid = true;
    for (int b = 0; b < batch; ++b) {
        gemm_ref(host_a.get() + static_cast<std::size_t>(b) * M * K,
                 host_b.get() + static_cast<std::size_t>(b) * N * K,
                 host_sfa.get() + static_cast<std::size_t>(b) * kargs.stride_sfa_batch,
                 host_sfb.get() + static_cast<std::size_t>(b) * kargs.stride_sfb_batch,
                 host_c.get() + static_cast<std::size_t>(b) * M * N,
                 M, N, K, K, K, N, kargs.stride_sfa, kargs.stride_sfb, GROUP_M, GROUP_N, GROUP_K);
        const bool valid = valid_vector(host_c.get() + static_cast<std::size_t>(b) * M * N,
                                        host_c_out.get() + static_cast<std::size_t>(b) * M * N, M * N, 1e-3f);
        std::printf("[GEMM batch %d/%d: %dx%dx%d, block_%dx%dx%d] %s\n",
                    b + 1, batch, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, valid ? "VALID" : "FAIL");
        all_valid = all_valid && valid;
    }

    std::printf("\n[Overall] %s\n", all_valid ? "ALL BATCHES VALID" : "SOME BATCHES FAILED");

    std::printf("\n");
    benchmark_kernel<GemmTraits>(kargs, grid, block);
    std::printf("\n");

    CHECK_HIP(hipFree(dev_a));
    CHECK_HIP(hipFree(dev_b));
    CHECK_HIP(hipFree(dev_c));
    CHECK_HIP(hipFree(dev_sfa));
    CHECK_HIP(hipFree(dev_sfb));

    return 0;
}
