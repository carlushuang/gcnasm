#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <random>
#include <iostream>
#include <memory>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <stdio.h>
#include <omp.h>
#include "half.hpp"

#include "opus/opus.hpp"
#include "reg_access_uitls.h"
#include "tcopy_desc_utils.h"

#define CHECK_HIP(call)                                                                                   \
    do {                                                                                                  \
        hipError_t status_ = call;                                                                        \
        if (status_ != hipSuccess) {                                                                      \
            fprintf(stderr, "HIP error (%s:%d): %s\n", __FILE__, __LINE__, hipGetErrorString(status_));   \
            exit(1);                                                                                      \
        }                                                                                                 \
    } while(0)

#define CHECK_HIP_KERNEL_LAUNCH() CHECK_HIP(hipGetLastError())

using fp32_t    = float;
using float16   = half_float::half; // cpu type, half_float::half for host only

// All vector types (fp16_t, fp16x2_t .. fp16x16_t, fp32x4_t, etc.)
// are pulled from opus, which selects __fp16 or _Float16 based on clang version.
using namespace opus;

using int32x4_t = int32_t __attribute__((ext_vector_type(4)));
using int32x8_t = int32_t __attribute__((ext_vector_type(8)));
#define BUFFER_LOAD_DWORD3 0x00020000   // This is valid for 
struct buffer_resource {
    const void * ptr;
    uint32_t range;
    uint32_t config;
};
__device__ int32x4_t make_buffer_resource(const void * ptr, uint32_t size = 0xffffffff)
{
    buffer_resource res {ptr, size, BUFFER_LOAD_DWORD3};
    return __builtin_bit_cast(int32x4_t, res);
}



__global__ void 
wmma_kernel_standard(const void* __restrict__ ptr_a,
                   const void* __restrict__ ptr_b,
                   void* __restrict__ ptr_c,
                   int stride_a, // stride in unit of pixel
                   int stride_b,
                   int stride_c)
{
    // 16x16x32 gemm, assume only launched 1 wave
    // A: 16 rows x 32 cols (row-major), each thread holds 16 fp16 elements
    // thread lane = threadIdx.x % 16 → selects the row
    // threadIdx.x / 16 ∈ {0,1} → selects the first or second group of 8 K-elements
    __shared__ char Smem[32*16*2*sizeof(fp16_t)];
    uintptr_t smembase = reinterpret_cast<uintptr_t>(Smem);

    // TileDim0=32(K), TileDim1=16(M/N), data_size=1(fp16) — all compile-time
    using Wmma16x16x32Tcopy = TcopyDesc<fp16_t, 32, 16>;

    Wmma16x16x32Tcopy tcopy_a, tcopy_b;
    tcopy_a.make(smembase,                          ptr_a, 32, 16, stride_a);
    tcopy_b.make(smembase + 16*32*sizeof(fp16_t),   ptr_b, 32, 16, stride_b);



    int row_a = threadIdx.x % 16;
    int grp_a = threadIdx.x / 16;  // 0 or 1, each group covers 8 K elements
    // A is at Smem[0], laid out as 16 rows x 32 cols (fp16), stride = 32
    // offset in bytes for ds_read
    int smem_stride_a = 32;
    int a_sld_os0 = (row_a * smem_stride_a + grp_a * 8     ) * sizeof(fp16_t);
    int a_sld_os1 = (row_a * smem_stride_a + grp_a * 8 + 16) * sizeof(fp16_t);

    int row_b = threadIdx.x % 16;
    int grp_b = threadIdx.x / 16;
    // B is at Smem[16*32 fp16], i.e. byte offset 16*32*sizeof(fp16_t)
    constexpr int smem_b_base_bytes = 16 * 32 * sizeof(fp16_t);
    int smem_stride_b = 32;
    int b_sld_os0 = smem_b_base_bytes + (row_b * smem_stride_b + grp_b * 8     ) * sizeof(fp16_t);
    int b_sld_os1 = smem_b_base_bytes + (row_b * smem_stride_b + grp_b * 8 + 16) * sizeof(fp16_t);

    reg_utils::Fp16x16Packer convertA;
    reg_utils::Fp16x16Packer convertB;

    __builtin_amdgcn_tensor_load_to_lds(tcopy_a.sg0.as<int32x4_t>(), tcopy_a.sg1.as<int32x8_t>(), {0,0,0,0}, {0,0,0,0}, 27);
    __builtin_amdgcn_tensor_load_to_lds(tcopy_b.sg0.as<int32x4_t>(), tcopy_b.sg1.as<int32x8_t>(), {0,0,0,0}, {0,0,0,0}, 27);

    __builtin_amdgcn_s_wait_tensorcnt(0);

    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_barrier_wait(-1);

    // ds_read_b128: load 8x fp16 (128-bit) per issue, two issues per matrix fragment
    fp16x8_t sld_a0, sld_a1;
    fp16x8_t sld_b0, sld_b1;
    asm volatile(
        "ds_read_b128 %[a0], %[a_os0]\n\t"
        "ds_read_b128 %[a1], %[a_os1]\n\t"
        "ds_read_b128 %[b0], %[b_os0]\n\t"
        "ds_read_b128 %[b1], %[b_os1]\n\t"
        : [a0]"=v"(sld_a0), [a1]"=v"(sld_a1),
          [b0]"=v"(sld_b0), [b1]"=v"(sld_b1)
        : [a_os0]"v"(a_sld_os0), [a_os1]"v"(a_sld_os1),
          [b_os0]"v"(b_sld_os0), [b_os1]"v"(b_sld_os1)
        : "memory"
    );
    asm volatile("" : : "v"(a_sld_os0), "v"(a_sld_os1), "v"(b_sld_os0), "v"(b_sld_os1) : "memory");
    asm volatile("s_wait_dscnt(0)" ::: "memory");
    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_barrier_wait(-1);

    convertA = __builtin_bit_cast(reg_utils::Fp16x16Packer, opus::array<fp16x8_t,2>{sld_a0, sld_a1});
    convertB = __builtin_bit_cast(reg_utils::Fp16x16Packer, opus::array<fp16x8_t,2>{sld_b0, sld_b1});
    

    fp16x8_t v_c = {.0f};  // clear

    __builtin_amdgcn_sched_barrier(0);
    v_c = __builtin_amdgcn_wmma_f16_16x16x32_f16(0, convertB.vec16, 0, convertA.vec16, 0, v_c, false, false);

    int col_id_c = threadIdx.x / 16;
    int row_id_c = threadIdx.x % 16;

    *(reinterpret_cast<fp16x8_t*>(ptr_c) + col_id_c + row_id_c * 2) = v_c;
}

// Fill 2D matrix with random fp16 values in specified range
template<typename T>
void rand_vector_2d(T* ptr, int m, int n, int ld, float min_val = 0.0f, float max_val = 1.0f) {
    #pragma omp parallel
    {
        std::random_device rd;
        std::mt19937 gen(rd() + omp_get_thread_num());
        std::uniform_real_distribution<float> dis(min_val, max_val);
        #pragma omp for collapse(2)
        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) {
                ptr[i * ld + j] = static_cast<T>(dis(gen));
            }
        }
    }
}

// Validate computed fp16 results against float reference
bool valid_vector(const float* ref, const float16* result, int n, float threshold = 1e-3f) {
    int errors = 0;
    for(int i = 0; i < n; i++) {
        float diff = std::abs(ref[i] - static_cast<float>(result[i]));
        if(diff > threshold) {
            if(errors < 10) {
                printf("Error at %d: ref=%.6f, result=%.6f, diff=%.6f\n",
                       i, ref[i], static_cast<float>(result[i]), diff);
            }
            errors++;
            if(errors >= 10) break;
        }
    }
    return errors == 0;
}

// CPU reference GEMM (RCR layout: A row-major, B row-major transposed)
void gemm_ref(const float* a, const float* b, float* c, int m, int n, int k, int lda, int ldb, int ldc) {
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            float sum = 0.0f;
            for(int p = 0; p < k; p++) {
                sum += a[i * lda + p] * b[j * ldb + p];
            }
            c[i * ldc + j] = sum;
        }
    }
}

// Benchmark kernel performance with warm-up and hipEvent timing
void benchmark_kernel(float16* dev_a, float16* dev_b, float16* dev_c,
                      int lda, int ldb, int ldc, int m, int n, int k,
                      int warmup = 5, int iterations = 20) {
    for(int i = 0; i < warmup; ++i) {
        wmma_kernel_standard<<<1, 32>>>(dev_a, dev_b, dev_c, lda, ldb, ldc);
        CHECK_HIP_KERNEL_LAUNCH();
    }

    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));

    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipEventRecord(start));

    for(int i = 0; i < iterations; ++i) {
        wmma_kernel_standard<<<1, 32>>>(dev_a, dev_b, dev_c, lda, ldb, ldc);
        CHECK_HIP_KERNEL_LAUNCH();
    }

    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));

    float total_ms = 0;
    CHECK_HIP(hipEventElapsedTime(&total_ms, start, stop));
    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));

    const float avg_ms = total_ms / iterations;
    const float tflops = 2.0f * m * n * k / 1.0e9f / avg_ms;
    printf("Kernel Performance: avg_time=%.4f ms, %.2f TFlops\n", avg_ms, tflops);
}

int main(int argc, char** argv) {
    int m = 16;
    int n = 16;
    int k = 32;

    // Parse command line arguments: -m -n -k
    for(int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if((std::strcmp(arg, "-m") == 0 || std::strcmp(arg, "--m") == 0) && i + 1 < argc) {
            m = std::atoi(argv[++i]);
        } else if((std::strcmp(arg, "-n") == 0 || std::strcmp(arg, "--n") == 0) && i + 1 < argc) {
            n = std::atoi(argv[++i]);
        } else if((std::strcmp(arg, "-k") == 0 || std::strcmp(arg, "--k") == 0) && i + 1 < argc) {
            k = std::atoi(argv[++i]);
        }
    }

    if(m <= 0 || n <= 0 || k <= 0) {
        fprintf(stderr, "Invalid problem size: m, n, k must be positive.\n");
        return 1;
    }

    int lda = k;
    int ldb = k;
    int ldc = n;

    // Allocate host memory
    auto host_a    = std::make_unique<float[]>(lda * m);
    auto host_b    = std::make_unique<float[]>(ldb * n);
    auto host_c    = std::make_unique<float[]>(ldc * m);
    auto fp16_a    = std::make_unique<float16[]>(lda * m);
    auto fp16_b    = std::make_unique<float16[]>(ldb * n);
    auto fp16_c    = std::make_unique<float16[]>(ldc * m);

    // Initialize fp32 host data
    rand_vector_2d(host_a.get(), m, k, lda, 0.0f, 1.0f);
    rand_vector_2d(host_b.get(), n, k, ldb, -0.5f, 0.5f);

    // Convert fp32 → fp16 on host
    for(int i = 0; i < lda * m; i++) fp16_a[i] = __float2half_rn(host_a[i]);
    for(int i = 0; i < ldb * n; i++) fp16_b[i] = __float2half_rn(host_b[i]);

    // Allocate device memory
    float16 *dev_a, *dev_b, *dev_c;
    CHECK_HIP(hipMalloc(&dev_a, lda * m * sizeof(float16)));
    CHECK_HIP(hipMalloc(&dev_b, ldb * n * sizeof(float16)));
    CHECK_HIP(hipMalloc(&dev_c, ldc * m * sizeof(float16)));

    // Copy fp16 data to device
    CHECK_HIP(hipMemcpy(dev_a, fp16_a.get(), lda * m * sizeof(float16), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dev_b, fp16_b.get(), ldb * n * sizeof(float16), hipMemcpyHostToDevice));

    printf("m:%d, n:%d, k:%d, lda:%d, ldb:%d, ldc:%d\n", m, n, k, lda, ldb, ldc);

    // CPU reference
    gemm_ref(host_a.get(), host_b.get(), host_c.get(), m, n, k, lda, ldb, ldc);

    // Launch kernel
    wmma_kernel_standard<<<1, 32>>>(dev_a, dev_b, dev_c, lda, ldb, ldc);
    CHECK_HIP_KERNEL_LAUNCH();

    // Copy results back and validate
    CHECK_HIP(hipMemcpy(fp16_c.get(), dev_c, ldc * m * sizeof(float16), hipMemcpyDeviceToHost));
    bool valid = valid_vector(host_c.get(), fp16_c.get(), m * n, 1e-3f);
    printf("[16x16x32, standard] %s\n", valid ? "✓ VALID" : "✗ FAIL");

    // Benchmark
    printf("\n");
    benchmark_kernel(dev_a, dev_b, dev_c, lda, ldb, ldc, m, n, k);

    // Cleanup
    CHECK_HIP(hipFree(dev_a));
    CHECK_HIP(hipFree(dev_b));
    CHECK_HIP(hipFree(dev_c));

    return valid ? 0 : 1;
}
