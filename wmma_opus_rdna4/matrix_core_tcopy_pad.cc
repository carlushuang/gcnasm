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
    using opus::operator""_I;

    // 16x16x128: K=128 = 4 × WMMA(16x16x32), single wave (32 lanes)
    constexpr int Block_K = 128;
    constexpr int Block_M = 16;
    constexpr int Block_N = 16;
    constexpr int32_t warpNum = 1;
    const int lane_id = threadIdx.x % opus::get_warp_size();
    const int wave_id = threadIdx.x / opus::get_warp_size();

    __shared__ char Smem[16 * Block_K * 2 * sizeof(fp16_t) + 16 * 8 * sizeof(fp16_t)];
    uintptr_t smembase = reinterpret_cast<uintptr_t>(Smem);

    // TileDim0=128(K), TileDim1=16(M/N); LdsPadEn=1, PadInterval=5(256B), PadAmount=3(16B)
    using NoSelectedWgs = opus::seq<>;
    using Wmma16x16x128Tcopy = TcopyDesc<fp16_t, 128, 16, 0, 0, 0,
    1, 0, 0, 0, 1, 0, 0, 0,
    0, 1, 5, 3, NoSelectedWgs>;

    Wmma16x16x128Tcopy tcopy_a, tcopy_b;
    tcopy_a.make(smembase, ptr_a, Block_K, 16, stride_a);
    tcopy_b.make(smembase + 16 * Block_K * sizeof(fp16_t) + 16 * 8 * sizeof(fp16_t), ptr_b, Block_K, 16, stride_b);

    // ── block_sld A/B: same construction as matrix_core_gfx942.cc (smem load windows)
    // 32-lane wave: AKSldLane×AMSldLane = 2×16 (gfx942 uses 4×16 on 64-lane)
    constexpr int32_t AKSldPack = 16 / static_cast<int32_t>(sizeof(fp16_t));
    constexpr int32_t AKSldLane = 16 / AKSldPack;
    constexpr int32_t AMSldLane = opus::get_warp_size() / AKSldLane;
    constexpr int32_t AMSldRepeat = Block_M / (AMSldLane * warpNum);
    constexpr int32_t AKSldRepeat = Block_K / (AKSldPack * AKSldLane);
    static_assert(AKSldLane * AMSldLane == opus::get_warp_size(), "A sld lane product");

    // LDS row pitch (fp16 elements) = logical K + Tensor Copy pad per row; keep A/B sld strides consistent
    constexpr int32_t SMemKPitch = Block_K + 8;

    auto block_sld_shape_a = opus::make_tuple(opus::number<AMSldRepeat>{},
                                                opus::number<warpNum>{},
                                                opus::number<AKSldRepeat>{},
                                                opus::number<AKSldLane>{},
                                                opus::number<AMSldLane>{},
                                                opus::number<AKSldPack>{});
    auto block_sld_stride_a = opus::make_tuple(AMSldLane * SMemKPitch * warpNum,
                                                AMSldLane * SMemKPitch,
                                                AKSldPack * AKSldLane,
                                                AKSldPack,
                                                SMemKPitch,
                                                1_I);
    auto block_sld_win_a = opus::make_layout<0>(block_sld_shape_a, block_sld_stride_a);

    constexpr int32_t BSldKPack = 16 / static_cast<int32_t>(sizeof(fp16_t));
    constexpr int32_t BSldKLane = AKSldLane;
    constexpr int32_t BSldNLane = AMSldLane;
    constexpr int32_t BSldKRepeat = Block_K / (BSldKPack * BSldKLane);
    constexpr int32_t BSldNRepeat = Block_N / BSldNLane;
    static_assert(BSldKLane * BSldNLane == opus::get_warp_size(), "B sld lane product");
    static_assert(BSldKPack * BSldKLane * BSldKRepeat == Block_K, "B K tile");
    // static_assert(BSldNLane * BSldNRepeat == Block_N, "B N tile");

    auto block_sld_shape_b = opus::make_tuple(opus::number<BSldNRepeat>{},
                                                opus::number<BSldKRepeat>{},
                                                opus::number<BSldKLane>{},
                                                opus::number<BSldNLane>{},
                                                opus::number<BSldKPack>{});
    auto block_sld_stride_b = opus::make_tuple(SMemKPitch * BSldNLane,
                                                BSldKPack * BSldKLane,
                                                BSldKPack,
                                                SMemKPitch,
                                                1_I);
    auto block_sld_win_b = opus::make_layout<0>(block_sld_shape_b, block_sld_stride_b);

    constexpr int smem_b_base_bytes = 16 * Block_K * static_cast<int>(sizeof(fp16_t))
                                      + 16 * 8 * static_cast<int>(sizeof(fp16_t));

    __builtin_amdgcn_tensor_load_to_lds(tcopy_a.sg0.as<int32x4_t>(), tcopy_a.sg1.as<int32x8_t>(), {0,0,0,0}, {0,0,0,0}, 27);
    __builtin_amdgcn_tensor_load_to_lds(tcopy_b.sg0.as<int32x4_t>(), tcopy_b.sg1.as<int32x8_t>(), {0,0,0,0}, {0,0,0,0}, 27);

    __builtin_amdgcn_s_wait_tensorcnt(0);

    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_barrier_wait(-1);

    fp16x8_t v_c = {.0f};
    constexpr int KtileElems = 32; // per WMMA along K
    static_assert(Block_K % KtileElems == 0, "K must be multiple of 32");
    constexpr int K_WmmaTiles = Block_K / KtileElems;

    #pragma unroll
    for (int kt = 0; kt < K_WmmaTiles; ++kt) {
        // One WMMA K-slab = 32 fp16 = two KRepeat steps (16 fp16 each), same as gfx942 +0 / +32 B offsets pattern
        const int32_t kr0 = 2 * kt;
        const int32_t kr1 = kr0 + 1;

        const int32_t a_sld_os0 =
            block_sld_win_a(0_I, wave_id, kr0, lane_id / AMSldLane, lane_id % AMSldLane, 0_I)
            * static_cast<int32_t>(sizeof(fp16_t));
        const int32_t a_sld_os1 =
            block_sld_win_a(0_I, wave_id, kr1, lane_id / AMSldLane, lane_id % AMSldLane, 0_I)
            * static_cast<int32_t>(sizeof(fp16_t));
        const int32_t b_sld_os0 =
            smem_b_base_bytes
            + block_sld_win_b(0_I, kr0, lane_id / BSldNLane, lane_id % BSldNLane, 0_I)
                  * static_cast<int32_t>(sizeof(fp16_t));
        const int32_t b_sld_os1 =
            smem_b_base_bytes
            + block_sld_win_b(0_I, kr1, lane_id / BSldNLane, lane_id % BSldNLane, 0_I)
                  * static_cast<int32_t>(sizeof(fp16_t));

        fp16x8_t sld_a0, sld_a1, sld_b0, sld_b1;
        asm volatile(
            "ds_read_b128 %[a0], %[a_os0]\n\t"
            "ds_read_b128 %[a1], %[a_os1]\n\t"
            "ds_read_b128 %[b0], %[b_os0]\n\t"
            "ds_read_b128 %[b1], %[b_os1]\n\t"
            : [a0]"=v"(sld_a0), [a1]"=v"(sld_a1),
              [b0]"=v"(sld_b0), [b1]"=v"(sld_b1)
            : [a_os0]"v"(a_sld_os0), [a_os1]"v"(a_sld_os1),
              [b_os0]"v"(b_sld_os0), [b_os1]"v"(b_sld_os1)
            : "memory");
        asm volatile("" : : "v"(a_sld_os0), "v"(a_sld_os1), "v"(b_sld_os0), "v"(b_sld_os1) : "memory");
        asm volatile("s_wait_dscnt(0)" ::: "memory");
        __builtin_amdgcn_s_barrier_signal(-1);
        __builtin_amdgcn_s_barrier_wait(-1);

        reg_utils::Fp16x16Packer convertA = __builtin_bit_cast(
            reg_utils::Fp16x16Packer, opus::array<fp16x8_t, 2>{sld_a0, sld_a1});
        reg_utils::Fp16x16Packer convertB = __builtin_bit_cast(
            reg_utils::Fp16x16Packer, opus::array<fp16x8_t, 2>{sld_b0, sld_b1});

        __builtin_amdgcn_sched_barrier(0);
        v_c = __builtin_amdgcn_wmma_f16_16x16x32_f16(0, convertB.vec16, 0, convertA.vec16, 0, v_c, false, false);
    }

    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_barrier_wait(-1);

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
    int k = 128;

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
    printf("[16x16x128, Tensor Copy] %s\n", valid ? "✓ VALID" : "✗ FAIL");

    // Benchmark
    printf("\n");
    benchmark_kernel(dev_a, dev_b, dev_c, lda, ldb, ldc, m, n, k);

    // Cleanup
    CHECK_HIP(hipFree(dev_a));
    CHECK_HIP(hipFree(dev_b));
    CHECK_HIP(hipFree(dev_c));

    return valid ? 0 : 1;
}
