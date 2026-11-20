// SPDX-License-Identifier: MIT
// Microbenchmark: wmma_f32_16x16x16_f16 and v_exp2 throughput on gfx1201.
//
// One workgroup = 1 wave = 32 lanes. Each lane runs a tight loop of the
// target op with inputs that don't allow CSE / dead code elim. Lots of
// workgroups in flight to saturate the chip; we measure achieved ops/sec.
//
// Build:
//   hipcc -x hip -std=c++17 -O3 --offload-arch=gfx1201 ubench_gfx1201.cc -o ubench
//
// Run:
//   ./ubench [wmma|exp2|both]

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using fp16_t   = _Float16;
using fp32_t   = float;
using fp16x8_t = fp16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

#define HIP_CALL(c) do { hipError_t e = (c); \
    if (e != hipSuccess) { fprintf(stderr, "HIP %s\n", hipGetErrorString(e)); std::exit(1); } } while(0)

struct UBKArgs {
    fp32_t* sink;
    int iters;
};

// ---- WMMA throughput ----
// Each loop iter issues N_PIPE independent wmma calls so the SIMD can
// pipeline them (consecutive wmmas with no data dep). N_PIPE=8 means we
// hold 8 fp32x8_t accumulators in registers.
//
// Per iter: N_PIPE wmma_f32_16x16x16_f16 = N_PIPE * 4096 FLOPS.

template<int N_PIPE>
__launch_bounds__(32, 1)
__global__ void ubench_wmma_kernel(UBKArgs k)
{
#if defined(__gfx1201__) || defined(__gfx1200__)
    fp16x8_t a, b;
    #pragma unroll
    for (int j = 0; j < 8; ++j) { a[j] = (fp16_t)((threadIdx.x + j) * 0.01f); b[j] = (fp16_t)((threadIdx.x + j + 1) * 0.01f); }

    fp32x8_t c[N_PIPE];
    #pragma unroll
    for (int i = 0; i < N_PIPE; ++i) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) c[i][j] = (fp32_t)i;
    }

    for (int it = 0; it < k.iters; ++it) {
        #pragma unroll
        for (int i = 0; i < N_PIPE; ++i) {
            c[i] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c[i]);
        }
    }

    // Anti-DCE: store sum to sink only if some impossible condition is met
    fp32_t acc = 0.0f;
    #pragma unroll
    for (int i = 0; i < N_PIPE; ++i) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) acc += c[i][j];
    }
    if (acc < -1e30f) k.sink[blockIdx.x * 32 + threadIdx.x] = acc;
#else
    (void)k;
#endif
}

// ---- v_exp2_f32 throughput ----
//
// Per iter: N_PIPE * 8 v_exp2_f32 = N_PIPE * 8 exp ops.

template<int N_PIPE>
__launch_bounds__(32, 1)
__global__ void ubench_exp2_kernel(UBKArgs k)
{
#if defined(__gfx1201__) || defined(__gfx1200__)
    fp32x8_t v[N_PIPE];
    #pragma unroll
    for (int i = 0; i < N_PIPE; ++i) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) v[i][j] = (fp32_t)(threadIdx.x + i + j) * 0.001f;
    }

    for (int it = 0; it < k.iters; ++it) {
        #pragma unroll
        for (int i = 0; i < N_PIPE; ++i) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) v[i][j] = __builtin_amdgcn_exp2f(v[i][j]);
        }
    }

    fp32_t acc = 0.0f;
    #pragma unroll
    for (int i = 0; i < N_PIPE; ++i) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) acc += v[i][j];
    }
    if (acc < -1e30f) k.sink[blockIdx.x * 32 + threadIdx.x] = acc;
#else
    (void)k;
#endif
}

template<class K>
static float bench_kernel(K kern, int grid_blocks, int iters, int warmups = 3, int rep = 20) {
    UBKArgs ka{};
    HIP_CALL(hipMalloc(&ka.sink, grid_blocks * 32 * sizeof(fp32_t)));
    ka.iters = iters;
    for (int i = 0; i < warmups; ++i) kern<<<grid_blocks, 32>>>(ka);
    HIP_CALL(hipDeviceSynchronize());
    hipEvent_t e0, e1;
    HIP_CALL(hipEventCreate(&e0)); HIP_CALL(hipEventCreate(&e1));
    HIP_CALL(hipEventRecord(e0));
    for (int i = 0; i < rep; ++i) kern<<<grid_blocks, 32>>>(ka);
    HIP_CALL(hipEventRecord(e1));
    HIP_CALL(hipEventSynchronize(e1));
    float ms = 0; HIP_CALL(hipEventElapsedTime(&ms, e0, e1));
    HIP_CALL(hipFree(ka.sink));
    return ms / rep;
}

int main(int argc, char** argv) {
    const char* mode = (argc > 1) ? argv[1] : "both";

    int dev;
    HIP_CALL(hipGetDevice(&dev));
    hipDeviceProp_t prop;
    HIP_CALL(hipGetDeviceProperties(&prop, dev));
    printf("device: %s  CUs=%d  clk_mhz=%d\n", prop.name, prop.multiProcessorCount, prop.clockRate / 1000);

    const int CUs = prop.multiProcessorCount;

    if (strcmp(mode, "wmma") == 0 || strcmp(mode, "both") == 0) {
        printf("\n=== WMMA f32_16x16x16_f16 throughput ===\n");
        printf("%-8s %-12s %-12s %-12s %-12s\n", "n_pipe", "grid", "iters", "avg_ms", "TFLOPS");

        // Sweep N_PIPE and grid size. Each wmma = 16*16*16*2 = 8192 FLOPS.
        // Per kernel: grid * N_PIPE * iters * 8192 / time = FLOPS/sec.
        const int iters = 1024;
        for (int n_pipe : {1, 2, 4, 8}) {
            for (int grid_mul : {1, 4, 8, 16}) {
                int grid = CUs * grid_mul * 8;  // 8 WGs/CU as a baseline, scale up
                float ms = 0;
                if (n_pipe == 1) ms = bench_kernel(ubench_wmma_kernel<1>, grid, iters);
                if (n_pipe == 2) ms = bench_kernel(ubench_wmma_kernel<2>, grid, iters);
                if (n_pipe == 4) ms = bench_kernel(ubench_wmma_kernel<4>, grid, iters);
                if (n_pipe == 8) ms = bench_kernel(ubench_wmma_kernel<8>, grid, iters);
                double total_flops = (double)grid * n_pipe * iters * 8192.0;
                double tflops = total_flops / (ms * 1e9);
                printf("%-8d %-12d %-12d %-12.4f %-12.2f\n", n_pipe, grid, iters, ms, tflops);
            }
        }
    }

    if (strcmp(mode, "exp2") == 0 || strcmp(mode, "both") == 0) {
        printf("\n=== v_exp2_f32 throughput ===\n");
        printf("%-8s %-12s %-12s %-12s %-12s\n", "n_pipe", "grid", "iters", "avg_ms", "Gexp/s");
        const int iters = 4096;
        for (int n_pipe : {1, 2, 4, 8}) {
            for (int grid_mul : {1, 4, 8, 16}) {
                int grid = CUs * grid_mul * 8;
                float ms = 0;
                if (n_pipe == 1) ms = bench_kernel(ubench_exp2_kernel<1>, grid, iters);
                if (n_pipe == 2) ms = bench_kernel(ubench_exp2_kernel<2>, grid, iters);
                if (n_pipe == 4) ms = bench_kernel(ubench_exp2_kernel<4>, grid, iters);
                if (n_pipe == 8) ms = bench_kernel(ubench_exp2_kernel<8>, grid, iters);
                // Per kernel: grid * 32 lanes * n_pipe * 8 elems * iters exps
                double total_ops = (double)grid * 32.0 * n_pipe * 8.0 * iters;
                double gops = total_ops / (ms * 1e6);
                printf("%-8d %-12d %-12d %-12.4f %-12.2f\n", n_pipe, grid, iters, ms, gops);
            }
        }
    }

    return 0;
}
