#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <hip/hip_runtime.h>
#include "opus/opus.hpp"

// HIP error checking macro
#define CHECK(call) {                                                     \
    hipError_t err = call;                                                \
    if (err != hipSuccess) {                                              \
        printf("HIP Error in %s at line %d: %s\n", __FILE__, __LINE__,    \
               hipGetErrorString(err));                                   \
        exit(EXIT_FAILURE);                                               \
    }                                                                     \
}

#if 0
// Kernel for element-wise addition
__global__ void addVectors(const float* a, const float* b, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Grid-stride loop to handle any size array
    for (int i = idx; i < n; i += gridDim.x * blockDim.x) {
        result[i] = a[i] + b[i];
    }
}
#endif

OPUS_USING_COMMON_TYPES_ALL

#if 0
template<int BLOCK_SIZE>
__global__ void addVectors_async(const float* a, const float* b, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float smem[BLOCK_SIZE * 2];

    auto g_a = opus::make_gmem(a);

    int wave_id = threadIdx.x / 64;
    int lane_id = threadIdx.x % 64;

    // Grid-stride loop to handle any size array
    for (int i = idx; i < n; i += gridDim.x * blockDim.x) {
        // auto x = g_a.load(i);//a[i];
        g_a.async_load(smem + wave_id * 65 + lane_id, i);
        auto x = smem[wave_id * 65 + lane_id];
        auto y = b[i];
        result[i] = x + y;
    }
}
#endif
OPUS_USING_COMMON_TYPES_ALL

// template<typename... T> __host__ __device__ tup(T&&...) -> opus::tuple<opus::remove_cvref_t<T>...>;

template<int BLOCK_SIZE>
__global__ void addVectors_async(const float* a, const float* b, float* result, int n) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int stride = gridDim.x * BLOCK_SIZE;

    constexpr int n_waves = BLOCK_SIZE / 64;
    int wave_id = threadIdx.x / 64;
    int lane_id = threadIdx.x % 64;

    // NOTE! below declaration will not have correct result!
    // __shared__ float smem[n_waves * 65 * 2];
    // float* smem_1 = smem + wave_id * 65 + lane_id;
    // float* smem_2 = smem + wave_id * 65 + lane_id + n_waves * 65;

    // NOTE! if using multiple buffer, must use multiple __shared__ variable
    // otherwise the compiler may not generate proper dependency (vmcnt)
    __shared__ float smem_1_[n_waves * 65];
    __shared__ float smem_2_[n_waves * 65];
    float* smem_1 = smem_1_ + wave_id * 65; // NOTE: per-wave smem pointer passed to async_load API
    float* smem_2 = smem_2_ + wave_id * 65; // NOTE: per-wave smem pointer passed to async_load API

    auto g_a = opus::make_gmem(a);
    auto g_b = opus::make_gmem(b);
    auto g_r = opus::make_gmem(result);

    int num_loops = (n + stride - 1) / stride;

    g_a.async_load(smem_1, idx);

    int i = 0;
    for ( ; i < num_loops / 2 - 1; i++) {
        auto y0 = g_b.load(idx + (2 * i + 0) * stride);
        __builtin_amdgcn_sched_group_barrier(0x0020, 1, 0); // 1x VMEM read
        g_a.async_load(smem_2, idx + (2 * i + 1) * stride);
        __builtin_amdgcn_sched_group_barrier(0x0020, 1, 0); // 1x VMEM read

        opus::s_waitcnt_vmcnt(2_I);
        auto x0 = smem_1[lane_id];

        opus::s_waitcnt_vmcnt(1_I);
        g_r.store(x0 + y0[0], idx + (2 * i + 0) * stride);
        __builtin_amdgcn_sched_group_barrier(0x0040, 1, 0); // 1x VMEM write

        auto y1 = g_b.load(idx + (2 * i + 1) * stride);
        __builtin_amdgcn_sched_group_barrier(0x0020, 1, 0); // 1x VMEM read
        g_a.async_load(smem_1, idx + (2 * i + 2) * stride);
        __builtin_amdgcn_sched_group_barrier(0x0020, 1, 0); // 1x VMEM read

        opus::s_waitcnt_vmcnt(2_I);  // consume the write vmcnt
        auto x1 = smem_2[lane_id];

        opus::s_waitcnt_vmcnt(1_I);
        g_r.store(x1 + y1[0], idx + (2 * i + 1) * stride);
        __builtin_amdgcn_sched_group_barrier(0x0040, 1, 0); // 1x VMEM write
    }

    g_a.async_load(smem_2, idx + (2 * i + 1) * stride);
    auto y0 = g_b.load(idx + (2 * i + 0) * stride);
    auto x0 = smem_1[lane_id];

    result[idx + (2 * i + 0) * stride] = x0 + y0[0];
    auto y1 = g_b.load(idx + (2 * i + 1) * stride);
    auto x1 = smem_2[lane_id];

    result[idx + (2 * i + 1) * stride] = x1 + y1[0];
}

// Host function with full error checking
void launchVectorAdd(int n) {
    size_t size = n * sizeof(float);
    
    // Allocate host memory
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_result = (float*)malloc(size);
    
    if (!h_a || !h_b || !h_result) {
        printf("Host memory allocation failed!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize host arrays
    for (int i = 0; i < n; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
    }

    // Allocate device memory with error checking
    float *d_a = nullptr, *d_b = nullptr, *d_result = nullptr;
    CHECK(hipMalloc(&d_a, size));
    CHECK(hipMalloc(&d_b, size));
    CHECK(hipMalloc(&d_result, size));
    
    // Copy data to device
    CHECK(hipMemcpy(d_a, h_a, size, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_b, h_b, size, hipMemcpyHostToDevice));
    
    // Configure kernel launch parameters
    constexpr int threadsPerBlock = 256;
    // int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGrid = 80;
    
    // Validate launch parameters
    if (threadsPerBlock > 1024) {  // Typical max threads per block
        printf("Error: threadsPerBlock (%d) exceeds hardware limit\n", threadsPerBlock);
        exit(EXIT_FAILURE);
    }
    
    printf("Launching kernel with %d blocks, %d threads per block\n", 
           blocksPerGrid, threadsPerBlock);
    
    // Launch kernel with error checking
    addVectors_async<threadsPerBlock><<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_result, n);
    
    // Check for kernel launch errors
    CHECK(hipGetLastError());
    
    // Wait for kernel completion
    CHECK(hipDeviceSynchronize());
    
    // Copy result back to host
    CHECK(hipMemcpy(h_result, d_result, size, hipMemcpyDeviceToHost));
    
    // Verify results
    int errors = 0;
    for (int i = 0; i < n; i++) {
        float expected = h_a[i] + h_b[i];
        if (fabs(h_result[i] - expected) > 1e-5) {
            errors++;
            if (errors < 10) {
                printf("Error at index %d: %f + %f = %f (expected %f)\n",
                       i, h_a[i], h_b[i], h_result[i], expected);
            }
        }
    }
    
    if (errors == 0) {
        printf("Success! All %d elements added correctly.\n", n);
    } else {
        printf("Found %d errors\n", errors);
    }
    
    // Cleanup with error checking
    free(h_a);
    free(h_b);
    free(h_result);
    CHECK(hipFree(d_a));
    CHECK(hipFree(d_b));
    CHECK(hipFree(d_result));
}

int main() {
    int n = 1310720;  // 1 million elements
    
    // Initialize HIP
    CHECK(hipInit(0));
    
    // Get device count and info
    int deviceCount = 0;
    CHECK(hipGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        printf("Error: No HIP devices found!\n");
        return EXIT_FAILURE;
    }
    
    printf("Found %d HIP device(s):\n", deviceCount);
    
    // Print device info for each device
    for (int i = 0; i < deviceCount; i++) {
        hipDeviceProp_t props;
        CHECK(hipGetDeviceProperties(&props, i));
        printf("  Device %d: %s\n", i, props.name);
    }
    
    // Set device 0 as current
    CHECK(hipSetDevice(0));
    
    // Get current device info
    hipDeviceProp_t props;
    CHECK(hipGetDeviceProperties(&props, 0));
    
    printf("\nUsing device: %s\n", props.name);
    printf("  Global Memory: %.1f GB\n", props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Compute Capability: %d.%d\n", props.major, props.minor);
    printf("  Max Threads per Block: %d\n", props.maxThreadsPerBlock);
    printf("  Max Threads Dim: %d x %d x %d\n", 
           props.maxThreadsDim[0], props.maxThreadsDim[1], props.maxThreadsDim[2]);
    printf("  Max Grid Size: %d x %d x %d\n", 
           props.maxGridSize[0], props.maxGridSize[1], props.maxGridSize[2]);
    
    // Launch vector addition
    printf("\nRunning vector addition on %d elements...\n", n);
    launchVectorAdd(n);
    
    // Reset device
    CHECK(hipDeviceReset());
    
    printf("\nProgram completed successfully!\n");
    return EXIT_SUCCESS;
}