#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <hip/hip_runtime.h>
#include <random>
#include <vector>
#include <algorithm>
#include <numeric>

#define HIP_CALL(call) do {                                                 \
    hipError_t err = call;                                                  \
    if (err != hipSuccess) {                                                \
        printf("[HIP ERROR] (%d) %s  at %s:%d\n",                          \
               (int)err, hipGetErrorString(err), __FILE__, __LINE__);       \
        exit(EXIT_FAILURE);                                                 \
    }                                                                       \
} while (0)

// Generate a pointer-chase array where consecutive accesses are stride_bytes apart.
// The chase pointer is stored as the first dword at each stride-aligned position.
// buffer must be pre-allocated to total_bytes / 4 dwords.
static void generate_chase_with_stride(uint32_t* buffer, uint32_t total_bytes,
                                        uint32_t stride_bytes, unsigned seed = 42) {
    for (uint32_t i = 0; i < total_bytes / 4; i++) buffer[i] = 0;
    uint32_t num_nodes = total_bytes / stride_bytes;

    // Sattolo permutation for single-cycle traversal
    std::vector<uint32_t> perm(num_nodes);
    std::iota(perm.begin(), perm.end(), 0);
    std::mt19937 rng(seed);
    for (uint32_t i = num_nodes - 1; i >= 1; i--) {
        std::uniform_int_distribution<uint32_t> dist(0, i - 1);
        std::swap(perm[i], perm[dist(rng)]);
    }

    // Store chase pointers
    for (uint32_t i = 0; i < num_nodes; i++) {
        uint32_t dword_idx = (i * stride_bytes) / 4;
        buffer[dword_idx] = perm[i] * stride_bytes;  // byte offset of next node
    }
}

static double measure_nop_loop(uint32_t num_iters) {
    hipModule_t module;
    hipFunction_t func;
    HIP_CALL(hipModuleLoad(&module, "nop_loop.hsaco"));
    HIP_CALL(hipModuleGetFunction(&func, module, "nop_loop_global_kernel"));

    uint64_t* d_out = nullptr;
    HIP_CALL(hipMalloc(&d_out, 16));
    HIP_CALL(hipMemset(d_out, 0, 16));

    struct __attribute__((packed)) { uint64_t* out; uint32_t iters; uint32_t mode; } args;
    args.out = d_out; args.iters = num_iters; args.mode = 0;
    size_t sz = sizeof(args);
    void* cfg[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END };

    HIP_CALL(hipModuleLaunchKernel(func, 1,1,1, 64,1,1, 0,0, NULL, (void**)&cfg));
    HIP_CALL(hipDeviceSynchronize());

    uint64_t h[2];
    HIP_CALL(hipMemcpy(h, d_out, 16, hipMemcpyDeviceToHost));
    double overhead = (double)(h[1] - h[0]) / num_iters;
    HIP_CALL(hipFree(d_out));
    HIP_CALL(hipModuleUnload(module));
    return overhead;
}

static double measure_global_latency_stride(uint32_t total_bytes, uint32_t stride_bytes,
                                             uint32_t num_iters, uint32_t warmup_iters) {
    hipModule_t module;
    hipFunction_t func;
    HIP_CALL(hipModuleLoad(&module, "global_load_latency.hsaco"));
    HIP_CALL(hipModuleGetFunction(&func, module, "global_load_latency_kernel"));

    uint32_t total_dwords = total_bytes / 4;
    std::vector<uint32_t> h_buf(total_dwords, 0);
    generate_chase_with_stride(h_buf.data(), total_bytes, stride_bytes);

    uint32_t* d_chase = nullptr;
    uint64_t* d_out = nullptr;
    HIP_CALL(hipMalloc(&d_chase, total_bytes));
    HIP_CALL(hipMalloc(&d_out, 16));
    HIP_CALL(hipMemcpy(d_chase, h_buf.data(), total_bytes, hipMemcpyHostToDevice));
    HIP_CALL(hipMemset(d_out, 0, 16));

    struct __attribute__((packed)) {
        uint32_t* chase; uint64_t* out; uint32_t iters; uint32_t warmup;
    } args;
    args.chase = d_chase; args.out = d_out;
    args.iters = num_iters; args.warmup = warmup_iters;

    size_t sz = sizeof(args);
    void* cfg[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END };

    HIP_CALL(hipModuleLaunchKernel(func, 1,1,1, 64,1,1, 0,0, NULL, (void**)&cfg));
    HIP_CALL(hipDeviceSynchronize());

    uint64_t h[2];
    HIP_CALL(hipMemcpy(h, d_out, 16, hipMemcpyDeviceToHost));

    double ticks = (double)(h[1] - h[0]) / num_iters;

    HIP_CALL(hipFree(d_chase));
    HIP_CALL(hipFree(d_out));
    HIP_CALL(hipModuleUnload(module));
    return ticks;
}

static double calibrate_tick_freq() {
    hipModule_t module;
    hipFunction_t func;
    HIP_CALL(hipModuleLoad(&module, "global_load_latency.hsaco"));
    HIP_CALL(hipModuleGetFunction(&func, module, "global_load_latency_kernel"));

    uint32_t N = 4096, iters = 50000;
    std::vector<uint32_t> h_chase(N);
    std::iota(h_chase.begin(), h_chase.end(), 0);
    std::mt19937 rng(42);
    for (uint32_t i = N - 1; i >= 1; i--) {
        std::uniform_int_distribution<uint32_t> d(0, i - 1);
        std::swap(h_chase[i], h_chase[d(rng)]);
    }
    for (uint32_t i = 0; i < N; i++) h_chase[i] *= 4;

    uint32_t* d_chase; uint64_t* d_out;
    HIP_CALL(hipMalloc(&d_chase, N * 4));
    HIP_CALL(hipMalloc(&d_out, 16));
    HIP_CALL(hipMemcpy(d_chase, h_chase.data(), N * 4, hipMemcpyHostToDevice));
    HIP_CALL(hipMemset(d_out, 0, 16));

    struct __attribute__((packed)) {
        uint32_t* chase; uint64_t* out; uint32_t iters; uint32_t warmup;
    } args = { d_chase, d_out, iters, 500 };
    size_t sz = sizeof(args);
    void* cfg[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END };

    HIP_CALL(hipModuleLaunchKernel(func, 1,1,1, 64,1,1, 0,0, NULL, (void**)&cfg));
    HIP_CALL(hipDeviceSynchronize());
    HIP_CALL(hipMemset(d_out, 0, 16));

    hipEvent_t e0, e1;
    HIP_CALL(hipEventCreate(&e0)); HIP_CALL(hipEventCreate(&e1));
    HIP_CALL(hipEventRecord(e0));
    HIP_CALL(hipModuleLaunchKernel(func, 1,1,1, 64,1,1, 0,0, NULL, (void**)&cfg));
    HIP_CALL(hipEventRecord(e1));
    HIP_CALL(hipEventSynchronize(e1));

    float ms; HIP_CALL(hipEventElapsedTime(&ms, e0, e1));
    uint64_t h[2]; HIP_CALL(hipMemcpy(h, d_out, 16, hipMemcpyDeviceToHost));
    double freq = (double)(h[1] - h[0]) / (ms * 1000.0);

    HIP_CALL(hipFree(d_chase)); HIP_CALL(hipFree(d_out));
    HIP_CALL(hipEventDestroy(e0)); HIP_CALL(hipEventDestroy(e1));
    HIP_CALL(hipModuleUnload(module));
    return freq;
}

int main() {
    printf("=== Cache Line Stride Analysis (gfx942 / MI308) ===\n\n");

    HIP_CALL(hipInit(0));
    HIP_CALL(hipSetDevice(0));

    hipDeviceProp_t props;
    HIP_CALL(hipGetDeviceProperties(&props, 0));
    printf("Device: %s (%s)\n", props.name, props.gcnArchName);
    double shader_mhz = props.clockRate / 1000.0;
    printf("Shader clock: %.0f MHz\n\n", shader_mhz);

    double tick_freq = calibrate_tick_freq();
    double tick_to_ns = 1000.0 / tick_freq;
    double tick_to_cy = shader_mhz / tick_freq;
    printf("Clock: %.1f MHz (1 tick = %.2f ns = %.2f shader cy)\n\n",
           tick_freq, tick_to_ns, tick_to_cy);

    double overhead = measure_nop_loop(10000);
    printf("Global loop overhead: %.2f ticks/iter\n\n", overhead);

    uint32_t strides[] = { 4, 8, 16, 32, 64, 128, 256 };
    int num_strides = sizeof(strides) / sizeof(strides[0]);

    // --- L1 cache (16 KB working set) ---
    {
        uint32_t total = 16 * 1024;
        uint32_t iters = 10000;
        printf("L1 Cache Latency vs Access Stride (working set = %u KB)\n", total / 1024);
        printf("====================================================================\n");
        printf("  %8s | %8s | %8s | %8s | %8s | %s\n",
               "Stride", "Raw(tick)", "Net(tick)", "ns", "Shdr(cy)", "Nodes");
        printf("  %8s-+-%8s-+-%8s-+-%8s-+-%8s-+-%s\n",
               "--------", "--------", "--------", "--------", "--------", "--------");

        for (int i = 0; i < num_strides; i++) {
            uint32_t stride = strides[i];
            if (stride > total) break;
            uint32_t nodes = total / stride;
            if (nodes < 2) break;

            double raw = measure_global_latency_stride(total, stride, iters, 500);
            double net = raw - overhead;
            printf("  %6u B | %8.2f | %8.2f | %8.1f | %8.0f | %u\n",
                   stride, raw, net, net * tick_to_ns, net * tick_to_cy, nodes);
        }
        printf("====================================================================\n\n");
    }

    // --- L2 cache (256 KB working set) ---
    {
        uint32_t total = 256 * 1024;
        uint32_t iters = 5000;
        printf("L2 Cache Latency vs Access Stride (working set = %u KB)\n", total / 1024);
        printf("====================================================================\n");
        printf("  %8s | %8s | %8s | %8s | %8s | %s\n",
               "Stride", "Raw(tick)", "Net(tick)", "ns", "Shdr(cy)", "Nodes");
        printf("  %8s-+-%8s-+-%8s-+-%8s-+-%8s-+-%s\n",
               "--------", "--------", "--------", "--------", "--------", "--------");

        for (int i = 0; i < num_strides; i++) {
            uint32_t stride = strides[i];
            if (stride > total) break;
            uint32_t nodes = total / stride;
            if (nodes < 2) break;

            double raw = measure_global_latency_stride(total, stride, iters, 500);
            double net = raw - overhead;
            printf("  %6u B | %8.2f | %8.2f | %8.1f | %8.0f | %u\n",
                   stride, raw, net, net * tick_to_ns, net * tick_to_cy, nodes);
        }
        printf("====================================================================\n\n");
    }

    printf("Note: A jump in latency between stride sizes reveals the cache line boundary.\n");
    printf("      If latency jumps at stride=X, the cache line size is likely X bytes.\n");

    HIP_CALL(hipDeviceReset());
    return 0;
}
