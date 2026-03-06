#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
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

// Generate chase array with configurable stride between entries.
// For b32: stride=4, b64: stride=8, b128: stride=16.
// Each entry's first dword holds the byte offset of the next entry.
// Total LDS usage: num_entries * entry_stride bytes.
static void generate_chase_array_strided(uint32_t* arr, uint32_t total_dwords,
                                          uint32_t num_entries, uint32_t entry_stride,
                                          unsigned seed = 42) {
    // Zero-fill the entire buffer
    for (uint32_t i = 0; i < total_dwords; i++) arr[i] = 0;

    // Create Sattolo permutation
    std::vector<uint32_t> perm(num_entries);
    std::iota(perm.begin(), perm.end(), 0);
    std::mt19937 rng(seed);
    for (uint32_t i = num_entries - 1; i >= 1; i--) {
        std::uniform_int_distribution<uint32_t> dist(0, i - 1);
        std::swap(perm[i], perm[dist(rng)]);
    }

    // Store chase pointers: entry i is at byte offset i*entry_stride
    // Its first dword = perm[i] * entry_stride (byte offset of next entry)
    for (uint32_t i = 0; i < num_entries; i++) {
        uint32_t dword_idx = (i * entry_stride) / 4;
        arr[dword_idx] = perm[i] * entry_stride;
    }
}

struct LatResult {
    double ticks_per_iter;
};

static double measure_nop_loop(const char* kernel_name, uint32_t num_iters) {
    hipModule_t module;
    hipFunction_t func;
    HIP_CALL(hipModuleLoad(&module, "nop_loop.hsaco"));
    HIP_CALL(hipModuleGetFunction(&func, module, kernel_name));

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

// Measure LDS read latency for a given kernel (pointer chase)
static LatResult measure_lds_read(const char* kernel_name, uint32_t entry_stride,
                                   uint32_t num_entries, uint32_t num_iters) {
    hipModule_t module;
    hipFunction_t func;
    HIP_CALL(hipModuleLoad(&module, "lds_detailed.hsaco"));
    HIP_CALL(hipModuleGetFunction(&func, module, kernel_name));

    uint32_t total_dwords = (num_entries * entry_stride) / 4;
    std::vector<uint32_t> h_chase(total_dwords, 0);
    generate_chase_array_strided(h_chase.data(), total_dwords, num_entries, entry_stride);

    uint32_t* d_chase = nullptr;
    uint64_t* d_out = nullptr;
    HIP_CALL(hipMalloc(&d_chase, total_dwords * 4));
    HIP_CALL(hipMalloc(&d_out, 16));
    HIP_CALL(hipMemcpy(d_chase, h_chase.data(), total_dwords * 4, hipMemcpyHostToDevice));
    HIP_CALL(hipMemset(d_out, 0, 16));

    struct __attribute__((packed)) {
        uint32_t* chase; uint64_t* out; uint32_t num_entries; uint32_t num_iters;
    } args;
    args.chase = d_chase; args.out = d_out;
    args.num_entries = total_dwords;  // total dwords to copy to LDS
    args.num_iters = num_iters;

    size_t sz = sizeof(args);
    void* cfg[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END };

    HIP_CALL(hipModuleLaunchKernel(func, 1,1,1, 64,1,1, 0,0, NULL, (void**)&cfg));
    HIP_CALL(hipDeviceSynchronize());

    uint64_t h[2];
    HIP_CALL(hipMemcpy(h, d_out, 16, hipMemcpyDeviceToHost));

    LatResult r;
    r.ticks_per_iter = (double)(h[1] - h[0]) / num_iters;

    HIP_CALL(hipFree(d_chase));
    HIP_CALL(hipFree(d_out));
    HIP_CALL(hipModuleUnload(module));
    return r;
}

// Measure LDS write latency (no chase array needed)
static LatResult measure_lds_write(const char* kernel_name, uint32_t num_iters) {
    hipModule_t module;
    hipFunction_t func;
    HIP_CALL(hipModuleLoad(&module, "lds_detailed.hsaco"));
    HIP_CALL(hipModuleGetFunction(&func, module, kernel_name));

    uint64_t* d_out = nullptr;
    HIP_CALL(hipMalloc(&d_out, 16));
    HIP_CALL(hipMemset(d_out, 0, 16));

    // Write kernels still use same kernarg layout; chase_array unused
    struct __attribute__((packed)) {
        void* chase; uint64_t* out; uint32_t num_entries; uint32_t num_iters;
    } args;
    args.chase = nullptr; args.out = d_out;
    args.num_entries = 0; args.num_iters = num_iters;

    size_t sz = sizeof(args);
    void* cfg[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END };

    HIP_CALL(hipModuleLaunchKernel(func, 1,1,1, 64,1,1, 0,0, NULL, (void**)&cfg));
    HIP_CALL(hipDeviceSynchronize());

    uint64_t h[2];
    HIP_CALL(hipMemcpy(h, d_out, 16, hipMemcpyDeviceToHost));

    LatResult r;
    r.ticks_per_iter = (double)(h[1] - h[0]) / num_iters;

    HIP_CALL(hipFree(d_out));
    HIP_CALL(hipModuleUnload(module));
    return r;
}

// Calibrate s_memrealtime tick frequency
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

    // Warmup
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
    printf("=== LDS Detailed Latency Benchmark (gfx942 / MI308) ===\n\n");

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
    printf("Clock calibration: %.1f MHz (1 tick = %.2f ns = %.2f shader cy)\n\n",
           tick_freq, tick_to_ns, tick_to_cy);

    double overhead = measure_nop_loop("nop_loop_lds_kernel", 10000);
    printf("Loop overhead: %.2f ticks/iter\n\n", overhead);

    uint32_t num_iters = 10000;

    // LDS entries: use 256 entries for b32, 128 for b64, 64 for b128
    // to keep total LDS usage reasonable (1KB each)
    struct TestConfig {
        const char* name;
        const char* kernel;
        uint32_t entry_stride;
        uint32_t num_entries;
        bool is_read;
    };

    TestConfig tests[] = {
        { "ds_read_b32",   "lds_read_b32_kernel",   4,  256, true  },
        { "ds_read_b64",   "lds_read_b64_kernel",   8,  256, true  },
        { "ds_read_b128",  "lds_read_b128_kernel",  16, 256, true  },
        { "ds_write_b32",  "lds_write_b32_kernel",  0,  0,   false },
        { "ds_write_b64",  "lds_write_b64_kernel",  0,  0,   false },
        { "ds_write_b128", "lds_write_b128_kernel", 0,  0,   false },
    };

    printf("====================================================================\n");
    printf("  %-16s | %8s | %8s | %8s | %8s\n",
           "Instruction", "Raw(tick)", "Net(tick)", "ns", "Shdr(cy)");
    printf("  %-16s-+-%8s-+-%8s-+-%8s-+-%8s\n",
           "----------------", "--------", "--------", "--------", "--------");

    for (auto& t : tests) {
        LatResult r;
        if (t.is_read)
            r = measure_lds_read(t.kernel, t.entry_stride, t.num_entries, num_iters);
        else
            r = measure_lds_write(t.kernel, num_iters);

        double net = r.ticks_per_iter - overhead;
        printf("  %-16s | %8.2f | %8.2f | %8.1f | %8.0f\n",
               t.name, r.ticks_per_iter, net, net * tick_to_ns, net * tick_to_cy);
    }
    printf("====================================================================\n");
    printf("  Net = Raw - loop overhead (%.2f ticks)\n", overhead);

    HIP_CALL(hipDeviceReset());
    return 0;
}
