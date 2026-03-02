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

// Generate a random single-cycle permutation stored as byte offsets (index * 4)
// Uses Sattolo's algorithm to guarantee a single cycle visiting all entries.
static void generate_chase_array(uint32_t* arr, uint32_t num_entries, unsigned seed = 42) {
    std::vector<uint32_t> perm(num_entries);
    std::iota(perm.begin(), perm.end(), 0);

    std::mt19937 rng(seed);
    for (uint32_t i = num_entries - 1; i >= 1; i--) {
        std::uniform_int_distribution<uint32_t> dist(0, i - 1);
        uint32_t j = dist(rng);
        std::swap(perm[i], perm[j]);
    }

    for (uint32_t i = 0; i < num_entries; i++) {
        arr[i] = perm[i] * 4;  // byte offset
    }
}

struct LatencyResult {
    double cycles;      // ticks per iteration
    uint64_t start;
    uint64_t end;
    uint32_t num_iters;
};

// Measure nop loop overhead (no memory access, just loop control)
static double measure_nop_loop(const char* kernel_name, uint32_t num_iters) {
    hipModule_t module;
    hipFunction_t kernel_func;

    HIP_CALL(hipModuleLoad(&module, "nop_loop.hsaco"));
    HIP_CALL(hipModuleGetFunction(&kernel_func, module, kernel_name));

    uint64_t* d_output = nullptr;
    HIP_CALL(hipMalloc(&d_output, 2 * sizeof(uint64_t)));
    HIP_CALL(hipMemset(d_output, 0, 2 * sizeof(uint64_t)));

    struct __attribute__((packed)) {
        uint64_t* output;
        uint32_t  num_iters;
        uint32_t  mode;
    } args;
    args.output = d_output;
    args.num_iters = num_iters;
    args.mode = 0;

    size_t arg_size = sizeof(args);
    void* config[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
        HIP_LAUNCH_PARAM_BUFFER_SIZE, &arg_size,
        HIP_LAUNCH_PARAM_END
    };

    HIP_CALL(hipModuleLaunchKernel(kernel_func, 1, 1, 1, 64, 1, 1, 0, 0, NULL, (void**)&config));
    HIP_CALL(hipDeviceSynchronize());

    uint64_t h_output[2];
    HIP_CALL(hipMemcpy(h_output, d_output, 2 * sizeof(uint64_t), hipMemcpyDeviceToHost));

    double overhead = (double)(h_output[1] - h_output[0]) / (double)num_iters;

    HIP_CALL(hipFree(d_output));
    HIP_CALL(hipModuleUnload(module));
    return overhead;
}

// Run LDS latency benchmark
static LatencyResult measure_lds_latency(uint32_t num_entries, uint32_t num_iters) {
    hipModule_t module;
    hipFunction_t kernel_func;

    HIP_CALL(hipModuleLoad(&module, "lds_latency.hsaco"));
    HIP_CALL(hipModuleGetFunction(&kernel_func, module, "lds_latency_kernel"));

    std::vector<uint32_t> h_chase(num_entries);
    generate_chase_array(h_chase.data(), num_entries);

    uint32_t* d_chase = nullptr;
    uint64_t* d_output = nullptr;
    HIP_CALL(hipMalloc(&d_chase, num_entries * sizeof(uint32_t)));
    HIP_CALL(hipMalloc(&d_output, 2 * sizeof(uint64_t)));
    HIP_CALL(hipMemcpy(d_chase, h_chase.data(), num_entries * sizeof(uint32_t), hipMemcpyHostToDevice));
    HIP_CALL(hipMemset(d_output, 0, 2 * sizeof(uint64_t)));

    struct __attribute__((packed)) {
        uint32_t* chase_array;
        uint64_t* output;
        uint32_t  num_entries;
        uint32_t  num_iters;
    } args;
    args.chase_array = d_chase;
    args.output = d_output;
    args.num_entries = num_entries;
    args.num_iters = num_iters;

    size_t arg_size = sizeof(args);
    void* config[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
        HIP_LAUNCH_PARAM_BUFFER_SIZE, &arg_size,
        HIP_LAUNCH_PARAM_END
    };

    HIP_CALL(hipModuleLaunchKernel(kernel_func, 1, 1, 1, 64, 1, 1, 0, 0, NULL, (void**)&config));
    HIP_CALL(hipDeviceSynchronize());

    uint64_t h_output[2];
    HIP_CALL(hipMemcpy(h_output, d_output, 2 * sizeof(uint64_t), hipMemcpyDeviceToHost));

    LatencyResult result;
    result.start = h_output[0];
    result.end = h_output[1];
    result.num_iters = num_iters;
    result.cycles = (double)(result.end - result.start) / (double)num_iters;

    HIP_CALL(hipFree(d_chase));
    HIP_CALL(hipFree(d_output));
    HIP_CALL(hipModuleUnload(module));
    return result;
}

// Run global memory latency benchmark
// warmup_iters: number of in-kernel warmup iterations (set high for L1/L2, 0 for HBM)
static LatencyResult measure_global_latency(uint32_t num_entries, uint32_t num_iters,
                                             uint32_t warmup_iters) {
    hipModule_t module;
    hipFunction_t kernel_func;

    HIP_CALL(hipModuleLoad(&module, "global_load_latency.hsaco"));
    HIP_CALL(hipModuleGetFunction(&kernel_func, module, "global_load_latency_kernel"));

    std::vector<uint32_t> h_chase(num_entries);
    generate_chase_array(h_chase.data(), num_entries);

    uint32_t* d_chase = nullptr;
    uint64_t* d_output = nullptr;
    HIP_CALL(hipMalloc(&d_chase, (size_t)num_entries * sizeof(uint32_t)));
    HIP_CALL(hipMalloc(&d_output, 2 * sizeof(uint64_t)));
    HIP_CALL(hipMemcpy(d_chase, h_chase.data(), (size_t)num_entries * sizeof(uint32_t), hipMemcpyHostToDevice));
    HIP_CALL(hipMemset(d_output, 0, 2 * sizeof(uint64_t)));

    // For L1/L2: do extra warmup by launching a warmup kernel first
    if (warmup_iters > 0) {
        struct __attribute__((packed)) {
            uint32_t* chase_array;
            uint64_t* output;
            uint32_t  num_iters;
            uint32_t  warmup_iters;
        } warmup_args;
        warmup_args.chase_array = d_chase;
        warmup_args.output = d_output;
        warmup_args.num_iters = warmup_iters;
        warmup_args.warmup_iters = warmup_iters;

        size_t warg_size = sizeof(warmup_args);
        void* wconfig[] = {
            HIP_LAUNCH_PARAM_BUFFER_POINTER, &warmup_args,
            HIP_LAUNCH_PARAM_BUFFER_SIZE, &warg_size,
            HIP_LAUNCH_PARAM_END
        };
        HIP_CALL(hipModuleLaunchKernel(kernel_func, 1, 1, 1, 64, 1, 1, 0, 0, NULL, (void**)&wconfig));
        HIP_CALL(hipDeviceSynchronize());
    }

    HIP_CALL(hipMemset(d_output, 0, 2 * sizeof(uint64_t)));

    struct __attribute__((packed)) {
        uint32_t* chase_array;
        uint64_t* output;
        uint32_t  num_iters;
        uint32_t  warmup_iters;
    } args;
    args.chase_array = d_chase;
    args.output = d_output;
    args.num_iters = num_iters;
    args.warmup_iters = warmup_iters;

    size_t arg_size = sizeof(args);
    void* config[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
        HIP_LAUNCH_PARAM_BUFFER_SIZE, &arg_size,
        HIP_LAUNCH_PARAM_END
    };

    HIP_CALL(hipModuleLaunchKernel(kernel_func, 1, 1, 1, 64, 1, 1, 0, 0, NULL, (void**)&config));
    HIP_CALL(hipDeviceSynchronize());

    uint64_t h_output[2];
    HIP_CALL(hipMemcpy(h_output, d_output, 2 * sizeof(uint64_t), hipMemcpyDeviceToHost));

    LatencyResult result;
    result.start = h_output[0];
    result.end = h_output[1];
    result.num_iters = num_iters;
    result.cycles = (double)(result.end - result.start) / (double)num_iters;

    HIP_CALL(hipFree(d_chase));
    HIP_CALL(hipFree(d_output));
    HIP_CALL(hipModuleUnload(module));
    return result;
}

int main(int argc, char** argv) {
    printf("=== AMDGPU Memory Latency Micro-Benchmark (gfx942 / MI308) ===\n\n");

    HIP_CALL(hipInit(0));

    int device_count = 0;
    HIP_CALL(hipGetDeviceCount(&device_count));
    if (device_count == 0) {
        printf("ERROR: No HIP devices found!\n");
        return EXIT_FAILURE;
    }
    HIP_CALL(hipSetDevice(0));

    hipDeviceProp_t props;
    HIP_CALL(hipGetDeviceProperties(&props, 0));
    printf("Device:        %s\n", props.name);
    printf("GCN Arch:      %s\n", props.gcnArchName);
    printf("Compute Units: %d\n", props.multiProcessorCount);
    printf("Clock Rate:    %d MHz\n", props.clockRate / 1000);
    printf("\n");

    // ---------------------------------------------------------------
    // Measure loop overhead (nop loops)
    // ---------------------------------------------------------------
    printf("Measuring loop overhead (nop loops)...\n");
    double lds_overhead = measure_nop_loop("nop_loop_lds_kernel", 10000);
    double global_overhead = measure_nop_loop("nop_loop_global_kernel", 10000);
    printf("  LDS-style loop overhead:    %.2f ticks/iter\n", lds_overhead);
    printf("  Global-style loop overhead: %.2f ticks/iter\n\n", global_overhead);

    // ---------------------------------------------------------------
    // LDS Latency (1 KB working set)
    // ---------------------------------------------------------------
    printf("Measuring LDS latency (256 entries = 1 KB)...\n");
    LatencyResult lds = measure_lds_latency(256, 10000);
    printf("  Raw: %.1f ticks/iter, overhead: %.1f, net: %.1f ticks\n\n",
           lds.cycles, lds_overhead, lds.cycles - lds_overhead);

    // ---------------------------------------------------------------
    // L1 Cache Latency (16 KB working set, fits in 32KB L1)
    // ---------------------------------------------------------------
    printf("Measuring L1 cache latency (4096 entries = 16 KB)...\n");
    LatencyResult l1 = measure_global_latency(4096, 10000, 500);
    printf("  Raw: %.1f ticks/iter, overhead: %.1f, net: %.1f ticks\n\n",
           l1.cycles, global_overhead, l1.cycles - global_overhead);

    // ---------------------------------------------------------------
    // L2 Cache Latency (256 KB working set, exceeds L1, fits in L2)
    // ---------------------------------------------------------------
    printf("Measuring L2 cache latency (65536 entries = 256 KB)...\n");
    LatencyResult l2 = measure_global_latency(65536, 5000, 500);
    printf("  Raw: %.1f ticks/iter, overhead: %.1f, net: %.1f ticks\n\n",
           l2.cycles, global_overhead, l2.cycles - global_overhead);

    // ---------------------------------------------------------------
    // Global/HBM Latency (512 MB, exceeds L2)
    // ---------------------------------------------------------------
    printf("Measuring Global/HBM latency (128M entries = 512 MB)...\n");
    LatencyResult hbm = measure_global_latency(128 * 1024 * 1024, 2000, 0);
    printf("  Raw: %.1f ticks/iter, overhead: %.1f, net: %.1f ticks\n\n",
           hbm.cycles, global_overhead, hbm.cycles - global_overhead);

    // ---------------------------------------------------------------
    // Clock calibration
    // ---------------------------------------------------------------
    double shader_clock_mhz = props.clockRate / 1000.0;

    {
        hipModule_t cal_module;
        hipFunction_t cal_func;
        HIP_CALL(hipModuleLoad(&cal_module, "global_load_latency.hsaco"));
        HIP_CALL(hipModuleGetFunction(&cal_func, cal_module, "global_load_latency_kernel"));

        uint32_t cal_entries = 4096;
        uint32_t cal_iters = 50000;
        std::vector<uint32_t> cal_chase(cal_entries);
        generate_chase_array(cal_chase.data(), cal_entries);

        uint32_t* d_cal_chase = nullptr;
        uint64_t* d_cal_out = nullptr;
        HIP_CALL(hipMalloc(&d_cal_chase, cal_entries * sizeof(uint32_t)));
        HIP_CALL(hipMalloc(&d_cal_out, 2 * sizeof(uint64_t)));
        HIP_CALL(hipMemcpy(d_cal_chase, cal_chase.data(), cal_entries * sizeof(uint32_t), hipMemcpyHostToDevice));
        HIP_CALL(hipMemset(d_cal_out, 0, 2 * sizeof(uint64_t)));

        struct __attribute__((packed)) {
            uint32_t* chase_array;
            uint64_t* output;
            uint32_t  num_iters;
            uint32_t  warmup_iters;
        } cal_args;
        cal_args.chase_array = d_cal_chase;
        cal_args.output = d_cal_out;
        cal_args.num_iters = cal_iters;
        cal_args.warmup_iters = 500;

        size_t cal_arg_size = sizeof(cal_args);
        void* cal_config[] = {
            HIP_LAUNCH_PARAM_BUFFER_POINTER, &cal_args,
            HIP_LAUNCH_PARAM_BUFFER_SIZE, &cal_arg_size,
            HIP_LAUNCH_PARAM_END
        };

        HIP_CALL(hipModuleLaunchKernel(cal_func, 1, 1, 1, 64, 1, 1, 0, 0, NULL, (void**)&cal_config));
        HIP_CALL(hipDeviceSynchronize());
        HIP_CALL(hipMemset(d_cal_out, 0, 2 * sizeof(uint64_t)));

        hipEvent_t evt0, evt1;
        HIP_CALL(hipEventCreate(&evt0));
        HIP_CALL(hipEventCreate(&evt1));
        HIP_CALL(hipEventRecord(evt0));
        HIP_CALL(hipModuleLaunchKernel(cal_func, 1, 1, 1, 64, 1, 1, 0, 0, NULL, (void**)&cal_config));
        HIP_CALL(hipEventRecord(evt1));
        HIP_CALL(hipEventSynchronize(evt1));

        float elapsed_ms;
        HIP_CALL(hipEventElapsedTime(&elapsed_ms, evt0, evt1));
        uint64_t cal_output[2];
        HIP_CALL(hipMemcpy(cal_output, d_cal_out, 2 * sizeof(uint64_t), hipMemcpyDeviceToHost));
        uint64_t cal_ticks = cal_output[1] - cal_output[0];

        double tick_freq_mhz = (double)cal_ticks / (elapsed_ms * 1000.0);
        double tick_to_ns = 1000.0 / tick_freq_mhz;
        double tick_to_shader = shader_clock_mhz / tick_freq_mhz;

        printf("Clock calibration:\n");
        printf("  s_memrealtime freq: %.1f MHz  (1 tick = %.2f ns)\n", tick_freq_mhz, tick_to_ns);
        printf("  Shader clock:       %.0f MHz  (1 tick = %.2f shader cycles)\n\n",
               shader_clock_mhz, tick_to_shader);

        HIP_CALL(hipFree(d_cal_chase));
        HIP_CALL(hipFree(d_cal_out));
        HIP_CALL(hipEventDestroy(evt0));
        HIP_CALL(hipEventDestroy(evt1));
        HIP_CALL(hipModuleUnload(cal_module));

        // Compute net latencies (raw - overhead)
        double lds_net  = lds.cycles - lds_overhead;
        double l1_net   = l1.cycles - global_overhead;
        double l2_net   = l2.cycles - global_overhead;
        double hbm_net  = hbm.cycles - global_overhead;

        printf("====================================================================\n");
        printf("  Memory Latency Summary (MI308X, gfx942 @ %.0f MHz)\n", shader_clock_mhz);
        printf("====================================================================\n");
        printf("  %-16s | %8s | %8s | %8s | %8s | %s\n",
               "Memory Level", "Raw(tick)", "Net(tick)", "ns", "Shdr(cy)", "WorkSet");
        printf("  %-16s-+-%8s-+-%8s-+-%8s-+-%8s-+-%s\n",
               "----------------", "--------", "--------", "--------", "--------", "--------");
        printf("  %-16s | %8.1f | %8.1f | %8.1f | %8.0f | %s\n",
               "LDS",          lds.cycles, lds_net, lds_net*tick_to_ns, lds_net*tick_to_shader, "1 KB");
        printf("  %-16s | %8.1f | %8.1f | %8.1f | %8.0f | %s\n",
               "L1 Cache",     l1.cycles,  l1_net,  l1_net*tick_to_ns,  l1_net*tick_to_shader,  "16 KB");
        printf("  %-16s | %8.1f | %8.1f | %8.1f | %8.0f | %s\n",
               "L2 Cache",     l2.cycles,  l2_net,  l2_net*tick_to_ns,  l2_net*tick_to_shader,  "256 KB");
        printf("  %-16s | %8.1f | %8.1f | %8.1f | %8.0f | %s\n",
               "Global (HBM)", hbm.cycles, hbm_net, hbm_net*tick_to_ns, hbm_net*tick_to_shader, "512 MB");
        printf("====================================================================\n");
        printf("  Net = Raw - loop overhead (LDS: %.1f ticks, Global: %.1f ticks)\n",
               lds_overhead, global_overhead);
    }

    HIP_CALL(hipDeviceReset());
    return EXIT_SUCCESS;
}
