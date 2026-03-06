#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <hip/hip_runtime.h>

#define HIP_CALL(call) do {                                                 \
    hipError_t err = call;                                                  \
    if (err != hipSuccess) {                                                \
        printf("[HIP ERROR] (%d) %s  at %s:%d\n",                          \
               (int)err, hipGetErrorString(err), __FILE__, __LINE__);       \
        exit(EXIT_FAILURE);                                                 \
    }                                                                       \
} while (0)

// Calibrate s_memrealtime tick frequency using HIP event timing
static double calibrate_tick_freq() {
    hipModule_t module;
    hipFunction_t func;
    HIP_CALL(hipModuleLoad(&module, "lds_throughput.hsaco"));
    HIP_CALL(hipModuleGetFunction(&func, module, "lds_tp_nop_kernel"));

    uint64_t* d_out = nullptr;
    HIP_CALL(hipMalloc(&d_out, 16));
    uint64_t zeros[2] = {0, 0};
    HIP_CALL(hipMemcpy(d_out, zeros, 16, hipMemcpyHostToDevice));

    uint32_t iters = 100000;
    struct __attribute__((packed)) { uint64_t* out; uint32_t iters; } args;
    args.out = d_out; args.iters = iters;
    size_t sz = sizeof(args);
    void* cfg[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END };

    // Warmup
    HIP_CALL(hipModuleLaunchKernel(func, 1,1,1, 64,1,1, 0,0, NULL, (void**)&cfg));
    HIP_CALL(hipDeviceSynchronize());
    HIP_CALL(hipMemcpy(d_out, zeros, 16, hipMemcpyHostToDevice));

    hipEvent_t e0, e1;
    HIP_CALL(hipEventCreate(&e0));
    HIP_CALL(hipEventCreate(&e1));
    HIP_CALL(hipEventRecord(e0));
    HIP_CALL(hipModuleLaunchKernel(func, 1,1,1, 64,1,1, 0,0, NULL, (void**)&cfg));
    HIP_CALL(hipEventRecord(e1));
    HIP_CALL(hipEventSynchronize(e1));

    float ms;
    HIP_CALL(hipEventElapsedTime(&ms, e0, e1));
    uint64_t h[2];
    HIP_CALL(hipMemcpy(h, d_out, 16, hipMemcpyDeviceToHost));
    double freq = (double)(h[1] - h[0]) / (ms * 1000.0);

    HIP_CALL(hipFree(d_out));
    HIP_CALL(hipEventDestroy(e0));
    HIP_CALL(hipEventDestroy(e1));
    HIP_CALL(hipModuleUnload(module));
    return freq;
}

// Measure raw ticks for a kernel, return minimum across trials
static double measure_kernel(hipModule_t module, const char* kernel_name,
                             uint32_t num_iters, int num_trials) {
    hipFunction_t func;
    HIP_CALL(hipModuleGetFunction(&func, module, kernel_name));

    uint64_t* d_out = nullptr;
    HIP_CALL(hipMalloc(&d_out, 16));

    struct __attribute__((packed)) { uint64_t* out; uint32_t iters; } args;
    args.out = d_out; args.iters = num_iters;
    size_t sz = sizeof(args);
    void* cfg[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END };

    double min_ticks = 1e18;
    for (int t = 0; t < num_trials; t++) {
        uint64_t zeros[2] = {0, 0};
        HIP_CALL(hipMemcpy(d_out, zeros, 16, hipMemcpyHostToDevice));
        HIP_CALL(hipModuleLaunchKernel(func, 1,1,1, 64,1,1, 0,0, NULL, (void**)&cfg));
        HIP_CALL(hipDeviceSynchronize());

        uint64_t h[2];
        HIP_CALL(hipMemcpy(h, d_out, 16, hipMemcpyDeviceToHost));
        double ticks = (double)(h[1] - h[0]);
        if (ticks < min_ticks) min_ticks = ticks;
    }

    HIP_CALL(hipFree(d_out));
    return min_ticks;
}

int main() {
    printf("=== LDS Throughput Benchmark (gfx942 / MI308) ===\n\n");

    HIP_CALL(hipInit(0));
    HIP_CALL(hipSetDevice(0));

    hipDeviceProp_t props;
    HIP_CALL(hipGetDeviceProperties(&props, 0));
    printf("Device: %s (%s)\n", props.name, props.gcnArchName);
    double shader_mhz = props.clockRate / 1000.0;
    printf("Shader clock: %.0f MHz\n\n", shader_mhz);

    double tick_freq = calibrate_tick_freq();
    double tick_to_shader = shader_mhz / tick_freq;
    printf("Clock calibration: tick_freq = %.1f MHz, "
           "shader_freq = %.0f MHz (ratio = %.2f)\n\n",
           tick_freq, shader_mhz, tick_to_shader);

    hipModule_t module;
    HIP_CALL(hipModuleLoad(&module, "lds_throughput.hsaco"));

    uint32_t num_iters = 10000;
    int num_trials = 5;

    // Measure NOP baseline
    double nop_ticks = measure_kernel(module, "lds_tp_nop_kernel", num_iters, num_trials);
    printf("NOP baseline: %.0f ticks for %u iters (%.2f ticks/iter)\n\n",
           nop_ticks, num_iters, nop_ticks / num_iters);

    struct TestConfig {
        const char* kernel_name;
        const char* display_name;
        int unroll_factor;
        int dwords_per_op;  // per lane per instruction
    };

    TestConfig tests[] = {
        { "lds_tp_read_b32_kernel",   "ds_read_b32",   32, 1 },
        { "lds_tp_read_b64_kernel",   "ds_read_b64",   16, 2 },
        { "lds_tp_read_b128_kernel",  "ds_read_b128",   8, 4 },
        { "lds_tp_write_b32_kernel",  "ds_write_b32",  32, 1 },
        { "lds_tp_write_b64_kernel",  "ds_write_b64",  16, 2 },
        { "lds_tp_write_b128_kernel", "ds_write_b128",  8, 4 },
    };

    printf("LDS Throughput (bank-conflict-free, wavefront=64, %u iters x unroll)\n\n",
           num_iters);
    printf("  %-16s | %12s | %11s | %11s\n",
           "Instruction", "Dwords/cycle", "Bytes/cycle", "Cycles/inst");
    printf("  %-16s-+-%12s-+-%11s-+-%11s\n",
           "----------------", "------------", "-----------", "-----------");

    for (int i = 0; i < 6; i++) {
        auto& t = tests[i];
        double raw_ticks = measure_kernel(module, t.kernel_name, num_iters, num_trials);
        double net_ticks = raw_ticks - nop_ticks;
        if (net_ticks < 1.0) net_ticks = 1.0;  // guard

        double net_shader_cycles = net_ticks * tick_to_shader;
        uint64_t total_ops = (uint64_t)num_iters * t.unroll_factor;
        uint64_t total_dwords = total_ops * 64 * t.dwords_per_op;

        double dwords_per_cycle = (double)total_dwords / net_shader_cycles;
        double bytes_per_cycle = dwords_per_cycle * 4.0;
        double cycles_per_inst = net_shader_cycles / (double)total_ops;

        printf("  %-16s | %12.1f | %11.1f | %11.2f\n",
               t.display_name, dwords_per_cycle, bytes_per_cycle, cycles_per_inst);
    }
    printf("\n");

    HIP_CALL(hipModuleUnload(module));
    HIP_CALL(hipDeviceReset());
    return 0;
}
