#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <hip/hip_runtime.h>
#include <random>

// ---------------------------------------------------------------------------
// HIP error-checking macro
// ---------------------------------------------------------------------------
#define HIP_CALL(call) do {                                                 \
    hipError_t err = call;                                                  \
    if (err != hipSuccess) {                                                \
        printf("[HIP ERROR] (%d) %s  at %s:%d\n",                          \
               (int)err, hipGetErrorString(err), __FILE__, __LINE__);       \
        exit(EXIT_FAILURE);                                                 \
    }                                                                       \
} while (0)

// ---------------------------------------------------------------------------
// Paths to the pre-assembled kernel code object
// ---------------------------------------------------------------------------
#define HSACO       "vector_add_kernel.hsaco"
#define KERNEL_NAME "vector_add_kernel"

// ---------------------------------------------------------------------------
// Test configurations
// ---------------------------------------------------------------------------
struct TestCase {
    int N;
    const char* description;
};

static TestCase test_cases[] = {
    {        1, "single element"          },
    {       64, "one wavefront"           },
    {      256, "one workgroup"           },
    {      257, "one workgroup + 1"       },
    {     1000, "non-power-of-two"        },
    {     1024, "small power-of-two"      },
    {    65536, "64K elements"            },
    {  1048576, "1M elements"             },
    {  4194304, "4M elements"             },
};
static const int num_tests = sizeof(test_cases) / sizeof(test_cases[0]);

// ---------------------------------------------------------------------------
// Run a single test case (persistent kernel: grid = num_CUs)
//   Returns 0 on success, 1 on failure.
// ---------------------------------------------------------------------------
int run_test(hipFunction_t kernel_func, int num_cu, int N) {
    size_t size = (size_t)N * sizeof(float);

    // Allocate host memory
    float* h_a   = new float[N];
    float* h_b   = new float[N];
    float* h_c   = new float[N];
    float* h_ref = new float[N];

    // Initialize with deterministic but non-trivial values
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    for (int i = 0; i < N; i++) {
        h_a[i]   = dist(rng);
        h_b[i]   = dist(rng);
        h_ref[i] = h_a[i] + h_b[i];   // reference result
    }

    // Allocate device memory
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    HIP_CALL(hipMalloc(&d_a, size));
    HIP_CALL(hipMalloc(&d_b, size));
    HIP_CALL(hipMalloc(&d_c, size));

    // Copy inputs to device
    HIP_CALL(hipMemcpy(d_a, h_a, size, hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_b, h_b, size, hipMemcpyHostToDevice));
    HIP_CALL(hipMemset(d_c, 0, size));

    // Prepare kernel arguments  (must match the .s kernarg layout exactly)
    //   offset  0: float* A       (8 bytes)
    //   offset  8: float* B       (8 bytes)
    //   offset 16: float* C       (8 bytes)
    //   offset 24: uint32 N       (4 bytes)  -- number of elements
    //   offset 28: uint32 stride  (4 bytes)  -- total threads = num_CUs * blockDim
    int bdx = 256;
    int gdx = num_cu;                         // persistent: one workgroup per CU
    uint32_t stride = (uint32_t)(gdx * bdx);  // grid-stride loop step

    struct __attribute__((packed)) {
        float*   A;
        float*   B;
        float*   C;
        uint32_t N;
        uint32_t stride;
    } args;

    args.A      = d_a;
    args.B      = d_b;
    args.C      = d_c;
    args.N      = (uint32_t)N;
    args.stride = stride;

    size_t arg_size = sizeof(args);
    void* config[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
        HIP_LAUNCH_PARAM_BUFFER_SIZE,    &arg_size,
        HIP_LAUNCH_PARAM_END
    };

    HIP_CALL(hipModuleLaunchKernel(
        kernel_func,
        gdx, 1, 1,       // grid  = num_CUs  (persistent)
        bdx, 1, 1,       // block = 256
        0,                // shared memory
        0,                // stream
        NULL,             // kernel params (unused with config)
        (void**)&config   // extra config
    ));

    HIP_CALL(hipDeviceSynchronize());

    // Copy result back
    HIP_CALL(hipMemcpy(h_c, d_c, size, hipMemcpyDeviceToHost));

    // Verify results
    int errors = 0;
    float max_abs_err = 0.0f;
    for (int i = 0; i < N; i++) {
        float err = fabsf(h_c[i] - h_ref[i]);
        if (err > max_abs_err) max_abs_err = err;

        if (err > 1e-5f) {
            if (errors < 5) {
                printf("    ERROR at [%d]: A=%.6f, B=%.6f, got=%.6f, expected=%.6f (err=%.2e)\n",
                       i, h_a[i], h_b[i], h_c[i], h_ref[i], err);
            }
            errors++;
        }
    }

    // Cleanup
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    delete[] h_ref;
    HIP_CALL(hipFree(d_a));
    HIP_CALL(hipFree(d_b));
    HIP_CALL(hipFree(d_c));

    if (errors == 0) {
        printf("    PASSED  (N=%d, grid=%d, stride=%u)\n", N, gdx, stride);
        return 0;
    } else {
        printf("    FAILED  (N=%d, %d errors out of %d)\n", N, errors, N);
        return 1;
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    printf("=== Persistent Vector Add Assembly Kernel Unit Test (gfx942) ===\n\n");

    // Initialize HIP
    HIP_CALL(hipInit(0));

    int device_count = 0;
    HIP_CALL(hipGetDeviceCount(&device_count));
    if (device_count == 0) {
        printf("ERROR: No HIP devices found!\n");
        return EXIT_FAILURE;
    }

    HIP_CALL(hipSetDevice(0));

    // Print device info and detect CU count
    hipDeviceProp_t props;
    HIP_CALL(hipGetDeviceProperties(&props, 0));
    int num_cu = props.multiProcessorCount;

    printf("Device:           %s\n", props.name);
    printf("Compute Units:    %d\n", num_cu);
    printf("Global Memory:    %.1f GB\n",
           props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("GCN Architecture: %s\n", props.gcnArchName);
    printf("Grid (persistent): %d workgroups (= num CUs)\n", num_cu);
    printf("Stride:           %d elements\n", num_cu * 256);
    printf("\n");

    // Load the pre-assembled code object
    hipModule_t   module;
    hipFunction_t kernel_func;

    HIP_CALL(hipModuleLoad(&module, HSACO));
    HIP_CALL(hipModuleGetFunction(&kernel_func, module, KERNEL_NAME));
    printf("Loaded kernel '%s' from '%s'\n\n", KERNEL_NAME, HSACO);

    // Run all test cases
    int total_pass = 0;
    int total_fail = 0;

    for (int t = 0; t < num_tests; t++) {
        printf("Test %d/%d: N=%d (%s)\n",
               t + 1, num_tests, test_cases[t].N, test_cases[t].description);

        int result = run_test(kernel_func, num_cu, test_cases[t].N);
        if (result == 0)
            total_pass++;
        else
            total_fail++;
    }

    // Summary
    printf("\n=== Summary ===\n");
    printf("Passed: %d / %d\n", total_pass, num_tests);
    printf("Failed: %d / %d\n", total_fail, num_tests);

    // Cleanup
    HIP_CALL(hipModuleUnload(module));
    HIP_CALL(hipDeviceReset());

    if (total_fail == 0) {
        printf("\nAll tests PASSED!\n");
        return EXIT_SUCCESS;
    } else {
        printf("\nSome tests FAILED!\n");
        return EXIT_FAILURE;
    }
}
