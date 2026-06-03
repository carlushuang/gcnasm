// pdl_bench (AMD / MI355X, gfx950) -- software emulation of NVIDIA's
// Programmatic Dependent Launch (PDL) using a single global semaphore.
//
// AMD GPUs have no PDL hardware. We reproduce the *effect* (overlap the
// producer's post-trigger "tail" with the consumer's pre-sync "head") with:
//
//   * a single uint32 device buffer  `pdl_semaphore`
//   * initialised to the producer grid size (number of workgroups)
//   * each producer workgroup decrements it once, at the point where CUDA
//     would call cudaTriggerProgrammaticLaunchCompletion()
//   * each consumer workgroup spin-waits for it to reach 0, at the point
//     where CUDA would call cudaGridDependencySynchronize()
//
// Because same-stream kernels serialize on AMD, the overlap only materialises
// when producer and consumer run on *separate streams*. The semaphore then
// replaces the hardware grid-dependency edge: it lets the consumer race ahead
// through its independent head compute, but blocks the dependent epilogue until
// every producer workgroup has published its writes.

#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>   // hipExtLaunchKernel + hipExtAnyOrderLaunch
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>

constexpr int BLOCK_SIZE = 256;

// Prevent the compiler from optimizing away the timing loops.
__device__ __forceinline__ float dummy_compute(float x, int iters) {
    float acc = x;
    #pragma unroll 1
    for (int i = 0; i < iters; ++i) {
        acc = fmaf(acc, 1.0001f, float(i & 0xFF));
    }
    return acc;
}

// Set the semaphore to `v` (one thread). Atomic store keeps it ordered w.r.t.
// the producer's atomic decrement / consumer's atomic load.
__global__ void reset_sem_kernel(unsigned int* sem, unsigned int v) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        __hip_atomic_store(sem, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
}

// Producer: small write, "trigger" (decrement semaphore), then long tail compute.
// USE_PDL=false reproduces a plain producer with no semaphore traffic (baseline).
template <bool USE_PDL>
__global__ void producer_kernel(float* __restrict__ out, int N, int tail_iters,
                                unsigned int* __restrict__ sem) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = 1.0f;
    }

    if (USE_PDL) {
        // === equivalent of cudaTriggerProgrammaticLaunchCompletion() ===
        // Make every thread's write above visible within the workgroup, then
        // the lead thread publishes it agent-wide via a release decrement.
        __syncthreads();
        if (threadIdx.x == 0) {  // one lead thread per workgroup (threadIdx.x % BLOCK_SIZE == 0)
            __hip_atomic_fetch_add(sem, -1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        }
    }

    if (tail_iters > 0) {
        float acc = dummy_compute(__int_as_float(idx), tail_iters);
        if (acc == 1.23456789e30f && idx < N) {
            out[idx] = acc;
        }
    }
}

// Consumer: long head compute (independent), grid-dependency sync, then small
// dependent work. USE_PDL=false skips the semaphore wait (baseline).
template <bool USE_PDL>
__global__ void consumer_kernel(const float* __restrict__ in,
                                float* __restrict__ out,
                                int N, int head_iters,
                                unsigned int* __restrict__ sem) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.0f;
    if (head_iters > 0) {
        acc = dummy_compute(__int_as_float(idx ^ 0x1A2B3C4D), head_iters);
    }

    if (USE_PDL) {
        // === equivalent of cudaGridDependencySynchronize() ===
        // Barrier so the whole workgroup has finished its head compute, then the
        // lead thread spins until all producer workgroups have signalled.
        __syncthreads();
        if (threadIdx.x == 0) {
            while (__hip_atomic_load(sem, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != 0u) {
                /* spin */
            }
            // Acquire so the producers' writes (released via the decrement) are
            // visible to this thread before we read `in` below. The spin itself
            // uses relaxed loads to keep the busy-wait cheap.
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
        }
        // Propagate the lead thread's acquire to the rest of the workgroup so no
        // non-lead thread reads `in` before the dependency is satisfied.
        __syncthreads();
    }

    if (idx < N) {
        out[idx] = in[idx] + acc;
    }
}

static void checkHip(hipError_t err, const char* file, int line) {
    if (err != hipSuccess) {
        fprintf(stderr, "HIP error at %s:%d: %s\n", file, line, hipGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
#define CHECK(x) checkHip(x, __FILE__, __LINE__)

using Clock = std::chrono::high_resolution_clock;
static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

int main(int argc, char** argv) {
    int N = 1 << 16;
    int tail_iters = 16384;
    int head_iters = 16384;
    int warmup = 10;
    int iters = 100;

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) tail_iters = atoi(argv[2]);
    if (argc > 3) head_iters = atoi(argv[3]);
    if (argc > 4) iters = atoi(argv[4]);

    int dev = 0;
    CHECK(hipSetDevice(dev));
    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop, dev));
    printf("Device: %s (%s), %d CUs/WGPs\n", prop.name, prop.gcnArchName, prop.multiProcessorCount);
    // hipExtAnyOrderLaunch clears the AQL barrier bit; this gives real same-stream
    // overlap on gfx10/11/12 (RDNA) but is a no-op on gfx9xx (CDNA/Vega), whose
    // command processor serializes same-queue dispatches in hardware regardless.
    bool anyorder_hw = (strncmp(prop.gcnArchName, "gfx9", 4) != 0);
    const char* anyorder_note = anyorder_hw ? "[anyOrder active: same-stream overlap supported]"
                                            : "[anyOrder no-op on gfx9xx: same-queue serializes in HW]";

    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 block(BLOCK_SIZE);
    dim3 gdim(grid);

    // Co-residency guard. The software handshake spins, so a consumer workgroup
    // that becomes resident holds its CU until the semaphore hits 0. If producer
    // and consumer workgroups cannot all be co-resident, resident spinning
    // consumers starve the not-yet-scheduled producers -> deadlock (unlike
    // hardware PDL, which the scheduler resolves). Require both grids to fit.
    int occP = 0, occC = 0;
    CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&occP, producer_kernel<true>, BLOCK_SIZE, 0));
    CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&occC, consumer_kernel<true>, BLOCK_SIZE, 0));
    int co_resident_budget = prop.multiProcessorCount * (occP < occC ? occP : occC);
    bool pdl_safe = (2 * grid) <= co_resident_budget;
    printf("Occupancy: producer %d blk/CU, consumer %d blk/CU -> co-resident budget %d blocks; need %d (2x%d)%s\n",
           occP, occC, co_resident_budget, 2 * grid, grid, pdl_safe ? "" : "  [UNSAFE]");

    size_t bytes = static_cast<size_t>(N) * sizeof(float);
    float *d_producer_out, *d_consumer_out;
    unsigned int* d_sem;
    CHECK(hipMalloc(&d_producer_out, bytes));
    CHECK(hipMalloc(&d_consumer_out, bytes));
    CHECK(hipMalloc(&d_sem, sizeof(unsigned int)));

    // Stream A drives producers (+ semaphore reset); stream B drives consumers.
    hipStream_t streamA, streamB;
    CHECK(hipStreamCreate(&streamA));
    CHECK(hipStreamCreate(&streamB));

    // Ping-pong events: evReset gates the consumer behind this pair's reset;
    // evPair gates the next pair's reset behind this pair's consumer (so the
    // single shared semaphore is never reused across an in-flight pair).
    hipEvent_t evReset, evPair;
    CHECK(hipEventCreate(&evReset));
    CHECK(hipEventCreate(&evPair));

    // -------- Baseline: producer then consumer, same stream, no semaphore -----
    auto run_baseline = [&]() {
        producer_kernel<false><<<gdim, block, 0, streamA>>>(d_producer_out, N, tail_iters, d_sem);
        consumer_kernel<false><<<gdim, block, 0, streamA>>>(d_producer_out, d_consumer_out, N, head_iters, d_sem);
    };

    // -------- Software PDL: two streams + semaphore --------------------------
    auto run_pdl_pair = [&](int i) {
        if (i > 0) CHECK(hipStreamWaitEvent(streamA, evPair, 0));   // wait prev consumer
        reset_sem_kernel<<<1, 1, 0, streamA>>>(d_sem, (unsigned int)grid);
        CHECK(hipEventRecord(evReset, streamA));
        CHECK(hipStreamWaitEvent(streamB, evReset, 0));             // consumer waits for reset
        producer_kernel<true><<<gdim, block, 0, streamA>>>(d_producer_out, N, tail_iters, d_sem);
        consumer_kernel<true><<<gdim, block, 0, streamB>>>(d_producer_out, d_consumer_out, N, head_iters, d_sem);
        CHECK(hipEventRecord(evPair, streamB));
    };

    // -------- Same-stream "any-order" launch (the PDL-on-same-stream idea) ----
    // hipExtLaunchKernel(..., hipExtAnyOrderLaunch) asks the runtime to relax the
    // in-order stream guarantee so the consumer may start before the producer
    // finishes -- the closest analogue to NVIDIA PDL (same stream, overlap). The
    // semaphore still provides the data dependency. NOTE: this flag is documented
    // as unsupported on GFX9xx and is a no-op on gfx950 (kernels still serialize),
    // so this path measures ~baseline here. It is kept to demonstrate that.
    auto run_anyorder_pair = [&]() {
        reset_sem_kernel<<<1, 1, 0, streamA>>>(d_sem, (unsigned int)grid);
        float* p_out = d_producer_out; float* c_out = d_consumer_out;
        const float* c_in = d_producer_out; unsigned int* sem = d_sem;
        int n = N, ti = tail_iters, hi = head_iters;
        void* pargs[] = {&p_out, &n, &ti, &sem};
        void* cargs[] = {&c_in, &c_out, &n, &hi, &sem};
        CHECK(hipExtLaunchKernel((void*)producer_kernel<true>, gdim, block, pargs, 0,
                                 streamA, nullptr, nullptr, hipExtAnyOrderLaunch));
        CHECK(hipExtLaunchKernel((void*)consumer_kernel<true>, gdim, block, cargs, 0,
                                 streamA, nullptr, nullptr, hipExtAnyOrderLaunch));
    };

    // --- Warmup ---
    for (int i = 0; i < warmup; ++i) run_baseline();
    CHECK(hipStreamSynchronize(streamA));
    for (int i = 0; i < warmup; ++i) run_anyorder_pair();
    CHECK(hipStreamSynchronize(streamA));
    if (pdl_safe) {
        for (int i = 0; i < warmup; ++i) run_pdl_pair(i);
        CHECK(hipDeviceSynchronize());
    }

    // --- Baseline timing (wall clock, end-to-end) ---
    auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) run_baseline();
    CHECK(hipStreamSynchronize(streamA));
    double baseline_ms = ms_since(t0);

    // --- Same-stream any-order timing ---
    t0 = Clock::now();
    for (int i = 0; i < iters; ++i) run_anyorder_pair();
    CHECK(hipStreamSynchronize(streamA));
    double anyorder_ms = ms_since(t0);

    // --- Software-PDL timing ---
    double pdl_ms = 0;
    if (pdl_safe) {
        t0 = Clock::now();
        for (int i = 0; i < iters; ++i) run_pdl_pair(i);
        CHECK(hipDeviceSynchronize());
        pdl_ms = ms_since(t0);
    }

    // --- Correctness: does the consumer observe the producer's writes? -------
    // Producer writes 1.0f; consumer (PDL) copies in[idx] -> out[idx] with no
    // head compute. With a correct semaphore handshake every element must be 1.
    if (pdl_safe) {
        CHECK(hipMemset(d_consumer_out, 0, bytes));
        reset_sem_kernel<<<1, 1, 0, streamA>>>(d_sem, (unsigned int)grid);
        CHECK(hipEventRecord(evReset, streamA));
        CHECK(hipStreamWaitEvent(streamB, evReset, 0));
        producer_kernel<true><<<gdim, block, 0, streamA>>>(d_producer_out, N, /*tail*/4096, d_sem);
        consumer_kernel<true><<<gdim, block, 0, streamB>>>(d_producer_out, d_consumer_out, N, /*head*/0, d_sem);
        CHECK(hipDeviceSynchronize());

        float* h = (float*)malloc(bytes);
        CHECK(hipMemcpy(h, d_consumer_out, bytes, hipMemcpyDeviceToHost));
        int bad = 0;
        for (int k = 0; k < N; ++k) if (h[k] != 1.0f) { ++bad; }
        printf("Correctness (consumer sees producer writes): %s (%d/%d mismatched)\n",
               bad == 0 ? "PASS" : "FAIL", bad, N);
        free(h);
    }

    // --- Results ---
    printf("\n=== Software-PDL Microbenchmark (single global semaphore) ===\n");
    printf("N=%d, grid=%d blocks x %d, tail_iters=%d, head_iters=%d, iterations=%d\n",
           N, grid, BLOCK_SIZE, tail_iters, head_iters, iters);
    printf("Baseline (1 stream, sequential)     : %.3f ms total, %.3f ms avg\n",
           baseline_ms, baseline_ms / iters);
    printf("Same stream + anyOrder + sem        : %.3f ms total, %.3f ms avg (%.2fx)  %s\n",
           anyorder_ms, anyorder_ms / iters, baseline_ms / anyorder_ms, anyorder_note);
    if (pdl_safe) {
        printf("Software PDL (2 streams + sem)       : %.3f ms total, %.3f ms avg (%.2fx)\n",
               pdl_ms, pdl_ms / iters, baseline_ms / pdl_ms);
    } else {
        printf("Software PDL (2 streams + sem)       : SKIPPED -- grids cannot be co-resident "
               "(deadlock risk). Reduce N or BLOCK_SIZE so 2*grid <= %d.\n", co_resident_budget);
    }

    CHECK(hipFree(d_producer_out));
    CHECK(hipFree(d_consumer_out));
    CHECK(hipFree(d_sem));
    CHECK(hipEventDestroy(evReset));
    CHECK(hipEventDestroy(evPair));
    CHECK(hipStreamDestroy(streamA));
    CHECK(hipStreamDestroy(streamB));
    return 0;
}
