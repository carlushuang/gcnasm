// SPX + NPS1 interleave probe for MI355X (gfx950).
// Times single cold global loads from one XCC at a time (others exit), with a cache flush between launches, to find
// (a) which XCC_IDs share an IOD and (b) how a byte offset of a 2 MiB-aligned allocation maps to its home IOD.
// Usage (SPX + NPS1): ./spx_interleave_probe [dev=0]  -> writes partA.csv / partB.csv; then python3 analyze_interleave.py <cycles_per_ns printed at the end>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>

#define CHECK(x)                                                                                   \
  do {                                                                                             \
    hipError_t e_ = (x);                                                                           \
    if (e_ != hipSuccess) {                                                                        \
      fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_));          \
      exit(1);                                                                                     \
    }                                                                                              \
  } while (0)

#define GETREG_FULL(REG) ((31 << 11) | (0 << 6) | (REG))
__device__ __forceinline__ uint32_t read_xcc() { return __builtin_amdgcn_s_getreg(GETREG_FULL(20)) & 0xf; }

__global__ void id_kernel(uint32_t* out) {
  if (threadIdx.x == 0) out[blockIdx.x] = read_xcc();
}

__global__ void calib_kernel(uint64_t* out, uint64_t spin_rt_ticks) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  const uint64_t r0 = wall_clock64(), c0 = clock64();
  while (wall_clock64() - r0 < spin_rt_ticks) {
  }
  const uint64_t r1 = wall_clock64(), c1 = clock64();
  out[0] = r1 - r0;
  out[1] = c1 - c0;
}

// One lane per workgroup; only workgroups on `target_xcc` work; samples are claimed with an atomic counter.
__global__ __launch_bounds__(64) void line_lat_kernel(const char* base, const uint64_t* offs, uint32_t n,
                                                       uint32_t target_xcc, uint32_t* counter, uint32_t* out) {
  if (threadIdx.x != 0) return;
  if (read_xcc() != target_xcc) return;
  while (true) {
    const uint32_t i = atomicAdd(counter, 1u);
    if (i >= n) break;
    const uint32_t* p = reinterpret_cast<const uint32_t*>(base + offs[i]);
    uint64_t t0, t1;
    uint32_t v;
    asm volatile(
        "s_waitcnt vmcnt(0) lgkmcnt(0)\n\t"
        "s_memtime %0\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        "global_load_dword %2, %3, off\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        "s_memtime %1\n\t"
        "s_waitcnt lgkmcnt(0)"
        : "=&s"(t0), "=&s"(t1), "=&v"(v)
        : "v"(p)
        : "memory");
    out[i] = static_cast<uint32_t>(t1 - t0) + (v == 0xdeadbeefu ? 1u : 0u);
  }
}

int main(int argc, char** argv) {
  const int dev = argc > 1 ? atoi(argv[1]) : 0;
  CHECK(hipSetDevice(dev));
  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, dev));
  printf("dev %d: %s %s CUs=%d mem=%.1f GiB\n", dev, prop.name, prop.gcnArchName, prop.multiProcessorCount,
         prop.totalGlobalMem / 1073741824.0);

  // XCC_ID semantics and dispatch order in SPX
  {
    const int nb = 4096;
    uint32_t* d;
    CHECK(hipMalloc(&d, nb * 4));
    hipLaunchKernelGGL(id_kernel, dim3(nb), dim3(64), 0, 0, d);
    CHECK(hipDeviceSynchronize());
    std::vector<uint32_t> h(nb);
    CHECK(hipMemcpy(h.data(), d, nb * 4, hipMemcpyDeviceToHost));
    CHECK(hipFree(d));
    std::set<uint32_t> s(h.begin(), h.end());
    int rr_ok = 0;
    for (int b = 0; b < nb; ++b) rr_ok += (h[b] == (h[0] + b) % 8);
    printf("XCC_ID values:");
    for (auto x : s) printf(" %u", x);
    printf("  wg0..15 ->");
    for (int b = 0; b < 16; ++b) printf(" %u", h[b]);
    printf("  round-robin (xcc == (xcc0 + wg) %% 8) for %d/%d WGs\n", rr_ok, nb);
  }

  // s_memtime vs s_memrealtime calibration
  double cyc_per_ns = 0;
  {
    uint64_t* d;
    CHECK(hipMalloc(&d, 16));
    int wclk_khz = 0;
    CHECK(hipDeviceGetAttribute(&wclk_khz, hipDeviceAttributeWallClockRate, dev));
    hipLaunchKernelGGL(calib_kernel, dim3(1), dim3(64), 0, 0, d, (uint64_t)wclk_khz * 20);  // ~20 ms
    CHECK(hipDeviceSynchronize());
    uint64_t h[2];
    CHECK(hipMemcpy(h, d, 16, hipMemcpyDeviceToHost));
    CHECK(hipFree(d));
    const double ns = (double)h[0] * 1e6 / wclk_khz;
    cyc_per_ns = (double)h[1] / ns;
    printf("s_memtime rate: %.3f cycles/ns (%.1f MHz)\n", cyc_per_ns, cyc_per_ns * 1e3);
  }

  const size_t buf_bytes = 1ull << 30, flush_bytes = 2ull << 30;
  char* buf;
  void* scratch;
  uint32_t *counter, *dout;
  uint64_t* doffs;
  CHECK(hipMalloc(&buf, buf_bytes));
  CHECK(hipMemset(buf, 0, buf_bytes));
  CHECK(hipMalloc(&scratch, flush_bytes));
  CHECK(hipMalloc(&counter, 4));
  const uint32_t kMaxSamples = 1u << 17;
  CHECK(hipMalloc(&dout, kMaxSamples * 4));
  CHECK(hipMalloc(&doffs, kMaxSamples * 8));
  printf("buffer base %p (base %% 2MiB = %llu)\n", (void*)buf,
         (unsigned long long)(reinterpret_cast<uintptr_t>(buf) % (2u << 20)));

  int flush_val = 1;
  auto measure = [&](const std::vector<uint64_t>& offs, uint32_t xcc, std::vector<uint32_t>& res) {
    CHECK(hipMemcpy(doffs, offs.data(), offs.size() * 8, hipMemcpyHostToDevice));
    CHECK(hipMemsetAsync(scratch, (flush_val++) & 0xff, flush_bytes, 0));
    CHECK(hipMemsetAsync(counter, 0, 4, 0));
    hipLaunchKernelGGL(line_lat_kernel, dim3(256), dim3(64), 0, 0, (const char*)buf, (const uint64_t*)doffs,
                       (uint32_t)offs.size(), xcc, counter, dout);
    CHECK(hipDeviceSynchronize());
    res.resize(offs.size());
    CHECK(hipMemcpy(res.data(), dout, offs.size() * 4, hipMemcpyDeviceToHost));
  };

  // Part A: every 128 B line of the first 1 MiB, 4 passes (lines 512 B apart within a launch), from every XCC
  {
    const int kLines = 8192;
    std::vector<std::vector<uint32_t>> lat(8, std::vector<uint32_t>(kLines));
    for (uint32_t x = 0; x < 8; ++x) {
      for (int p = 0; p < 4; ++p) {
        std::vector<uint64_t> offs;
        for (int l = p; l < kLines; l += 4) offs.push_back((uint64_t)l * 128);
        std::vector<uint32_t> r;
        measure(offs, x, r);
        for (size_t k = 0; k < offs.size(); ++k) lat[x][offs[k] / 128] = r[k];
      }
    }
    FILE* f = fopen("partA.csv", "w");
    fprintf(f, "offset");
    for (int x = 0; x < 8; ++x) fprintf(f, ",x%d", x);
    fprintf(f, "\n");
    for (int l = 0; l < kLines; ++l) {
      fprintf(f, "%d", l * 128);
      for (int x = 0; x < 8; ++x) fprintf(f, ",%u", lat[x][l]);
      fprintf(f, "\n");
    }
    fclose(f);
    printf("part A: median ns per XCC:");
    for (int x = 0; x < 8; ++x) {
      auto v = lat[x];
      std::sort(v.begin(), v.end());
      printf(" %d:%.0f(p10 %.0f p90 %.0f)", x, v[v.size() / 2] / cyc_per_ns, v[v.size() / 10] / cyc_per_ns,
             v[v.size() * 9 / 10] / cyc_per_ns);
    }
    printf("\n");
  }

  // Part B: offset 0 of every 4 KiB block over 256 MiB (128 x 2 MiB pages), from every XCC
  {
    const int kBlocks = 65536;
    std::vector<uint64_t> offs(kBlocks);
    for (int i = 0; i < kBlocks; ++i) offs[i] = (uint64_t)i * 4096;
    std::vector<std::vector<uint32_t>> lat(8);
    for (uint32_t x = 0; x < 8; ++x) measure(offs, x, lat[x]);
    FILE* f = fopen("partB.csv", "w");
    fprintf(f, "offset");
    for (int x = 0; x < 8; ++x) fprintf(f, ",x%d", x);
    fprintf(f, "\n");
    for (int i = 0; i < kBlocks; ++i) {
      fprintf(f, "%llu", (unsigned long long)offs[i]);
      for (int x = 0; x < 8; ++x) fprintf(f, ",%u", lat[x][i]);
      fprintf(f, "\n");
    }
    fclose(f);
    printf("part B written (%d samples x 8 XCCs)\n", kBlocks);
  }
  printf("cyc_per_ns=%.4f\n", cyc_per_ns);
  return 0;
}
