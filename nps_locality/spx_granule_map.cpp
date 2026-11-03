// SPX + NPS1 per-granule (8 KiB) home-IOD map of a large buffer on MI355X (gfx950), then a power/bandwidth PoC
// using exact per-granule lists: naive (every XCC reads everything), local (XCC 0-3 -> IOD-0 granules,
// XCC 4-7 -> IOD-1 granules), remote (swapped).
// Usage (SPX + NPS1): ./spx_granule_map [dev=0] [GiB=16] [seconds_per_mode=6]
#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dirent.h>
#include <map>
#include <string>
#include <thread>
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

__global__ void calib_kernel(uint64_t* out, uint64_t spin_rt_ticks) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  const uint64_t r0 = wall_clock64(), c0 = clock64();
  while (wall_clock64() - r0 < spin_rt_ticks) {
  }
  const uint64_t r1 = wall_clock64(), c1 = clock64();
  out[0] = r1 - r0;
  out[1] = c1 - c0;
}

// s_memtime (clock64) cycles per ns; s_memrealtime (wall_clock64) ticks at hipDeviceAttributeWallClockRate
static double measure_cycles_per_ns(int dev) {
  uint64_t* d;
  CHECK(hipMalloc(&d, 16));
  int wclk_khz = 0;
  CHECK(hipDeviceGetAttribute(&wclk_khz, hipDeviceAttributeWallClockRate, dev));
  hipLaunchKernelGGL(calib_kernel, dim3(1), dim3(64), 0, 0, d, (uint64_t)wclk_khz * 20);
  CHECK(hipDeviceSynchronize());
  uint64_t h[2];
  CHECK(hipMemcpy(h, d, 16, hipMemcpyDeviceToHost));
  CHECK(hipFree(d));
  return (double)h[1] / ((double)h[0] * 1e6 / wclk_khz);
}

constexpr uint64_t kGranule = 8192, kPage = 2u << 20, kGranPerPage = kPage / kGranule;

// Times one cold load at offset (i * stride + in_off) for i in [0, n), from `target_xcc` only.
__global__ __launch_bounds__(64) void stride_lat_kernel(const char* base, uint64_t stride, uint64_t in_off, uint32_t n,
                                                         uint32_t target_xcc, uint32_t* counter, uint16_t* out) {
  if (threadIdx.x != 0) return;
  if (read_xcc() != target_xcc) return;
  while (true) {
    const uint32_t i = atomicAdd(counter, 1u);
    if (i >= n) break;
    const uint32_t* p = reinterpret_cast<const uint32_t*>(base + (uint64_t)i * stride + in_off);
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
    const uint64_t d = (t1 - t0) + (v == 0xdeadbeefu ? 1u : 0u);
    out[i] = static_cast<uint16_t>(d > 65535 ? 65535 : d);
  }
}

// mode 0: naive contiguous claims of 32 granules; mode 1/2: group g claims 32 granules from list[g] / list[g ^ 1]
__global__ __launch_bounds__(256) void bw_kernel(const uint4* __restrict__ base, uint32_t num_granules, int mode,
                                                  const uint32_t* __restrict__ list0, uint32_t n0,
                                                  const uint32_t* __restrict__ list1, uint32_t n1, uint32_t* ctr,
                                                  uint32_t* sink) {
  constexpr uint32_t kVec = kGranule / sizeof(uint4), kPerClaim = 32;
  const uint32_t grp = read_xcc() >> 2, tid = threadIdx.x;
  const uint32_t sel = mode == 2 ? (grp ^ 1u) : grp;
  const uint32_t* list = sel ? list1 : list0;
  const uint32_t n = mode == 0 ? num_granules : (sel ? n1 : n0);
  __shared__ uint32_t s_k;
  uint32_t acc = 0;
  for (;;) {
    if (tid == 0) s_k = atomicAdd(&ctr[mode == 0 ? 0 : grp], 1u);
    __syncthreads();
    const uint32_t k = s_k;
    __syncthreads();
    const uint32_t first = k * kPerClaim;
    if (first >= n) break;
    const uint32_t cnt = min(kPerClaim, n - first);
#pragma unroll 8
    for (uint32_t j = 0; j < cnt; ++j) {
      const uint32_t gran = mode == 0 ? first + j : list[first + j];
      const uint4* p = base + (size_t)gran * kVec;
      const uint4 v0 = p[tid], v1 = p[tid + 256];
      acc ^= v0.x ^ v0.y ^ v0.z ^ v0.w ^ v1.x ^ v1.y ^ v1.z ^ v1.w;
    }
  }
  if (acc == 0x9e3779b9u) sink[0] = acc;
}

static std::string find_hwmon(const hipDeviceProp_t& p) {
  char dir[128];
  snprintf(dir, sizeof(dir), "/sys/bus/pci/devices/%04x:%02x:%02x.0/hwmon", p.pciDomainID, p.pciBusID, p.pciDeviceID);
  DIR* d = opendir(dir);
  if (!d) return "";
  std::string res;
  while (dirent* e = readdir(d))
    if (std::string(e->d_name).rfind("hwmon", 0) == 0) res = std::string(dir) + "/" + e->d_name;
  closedir(d);
  return res;
}
static double read_num(const std::string& f) {
  FILE* fp = fopen(f.c_str(), "r");
  if (!fp) return -1;
  double v = -1;
  if (fscanf(fp, "%lf", &v) != 1) v = -1;
  fclose(fp);
  return v;
}

int main(int argc, char** argv) {
  const int dev = argc > 1 ? atoi(argv[1]) : 0;
  const size_t gib = argc > 2 ? atoi(argv[2]) : 16;
  const double seconds = argc > 3 ? atof(argv[3]) : 6.0;
  CHECK(hipSetDevice(dev));
  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, dev));
  const double cpn = measure_cycles_per_ns(dev);
  printf("s_memtime rate: %.3f cycles/ns\n", cpn);
  const std::string hw = find_hwmon(prop);
  const std::string pfile = hw + "/power1_input";

  const size_t bytes = gib << 30, num_gran = bytes / kGranule, num_pages = bytes / kPage;
  char* buf;
  void* scratch;
  CHECK(hipMalloc(&buf, bytes));
  CHECK(hipMemset(buf, 0x11, bytes));
  CHECK(hipMalloc(&scratch, 2ull << 30));
  uint32_t* counter;
  uint16_t* dout;
  CHECK(hipMalloc(&counter, 4));
  CHECK(hipMalloc(&dout, num_gran * 2));
  printf("dev %d: %d CUs; buffer %zu GiB at %p, %zu granules, %zu pages\n", dev, prop.multiProcessorCount, gib,
         (void*)buf, num_gran, num_pages);

  auto probe = [&](uint32_t xcc, uint64_t in_off, std::vector<uint16_t>& r) {
    CHECK(hipMemsetAsync(scratch, (int)(xcc * 7 + 1) & 0xff, 2ull << 30, 0));
    CHECK(hipMemsetAsync(counter, 0, 4, 0));
    hipLaunchKernelGGL(stride_lat_kernel, dim3(256), dim3(64), 0, 0, (const char*)buf, kGranule, in_off,
                       (uint32_t)num_gran, xcc, counter, dout);
    CHECK(hipDeviceSynchronize());
    r.resize(num_gran);
    CHECK(hipMemcpy(r.data(), dout, num_gran * 2, hipMemcpyDeviceToHost));
  };

  // ---- 1) map: every granule from every XCC
  const auto tp0 = std::chrono::steady_clock::now();
  std::vector<std::vector<uint16_t>> lat(8);
  for (uint32_t x = 0; x < 8; ++x) probe(x, 0, lat[x]);
  const double probe_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - tp0).count();
  std::vector<uint8_t> home(num_gran);
  std::vector<float> margin(num_gran);
  for (size_t g = 0; g < num_gran; ++g) {
    double m0 = 0, m1 = 0;
    for (int x = 0; x < 4; ++x) m0 += lat[x][g], m1 += lat[x + 4][g];
    const double d = (m0 - m1) / 4 / cpn;  // ns; > 0 means IOD 1 (XCC 4-7) is closer
    home[g] = d > 0;
    margin[g] = std::abs(d);
  }
  std::vector<float> ms(margin);
  std::sort(ms.begin(), ms.end());
  const size_t n1 = std::count(home.begin(), home.end(), 1);
  printf("map: %.2f s for 8 x %zu samples; IOD1 share %.4f; margin p0.1 %.0f p1 %.0f p10 %.0f p50 %.0f ns\n", probe_s,
         num_gran, (double)n1 / num_gran, ms[num_gran / 1000], ms[num_gran / 100], ms[num_gran / 10], ms[num_gran / 2]);

  // repeatability: second pass from XCC 1 (IOD 0) and XCC 5 (IOD 1) at a different in-granule offset (+4 KiB)
  {
    std::vector<uint16_t> r1, r5;
    probe(1, 4096, r1);
    probe(5, 4096, r5);
    size_t agree = 0, conf = 0, conf_agree = 0;
    for (size_t g = 0; g < num_gran; ++g) {
      const uint8_t h = r5[g] < r1[g];
      agree += (h == home[g]);
      if (margin[g] > 40) {
        ++conf;
        conf_agree += (h == home[g]);
      }
    }
    printf("repeat (XCC1 vs XCC5, +4 KiB inside the granule): agreement %.4f overall, %.4f on confident granules\n",
           (double)agree / num_gran, (double)conf_agree / conf);
  }

  // ---- 2) structure: per-page linear fit over in-page granule bits 13..20 (GF(2) mask + const)
  std::map<int, size_t> mask_hist;
  size_t perfect = 0, near_perfect = 0;
  std::vector<int> page_mask(num_pages);
  for (size_t p = 0; p < num_pages; ++p) {
    int best_mask = 0, best_err = 1 << 30;
    for (int m = 0; m < 256; ++m) {
      int e0 = 0;
      for (size_t i = 0; i < kGranPerPage; ++i) e0 += (__builtin_popcount(i & m) & 1) != home[p * kGranPerPage + i];
      const int e = std::min(e0, (int)kGranPerPage - e0);
      if (e < best_err) best_err = e, best_mask = m;
    }
    page_mask[p] = best_mask;
    mask_hist[best_mask]++;
    perfect += best_err == 0;
    near_perfect += best_err <= 2;
  }
  printf("per-page fit home = parity(granule_idx & mask) ^ c: perfect %zu/%zu, <=2 errors %zu/%zu; masks (bit0 = addr bit13):",
         perfect, num_pages, near_perfect, num_pages);
  for (auto& [m, c] : mask_hist) printf(" 0x%02x:%zu", m, c);
  printf("\n");
  // print a few non-default pages' in-page patterns (first 64 granules)
  int shown = 0;
  for (size_t p = 0; p < num_pages && shown < 4; ++p) {
    if (page_mask[p] == 1) continue;
    printf("  page %zu (mask 0x%02x): ", p, page_mask[p]);
    for (size_t i = 0; i < 64; ++i) printf("%d", home[p * kGranPerPage + i]);
    printf("\n");
    ++shown;
  }
  // does the page mask follow the page index (address bits >= 21)?
  printf("page masks of pages 0..63: ");
  for (int p = 0; p < 64; ++p) printf("%x", page_mask[p] == 1 ? 1 : 0);
  printf("  (1 = plain 8 KiB alternation)\n");

  // ---- 3) power / bandwidth PoC with exact per-granule lists
  std::vector<uint32_t> l0, l1;
  for (size_t g = 0; g < num_gran; ++g) (home[g] ? l1 : l0).push_back((uint32_t)g);
  uint32_t *dl0, *dl1, *ctr, *sink;
  CHECK(hipMalloc(&dl0, l0.size() * 4));
  CHECK(hipMalloc(&dl1, l1.size() * 4));
  CHECK(hipMemcpy(dl0, l0.data(), l0.size() * 4, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(dl1, l1.data(), l1.size() * 4, hipMemcpyHostToDevice));
  CHECK(hipMalloc(&ctr, 8));
  CHECK(hipMalloc(&sink, 4));
  printf("lists: IOD0 %zu granules, IOD1 %zu granules\n", l0.size(), l1.size());
  const char* names[3] = {"naive (every XCC, every granule)", "local (exact per-granule lists)",
                          "remote (lists swapped)"};
  const int grid = prop.multiProcessorCount * 8;
  for (int rep = 0; rep < 2; ++rep) {
    for (int mode : {0, 1, 2}) {
      auto launch = [&] {
        CHECK(hipMemsetAsync(ctr, 0, 8, 0));
        hipLaunchKernelGGL(bw_kernel, dim3(grid), dim3(256), 0, 0, (const uint4*)buf, (uint32_t)num_gran, mode,
                           (const uint32_t*)dl0, (uint32_t)l0.size(), (const uint32_t*)dl1, (uint32_t)l1.size(), ctr,
                           sink);
      };
      launch();
      CHECK(hipDeviceSynchronize());
      std::atomic<bool> stop{false};
      std::vector<double> pw, f1;
      std::thread sampler([&] {
        while (!stop.load()) {
          const double v = read_num(pfile);
          if (v > 0) pw.push_back(v / 1e6);
          f1.push_back(read_num(hw + "/freq1_input") / 1e6);
          std::this_thread::sleep_for(std::chrono::milliseconds(25));
        }
      });
      const auto t0 = std::chrono::steady_clock::now();
      size_t launches = 0;
      double el = 0;
      do {
        launch();
        ++launches;
        if (launches % 8 == 0) {
          CHECK(hipDeviceSynchronize());
          el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        }
      } while (el < seconds);
      CHECK(hipDeviceSynchronize());
      el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      stop = true;
      sampler.join();
      std::vector<double> steady(pw.begin() + pw.size() / 5, pw.end());
      double mean = 0;
      for (double v : steady) mean += v;
      mean /= std::max<size_t>(1, steady.size());
      std::sort(f1.begin(), f1.end());
      const double gbps = (double)bytes * launches / el / 1e9;
      printf("  [rep %d] %-34s %7.1f GB/s   power %.0f W   %.2f GB/J   sclk %.0f MHz\n", rep, names[mode], gbps, mean,
             gbps / mean, f1.empty() ? 0.0 : f1[f1.size() / 2]);
    }
  }
  return 0;
}
