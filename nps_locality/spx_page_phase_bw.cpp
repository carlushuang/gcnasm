// SPX + NPS1 address-aware locality PoC on MI355X (gfx950).
// In NPS1 the home IOD of an 8 KiB granule is addr[13] ^ addr[21] (^ const), and XCC 0-3 / 4-7 sit on IOD 0 / 1.
// 1) probe the home of granule 0 of every 2 MiB page of a large buffer (also checks the in-page alternation),
// 2) stream-read the buffer with one full-GPU kernel in three modes and sample board power from hwmon:
//    naive (every XCC reads everything), local (XCC group g reads granules homed on IOD g), remote (the opposite).
// Usage (SPX + NPS1): ./spx_page_phase_bw [dev=0] [GiB=16] [seconds_per_mode=6]
#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dirent.h>
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

constexpr uint64_t kGranule = 8192, kPage = 2u << 20;

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

// mode 0: naive, claims of 8 contiguous granules from one queue
// mode 1: local, XCC group g claims 8 granule pairs at a time and reads the granule of each pair homed on IOD g
// mode 2: remote, same but reads the granule homed on the other IOD
__global__ __launch_bounds__(256) void bw_kernel(const uint4* __restrict__ base, uint32_t num_pairs, int mode,
                                                  const uint8_t* __restrict__ page_home, uint32_t* ctr,
                                                  uint32_t* sink) {
  constexpr uint32_t kVecPerGranule = kGranule / sizeof(uint4);  // 512
  const uint32_t grp = read_xcc() >> 2;
  const uint32_t tid = threadIdx.x;
  const uint32_t num_claims = num_pairs / 32;
  __shared__ uint32_t s_k;
  uint32_t acc = 0;
  for (;;) {
    if (tid == 0) s_k = atomicAdd(&ctr[mode == 0 ? 0 : grp], 1u);
    __syncthreads();
    const uint32_t k = s_k;
    __syncthreads();
    if (mode == 0) {
      if (k >= 2 * num_claims) break;
      const uint4* p = base + (size_t)k * 32 * kVecPerGranule;
#pragma unroll 16
      for (int j = 0; j < 64; ++j) {
        const uint4 v = p[j * 256 + tid];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
      }
    } else {
      if (k >= num_claims) break;
      const uint32_t want = mode == 1 ? grp : (grp ^ 1u);
#pragma unroll 8
      for (int j = 0; j < 32; ++j) {
        const uint32_t m = k * 32 + j;
        const uint32_t gran = 2 * m + (want ^ page_home[m >> 7]);
        const uint4* p = base + (size_t)gran * kVecPerGranule;
        const uint4 v0 = p[tid], v1 = p[tid + 256];
        acc ^= v0.x ^ v0.y ^ v0.z ^ v0.w ^ v1.x ^ v1.y ^ v1.z ^ v1.w;
      }
    }
  }
  if (acc == 0x9e3779b9u) sink[0] = acc;
}


__global__ __launch_bounds__(256) void gs_kernel(const uint4* __restrict__ p, size_t n_vec, uint32_t* sink) {
  uint32_t acc = 0;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  for (; i + 3 * stride < n_vec; i += 4 * stride) {
    const uint4 a = p[i], b = p[i + stride], c = p[i + 2 * stride], d = p[i + 3 * stride];
    acc ^= a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w ^ c.x ^ c.y ^ c.z ^ c.w ^ d.x ^ d.y ^ d.z ^ d.w;
  }
  for (; i < n_vec; i += stride) {
    const uint4 a = p[i];
    acc ^= a.x ^ a.y ^ a.z ^ a.w;
  }
  if (acc == 0x9e3779b9u) sink[0] = acc;
}

static std::string find_hwmon(const hipDeviceProp_t& p) {
  char dir[128];
  snprintf(dir, sizeof(dir), "/sys/bus/pci/devices/%04x:%02x:%02x.0/hwmon", p.pciDomainID, p.pciBusID, p.pciDeviceID);
  DIR* d = opendir(dir);
  if (!d) return "";
  std::string res;
  while (dirent* e = readdir(d)) {
    if (std::string(e->d_name).rfind("hwmon", 0) == 0) res = std::string(dir) + "/" + e->d_name;
  }
  closedir(d);
  return res;
}

static double read_uw(const std::string& f) {
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
  std::string pfile = hw + "/power1_average";
  if (read_uw(pfile) < 0) pfile = hw + "/power1_input";
  {
    FILE* l1 = fopen((hw + "/freq1_label").c_str(), "r");
    FILE* l2 = fopen((hw + "/freq2_label").c_str(), "r");
    char a[32] = "?", b[32] = "?";
    if (l1) { if (fscanf(l1, "%31s", a) != 1) a[0] = 0; fclose(l1); }
    if (l2) { if (fscanf(l2, "%31s", b) != 1) b[0] = 0; fclose(l2); }
    printf("hwmon freq1=%s freq2=%s\n", a, b);
  }
  printf("dev %d: %s CUs=%d pci %02x; power file %s (now %.0f W)\n", dev, prop.gcnArchName, prop.multiProcessorCount,
         prop.pciBusID, pfile.c_str(), read_uw(pfile) / 1e6);

  const size_t bytes = gib << 30, num_pages = bytes / kPage, num_pairs = bytes / (2 * kGranule);
  char* buf;
  void* scratch;
  CHECK(hipMalloc(&buf, bytes));
  CHECK(hipMemset(buf, 0x11, bytes));
  CHECK(hipMalloc(&scratch, 2ull << 30));
  printf("buffer %zu GiB at %p (base %% 2MiB = %llu), %zu pages\n", gib, (void*)buf,
         (unsigned long long)(reinterpret_cast<uintptr_t>(buf) % kPage), num_pages);

  // ---- 1) per-page home probe: granule 0 and granule 1 of every page, from XCC 0 (IOD 0) and XCC 4 (IOD 1)
  uint32_t *counter, *dout;
  uint64_t* doffs;
  CHECK(hipMalloc(&counter, 4));
  CHECK(hipMalloc(&dout, num_pages * 4));
  CHECK(hipMalloc(&doffs, num_pages * 8));
  auto probe = [&](uint64_t in_page, uint32_t xcc, std::vector<uint32_t>& r) {
    std::vector<uint64_t> offs(num_pages);
    for (size_t p = 0; p < num_pages; ++p) offs[p] = p * kPage + in_page;
    CHECK(hipMemcpy(doffs, offs.data(), num_pages * 8, hipMemcpyHostToDevice));
    CHECK(hipMemsetAsync(scratch, (int)(xcc + in_page) & 0xff, 2ull << 30, 0));
    CHECK(hipMemsetAsync(counter, 0, 4, 0));
    hipLaunchKernelGGL(line_lat_kernel, dim3(256), dim3(64), 0, 0, (const char*)buf, (const uint64_t*)doffs,
                       (uint32_t)num_pages, xcc, counter, dout);
    CHECK(hipDeviceSynchronize());
    r.resize(num_pages);
    CHECK(hipMemcpy(r.data(), dout, num_pages * 4, hipMemcpyDeviceToHost));
  };
  // lat[g][x][page]: granule g (0..3) of every page measured from XCC x
  std::vector<std::vector<std::vector<uint32_t>>> lat(4, std::vector<std::vector<uint32_t>>(8));
  for (int g = 0; g < 4; ++g)
    for (uint32_t x = 0; x < 8; ++x) probe((uint64_t)g * kGranule, x, lat[g][x]);
  std::vector<uint8_t> page_home(num_pages);
  size_t weak = 0, alt_viol = 0;
  std::vector<double> margins;
  for (size_t p = 0; p < num_pages; ++p) {
    // score > 0 means granule 0 is homed on IOD 1 (XCC 4-7 faster on even granules, slower on odd ones)
    double score = 0;
    int votes1 = 0;
    for (int g = 0; g < 4; ++g) {
      double m0 = 0, m1 = 0;
      for (int x = 0; x < 4; ++x) m0 += lat[g][x][p], m1 += lat[g][x + 4][p];
      const double d = (m0 - m1) / 4;  // > 0: IOD 1 is closer for granule g
      score += (g & 1) ? -d : d;
      votes1 += ((d > 0) ^ (g & 1)) ? 1 : 0;
    }
    page_home[p] = score > 0 ? 1 : 0;
    alt_viol += (votes1 != 0 && votes1 != 4);
    margins.push_back(std::abs(score) / 4 / cpn);
    weak += std::abs(score) / 4 / cpn < 40;
  }
  std::sort(margins.begin(), margins.end());
  printf("page probe (4 granules x 8 XCCs per page): pages whose 4 granules do not alternate 0101/1010: %zu/%zu; "
         "weak (<40 ns) %zu; per-granule margin p1 %.0f p10 %.0f p50 %.0f ns\n",
         alt_viol, num_pages, weak, margins[num_pages / 100], margins[num_pages / 10], margins[num_pages / 2]);
  size_t bit21_ok = 0;
  for (size_t p = 0; p < num_pages; ++p) bit21_ok += (page_home[p] == ((p & 1) ^ page_home[0]));
  printf("home(page) == bit21 rule: %zu/%zu; fraction of pages with granule0 on IOD1: %.3f\n", bit21_ok, num_pages,
         (double)std::count(page_home.begin(), page_home.end(), 1) / num_pages);
  printf("page home vs single address bits (agreement with page_home ^ home0):");
  for (int b = 21; (1ull << b) < bytes; ++b) {
    size_t agree = 0;
    for (size_t p = 0; p < num_pages; ++p) agree += (((p * kPage) >> b & 1) == (page_home[p] ^ page_home[0]));
    printf(" b%d:%.2f", b, (double)agree / num_pages);
  }
  printf("\nfirst 64 page phases: ");
  for (int p = 0; p < 64; ++p) printf("%d", page_home[p]);
  printf("\n");
  uint8_t* dhome;
  CHECK(hipMalloc(&dhome, num_pages));
  CHECK(hipMemcpy(dhome, page_home.data(), num_pages, hipMemcpyHostToDevice));

  // ---- 2) bandwidth + power per mode
  uint32_t *ctr, *sink;
  CHECK(hipMalloc(&ctr, 8));
  CHECK(hipMalloc(&sink, 4));
  const char* names[4] = {"naive (all XCCs, all granules)", "local (XCC grp g -> IOD g granules)",
                          "remote (XCC grp g -> other IOD)", "naive grid-stride (DPX test kernel)"};
  const int grid = prop.multiProcessorCount * 8;
  for (int rep = 0; rep < 2; ++rep) {
    for (int mode : {0, 1, 2, 3}) {
      // warmup
      auto launch = [&] {
        if (mode == 3) {
          hipLaunchKernelGGL(gs_kernel, dim3(prop.multiProcessorCount * 8), dim3(256), 0, 0, (const uint4*)buf,
                             bytes / sizeof(uint4), sink);
        } else {
          CHECK(hipMemsetAsync(ctr, 0, 8, 0));
          hipLaunchKernelGGL(bw_kernel, dim3(grid), dim3(256), 0, 0, (const uint4*)buf, (uint32_t)num_pairs, mode,
                             (const uint8_t*)dhome, ctr, sink);
        }
      };
      launch();
      CHECK(hipDeviceSynchronize());
      std::atomic<bool> stop{false};
      std::vector<double> pw, f1, f2;
      std::thread sampler([&] {
        while (!stop.load()) {
          const double v = read_uw(pfile);
          if (v > 0) pw.push_back(v / 1e6);
          f1.push_back(read_uw(hw + "/freq1_input") / 1e6);
          f2.push_back(read_uw(hw + "/freq2_input") / 1e6);
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
      // drop the first 20% of power samples (ramp)
      std::vector<double> steady(pw.begin() + pw.size() / 5, pw.end());
      double mean = 0;
      for (double v : steady) mean += v;
      mean /= std::max<size_t>(1, steady.size());
      const double gbps = (double)bytes * launches / el / 1e9;
      auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0.0 : v[v.size() / 2]; };
      printf("  [rep %d] %-38s %7.1f GB/s   power %.0f W (n=%zu)   %.2f GB/J   freq1 %.0f MHz  freq2 %.0f MHz\n", rep,
             names[mode], gbps, mean, steady.size(), gbps / mean, med(f1), med(f2));
    }
  }
  return 0;
}
