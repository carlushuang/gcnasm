// NPS-domain probe for MI355X (gfx950) in DPX/QPX/CPX + NPS2.
// 1) per-workgroup hardware IDs (HW_REG_XCC_ID, HW_REG_HW_ID) on every partition device,
// 2) pointer-chase latency and streaming bandwidth from a partition to memory allocated on
//    itself, on its sibling partition (other NPS domain, same package), and on another physical GPU.
// Usage (DPX + NPS2; HIP devices 2k / 2k+1 are the two partitions of one GPU): ./dpx_partition_probe [run_dev=0] [sibling_dev=1] [far_dev=2]
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#define CHECK(x)                                                                                   \
  do {                                                                                             \
    hipError_t e_ = (x);                                                                           \
    if (e_ != hipSuccess) {                                                                        \
      fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_));          \
      exit(1);                                                                                     \
    }                                                                                              \
  } while (0)

// s_getreg immediate: SIZE-1 in [15:11], OFFSET in [10:6], HWREG id in [5:0]
#define GETREG_FULL(REG) ((31 << 11) | (0 << 6) | (REG))
constexpr int kHwRegHwId = 4;
constexpr int kHwRegXccId = 20;

__device__ __forceinline__ uint32_t read_xcc_reg() { return __builtin_amdgcn_s_getreg(GETREG_FULL(kHwRegXccId)); }
__device__ __forceinline__ uint32_t read_hwid_reg() { return __builtin_amdgcn_s_getreg(GETREG_FULL(kHwRegHwId)); }

// ---------------------------------------------------------------- 1) IDs
__global__ __launch_bounds__(64) void id_kernel(uint32_t* out) {
  if (threadIdx.x == 0) {
    out[blockIdx.x * 3 + 0] = read_xcc_reg();
    out[blockIdx.x * 3 + 1] = read_hwid_reg();
    out[blockIdx.x * 3 + 2] = __smid();
  }
}

// ---------------------------------------------------------------- 2) latency
// Each workgroup owns `chunks` 4 KiB chunks. Inside a chunk, word 0 of line l points to line (l + 7) % 32,
// so 31 timed dependent hops touch 31 distinct cold 128 B lines (the untimed first load warms the TLB).
constexpr int kLineBytes = 128, kChunkBytes = 4096, kLines = kChunkBytes / kLineBytes, kWordsPerLine = kLineBytes / 4;
constexpr int kStride = 7, kTimedHops = kLines - 1;

__global__ __launch_bounds__(64) void chase_kernel(const uint32_t* __restrict__ buf, int chunks, uint64_t* ticks,
                                                    uint32_t* hops, uint32_t* xcc, uint32_t* hwid) {
  if (threadIdx.x != 0) return;
  uint64_t total = 0;
  uint32_t sink = 0;
  for (int c = 0; c < chunks; ++c) {
    const uint32_t* chunk = buf + ((size_t)blockIdx.x * chunks + c) * (kChunkBytes / 4);
    // NOTES: launder the index through a VGPR so the chase uses vector global loads, not scalar s_load
    uint32_t idx = 0;
    asm volatile("" : "+v"(idx));
    idx = chunk[idx];
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    const uint64_t t0 = wall_clock64();
#pragma unroll
    for (int i = 0; i < kTimedHops; ++i) {
      asm volatile("" : "+v"(idx));
      idx = chunk[idx];
    }
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    const uint64_t t1 = wall_clock64();
    total += t1 - t0;
    sink += idx;
  }
  ticks[blockIdx.x] = total;
  hops[blockIdx.x] = chunks * kTimedHops + (sink == 0xffffffffu ? 1 : 0);
  xcc[blockIdx.x] = read_xcc_reg();
  hwid[blockIdx.x] = read_hwid_reg();
}

// ---------------------------------------------------------------- 3) bandwidth
__global__ __launch_bounds__(256) void bw_kernel(const uint4* __restrict__ p, size_t n_vec, uint32_t* sink) {
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

// ---------------------------------------------------------------- host helpers
struct HwId {
  uint32_t xcc, se, sh, cu, simd, wave, queue, me;
};
static HwId decode(uint32_t xcc_reg, uint32_t hw) {
  return {xcc_reg & 0xf, (hw >> 13) & 0x3, (hw >> 12) & 0x1, (hw >> 8) & 0xf,
          (hw >> 4) & 0x3, hw & 0xf, (hw >> 24) & 0x7, (hw >> 30) & 0x3};
}

static double median(std::vector<double> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

static void flush_all(const std::vector<int>& devs, std::vector<void*>& scratch, size_t bytes) {
  for (size_t k = 0; k < devs.size(); ++k) {
    CHECK(hipSetDevice(devs[k]));
    CHECK(hipMemsetAsync(scratch[k], (int)k + 1, bytes, 0));
  }
  for (int d : devs) {
    CHECK(hipSetDevice(d));
    CHECK(hipDeviceSynchronize());
  }
}

int main(int argc, char** argv) {
  int n = 0;
  CHECK(hipGetDeviceCount(&n));
  printf("HIP devices: %d\n", n);
  for (int d = 0; d < n; ++d) {
    hipDeviceProp_t p;
    CHECK(hipGetDeviceProperties(&p, d));
    int wclk = 0;
    CHECK(hipDeviceGetAttribute(&wclk, hipDeviceAttributeWallClockRate, d));
    printf("  dev %2d: %s %s pci %04x:%02x:%02x CUs=%d mem=%.1f GiB wallclock=%d kHz\n", d, p.name, p.gcnArchName,
           p.pciDomainID, p.pciBusID, p.pciDeviceID, p.multiProcessorCount, p.totalGlobalMem / 1073741824.0, wclk);
  }

  // ---- 1) IDs on every device
  printf("\n== per-device hardware IDs (1024 WGs x 64 threads) ==\n");
  const int kIdBlocks = 1024;
  for (int d = 0; d < n; ++d) {
    CHECK(hipSetDevice(d));
    uint32_t* dout;
    CHECK(hipMalloc(&dout, kIdBlocks * 3 * sizeof(uint32_t)));
    hipLaunchKernelGGL(id_kernel, dim3(kIdBlocks), dim3(64), 0, 0, dout);
    CHECK(hipDeviceSynchronize());
    std::vector<uint32_t> h(kIdBlocks * 3);
    CHECK(hipMemcpy(h.data(), dout, h.size() * 4, hipMemcpyDeviceToHost));
    CHECK(hipFree(dout));
    std::set<uint32_t> xcc_raw, xccs, smids;
    std::set<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> cus;
    std::map<uint32_t, std::set<uint32_t>> cus_per_xcc;
    std::string first16;
    for (int b = 0; b < kIdBlocks; ++b) {
      const HwId id = decode(h[b * 3], h[b * 3 + 1]);
      xcc_raw.insert(h[b * 3]);
      xccs.insert(id.xcc);
      smids.insert(h[b * 3 + 2]);
      cus.insert({id.xcc, id.se, id.sh, id.cu});
      cus_per_xcc[id.xcc].insert((id.se << 5) | (id.sh << 4) | id.cu);
      if (b < 16) first16 += std::to_string(id.xcc) + (b < 15 ? "," : "");
    }
    printf("  dev %2d: XCC_ID raw32 = {", d);
    for (uint32_t v : xcc_raw) printf(" 0x%x", v);
    printf(" }  unique (xcc,se,sh,cu)=%zu  unique __smid=%zu  wg0..15 -> xcc [%s]  CUs/xcc:", cus.size(),
           smids.size(), first16.c_str());
    for (auto& [x, s] : cus_per_xcc) printf(" %u:%zu", x, s.size());
    printf("\n");
  }

  // ---- 2) + 3) need a run device, its sibling partition and a device on another physical GPU
  const int run0 = argc > 1 ? atoi(argv[1]) : 0;
  const int sib = argc > 2 ? atoi(argv[2]) : 1;
  const int far = argc > 3 ? atoi(argv[3]) : 2;
  std::vector<int> devs = {run0, sib, far};
  printf("\n== memory placement test: run devices {%d,%d}, memory on {%d,%d,%d} ==\n", run0, sib, run0, sib, far);
  for (int a : {run0, sib}) {
    for (int b : devs) {
      if (a == b) continue;
      int can = 0;
      CHECK(hipDeviceCanAccessPeer(&can, a, b));
      CHECK(hipSetDevice(a));
      hipError_t e = hipDeviceEnablePeerAccess(b, 0);
      if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) {
        printf("  enable peer %d->%d failed: %s\n", a, b, hipGetErrorString(e));
        return 1;
      }
      (void)hipGetLastError();
      int link_type = -1, hops = -1;
      uint32_t lt = 0, hc = 0;
      if (hipExtGetLinkTypeAndHopCount(a, b, &lt, &hc) == hipSuccess) {
        link_type = (int)lt;
        hops = (int)hc;
      }
      int perf = -1;
      (void)hipDeviceGetP2PAttribute(&perf, hipDevP2PAttrPerformanceRank, a, b);
      printf("  peer %d -> %d: canAccess=%d linkType=%d hops=%d perfRank=%d\n", a, b, can, link_type, hops, perf);
    }
  }

  // Allocate per memory device: chase buffer, bandwidth buffer, flush scratch
  const int kChaseBlocks = 512, kChunks = 32;
  const size_t chase_bytes = (size_t)kChaseBlocks * kChunks * kChunkBytes;
  const size_t bw_bytes = 4ull << 30, flush_bytes = 2ull << 30;
  std::vector<void*> chase(devs.size()), bwbuf(devs.size()), scratch(devs.size());
  std::vector<uint32_t> chain(chase_bytes / 4, 0);
  for (size_t w = 0; w < chain.size(); w += kChunkBytes / 4)
    for (int l = 0; l < kLines; ++l) chain[w + l * kWordsPerLine] = ((l + kStride) % kLines) * kWordsPerLine;
  for (size_t k = 0; k < devs.size(); ++k) {
    CHECK(hipSetDevice(devs[k]));
    CHECK(hipMalloc(&chase[k], chase_bytes));
    CHECK(hipMemcpy(chase[k], chain.data(), chase_bytes, hipMemcpyHostToDevice));
    CHECK(hipMalloc(&bwbuf[k], bw_bytes));
    CHECK(hipMemset(bwbuf[k], 0x5a, bw_bytes));
    CHECK(hipMalloc(&scratch[k], flush_bytes));
  }

  for (int run : {run0, sib}) {
    CHECK(hipSetDevice(run));
    int wclk_khz = 0;
    CHECK(hipDeviceGetAttribute(&wclk_khz, hipDeviceAttributeWallClockRate, run));
    uint64_t* dticks;
    uint32_t *dhops, *dxcc, *dhw, *dsink;
    CHECK(hipMalloc(&dticks, kChaseBlocks * 8));
    CHECK(hipMalloc(&dhops, kChaseBlocks * 4));
    CHECK(hipMalloc(&dxcc, kChaseBlocks * 4));
    CHECK(hipMalloc(&dhw, kChaseBlocks * 4));
    CHECK(hipMalloc(&dsink, 4));
    for (size_t k = 0; k < devs.size(); ++k) {
      const int mem = devs[k];
      // latency, repeated 3x with cache flushes on every involved device in between
      std::map<uint32_t, std::vector<double>> per_xcc;
      std::vector<double> all;
      for (int rep = 0; rep < 3; ++rep) {
        flush_all(devs, scratch, flush_bytes);
        CHECK(hipSetDevice(run));
        hipLaunchKernelGGL(chase_kernel, dim3(kChaseBlocks), dim3(64), 0, 0, (const uint32_t*)chase[k], kChunks,
                           dticks, dhops, dxcc, dhw);
        CHECK(hipDeviceSynchronize());
        std::vector<uint64_t> t(kChaseBlocks);
        std::vector<uint32_t> hp(kChaseBlocks), xc(kChaseBlocks);
        CHECK(hipMemcpy(t.data(), dticks, kChaseBlocks * 8, hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(hp.data(), dhops, kChaseBlocks * 4, hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(xc.data(), dxcc, kChaseBlocks * 4, hipMemcpyDeviceToHost));
        for (int b = 0; b < kChaseBlocks; ++b) {
          const double ns = (double)t[b] / hp[b] * 1e6 / wclk_khz;
          per_xcc[xc[b] & 0xf].push_back(ns);
          all.push_back(ns);
        }
      }
      // bandwidth
      CHECK(hipSetDevice(run));
      const size_t n_vec = bw_bytes / sizeof(uint4);
      hipEvent_t e0, e1;
      CHECK(hipEventCreate(&e0));
      CHECK(hipEventCreate(&e1));
      hipLaunchKernelGGL(bw_kernel, dim3(128 * 8), dim3(256), 0, 0, (const uint4*)bwbuf[k], n_vec, dsink);
      CHECK(hipDeviceSynchronize());
      float best_ms = 1e9f;
      for (int rep = 0; rep < 5; ++rep) {
        CHECK(hipEventRecord(e0, 0));
        hipLaunchKernelGGL(bw_kernel, dim3(128 * 8), dim3(256), 0, 0, (const uint4*)bwbuf[k], n_vec, dsink);
        CHECK(hipEventRecord(e1, 0));
        CHECK(hipEventSynchronize(e1));
        float ms = 0;
        CHECK(hipEventElapsedTime(&ms, e0, e1));
        best_ms = std::min(best_ms, ms);
      }
      CHECK(hipEventDestroy(e0));
      CHECK(hipEventDestroy(e1));
      const char* tag = mem == run ? "local (same partition / NPS domain)"
                        : (mem == run0 || mem == sib) ? "sibling partition (other NPS domain, same GPU)"
                                                      : "other physical GPU (XGMI)";
      printf("  run dev %d -> mem dev %d [%s]\n", run, mem, tag);
      printf("      chase latency median %.0f ns (p10 %.0f, p90 %.0f); per XCC_ID:", median(all),
             [&] { auto v = all; std::sort(v.begin(), v.end()); return v[v.size() / 10]; }(),
             [&] { auto v = all; std::sort(v.begin(), v.end()); return v[v.size() * 9 / 10]; }());
      for (auto& [x, v] : per_xcc) printf(" %u:%.0f", x, median(v));
      printf("\n      stream read %.1f GB/s (4 GiB, best of 5)\n", bw_bytes / (best_ms * 1e-3) / 1e9);
    }
    CHECK(hipFree(dticks));
    CHECK(hipFree(dhops));
    CHECK(hipFree(dxcc));
    CHECK(hipFree(dhw));
    CHECK(hipFree(dsink));
  }
  return 0;
}
