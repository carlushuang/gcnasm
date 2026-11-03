// Concurrent streaming-read bandwidth of two DPX partitions of one MI355X (NPS2):
// local+local vs cross+cross, to measure the inter-domain (IOD-IOD) link under bidirectional load.
// Usage (DPX + NPS2): ./dpx_concurrent_bw [dev0=0] [dev1=1]   (the two partitions of one GPU)
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

__global__ __launch_bounds__(256) void bw_kernel(const uint4* __restrict__ p, size_t n_vec, int iters, uint32_t* sink) {
  uint32_t acc = 0;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (int it = 0; it < iters; ++it) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i + 3 * stride < n_vec; i += 4 * stride) {
      const uint4 a = p[i], b = p[i + stride], c = p[i + 2 * stride], d = p[i + 3 * stride];
      acc ^= a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w ^ c.x ^ c.y ^ c.z ^ c.w ^ d.x ^ d.y ^ d.z ^ d.w;
    }
    for (; i < n_vec; i += stride) {
      const uint4 a = p[i];
      acc ^= a.x ^ a.y ^ a.z ^ a.w;
    }
  }
  if (acc == 0x9e3779b9u) sink[0] = acc;
}

int main(int argc, char** argv) {
  const int d0 = argc > 1 ? atoi(argv[1]) : 0, d1 = argc > 2 ? atoi(argv[2]) : 1;
  const size_t bytes = 4ull << 30, n_vec = bytes / sizeof(uint4);
  const int iters = 8;
  void *buf0, *buf1;
  uint32_t *s0, *s1;
  CHECK(hipSetDevice(d0));
  CHECK(hipMalloc(&buf0, bytes));
  CHECK(hipMemset(buf0, 1, bytes));
  CHECK(hipMalloc(&s0, 4));
  CHECK(hipDeviceEnablePeerAccess(d1, 0));
  CHECK(hipSetDevice(d1));
  CHECK(hipMalloc(&buf1, bytes));
  CHECK(hipMemset(buf1, 2, bytes));
  CHECK(hipMalloc(&s1, 4));
  CHECK(hipDeviceEnablePeerAccess(d0, 0));

  auto run = [&](const char* name, const void* src0, const void* src1, bool use0, bool use1) {
    for (int warm = 0; warm < 2; ++warm) {
      const auto t0 = std::chrono::steady_clock::now();
      if (use0) {
        CHECK(hipSetDevice(d0));
        hipLaunchKernelGGL(bw_kernel, dim3(128 * 8), dim3(256), 0, 0, (const uint4*)src0, n_vec, iters, s0);
      }
      if (use1) {
        CHECK(hipSetDevice(d1));
        hipLaunchKernelGGL(bw_kernel, dim3(128 * 8), dim3(256), 0, 0, (const uint4*)src1, n_vec, iters, s1);
      }
      if (use0) { CHECK(hipSetDevice(d0)); CHECK(hipDeviceSynchronize()); }
      if (use1) { CHECK(hipSetDevice(d1)); CHECK(hipDeviceSynchronize()); }
      const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      const double total = (double)bytes * iters * ((use0 ? 1 : 0) + (use1 ? 1 : 0));
      if (warm == 1) printf("  %-44s aggregate %7.1f GB/s  (%.0f ms)\n", name, total / s / 1e9, s * 1e3);
    }
  };
  printf("devices %d, %d (DPX partitions of one MI355X, NPS2); 4 GiB x %d reads per partition\n", d0, d1, iters);
  run("dev0<-local only", buf0, nullptr, true, false);
  run("dev0<-local + dev1<-local (concurrent)", buf0, buf1, true, true);
  run("dev0<-remote only", buf1, nullptr, true, false);
  run("dev0<-remote + dev1<-remote (concurrent)", buf1, buf0, true, true);
  run("dev0<-local + dev1<-remote(dev0 mem) (concurrent)", buf0, buf0, true, true);
  return 0;
}
