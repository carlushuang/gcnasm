// GEMM-like B-tile read bandwidth (no MFMA): a tile is 256 rows x 8 KiB, read as 64 k-steps of 128 B per row with
// 8 lanes x 16 B per row (the opus 8-wave kernel's B load pattern). Compares
//   layout 0: row-major (row stride 8 KiB, consecutive rows alternate IODs)
//   layout 1: IOD-placed-like (row stride 16 KiB: tile 2j uses the even 8 KiB granules of a 4 MiB region, tile 2j+1 the
//             odd ones, so the 128 rows of a tile inside one 2 MiB page share one IOD)
// optionally with a per-row XOR swizzle of the 128 B k-chunk, both L2-resident (every WG re-reads one tile) and
// HBM-streaming (distinct tiles over a 4 GiB buffer).
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CHECK(x)                                                                                   \
  do {                                                                                             \
    hipError_t e_ = (x);                                                                           \
    if (e_ != hipSuccess) {                                                                        \
      fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_));          \
      exit(1);                                                                                     \
    }                                                                                              \
  } while (0)

__device__ __forceinline__ size_t row_addr(uint32_t tile, uint32_t row, int layout) {
  return layout == 0 ? ((size_t)tile * 256 + row) * 8192
                     : (size_t)(tile >> 1) * (4u << 20) + (size_t)(2 * row + (tile & 1)) * 8192;
}

__global__ __launch_bounds__(512) void tile_read(const char* __restrict__ base, uint32_t num_tiles, int layout,
                                                  int swizzle, int reps, int resident, uint32_t* sink) {
  const uint32_t t = threadIdx.x, lane_k = t & 7, row0 = t >> 3;  // 64 rows per instruction, 8 lanes x 16 B per row
  uint32_t acc = 0;
  for (int r = 0; r < reps; ++r) {
    for (uint32_t tile = resident ? 0 : blockIdx.x; tile < num_tiles; tile += resident ? num_tiles : gridDim.x) {
      size_t ra[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) ra[i] = row_addr(tile, row0 + 64 * i, layout) + lane_k * 16;
#pragma unroll 4
      for (uint32_t k = 0; k < 64; ++k) {
        uint4 v[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const uint32_t chunk = swizzle ? (k ^ ((row0 + 64 * i) & 63)) : k;
          v[i] = *reinterpret_cast<const uint4*>(base + ra[i] + chunk * 128);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) acc ^= v[i].x ^ v[i].y ^ v[i].z ^ v[i].w;
      }
    }
  }
  if (acc == 0x9e3779b9u) sink[0] = acc;
}

int main(int argc, char** argv) {
  const int dev = argc > 1 ? atoi(argv[1]) : 0;
  CHECK(hipSetDevice(dev));
  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, dev));
  const size_t bytes = 4ull << 30;
  const uint32_t num_tiles = bytes / (2u << 20);  // 2 MiB of data per tile
  char* buf;
  uint32_t* sink;
  CHECK(hipMalloc(&buf, bytes));
  CHECK(hipMemset(buf, 1, bytes));
  CHECK(hipMalloc(&sink, 4));
  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0));
  CHECK(hipEventCreate(&e1));
  struct V {
    const char* name;
    int layout, swz;
  } vs[] = {{"row-major (8 KiB stride)", 0, 0},
            {"IOD-placed-like (16 KiB stride)", 1, 0},
            {"IOD-placed-like + k-chunk XOR swizzle", 1, 1},
            {"row-major + k-chunk XOR swizzle", 0, 1}};
  for (int resident = 1; resident >= 0; --resident) {
    const int grid = prop.multiProcessorCount * (resident ? 1 : 2), reps = resident ? 32 : 1;
    for (int round = 0; round < 2; ++round)
      for (const auto& v : vs) {
        hipLaunchKernelGGL(tile_read, dim3(grid), dim3(512), 0, 0, buf, num_tiles, v.layout, v.swz, reps, resident,
                           sink);
        CHECK(hipDeviceSynchronize());
        float best = 1e9f;
        for (int i = 0; i < 5; ++i) {
          CHECK(hipEventRecord(e0, 0));
          hipLaunchKernelGGL(tile_read, dim3(grid), dim3(512), 0, 0, buf, num_tiles, v.layout, v.swz, reps, resident,
                             sink);
          CHECK(hipEventRecord(e1, 0));
          CHECK(hipEventSynchronize(e1));
          float ms = 0;
          CHECK(hipEventElapsedTime(&ms, e0, e1));
          best = ms < best ? ms : best;
        }
        const double b = resident ? (double)grid * reps * (2u << 20) : (double)bytes;
        printf("[%s round %d] %-40s %8.1f GB/s\n", resident ? "L2-resident" : "HBM 4 GiB", round, v.name,
               b / (best * 1e-3) / 1e9);
      }
  }
  return 0;
}
