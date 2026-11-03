// Page-phase probe for MI355X in SPX + NPS1 (see gcnasm nps_locality): returns, for every 2 MiB page of a device
// buffer, the home IOD of the page's first 8 KiB granule (granule i of the page is homed on phase ^ (i & 1)).
// Built as a shared library and called from Python via ctypes on a torch-allocated buffer.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#define GETREG_FULL(REG) ((31 << 11) | (0 << 6) | (REG))

__global__ __launch_bounds__(64) void granule_lat_kernel(const char* base, uint32_t n, uint32_t target_xcc,
                                                          uint32_t* counter, uint16_t* out) {
    if (threadIdx.x != 0) return;
    if ((__builtin_amdgcn_s_getreg(GETREG_FULL(20)) & 0xf) != target_xcc) return;
    while (true) {
        const uint32_t i = atomicAdd(counter, 1u);
        if (i >= n) break;
        const uint32_t* p = reinterpret_cast<const uint32_t*>(base + (uint64_t)i * 8192);
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

// phase_out[p] = home IOD (0: XCC 0-3, 1: XCC 4-7) of granule 0 of page p; returns 0 on success
extern "C" int iod_page_phases(void* buf, size_t bytes, unsigned char* phase_out, int* weak_out) {
    const uint32_t ngran = (uint32_t)(bytes / 8192), npages = (uint32_t)(bytes >> 21);
    const size_t scratch_bytes = 2ull << 30;
    void* scratch = nullptr;
    uint32_t* counter = nullptr;
    uint16_t* dout = nullptr;
    if (hipMalloc(&scratch, scratch_bytes) != hipSuccess || hipMalloc(&counter, 4) != hipSuccess ||
        hipMalloc(&dout, (size_t)ngran * 2) != hipSuccess)
        return 1;
    std::vector<std::vector<uint16_t>> lat(8, std::vector<uint16_t>(ngran));
    for (uint32_t x = 0; x < 8; ++x) {
        if (hipMemsetAsync(scratch, (int)(x * 7 + 1), scratch_bytes, 0) != hipSuccess) return 2;
        if (hipMemsetAsync(counter, 0, 4, 0) != hipSuccess) return 2;
        granule_lat_kernel<<<256, 64>>>(reinterpret_cast<const char*>(buf), ngran, x, counter, dout);
        if (hipGetLastError() != hipSuccess || hipDeviceSynchronize() != hipSuccess) return 3;
        if (hipMemcpy(lat[x].data(), dout, (size_t)ngran * 2, hipMemcpyDeviceToHost) != hipSuccess) return 4;
    }
    (void)hipFree(scratch);
    (void)hipFree(counter);
    (void)hipFree(dout);
    int weak = 0;
    for (uint32_t p = 0; p < npages; ++p) {
        double score = 0;
        for (uint32_t i = 0; i < 256; ++i) {
            const uint32_t gi = p * 256 + i;
            double m0 = 0, m1 = 0;
            for (int x = 0; x < 4; ++x) m0 += lat[x][gi], m1 += lat[x + 4][gi];
            score += (i & 1) ? (m1 - m0) : (m0 - m1);  // > 0: granule 0 is closer to XCC 4..7 (IOD 1)
        }
        phase_out[p] = score > 0;
        weak += std::abs(score) / 256 < 4 * 48;
    }
    *weak_out = weak;
    return 0;
}
