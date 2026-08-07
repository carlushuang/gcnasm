// fabric_symm.hip -- HIP VMM fabric-handle symmetric window + cross-GPU copy kernels.
//
// Three ways to obtain a fabric-capable buffer, and one shared path for using it.
//
//   fs_alloc          allocate a fabric window ourselves          (method "own")
//   -- none --        torch allocates it fabric-capable, because
//                     fabric_shim.cpp rewrote the handle type      (method "shim")
//   fs_rebind_fabric  rebind a torch buffer's VA onto fabric
//                     backing, in place                            (method "rebind")
//
// After that the path is identical for all three: fs_export_ptr turns the buffer into a
// 64-byte POD fabric handle, fs_import maps a peer's handle, and the copy kernel moves
// data across. Ranks exchange handles however they like -- main.py uses torch.distributed.
//
// This file is the raw HIP half; hip_fabric.py binds it with ctypes and main.py drives it.
// Everything is a plain C ABI so ctypes can call it without pybind11.

#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdint>

// hipMemFabricHandle_t is `unsigned char data[HIP_IPC_HANDLE_SIZE]`
#define FS_FABRIC_HANDLE_BYTES 64

#define FS_THREADS 512
#define FS_BLOCKS_PER_CU 16

// Last HIP error, so the Python side can print something better than an errno.
static hipError_t g_last = hipSuccess;

// Run a HIP call, stash the error and bail out with it as the return code.
#define TRY(x)                          \
    do {                                \
        g_last = (x);                   \
        if (g_last != hipSuccess) {     \
            return (int)g_last;         \
        }                               \
    } while (0)

extern "C" const char* fs_last_error() { return hipGetErrorString(g_last); }

// Both probes below deliberately make HIP calls that are expected to fail. A failed
// API call leaves hipGetLastError() armed, and the next torch kernel launch checks it
// and reports our probe failure as a spurious "CUDA error" from unrelated code. So
// swallow the error we caused before handing control back.
static inline void clear_sticky_error() {
    hipError_t ignored = hipGetLastError();
    (void)ignored;
}

extern "C" int fs_device_count(int* n) {
    TRY(hipGetDeviceCount(n));
    return 0;
}

// Is fabric usable on this device? Probed functionally -- one granularity-sized
// alloc/export/import round trip -- rather than via hipDeviceGetAttribute.
//
// The attribute route is deliberately avoided: hipDeviceAttribute_t is a long
// sequential enum whose values shift between HIP releases, and a torch wheel often
// loads a different libamdhip64 than the one this file was compiled against (e.g. a
// pip _rocm_sdk_core runtime vs /opt/rocm). The VMM structs and the enums used below
// carry explicit values and are stable, so this probe is version-proof.
//
// failed_step: 0 ok, 1 granularity, 2 create, 3 export, 4 import. On failure the
// matching HIP error is left in fs_last_error().
extern "C" int fs_fabric_supported(int dev, int* supported, int* failed_step) {
    *supported = 0;
    *failed_step = 0;
    TRY(hipSetDevice(dev));

    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    prop.requestedHandleTypes = hipMemHandleTypeFabric;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = dev;

    size_t gran = 0;
    g_last = hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityRecommended);
    if (g_last != hipSuccess) {
        *failed_step = 1;
        clear_sticky_error();
        return 0;
    }

    hipMemGenericAllocationHandle_t h;
    g_last = hipMemCreate(&h, gran, &prop, 0);
    if (g_last != hipSuccess) {
        *failed_step = 2;
        clear_sticky_error();
        return 0;
    }

    unsigned char fh[FS_FABRIC_HANDLE_BYTES];
    g_last = hipMemExportToShareableHandle(fh, h, hipMemHandleTypeFabric, 0);
    if (g_last == hipSuccess) {
        hipMemGenericAllocationHandle_t imported;
        g_last = hipMemImportFromShareableHandle(&imported, fh, hipMemHandleTypeFabric);
        if (g_last == hipSuccess) {
            *supported = 1;
            hipError_t rel = hipMemRelease(imported);
            (void)rel;
        } else {
            *failed_step = 4;
        }
    } else {
        *failed_step = 3;
    }

    hipError_t rel = hipMemRelease(h);
    (void)rel;
    clear_sticky_error();
    return 0;
}

// Free/total device memory, so the caller can price each method's physical cost.
extern "C" int fs_mem_info(size_t* free_bytes, size_t* total_bytes) {
    TRY(hipMemGetInfo(free_bytes, total_bytes));
    return 0;
}

// Allocate a fabric-capable window ourselves (method "own"). torch is not involved, so
// the result is not a symm_mem tensor -- main.py wraps it as a tensor via
// __cuda_array_interface__. Exactly one allocation, so no memory is wasted.
extern "C" int fs_alloc(int dev,
                        size_t bytes,
                        void** out_ptr,
                        uint64_t* out_handle,
                        size_t* out_total) {
    TRY(hipSetDevice(dev));

    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    prop.requestedHandleTypes = hipMemHandleTypeFabric;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = dev;

    size_t gran = 0;
    TRY(hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityRecommended));
    size_t total = ((bytes + gran - 1) / gran) * gran;

    hipMemGenericAllocationHandle_t h;
    TRY(hipMemCreate(&h, total, &prop, 0));

    void* p = nullptr;
    TRY(hipMemAddressReserve(&p, total, 0, 0, 0));
    TRY(hipMemMap(p, total, 0, h, 0));

    hipMemAccessDesc ad = {};
    ad.location.type = hipMemLocationTypeDevice;
    ad.location.id = dev;
    ad.flags = hipMemAccessFlagsProtReadWrite;
    TRY(hipMemSetAccess(p, total, &ad, 1));

    *out_ptr = p;
    *out_handle = (uint64_t)h;
    *out_total = total;
    return 0;
}

// Tear down a window from fs_alloc (we own the reservation, the mapping and the handle).
extern "C" int fs_free_own(void* ptr, size_t total, uint64_t handle) {
    TRY(hipMemUnmap(ptr, total));
    TRY(hipMemAddressFree(ptr, total));
    TRY(hipMemRelease((hipMemGenericAllocationHandle_t)handle));
    return 0;
}

// Make a buffer that torch already allocated fabric-capable, in place.
//
// The set of shareable handle types is frozen at hipMemCreate and there is no API to
// change it afterwards -- hipMemGetAllocationPropertiesFromHandle only reads it back. But
// VMM separates the virtual address from its physical backing, so instead of changing the
// flag we swap the backing: allocate fabric-capable memory, unmap torch's range, and map
// ours at the exact same address. torch's pointer is unchanged and its tensor keeps
// working; the allocation simply became exportable.
//
// Two consequences the caller must respect:
//   * the previous contents are gone, so call this before the buffer is filled;
//   * torch still references the original allocation, which is now unmapped but not
//     freed, so the buffer costs twice its size in physical memory until torch frees it.
//
// We release our own handle immediately: the mapping holds its own reference, so the
// memory lives exactly as long as the mapping does and torch's free path stays correct.
extern "C" int fs_rebind_fabric(void* ptr, void** out_base, size_t* out_size) {
    void* base = nullptr;
    size_t size = 0;
    TRY(hipMemGetAddressRange(&base, &size, ptr));

    int dev = 0;
    TRY(hipGetDevice(&dev));

    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    prop.requestedHandleTypes = hipMemHandleTypeFabric;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = dev;

    hipMemGenericAllocationHandle_t h;
    TRY(hipMemCreate(&h, size, &prop, 0));

    // From here the range is briefly unbacked; any failure leaves the tensor unusable,
    // so surface the error rather than trying to limp on.
    TRY(hipMemUnmap(base, size));
    TRY(hipMemMap(base, size, 0, h, 0));

    hipMemAccessDesc ad = {};
    ad.location.type = hipMemLocationTypeDevice;
    ad.location.id = dev;
    ad.flags = hipMemAccessFlagsProtReadWrite;
    TRY(hipMemSetAccess(base, size, &ad, 1));

    TRY(hipMemRelease(h));

    *out_base = base;
    *out_size = size;
    return 0;
}

// Export a buffer that somebody else allocated -- specifically a torch
// symmetric-memory tensor -- as a fabric handle.
//
// hipMemGetAddressRange recovers the base and length of the VMM allocation backing the
// pointer (torch gives each symm_mem.empty its own granularity-rounded allocation), and
// hipMemRetainAllocationHandle takes a reference on its physical handle. torch keeps
// ownership of the mapping; we only borrow the handle, so release it with
// fs_release_handle and never unmap.
//
// This only succeeds on a fabric-capable allocation, which for a torch buffer means
// fs_rebind_fabric has been called on it first.
extern "C" int fs_export_ptr(void* ptr,
                             unsigned char* out_fh,
                             uint64_t* out_handle,
                             void** out_base,
                             size_t* out_size) {
    void* base = nullptr;
    size_t size = 0;
    TRY(hipMemGetAddressRange(&base, &size, ptr));

    hipMemGenericAllocationHandle_t h;
    TRY(hipMemRetainAllocationHandle(&h, base));

    hipError_t err = hipMemExportToShareableHandle(out_fh, h, hipMemHandleTypeFabric, 0);
    if (err != hipSuccess) {
        g_last = err;
        hipError_t rel = hipMemRelease(h);
        (void)rel;
        clear_sticky_error();
        return (int)err;
    }

    *out_handle = (uint64_t)h;
    *out_base = base;
    *out_size = size;
    return 0;
}

// Drop the reference taken by fs_export_ptr. Does not unmap: torch owns the mapping.
extern "C" int fs_release_handle(uint64_t handle) {
    TRY(hipMemRelease((hipMemGenericAllocationHandle_t)handle));
    return 0;
}

// Map a peer's exported window into this rank's address space. The resulting pointer is
// an ordinary device pointer: kernels running on `dev` can load and store through it.
extern "C" int fs_import(int dev,
                         const unsigned char* fh,
                         size_t total,
                         void** out_ptr,
                         uint64_t* out_handle) {
    TRY(hipSetDevice(dev));

    hipMemGenericAllocationHandle_t h;
    TRY(hipMemImportFromShareableHandle(&h, (void*)fh, hipMemHandleTypeFabric));

    void* p = nullptr;
    TRY(hipMemAddressReserve(&p, total, 0, 0, 0));
    TRY(hipMemMap(p, total, 0, h, 0));

    hipMemAccessDesc ad = {};
    ad.location.type = hipMemLocationTypeDevice;
    ad.location.id = dev;
    ad.flags = hipMemAccessFlagsProtReadWrite;
    TRY(hipMemSetAccess(p, total, &ad, 1));

    *out_ptr = p;
    *out_handle = (uint64_t)h;
    return 0;
}

// Tear down a peer mapping created by fs_import (we own its VA reservation).
extern "C" int fs_release_import(void* ptr, size_t total, uint64_t handle) {
    TRY(hipMemUnmap(ptr, total));
    TRY(hipMemAddressFree(ptr, total));
    TRY(hipMemRelease((hipMemGenericAllocationHandle_t)handle));
    return 0;
}

// Ask whether an arbitrary device pointer is VMM-backed and fabric-exportable.
// Used to show what torch.distributed._symmetric_memory hands you on ROCm.
extern "C" int fs_probe_ptr(void* ptr, int* retain_rc, int* export_fabric_rc, int* export_fd_rc) {
    hipMemGenericAllocationHandle_t h = nullptr;
    *retain_rc = (int)hipMemRetainAllocationHandle(&h, ptr);
    if (*retain_rc != (int)hipSuccess) {
        *export_fabric_rc = -1;
        *export_fd_rc = -1;
        clear_sticky_error();
        return 0;
    }

    unsigned char fh[FS_FABRIC_HANDLE_BYTES];
    *export_fabric_rc = (int)hipMemExportToShareableHandle(fh, h, hipMemHandleTypeFabric, 0);

    int fd = -1;
    *export_fd_rc =
        (int)hipMemExportToShareableHandle(&fd, h, hipMemHandleTypePosixFileDescriptor, 0);

    hipError_t rel = hipMemRelease(h);
    (void)rel;
    clear_sticky_error();
    return 0;
}

// Grid-stride uint4 copy. dst or src (or both) may be a fabric-imported peer pointer.
__global__ void copy_u4(uint4* __restrict__ dst, const uint4* __restrict__ src, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        dst[i] = src[i];
    }
}

static inline int launch_copy(int num_cu, void* dst, const void* src, size_t bytes) {
    size_t n = bytes / sizeof(uint4);
    int blocks = num_cu * FS_BLOCKS_PER_CU;
    copy_u4<<<blocks, FS_THREADS>>>((uint4*)dst, (const uint4*)src, n);
    return 0;
}

// One-shot copy, synchronous. bytes must be a multiple of 16.
extern "C" int fs_copy(int dev, void* dst, const void* src, size_t bytes, int num_cu) {
    TRY(hipSetDevice(dev));
    launch_copy(num_cu, dst, src, bytes);
    TRY(hipGetLastError());
    TRY(hipDeviceSynchronize());
    return 0;
}

// Timed copy loop. Returns milliseconds per iteration; the caller turns that into GB/s.
// Wall clock around a synchronised loop, matching the usual ubench convention.
extern "C" int fs_bench(int dev,
                        void* dst,
                        const void* src,
                        size_t bytes,
                        int num_cu,
                        int warmup,
                        int loop,
                        double* out_ms) {
    TRY(hipSetDevice(dev));

    for (int i = 0; i < warmup; ++i) {
        launch_copy(num_cu, dst, src, bytes);
    }
    TRY(hipGetLastError());
    TRY(hipDeviceSynchronize());

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < loop; ++i) {
        launch_copy(num_cu, dst, src, bytes);
    }
    TRY(hipGetLastError());
    TRY(hipDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();

    *out_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / loop;
    return 0;
}
