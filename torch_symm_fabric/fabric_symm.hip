// fabric_symm.hip -- HIP VMM fabric handles for cross-GPU transfer. See README.md.
//
// Three ways to get a fabric-capable buffer:
//   fs_alloc          we allocate it                            (method "own")
//   -- none --        fabric_shim.cpp rewrote the handle type   (method "shim")
//   fs_rebind_fabric  remap a torch buffer's VA onto fabric     (method "rebind")
//
// Then one shared path: fs_export_ptr -> 64-byte handle, fs_import maps a peer, the copy
// kernel moves data. Plain C ABI so ctypes can bind it without pybind11.

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

// The probes below provoke failures on purpose. A failed HIP call leaves
// hipGetLastError() armed and torch reports it on its next kernel launch, as a bogus
// "CUDA error" from unrelated code -- so swallow the one we caused.
static inline void clear_sticky_error() {
    hipError_t ignored = hipGetLastError();
    (void)ignored;
}

extern "C" int fs_device_count(int* n) {
    TRY(hipGetDeviceCount(n));
    return 0;
}

// Is fabric usable here? Probed functionally rather than via hipDeviceGetAttribute:
// hipDeviceAttribute_t values shift between HIP releases (fabric is 95 in 7.15, absent
// in 7.12) and torch often loads a different libamdhip64 than we compiled against, so
// the attribute route can silently ask the wrong question. The enums used below carry
// explicit values and are stable.
//
// failed_step: 0 ok, 1 granularity, 2 create, 3 export, 4 import; error in fs_last_error().
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

// Method "own": we allocate. torch is not involved, so this is not a symm_mem tensor --
// main.py wraps it via __cuda_array_interface__. One allocation, so 1x memory.
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

// Method "rebind": make a torch buffer fabric-capable in place.
//
// Handle types are frozen at hipMemCreate and there is no API to change them, but VMM
// separates the VA from its backing -- so swap the backing instead. torch's pointer is
// unchanged and its tensor keeps working.
//
// Caller must respect: contents are discarded (call before filling), and torch still
// references the orphaned original, so the buffer costs 2x physical until torch frees it.
// Our handle is released at once; the mapping holds its own reference.
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

// Export any fabric-capable buffer as a 64-byte handle, whoever allocated it.
//
// hipMemGetAddressRange recovers the allocation behind the pointer (torch gives each
// symm_mem.empty its own granularity-rounded one) and hipMemRetainAllocationHandle
// borrows its physical handle -- the owner keeps the mapping, so release via
// fs_release_handle and never unmap.
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
