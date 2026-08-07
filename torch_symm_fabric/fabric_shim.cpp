// fabric_shim.cpp -- LD_PRELOAD shim that makes torch's symmetric memory fabric-exportable.
//
// torch.distributed._symmetric_memory allocates through hipMemCreate. On ROCm it always
// asks for the POSIX-fd handle type, because c10::cuda::get_fabric_access() is compiled
// out (`#if !defined(USE_ROCM)`), and the handle types a VMM allocation supports are
// frozen at creation. So a torch symm_mem buffer can never be exported over fabric.
//
// requestedHandleTypes is a bitmask, so the fix is one bit: intercept hipMemCreate and
// OR in hipMemHandleTypeFabric. torch still gets the fd handle it asked for and its own
// rendezvous is untouched; the allocation just gains a second, fabric-shaped door that
// hip_fabric.py can open later via hipMemRetainAllocationHandle.
//
// If the driver rejects the combined mask the original request is retried unchanged, so
// preloading this is safe even where fabric is unavailable.
//
//   LD_PRELOAD=./libfabric_shim.so python3 your_pure_torch_script.py

#include <dlfcn.h>
#include <hip/hip_runtime_api.h>

#include <cstdio>
#include <cstdlib>

namespace {

using hip_mem_create_fn = hipError_t (*)(hipMemGenericAllocationHandle_t*,
                                         size_t,
                                         const hipMemAllocationProp*,
                                         unsigned long long);

hip_mem_create_fn real_hip_mem_create() {
    static hip_mem_create_fn real = nullptr;
    if (real == nullptr) {
        real = reinterpret_cast<hip_mem_create_fn>(dlsym(RTLD_NEXT, "hipMemCreate"));
    }
    return real;
}

// FABRIC_SHIM_VERBOSE=1 to see every allocation the shim upgrades
bool verbose() {
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("FABRIC_SHIM_VERBOSE");
        v = (e != nullptr && *e == '1') ? 1 : 0;
    }
    return v == 1;
}

}  // namespace

extern "C" hipError_t hipMemCreate(hipMemGenericAllocationHandle_t* handle,
                                   size_t size,
                                   const hipMemAllocationProp* prop,
                                   unsigned long long flags) {
    hip_mem_create_fn real = real_hip_mem_create();
    if (real == nullptr) {
        return hipErrorNotSupported;
    }

    // Only touch pinned device allocations that asked for some shareable handle and did
    // not already ask for fabric. Everything else -- including expandable_segments, which
    // requests no shareable handle at all -- passes through untouched.
    const bool upgradable = prop != nullptr && prop->type == hipMemAllocationTypePinned &&
                            prop->requestedHandleTypes != 0 &&
                            (prop->requestedHandleTypes & hipMemHandleTypeFabric) == 0;

    if (upgradable) {
        hipMemAllocationProp upgraded = *prop;
        upgraded.requestedHandleTypes = hipMemHandleTypeFabric;

        hipError_t err = real(handle, size, &upgraded, flags);
        if (err == hipSuccess) {
            if (verbose()) {
                fprintf(stderr,
                        "[fabric_shim] upgraded %zu B on device %d: handle types 0x%x -> 0x%x\n",
                        size,
                        prop->location.id,
                        (unsigned)prop->requestedHandleTypes,
                        (unsigned)upgraded.requestedHandleTypes);
            }
            return err;
        }
        if (verbose()) {
            fprintf(stderr,
                    "[fabric_shim] combined mask rejected for %zu B, retrying as requested\n",
                    size);
        }
    }

    return real(handle, size, prop, flags);
}
