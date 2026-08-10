// fabric_shim.cpp -- LD_PRELOAD interposer making torch's symm_mem fabric-exportable.
//
// Method "shim". torch allocates symmetric memory through hipMemCreate and on ROCm always
// asks for POSIX fds (get_fabric_access() is compiled out), and handle types are frozen
// at creation -- so intercept the call and ask for fabric instead. torch still allocates
// and still owns the mapping; only the handle type changes.
//
// ROCm rejects the combined fd|fabric mask, so this is a swap, not an addition: torch's
// own fd-based rendezvous stops working on these buffers. Allocations that request no
// shareable handle (expandable_segments, the caching allocator) pass through untouched,
// and the original request is retried if the driver refuses the change.
//
//   LD_PRELOAD=./libfabric_shim.so python3 your_script.py     (./run.sh --method shim)

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

// FABRIC_SHIM_VERBOSE=1 logs every allocation upgraded
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

    // Only pinned allocations that asked for some shareable handle and not already fabric.
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
