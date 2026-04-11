// Minimal HIP device-side definitions to replace <hip/hip_runtime.h> for device-only compilation.
// This avoids pulling in the full HIP runtime header (~3s frontend overhead).
#pragma once

#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif

// __launch_bounds__ is the key macro missing from hipcc's implicit wrapper.
// Matches the exact HIP definition from amd_hip_runtime.h.
#ifndef __launch_bounds__
#define __launch_bounds_impl0__(requiredMaxThreadsPerBlock) \
    __attribute__((amdgpu_flat_work_group_size(1, requiredMaxThreadsPerBlock)))
#define __launch_bounds_impl1__(requiredMaxThreadsPerBlock, minBlocksPerMultiprocessor) \
    __attribute__((amdgpu_flat_work_group_size(1, requiredMaxThreadsPerBlock), \
                   amdgpu_waves_per_eu(minBlocksPerMultiprocessor)))
#define __launch_bounds_select__(_1, _2, impl_, ...) impl_
#define __launch_bounds__(...) \
    __launch_bounds_select__(__VA_ARGS__, __launch_bounds_impl1__, __launch_bounds_impl0__, )(__VA_ARGS__)
#endif

// __shared__ / __device__ / __global__ / __host__ are provided by hipcc's implicit
// __clang_hip_runtime_wrapper.h, but define them as fallbacks just in case.
#ifndef __shared__
#define __shared__ __attribute__((shared))
#endif

#ifndef __device__
#define __device__ __attribute__((device))
#endif

#ifndef __global__
#define __global__ __attribute__((global))
#endif

#ifndef __host__
#define __host__ __attribute__((host))
#endif

// Warp vote intrinsic — __all(predicate) returns non-zero iff predicate is
// non-zero for every active lane in the wavefront.
extern "C" __device__ int __ockl_wfall_i32(int);
__device__ inline int __all(int predicate) { return __ockl_wfall_i32(predicate); }

// hipLaunchKernel stub — hipcc's host pass requires this symbol when it sees
// a __global__ function, even if no <<<>>> launch appears in this TU.
#if !defined(__HIP_DEVICE_COMPILE__)
#include <cstdint>
typedef int hipError_t;
typedef void* hipStream_t;
struct dim3 { unsigned x, y, z; dim3(unsigned x_=1,unsigned y_=1,unsigned z_=1):x(x_),y(y_),z(z_){} };
hipError_t hipLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim,
                           void** args, size_t sharedMem, hipStream_t stream);
#endif
