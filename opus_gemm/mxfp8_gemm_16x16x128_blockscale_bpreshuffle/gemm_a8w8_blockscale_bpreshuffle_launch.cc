#include <hip/hip_runtime.h>
#include <climits>
#include <cstdint>
#include <cstring>
#include "gemm_a8w8_mxfp8_scale_common.h"

template<class Traits>
__global__ void gemm_a8w8_mxfp8_scale_kernel(opus_gemm_scale_kargs);

namespace {
bool bounded_count(int rows, int cols, int batch, int bytes) {
    const uint64_t limit = INT_MAX / bytes;
    if (rows <= 0 || cols <= 0 || batch <= 0 || uint64_t(rows) > limit / cols)
        return false;
    return uint64_t(batch) <= limit / (uint64_t(rows) * cols);
}
bool overlaps(const void* a, uint64_t a_bytes, const void* b, uint64_t b_bytes) {
    const auto begin_a = reinterpret_cast<uintptr_t>(a);
    const auto begin_b = reinterpret_cast<uintptr_t>(b);
    return begin_a < begin_b + b_bytes && begin_b < begin_a + a_bytes;
}
template<int Tiles, bool BF16>
void launch(const opus_gemm_scale_kargs& args, hipStream_t stream) {
    using T = gemm_a8w8_mxfp8_scale_traits<256, 256, 128, 1, 128, 128, Tiles, BF16>;
    const dim3 grid(ceil_div_scale(args.m / 256, Tiles) * (args.n / 256), 1, args.batch);
    gemm_a8w8_mxfp8_scale_kernel<T><<<grid, dim3(T::BLOCK_SIZE), 0, stream>>>(args);
}
}

// Input pointers must already have the documented layouts. This function
// allocates no memory, performs no preprocessing, and launches exactly one GEMM.
extern "C" hipError_t launch_blockscale_bpreshuffle(
    const opus_gemm_scale_kargs* args, int output_bf16, int tiles, hipStream_t stream) {
    if (!args || !args->ptr_a || !args->ptr_b || !args->ptr_c ||
        !args->ptr_sfa || !args->ptr_sfb || (output_bf16 != 0 && output_bf16 != 1) ||
        (tiles != 0 && tiles != 1 && tiles != 4)) return hipErrorInvalidValue;
    const auto& a = *args;
    if (a.m <= 0 || a.n <= 0 || a.k <= 0 || a.batch <= 0 ||
        a.m % 256 || a.n % 256 || a.k % 128 ||
        !bounded_count(a.m, a.k, a.batch, 1) ||
        !bounded_count(a.n, a.k, a.batch, 1) ||
        !bounded_count(a.m, a.n, a.batch, output_bf16 ? 2 : 4) ||
        !bounded_count(a.m, a.k / 128, a.batch, 1) ||
        !bounded_count(a.n / 128, a.k / 128, a.batch, 1)) return hipErrorInvalidValue;
    if (a.stride_a != a.k || a.stride_b != a.k || a.stride_c != a.n ||
        a.stride_a_batch != a.m * a.k || a.stride_b_batch != a.n * a.k ||
        a.stride_c_batch != a.m * a.n || a.stride_sfa != a.m ||
        a.stride_sfb != a.k / 128 || a.stride_sfa_batch != a.m * (a.k / 128) ||
        a.stride_sfb_batch != (a.n / 128) * (a.k / 128)) return hipErrorInvalidValue;
    const uint64_t c_bytes = uint64_t(a.m) * a.n * a.batch * (output_bf16 ? 2 : 4);
    if (overlaps(a.ptr_c, c_bytes, a.ptr_a, uint64_t(a.stride_a_batch) * a.batch) ||
        overlaps(a.ptr_c, c_bytes, a.ptr_b, uint64_t(a.stride_b_batch) * a.batch) ||
        overlaps(a.ptr_c, c_bytes, a.ptr_sfa, uint64_t(a.stride_sfa_batch) * a.batch) ||
        overlaps(a.ptr_c, c_bytes, a.ptr_sfb, uint64_t(a.stride_sfb_batch) * a.batch))
        return hipErrorInvalidValue;
    int device;
    hipError_t status = hipGetDevice(&device);
    if (status != hipSuccess) return status;
    hipDeviceProp_t props{};
    status = hipGetDeviceProperties(&props, device);
    if (status != hipSuccess) return status;
    if (std::strncmp(props.gcnArchName, "gfx950", 6)) return hipErrorInvalidDeviceFunction;
    if (tiles == 0)
        tiles = int64_t(ceil_div_scale(a.m / 256, 4)) * (a.n / 256) * a.batch >=
            props.multiProcessorCount ? 4 : 1;
    if (output_bf16) {
        if (tiles == 4) launch<4, true>(a, stream);
        else launch<1, true>(a, stream);
    } else {
        if (tiles == 4) launch<4, false>(a, stream);
        else launch<1, false>(a, stream);
    }
    return hipGetLastError();
}
