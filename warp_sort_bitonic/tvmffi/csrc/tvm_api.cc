#include <tvm/ffi/tvm_ffi.h>
#include "warp_bitonic_sort.hpp"

// TVM FFI binding for warp_bitonic_sort
// Uses TVM_FFI_DLL_EXPORT_TYPED_FUNC (from apache-tvm-ffi) to export a typed function.
// The function accepts tvm::ffi::Tensor (which is DLPack-compatible),
// so it works with any framework that supports DLPack (PyTorch, NumPy, JAX, etc.)

void warp_bitonic_sort_ffi(tvm::ffi::Tensor input, tvm::ffi::Tensor output, int is_descending) {
    // extract raw data pointers (accounting for byte_offset)
    void* i_ptr = static_cast<char*>(input.data_ptr()) + input.byte_offset();
    void* o_ptr = static_cast<char*>(output.data_ptr()) + output.byte_offset();

    int num_element = static_cast<int>(input.shape()[0]);

    warp_bitonic_sort_kernel(i_ptr, o_ptr, num_element, is_descending ? 1 : 0);
}

// Export as "warp_bitonic_sort" — accessible via mod["warp_bitonic_sort"] in Python
TVM_FFI_DLL_EXPORT_TYPED_FUNC(warp_bitonic_sort, warp_bitonic_sort_ffi);
