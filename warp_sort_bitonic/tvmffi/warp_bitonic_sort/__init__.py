"""
warp_bitonic_sort - TVM FFI based Python binding

Uses apache-tvm-ffi (pip install apache-tvm-ffi) for the FFI layer.
Tensors are passed via DLPack — works with PyTorch, NumPy, JAX, etc.

Usage (with PyTorch):
    import warp_bitonic_sort
    y = warp_bitonic_sort.warp_bitonic_sort(x_torch, is_descending=True)
"""

import os
import tvm_ffi

# Load the compiled shared library via tvm_ffi.load_module
# This discovers all TVM_FFI_DLL_EXPORT_TYPED_FUNC symbols in the .so
_lib_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build")
_lib_path = os.path.join(_lib_dir, "libwarp_bitonic_sort_tvm.so")

if not os.path.exists(_lib_path):
    raise RuntimeError(
        f"Cannot find {_lib_path}. Please build first:\n"
        f"  cd {os.path.dirname(_lib_dir)} && make"
    )

_mod = tvm_ffi.load_module(_lib_path)
_sort_func = _mod["warp_bitonic_sort"]


def warp_bitonic_sort(x, is_descending=True):
    """
    Sort a 1-D float32 GPU tensor using warp-level bitonic merge sort.

    Args:
        x: Any tensor supporting DLPack (torch.Tensor, etc.), 1-D float32 on GPU.
           Length must be power-of-2 in {2, 4, 8, 16, 32, 64, 128, 256}.
        is_descending: if True, sort descending; otherwise ascending.

    Returns:
        Sorted tensor (same type as input if PyTorch, otherwise tvm_ffi.Tensor).
    """
    import torch

    is_torch = isinstance(x, torch.Tensor)

    # Convert input to tvm_ffi.Tensor via DLPack (zero-copy)
    x_tvm = tvm_ffi.from_dlpack(x)

    # Allocate output (same shape/dtype/device) via torch, then wrap
    if is_torch:
        y_torch = torch.empty_like(x)
        y_tvm = tvm_ffi.from_dlpack(y_torch)
    else:
        # Fallback: use torch to allocate on same device
        y_torch = torch.empty(x_tvm.shape, dtype=torch.float32, device="cuda")
        y_tvm = tvm_ffi.from_dlpack(y_torch)

    _sort_func(x_tvm, y_tvm, int(is_descending))

    return y_torch if is_torch else tvm_ffi.from_dlpack(y_torch)
