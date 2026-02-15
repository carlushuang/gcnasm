"""
warp_bitonic_sort - TVM FFI based Python binding

Uses apache-tvm-ffi (pip install apache-tvm-ffi) for the FFI layer.
Kernels are JIT-compiled on first use via Ninja and cached.
Tensors are passed via DLPack — works with PyTorch, NumPy, JAX, etc.

Usage:
    import warp_bitonic_sort
    y = warp_bitonic_sort.warp_bitonic_sort(x_torch, is_descending=True)
"""

import functools

import tvm_ffi

from .jit import gen_sort_module


@functools.cache
def _get_sort_func():
    """JIT-compile (if needed) and load the warp_bitonic_sort module."""
    mod = gen_sort_module().build_and_load()
    return mod["warp_bitonic_sort"]


def warp_bitonic_sort(x, is_descending=True):
    """
    Sort a 1-D float32 GPU tensor using warp-level bitonic merge sort.

    Args:
        x: Any tensor supporting DLPack (torch.Tensor, etc.), 1-D float32 on GPU.
           Length must be power-of-2 in {2, 4, 8, 16, 32, 64, 128, 256}.
        is_descending: if True, sort descending; otherwise ascending.

    Returns:
        Sorted tensor (same type as input if PyTorch).
    """
    import torch

    is_torch = isinstance(x, torch.Tensor)

    # Convert input to tvm_ffi.Tensor via DLPack (zero-copy)
    x_tvm = tvm_ffi.from_dlpack(x)

    # Allocate output (same shape/dtype/device) via torch, then wrap
    if is_torch:
        y_torch = torch.empty_like(x)
    else:
        y_torch = torch.empty(x_tvm.shape, dtype=torch.float32, device="cuda")

    y_tvm = tvm_ffi.from_dlpack(y_torch)

    _get_sort_func()(x_tvm, y_tvm, int(is_descending))

    return y_torch if is_torch else tvm_ffi.from_dlpack(y_torch)
