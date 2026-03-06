"""
warp_bitonic_sort - pure pybind11 Python binding

Uses pybind11 (pip install pybind11) for the FFI layer.
Kernels are JIT-compiled on first use via Ninja and cached.

Usage:
    import warp_bitonic_sort
    y = warp_bitonic_sort.warp_bitonic_sort(x_torch, is_descending=True)
"""

import functools

from .jit import gen_sort_module


@functools.cache
def _get_module():
    """JIT-compile (if needed) and load the pybind11 extension."""
    return gen_sort_module().build_and_load()


def warp_bitonic_sort(x, is_descending=True):
    """
    Sort a 1-D float32 GPU tensor using warp-level bitonic merge sort.

    Args:
        x: torch.Tensor, 1-D float32 on GPU.
           Length must be power-of-2 in {2, 4, 8, 16, 32, 64, 128, 256}.
        is_descending: if True, sort descending; otherwise ascending.

    Returns:
        Sorted torch.Tensor (same shape/dtype/device as input).
    """
    import torch

    y = torch.empty_like(x)
    _get_module().warp_bitonic_sort(
        x.data_ptr(), y.data_ptr(), x.size(0), is_descending
    )
    return y
