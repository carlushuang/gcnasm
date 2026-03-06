"""
warp_bitonic_sort — pure ctypes Python binding (no C++ binding code)

Compiles the .hip kernel into a plain .so on first use, then loads it
via ctypes.CDLL and calls the extern "C" host function directly.

Usage:
    import warp_bitonic_sort
    y = warp_bitonic_sort.warp_bitonic_sort(x_torch, is_descending=True)
"""

from .jit import load_kernel_lib


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

    lib = load_kernel_lib()
    y = torch.empty_like(x)
    lib.warp_bitonic_sort_kernel(x.data_ptr(), y.data_ptr(), x.size(0), int(is_descending))
    return y
