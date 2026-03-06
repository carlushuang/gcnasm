"""
warp_bitonic_sort — pure Python kernel launch via HIP driver API

The .hip file is compiled to a code object (.hsaco) with --genco.
Zero host code. Python loads the code object with hipModuleLoad and
launches kernels with hipModuleLaunchKernel via ctypes.

Usage:
    import warp_bitonic_sort
    y = warp_bitonic_sort.warp_bitonic_sort(x_torch, is_descending=True)
"""

import ctypes
from .jit import load_kernel_lib, launch_kernel


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

    _, func_map = load_kernel_lib()
    num_element = x.size(0)

    key = (num_element, bool(is_descending))
    if key not in func_map:
        raise ValueError(f"Unsupported num_element={num_element}. "
                         f"Must be one of {sorted(set(k[0] for k in func_map))}")

    func_handle, block_size = func_map[key]

    y = torch.empty_like(x)

    i_ptr = ctypes.c_void_p(x.data_ptr())
    o_ptr = ctypes.c_void_p(y.data_ptr())

    launch_kernel(func_handle, 1, block_size, i_ptr, o_ptr)

    return y
