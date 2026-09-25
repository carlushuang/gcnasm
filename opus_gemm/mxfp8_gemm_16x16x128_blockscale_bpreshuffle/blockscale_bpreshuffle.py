"""Direct PyTorch adapter for the gfx950 E8M0 blockscale prototype.

B must already be shuffle_weight(B, layout=(16,16)). SFA has logical
[M,K/128] but column-major bytes; SFB is contiguous [N/128,K/128].
The adapter does not quantize, shuffle, expand, or convert input tensors.
It is a direct eager adapter, not an installed aiter backend registration.
"""
import ctypes
from functools import lru_cache
from pathlib import Path

import torch


class _Args(ctypes.Structure):
    _fields_ = (
        [(name, ctypes.c_void_p) for name in ("ptr_a", "ptr_b", "ptr_c")]
        + [(name, ctypes.c_int) for name in (
            "m", "n", "k", "batch", "stride_a", "stride_b", "stride_c",
            "stride_a_batch", "stride_b_batch", "stride_c_batch")]
        + [(name, ctypes.c_void_p) for name in ("ptr_sfa", "ptr_sfb")]
        + [(name, ctypes.c_int) for name in (
            "stride_sfa", "stride_sfb", "stride_sfa_batch", "stride_sfb_batch")]
    )


@lru_cache(None)
def _library():
    path = Path(__file__).resolve().parent / "build/libblockscale_bpreshuffle.so"
    lib = ctypes.CDLL(str(path))
    lib.launch_blockscale_bpreshuffle.argtypes = [
        ctypes.POINTER(_Args), ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    lib.launch_blockscale_bpreshuffle.restype = ctypes.c_int
    return lib


def gemm_a8w8_blockscale_bpreshuffle(
    A, B, A_scale, B_scale, dtype=torch.bfloat16, out=None, *, tiles=0
):
    """One GEMM launch, BF16/FP32 output; no input conversion or scale workspace.

    A_scale accepts either true column-major strides (1,M), or a contiguous
    [M,K/128] tensor whose bytes were transposed by the quantizer, as in aiter.
    A contiguous row-major scale with ordinary logical values is NOT that format.
    M/N must be multiples of 256 and K a multiple of 128. `tiles` may be 0/1/4.
    """
    tensors = (A, B, A_scale, B_scale)
    if any(t.ndim != 2 for t in tensors):
        raise ValueError("A/B/A_scale/B_scale must be 2-D")
    if not A.is_cuda or any(t.device != A.device for t in tensors):
        raise ValueError("all inputs must be on the same GPU")
    if A.dtype != torch.float8_e4m3fn or B.dtype not in (torch.float8_e4m3fn, torch.uint8):
        raise ValueError("A/B must contain E4M3FN FP8 bytes")
    scale_types = (torch.uint8, getattr(torch, "float8_e8m0fnu", torch.uint8))
    if A_scale.dtype not in scale_types or B_scale.dtype not in scale_types:
        raise ValueError("scales must be E8M0 bytes, not FP32")
    m, k = A.shape
    n, bk = B.shape
    if min(m, n, k) <= 0 or k != bk or m % 256 or n % 256 or k % 128:
        raise ValueError("requires matching K, M/N % 256 == 0, K % 128 == 0")
    if dtype not in (torch.bfloat16, torch.float32) or tiles not in (0, 1, 4):
        raise ValueError("dtype must be bf16/fp32; tiles must be 0, 1 or 4")
    if not A.is_contiguous() or not B.is_contiguous() or not B_scale.is_contiguous():
        raise ValueError("A, packed B and B_scale must be contiguous")
    if A_scale.shape != (m, k // 128) or B_scale.shape != (n // 128, k // 128):
        raise ValueError("requires A_scale [M,K/128], B_scale [N/128,K/128]")
    if not A_scale.is_contiguous() and A_scale.stride() != (1, m):
        raise ValueError("A_scale must use dense column-major bytes")
    output_bytes = 2 if dtype == torch.bfloat16 else 4
    if max(m*k, n*k, m*n*output_bytes) > 2**31 - 1:
        raise ValueError("prototype requires every tensor allocation below 2 GiB")
    if out is None:
        out = torch.empty((m, n), device=A.device, dtype=dtype)
    elif out.shape != (m, n) or out.dtype != dtype or out.device != A.device or not out.is_contiguous():
        raise ValueError("out must be contiguous [M,N] with the requested dtype/device")
    out_begin = out.data_ptr()
    out_end = out_begin + out.numel() * out.element_size()
    for tensor in tensors:
        begin = tensor.data_ptr()
        end = begin + tensor.numel() * tensor.element_size()
        if out_begin < end and begin < out_end:
            raise ValueError("out must not overlap any input tensor")
    args = _Args(A.data_ptr(), B.data_ptr(), out.data_ptr(), m, n, k, 1,
                 k, k, n, m*k, n*k, m*n, A_scale.data_ptr(), B_scale.data_ptr(),
                 m, k//128, m*(k//128), (n//128)*(k//128))
    with torch.cuda.device(A.device):
        stream = torch.cuda.current_stream(A.device)
        status = _library().launch_blockscale_bpreshuffle(
            ctypes.byref(args), int(dtype == torch.bfloat16), tiles,
            ctypes.c_void_p(stream.cuda_stream))
        # The launch crosses ctypes, so tell PyTorch's allocator about usage
        # on a stream other than the tensor's allocation stream.
        for tensor in (*tensors, out):
            tensor.record_stream(stream)
    if status:
        raise RuntimeError(f"blockscale bpreshuffle launch failed: HIP error {status}")
    return out
