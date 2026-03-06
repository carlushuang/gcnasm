"""
JIT compilation module for warp_bitonic_sort (ctypes, no C++ binding).

Compiles the .hip kernel source into a plain shared library (.so) with
only `hipcc`.  No pybind11, no torch, no TVM — only the HIP compiler
and Python's built-in ctypes are needed.

The compiled .so exposes `extern "C"` functions that Python calls via
ctypes.CDLL.
"""

import ctypes
import functools
import os
import subprocess
import sys
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # pyhip/
CSRC_DIR = _PACKAGE_ROOT / "csrc"

CACHE_DIR = Path(
    os.getenv(
        "WARP_SORT_CACHE_DIR",
        Path.home() / ".cache" / "warp_bitonic_sort_pyhip",
    )
)

ROCM_PATH = Path(os.getenv("ROCM_PATH", "/opt/rocm"))
OPUS_DIR = Path(os.getenv("OPUS_DIR", "/raid0/carhuang/repo/aiter/csrc/include"))
GPU_ARCH = os.getenv("GPU_ARCH", "native")


def _compile_so(src: Path, out: Path, verbose: bool = False) -> None:
    """Compile a .hip source into a shared library using hipcc."""
    out.parent.mkdir(parents=True, exist_ok=True)

    hipcc = str(ROCM_PATH / "bin" / "hipcc")
    cmd = [
        hipcc,
        "-shared", "-fPIC",
        f"--offload-arch={GPU_ARCH}",
        "-O3",
        "-D__HIPCC_RTC__",
        f"-I{ROCM_PATH / 'include'}",
        f"-I{OPUS_DIR}",
        f"-I{CSRC_DIR.resolve()}",
        str(src.resolve()),
        "-o", str(out.resolve()),
    ]

    if verbose:
        print(" ".join(cmd), flush=True)

    try:
        subprocess.run(
            cmd,
            stdout=None if verbose else subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        msg = f"hipcc compilation failed (exit {e.returncode})."
        if e.output:
            msg += "\n" + e.output
        raise RuntimeError(msg) from e


@functools.cache
def load_kernel_lib():
    """Compile (if needed) and load the kernel .so via ctypes."""
    so_path = CACHE_DIR / "libwarp_bitonic_sort.so"
    src_path = CSRC_DIR / "warp_bitonic_sort.hip"

    verbose = os.environ.get("WARP_SORT_JIT_VERBOSE", "0") == "1"

    # Recompile if .so missing or source is newer
    if not so_path.exists() or src_path.stat().st_mtime > so_path.stat().st_mtime:
        if verbose:
            print(f"Compiling {src_path} -> {so_path}", flush=True)
        _compile_so(src_path, so_path, verbose)

    lib = ctypes.CDLL(str(so_path))

    # void warp_bitonic_sort_kernel(void* i_ptr, void* o_ptr, int num_element, int is_descending)
    lib.warp_bitonic_sort_kernel.restype = None
    lib.warp_bitonic_sort_kernel.argtypes = [
        ctypes.c_void_p,  # i_ptr (device pointer)
        ctypes.c_void_p,  # o_ptr (device pointer)
        ctypes.c_int,     # num_element
        ctypes.c_int,     # is_descending
    ]

    return lib
