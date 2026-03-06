"""
JIT compilation + Python-side kernel launch via hipLaunchKernel.

No <<<>>> in C++ at all. The .hip file exports a table of kernel
host-stub addresses. Python reads this table and calls hipLaunchKernel
from libamdhip64.so directly through ctypes.
"""

import ctypes
import functools
import os
import subprocess
import struct
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # pyhip_v2/
CSRC_DIR = _PACKAGE_ROOT / "csrc"

CACHE_DIR = Path(
    os.getenv(
        "WARP_SORT_CACHE_DIR",
        Path.home() / ".cache" / "warp_bitonic_sort_pyhip_v2",
    )
)

ROCM_PATH = Path(os.getenv("ROCM_PATH", "/opt/rocm"))
OPUS_DIR = Path(os.getenv("OPUS_DIR", "/raid0/carhuang/repo/aiter/csrc/include"))
GPU_ARCH = os.getenv("GPU_ARCH", "native")

# ── HIP runtime types via ctypes ─────────────────────────────────────────────

class dim3(ctypes.Structure):
    _fields_ = [("x", ctypes.c_uint32),
                ("y", ctypes.c_uint32),
                ("z", ctypes.c_uint32)]

class kernel_entry_t(ctypes.Structure):
    _fields_ = [("lanegroup_size", ctypes.c_int),
                ("block_size",     ctypes.c_int),
                ("desc",           ctypes.c_void_p),
                ("asc",            ctypes.c_void_p)]


def _compile_so(src: Path, out: Path, verbose: bool = False) -> None:
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
def _load_hip_runtime():
    """Load libamdhip64.so and set up hipLaunchKernel signature."""
    hip = ctypes.CDLL(str(ROCM_PATH / "lib" / "libamdhip64.so"))

    # hipError_t hipLaunchKernel(const void* func, dim3 grid, dim3 block,
    #                            void** args, size_t sharedMem, hipStream_t stream)
    hip.hipLaunchKernel.restype = ctypes.c_int
    hip.hipLaunchKernel.argtypes = [
        ctypes.c_void_p,  # function_address (host stub)
        dim3,             # gridDim
        dim3,             # blockDim
        ctypes.POINTER(ctypes.c_void_p),  # args
        ctypes.c_size_t,  # sharedMemBytes
        ctypes.c_void_p,  # stream (NULL = default)
    ]
    return hip


@functools.cache
def load_kernel_lib():
    """Compile (if needed), load .so, and build kernel dispatch table."""
    so_path = CACHE_DIR / "libwarp_bitonic_sort.so"
    src_path = CSRC_DIR / "warp_bitonic_sort.hip"

    verbose = os.environ.get("WARP_SORT_JIT_VERBOSE", "0") == "1"

    if not so_path.exists() or src_path.stat().st_mtime > so_path.stat().st_mtime:
        if verbose:
            print(f"Compiling {src_path} -> {so_path}", flush=True)
        _compile_so(src_path, so_path, verbose)

    lib = ctypes.CDLL(str(so_path))
    hip = _load_hip_runtime()

    # Read kernel_table (array of kernel_entry_t) directly from the .so
    MAX_ENTRIES = 9  # 8 kernel configs + sentinel
    KernelArray = kernel_entry_t * MAX_ENTRIES
    table = KernelArray.in_dll(lib, "kernel_table")

    kernel_map = {}  # (num_element, is_descending) -> (func_ptr, block_size)
    for i in range(MAX_ENTRIES):
        entry = table[i]
        if entry.lanegroup_size == 0:
            break
        kernel_map[(entry.lanegroup_size, True)]  = (entry.desc, entry.block_size)
        kernel_map[(entry.lanegroup_size, False)] = (entry.asc,  entry.block_size)

    return lib, hip, kernel_map


def launch_kernel(func_ptr, grid, block, *args, shared_mem=0, stream=None):
    """Call hipLaunchKernel from Python — the entire launch happens here."""
    _, hip, _ = load_kernel_lib()

    grid_dim  = dim3(grid[0]  if isinstance(grid,  (tuple, list)) else grid, 1, 1)
    block_dim = dim3(block[0] if isinstance(block, (tuple, list)) else block, 1, 1)

    arg_ptrs = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])

    err = hip.hipLaunchKernel(
        func_ptr,
        grid_dim,
        block_dim,
        arg_ptrs,
        shared_mem,
        stream,
    )
    if err != 0:
        raise RuntimeError(f"hipLaunchKernel failed with error {err}")
