"""
JIT compilation + Python-side kernel launch via HIP driver API.

Compiles .hip to a code object (.hsaco) with --genco (device-only).
Zero host code generated. Kernels loaded with hipModuleLoad and
launched with hipModuleLaunchKernel — all from Python via ctypes.
"""

import ctypes
import functools
import os
import subprocess
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # pyhip_v3/
CSRC_DIR = _PACKAGE_ROOT / "csrc"

CACHE_DIR = Path(
    os.getenv(
        "WARP_SORT_CACHE_DIR",
        Path.home() / ".cache" / "warp_bitonic_sort_pyhip_v3",
    )
)

ROCM_PATH = Path(os.getenv("ROCM_PATH", "/opt/rocm"))
OPUS_DIR = Path(os.getenv("OPUS_DIR", "/raid0/carhuang/repo/aiter/csrc/include"))
GPU_ARCH = os.getenv("GPU_ARCH", "native")

# ── Kernel name table: (lanegroup_size, is_descending) -> (symbol_name, block_size)
KERNEL_MAP = {}
for _ls, _bs in [(2, 64), (4, 64), (8, 64), (16, 64),
                  (32, 64), (64, 64), (128, 128), (256, 256)]:
    KERNEL_MAP[(_ls, True)]  = (f"sort_f32_{_bs}_{_ls}_desc", _bs)
    KERNEL_MAP[(_ls, False)] = (f"sort_f32_{_bs}_{_ls}_asc",  _bs)


def _compile_hsaco(src: Path, out: Path, verbose: bool = False) -> None:
    """Compile .hip to a code object (.hsaco) with hipcc --genco."""
    out.parent.mkdir(parents=True, exist_ok=True)

    hipcc = str(ROCM_PATH / "bin" / "hipcc")
    cmd = [
        hipcc,
        "--genco",
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
        msg = f"hipcc --genco compilation failed (exit {e.returncode})."
        if e.output:
            msg += "\n" + e.output
        raise RuntimeError(msg) from e


@functools.cache
def _load_hip_runtime():
    """Load libamdhip64.so and set up driver API signatures."""
    hip = ctypes.CDLL(str(ROCM_PATH / "lib" / "libamdhip64.so"))

    # hipError_t hipModuleLoad(hipModule_t* module, const char* fname)
    hip.hipModuleLoad.restype = ctypes.c_int
    hip.hipModuleLoad.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),  # module*
        ctypes.c_char_p,                   # fname
    ]

    # hipError_t hipModuleGetFunction(hipFunction_t* function, hipModule_t module, const char* kname)
    hip.hipModuleGetFunction.restype = ctypes.c_int
    hip.hipModuleGetFunction.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),  # function*
        ctypes.c_void_p,                   # module
        ctypes.c_char_p,                   # kname
    ]

    # hipError_t hipModuleLaunchKernel(
    #     hipFunction_t f,
    #     unsigned int gridDimX, gridDimY, gridDimZ,
    #     unsigned int blockDimX, blockDimY, blockDimZ,
    #     unsigned int sharedMemBytes, hipStream_t stream,
    #     void** kernelParams, void** extra)
    hip.hipModuleLaunchKernel.restype = ctypes.c_int
    hip.hipModuleLaunchKernel.argtypes = [
        ctypes.c_void_p,                   # function
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,  # grid
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,  # block
        ctypes.c_uint,                     # sharedMemBytes
        ctypes.c_void_p,                   # stream
        ctypes.POINTER(ctypes.c_void_p),   # kernelParams
        ctypes.POINTER(ctypes.c_void_p),   # extra
    ]

    return hip


@functools.cache
def load_kernel_lib():
    """Compile (if needed), load .hsaco, and resolve all kernel functions."""
    hsaco_path = CACHE_DIR / "warp_bitonic_sort.hsaco"
    src_path = CSRC_DIR / "warp_bitonic_sort.hip"

    verbose = os.environ.get("WARP_SORT_JIT_VERBOSE", "0") == "1"

    if not hsaco_path.exists() or src_path.stat().st_mtime > hsaco_path.stat().st_mtime:
        if verbose:
            print(f"Compiling {src_path} -> {hsaco_path}", flush=True)
        _compile_hsaco(src_path, hsaco_path, verbose)

    hip = _load_hip_runtime()

    # Load the module
    module = ctypes.c_void_p()
    err = hip.hipModuleLoad(ctypes.byref(module), str(hsaco_path).encode())
    if err != 0:
        raise RuntimeError(f"hipModuleLoad failed with error {err}")

    # Resolve all kernel functions
    func_map = {}  # (lanegroup_size, is_descending) -> (hipFunction_t, block_size)
    for key, (sym_name, block_size) in KERNEL_MAP.items():
        func = ctypes.c_void_p()
        err = hip.hipModuleGetFunction(ctypes.byref(func), module, sym_name.encode())
        if err != 0:
            raise RuntimeError(f"hipModuleGetFunction('{sym_name}') failed with error {err}")
        func_map[key] = (func.value, block_size)

    return hip, func_map


def launch_kernel(func_handle, grid_x, block_x, *args, shared_mem=0, stream=None):
    """Call hipModuleLaunchKernel from Python."""
    hip, _ = load_kernel_lib()

    arg_ptrs = (ctypes.c_void_p * len(args))(
        *[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args]
    )

    err = hip.hipModuleLaunchKernel(
        func_handle,
        grid_x, 1, 1,
        block_x, 1, 1,
        shared_mem,
        stream,
        arg_ptrs,
        None,
    )
    if err != 0:
        raise RuntimeError(f"hipModuleLaunchKernel failed with error {err}")
