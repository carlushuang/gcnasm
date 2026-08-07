"""ctypes bindings for fabric_symm.hip, plus the zero-copy torch tensor wrapper.

Nothing in here imports torch at module scope except `tensor_from_ptr`, so the
low-level pieces can be reused from a plain HIP script if you want.
"""

import ctypes
import os

FABRIC_HANDLE_BYTES = 64

_lib = None


def load(path=None):
    """dlopen libfabric_symm.so (built by build.sh, next to this file by default)."""
    global _lib
    if _lib is not None:
        return _lib

    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libfabric_symm.so")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- run ./build.sh first")

    lib = ctypes.CDLL(path)

    lib.fs_last_error.restype = ctypes.c_char_p
    lib.fs_device_count.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.fs_fabric_supported.argtypes = [ctypes.c_int] + [ctypes.POINTER(ctypes.c_int)] * 2
    lib.fs_alloc.argtypes = [
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_char_p,
    ]
    lib.fs_import.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.fs_release.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint64]
    lib.fs_probe_ptr.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_int)] * 3
    lib.fs_copy.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    lib.fs_bench.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]

    _lib = lib
    return _lib


def _check(rc, what):
    if rc != 0:
        msg = _lib.fs_last_error().decode()
        raise RuntimeError(f"{what} failed: hip error {rc} ({msg})")


def device_count():
    n = ctypes.c_int()
    _check(load().fs_device_count(ctypes.byref(n)), "fs_device_count")
    return n.value


FABRIC_STEPS = {0: "ok", 1: "granularity", 2: "hipMemCreate", 3: "export", 4: "import"}


def fabric_supported(dev):
    """Can this device allocate, export and import a fabric handle?

    Returns (supported, detail); detail names the failing step and HIP error.
    """
    ok, step = ctypes.c_int(), ctypes.c_int()
    lib = load()
    _check(lib.fs_fabric_supported(dev, ctypes.byref(ok), ctypes.byref(step)), "fs_fabric_supported")
    if ok.value:
        return True, "ok"
    return False, f"{FABRIC_STEPS.get(step.value, step.value)} failed: {lib.fs_last_error().decode()}"


def probe_ptr(ptr):
    """Is `ptr` VMM-backed, and which shareable handle types can it export?

    Returns (retain_rc, export_fabric_rc, export_fd_rc); 0 means success.
    """
    a, b, c = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
    _check(
        load().fs_probe_ptr(ctypes.c_void_p(ptr), ctypes.byref(a), ctypes.byref(b), ctypes.byref(c)),
        "fs_probe_ptr",
    )
    return a.value, b.value, c.value


def copy(dev, dst, src, nbytes, num_cu):
    _check(load().fs_copy(dev, ctypes.c_void_p(dst), ctypes.c_void_p(src), nbytes, num_cu), "fs_copy")


def bench(dev, dst, src, nbytes, num_cu, warmup, loop):
    """Timed copy loop; returns milliseconds per iteration."""
    ms = ctypes.c_double()
    _check(
        load().fs_bench(
            dev,
            ctypes.c_void_p(dst),
            ctypes.c_void_p(src),
            nbytes,
            num_cu,
            warmup,
            loop,
            ctypes.byref(ms),
        ),
        "fs_bench",
    )
    return ms.value


class FabricWindow:
    """The local half of a symmetric window, plus every peer's imported pointer.

    local_ptr  -- this rank's buffer, mapped read/write on this rank's device
    handle     -- the 64-byte fabric handle to hand to peers
    peer_ptr[] -- filled in by import_peers(); peer_ptr[my_rank] is local_ptr
    """

    def __init__(self, dev, nbytes):
        self.dev = dev
        lib = load()
        buf = ctypes.create_string_buffer(FABRIC_HANDLE_BYTES)
        ptr, h, total = ctypes.c_void_p(), ctypes.c_uint64(), ctypes.c_size_t()
        _check(
            lib.fs_alloc(
                dev,
                nbytes,
                ctypes.byref(ptr),
                ctypes.byref(h),
                ctypes.byref(total),
                buf,
            ),
            "fs_alloc",
        )
        self.local_ptr = ptr.value
        self._handle = h.value
        self.total = total.value
        self.handle = buf.raw[:FABRIC_HANDLE_BYTES]
        self.peer_ptr = []
        self._imported = []

    def import_peers(self, handles, totals, my_rank):
        """Map every peer's window. `handles`/`totals` are all-gathered, indexed by rank."""
        lib = load()
        self.peer_ptr = []
        for rank, (fh, total) in enumerate(zip(handles, totals)):
            if rank == my_rank:
                self.peer_ptr.append(self.local_ptr)
                continue
            ptr, h = ctypes.c_void_p(), ctypes.c_uint64()
            _check(
                lib.fs_import(self.dev, fh, total, ctypes.byref(ptr), ctypes.byref(h)),
                f"fs_import(rank={rank})",
            )
            self.peer_ptr.append(ptr.value)
            self._imported.append((ptr.value, total, h.value))
        return self.peer_ptr

    def close(self):
        lib = load()
        for ptr, total, h in self._imported:
            lib.fs_release(ctypes.c_void_p(ptr), total, h)
        self._imported = []
        if self.local_ptr is not None:
            lib.fs_release(ctypes.c_void_p(self.local_ptr), self.total, self._handle)
            self.local_ptr = None


_TYPESTR = {
    "torch.int32": "<i4",
    "torch.int64": "<i8",
    "torch.float32": "<f4",
    "torch.float16": "<f2",
    "torch.bfloat16": "<f2",  # no bf16 code in the array interface; only the size matters here
    "torch.uint8": "|u1",
}


def tensor_from_ptr(ptr, numel, dtype, device):
    """Wrap a raw device pointer as a torch tensor with no copy.

    Uses __cuda_array_interface__, which torch.as_tensor consumes directly. The
    returned tensor aliases the fabric mapping, so torch ops read and write the
    exact memory peers see through their imported pointers.
    """
    import torch

    key = str(dtype)
    if key not in _TYPESTR:
        raise TypeError(f"unsupported dtype {dtype}")

    class _CAI:
        __cuda_array_interface__ = {
            "data": (ptr, False),
            "shape": (numel,),
            "typestr": _TYPESTR[key],
            "strides": None,
            "version": 3,
            "stream": None,
        }

    t = torch.as_tensor(_CAI(), device=device)
    if t.data_ptr() != ptr:
        raise RuntimeError("torch.as_tensor copied instead of aliasing the pointer")
    return t.view(dtype) if t.dtype != dtype else t
