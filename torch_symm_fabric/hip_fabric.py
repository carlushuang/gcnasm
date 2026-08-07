"""ctypes bindings for fabric_symm.hip.

The buffer is always allocated by torch. This module only makes it fabric-capable
(``rebind_to_fabric``), exports it, imports peers, and hands back peer pointers
(and torch views onto them). Nothing here intercepts or replaces a HIP entry point.
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
    lib.fs_export_ptr.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.fs_rebind_fabric.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.fs_release_handle.argtypes = [ctypes.c_uint64]
    lib.fs_import.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.fs_release_import.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint64]
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
        raise RuntimeError(f"{what} failed: hip error {rc} ({_lib.fs_last_error().decode()})")


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


def rebind_to_fabric(tensor):
    """Make a torch-allocated buffer fabric-capable, in place, without moving it.

    The handle types of a VMM allocation are fixed at hipMemCreate and torch asks for
    POSIX fds on ROCm, so the buffer cannot be exported over fabric as allocated. Rather
    than intercept torch's allocator, this rebinds the tensor's virtual address range to
    fabric-capable physical memory: same pointer, same tensor, exportable backing.

    Call it immediately after allocation -- the previous contents are discarded. While the
    tensor lives it costs twice its size in physical memory, because torch still holds a
    reference to the original (now unmapped) allocation.

    Returns (base, size) of the rebound allocation.
    """
    base, size = ctypes.c_void_p(), ctypes.c_size_t()
    _check(
        load().fs_rebind_fabric(
            ctypes.c_void_p(tensor.data_ptr()), ctypes.byref(base), ctypes.byref(size)
        ),
        "fs_rebind_fabric",
    )
    return base.value, size.value


def copy(dev, dst, src, nbytes, num_cu):
    _check(load().fs_copy(dev, ctypes.c_void_p(dst), ctypes.c_void_p(src), nbytes, num_cu), "fs_copy")


def bench(dev, dst, src, nbytes, num_cu, warmup, loop):
    """Timed copy loop; returns milliseconds per iteration."""
    ms = ctypes.c_double()
    _check(
        load().fs_bench(
            dev, ctypes.c_void_p(dst), ctypes.c_void_p(src), nbytes, num_cu, warmup, loop, ctypes.byref(ms)
        ),
        "fs_bench",
    )
    return ms.value


_TYPESTR = {
    "torch.int32": "<i4",
    "torch.int64": "<i8",
    "torch.float32": "<f4",
    "torch.float16": "<f2",
    "torch.bfloat16": "<f2",  # the array interface has no bf16 code; only the width matters
    "torch.uint8": "|u1",
}


def tensor_from_ptr(ptr, numel, dtype, device):
    """Wrap a raw device pointer as a torch tensor with no copy.

    Uses __cuda_array_interface__, which torch.as_tensor consumes directly. Applied to a
    fabric-imported peer pointer this yields a tensor that torch ops read and write
    straight across the fabric.
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


class SymmFabricWindow:
    """Fabric export/import for a torch symmetric-memory tensor.

    torch owns the buffer; this only borrows its allocation handle. Mirrors the shape of
    torch's own rendezvous handle -- ``get_buffer(peer, sizes, dtype)`` returns a tensor
    on the peer's memory -- but reaches peers over fabric instead of POSIX fds.

        win = SymmFabricWindow(t, dev)
        win.import_peers(all_gathered_descriptors, my_rank)
        win.get_buffer(peer, (n,), torch.int32)
    """

    def __init__(self, tensor, dev):
        self.dev = dev
        self.device = tensor.device
        self.tensor = tensor
        lib = load()

        fh = ctypes.create_string_buffer(FABRIC_HANDLE_BYTES)
        handle, base, size = ctypes.c_uint64(), ctypes.c_void_p(), ctypes.c_size_t()
        rc = lib.fs_export_ptr(
            ctypes.c_void_p(tensor.data_ptr()),
            fh,
            ctypes.byref(handle),
            ctypes.byref(base),
            ctypes.byref(size),
        )
        if rc != 0:
            raise RuntimeError(
                f"fabric export of the torch buffer failed: hip error {rc} "
                f"({lib.fs_last_error().decode()}). The allocation is not fabric-capable -- "
                f"call hf.rebind_to_fabric(tensor) right after allocating it."
            )

        self._handle = handle.value
        self.base = base.value
        self.size = size.value
        # torch may place the tensor at an offset inside its allocation; peers need it
        self.offset = tensor.data_ptr() - self.base
        self.handle = fh.raw[:FABRIC_HANDLE_BYTES]
        self.peer_ptr = []
        self._imported = []

    def descriptor(self):
        """What a peer needs in order to map this rank's buffer: (handle, size, offset)."""
        return (self.handle, self.size, self.offset)

    def import_peers(self, descriptors, my_rank):
        """Map every peer's buffer. `descriptors` is all-gathered, indexed by rank."""
        lib = load()
        self.peer_ptr = []
        for rank, (fh, size, offset) in enumerate(descriptors):
            if rank == my_rank:
                self.peer_ptr.append(self.tensor.data_ptr())
                continue
            ptr, h = ctypes.c_void_p(), ctypes.c_uint64()
            _check(
                lib.fs_import(self.dev, fh, size, ctypes.byref(ptr), ctypes.byref(h)),
                f"fs_import(rank={rank})",
            )
            self._imported.append((ptr.value, size, h.value))
            # the peer's tensor sits at the same offset inside its own allocation
            self.peer_ptr.append(ptr.value + offset)
        return self.peer_ptr

    def get_buffer(self, peer, sizes, dtype, storage_offset=0):
        """A torch tensor aliasing `peer`'s buffer, reachable over the fabric mapping."""
        numel = 1
        for s in sizes:
            numel *= s
        ptr = self.peer_ptr[peer] + storage_offset * _itemsize(dtype)
        t = tensor_from_ptr(ptr, numel, dtype, self.device)
        return t.view(*sizes) if tuple(sizes) != (numel,) else t

    def close(self):
        lib = load()
        for ptr, size, h in self._imported:
            lib.fs_release_import(ctypes.c_void_p(ptr), size, h)
        self._imported = []
        if self._handle is not None:
            lib.fs_release_handle(self._handle)
            self._handle = None


def _itemsize(dtype):
    import torch

    return torch.empty(0, dtype=dtype).element_size()
