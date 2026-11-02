"""fabric_mem -- symmetric buffers over HIP fabric handles, for in-kernel peer access.

Same shape as torch.distributed._symmetric_memory wherever the concept is the same, so
the code reads the same way and would port mechanically if torch ever gains fabric on
ROCm. The one addition is ``device_desc()``, because torch has no equivalent.

    import fabric_mem as fsm

    buf  = fsm.empty(N, dtype=torch.bfloat16, device="cuda")   # fabric-capable tensor
    buf[:] = ...                                                # ordinary torch ops
    hdl  = fsm.rendezvous(buf, group_name)                      # export, exchange, import, map
    peer = hdl.get_buffer(3, (N,), torch.bfloat16)              # rank 3's memory as a tensor
    d    = hdl.device_desc()                                    # base/stride/rank/world

Every rank lands in one flat span, so a kernel addresses peers arithmetically instead of
dereferencing a pointer array:

    peer r  ->  base + r * stride

The exchange is deliberately pluggable. A fabric handle is ~80 opaque bytes, which is the
entire reason it beats a POSIX fd, so moving it is the caller's choice -- torch.distributed
on one node, or a TCP socket across nodes with no launcher in common.

No collectives, no signal pads, no multicast: this hands you peer pointers and gets out of
the way.
"""

import ctypes
import math
import os
import socket
import struct
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hip_fabric as _hf  # noqa: E402

__all__ = ["empty", "rendezvous", "torch_dist", "tcp", "SymmetricMemory", "FabricWin"]

# handle(64) + allocation size + tensor offset within the allocation
_DESC = struct.Struct("<64sQQ")

# allocations we made, kept alive and findable at rendezvous time
_OWNED = {}


class FabricWin(ctypes.Structure):
    """POD handed to a kernel by value. Mirrors `struct FabricWin` in fabric_mem.h."""

    _fields_ = [
        ("base", ctypes.c_void_p),
        ("stride", ctypes.c_uint64),
        ("rank", ctypes.c_int),
        ("world", ctypes.c_int),
    ]

    def as_tuple(self):
        """(base, stride, rank, world) -- for Triton, which wants plain scalars."""
        return (self.base, self.stride, self.rank, self.world)


def _device_index(device):
    device = torch.device(device)
    if device.type not in ("cuda", "hip"):
        raise ValueError(f"expected a GPU device, got {device}")
    return torch.cuda.current_device() if device.index is None else device.index


def empty(*size, dtype=torch.float32, device=None):
    """Allocate a fabric-capable buffer and return it as a torch tensor.

    Same call shape as symm_mem.empty(). The allocation is ours (HIP VMM with
    hipMemHandleTypeFabric), not torch's, so it costs exactly its own size and needs no
    interposition -- but it is a normal tensor and ordinary torch ops work on it.
    """
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        size = tuple(size[0])
    numel = math.prod(size) if size else 0
    if numel == 0:
        raise ValueError("fabric_mem.empty needs a non-empty shape")

    dev = _device_index(device if device is not None else torch.cuda.current_device())
    supported, detail = _hf.fabric_supported(dev)
    if not supported:
        raise RuntimeError(f"device {dev}: fabric handles unusable -- {detail}")

    nbytes = numel * torch.empty(0, dtype=dtype).element_size()
    own = _hf.OwnFabricBuffer(dev, nbytes, dtype, torch.device("cuda", dev))
    t = own.tensor[:numel].view(*size)
    _OWNED[t.data_ptr()] = own
    return t


class torch_dist:
    """Exchange over a torch.distributed process group. The default on one node."""

    def __init__(self, group_name=None):
        import torch.distributed as dist

        self._dist = dist
        self._group = None
        if group_name is not None:
            self._group = dist.distributed_c10d._resolve_process_group(group_name)
        self.rank = dist.get_rank(self._group)
        self.world_size = dist.get_world_size(self._group)

    def all_gather(self, payload):
        out = [None] * self.world_size
        self._dist.all_gather_object(out, payload, group=self._group)
        return out


class tcp:
    """Exchange over a TCP socket, with no launcher in common.

    Rank 0 listens and hubs the all-gather; everyone else connects. This is the form that
    works across nodes -- point `host` at rank 0 and nothing else changes.
    """

    def __init__(self, rank, world_size, host="127.0.0.1", port=55600, timeout=120):
        self.rank = rank
        self.world_size = world_size
        self._host, self._port, self._timeout = host, port, timeout

    @staticmethod
    def _send(sock, blob):
        sock.sendall(struct.pack("<I", len(blob)) + blob)

    @staticmethod
    def _recv(sock):
        (n,) = struct.unpack("<I", tcp._recv_exact(sock, 4))
        return tcp._recv_exact(sock, n)

    @staticmethod
    def _recv_exact(sock, n):
        buf = b""
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("peer closed during exchange")
            buf += chunk
        return buf

    def all_gather(self, payload):
        if self.rank == 0:
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind(("0.0.0.0", self._port))
            srv.listen(self.world_size)
            srv.settimeout(self._timeout)

            gathered = [None] * self.world_size
            gathered[0] = payload
            conns = []
            for _ in range(self.world_size - 1):
                sock, _addr = srv.accept()
                (peer_rank,) = struct.unpack("<I", self._recv_exact(sock, 4))
                gathered[peer_rank] = self._recv(sock)
                conns.append(sock)
            srv.close()

            blob = struct.pack("<I", self.world_size) + b"".join(
                struct.pack("<I", len(g)) + g for g in gathered
            )
            for sock in conns:
                self._send(sock, blob)
                sock.close()
            return gathered

        sock = socket.create_connection((self._host, self._port), timeout=self._timeout)
        sock.sendall(struct.pack("<I", self.rank))
        self._send(sock, payload)
        blob = self._recv(sock)
        sock.close()

        (n,) = struct.unpack("<I", blob[:4])
        out, off = [], 4
        for _ in range(n):
            (ln,) = struct.unpack("<I", blob[off : off + 4])
            off += 4
            out.append(blob[off : off + ln])
            off += ln
        return out


class SymmetricMemory:
    """Peer buffers for a rendezvous'd allocation. Returned by rendezvous()."""

    def __init__(self, win, rank, world_size, buffer_size, device):
        self._win = win
        self.rank = rank
        self.world_size = world_size
        self.buffer_size = buffer_size
        self.device = device

    @property
    def flat_base(self):
        """Base of the span holding every rank; peer r sits at flat_base + r*stride."""
        return self._win.flat_base

    @property
    def stride(self):
        return self._win.stride

    def get_buffer(self, rank, sizes, dtype, storage_offset=0):
        """A tensor aliasing `rank`'s buffer. Same signature as torch's."""
        return self._win.get_buffer(rank, sizes, dtype, storage_offset)

    def device_desc(self):
        """The POD a kernel needs: one base pointer, a stride, and who we are.

        torch has no equivalent -- its device-side contract is buffer_ptrs_dev, an N-entry
        pointer array. A stride is cheaper in kernarg and lets the kernel address a rank
        computed at run time.
        """
        return FabricWin(
            base=ctypes.c_void_p(self.flat_base),
            stride=self.stride,
            rank=self.rank,
            world=self.world_size,
        )

    def close(self):
        self._win.close()


def rendezvous(tensor, group_name=None, exchange=None):
    """Make every rank's buffer mutually accessible. Collective over the exchange.

    Exports this rank's allocation as a fabric handle, all-gathers the ~80-byte
    descriptor, imports every peer, and maps them all into one flat span.

        hdl = rendezvous(buf, group_name)                       # torch.distributed
        hdl = rendezvous(buf, exchange=tcp(rank, world, host))  # cross-node, no launcher
    """
    if exchange is None:
        exchange = torch_dist(group_name)

    dev = tensor.device.index
    win = _hf.SymmFabricWindow(tensor, dev)
    handle, size, offset = win.descriptor()

    payloads = exchange.all_gather(_DESC.pack(handle, size, offset))
    if len(payloads) != exchange.world_size:
        raise RuntimeError(f"exchange returned {len(payloads)} descriptors, "
                           f"expected {exchange.world_size}")
    descriptors = [_DESC.unpack(p) for p in payloads]

    win.import_peers(descriptors, exchange.rank)
    return SymmetricMemory(
        win,
        rank=exchange.rank,
        world_size=exchange.world_size,
        buffer_size=tensor.numel() * tensor.element_size(),
        device=tensor.device,
    )
