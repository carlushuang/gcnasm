#!/usr/bin/env python3
"""Cross-process fabric exchange with no shared launcher -- the cross-node shape.

main.py uses torchrun, so its ranks share a parent and a rendezvous. This script does
not: two processes are started independently and meet over a TCP socket, which is all a
fabric handle needs. It is 64 opaque position-independent bytes, so the same code works
between hosts -- point --host at the other node instead of 127.0.0.1.

That is the property POSIX fds do not have: sharing an fd across processes needs
SCM_RIGHTS or pidfd_getfd, and neither crosses a machine boundary.

    # on one node
    python3 xproc.py serve   --gpu 0 --port 55600
    # on the other (or the same box, different GPU)
    python3 xproc.py connect --gpu 1 --port 55600 --host <server-ip>

Both sides verify: the client reads the server's sentinel, writes its own pattern back,
and the server checks it -- so the mapping is proven readable and writable in both
directions.
"""

import argparse
import os
import socket
import struct
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hip_fabric as hf  # noqa: E402

MIB = 1 << 20
MAGIC = b"FABX0001"
# magic, allocation size, tensor offset within it, 64-byte handle
WIRE = struct.Struct("<8sQQ64s")

SERVER_RANK, CLIENT_RANK = 0, 1


def sentinel(n, tag, device):
    return torch.arange(n, device=device, dtype=torch.int32) + tag * 100000


def send_all(sock, data):
    sock.sendall(data)


def recv_all(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed early")
        buf += chunk
    return buf


def make_buffer(method, dev, device, nbytes):
    """Same three methods as main.py; returns (tensor, owner_or_None)."""
    if method == "own":
        own = hf.OwnFabricBuffer(dev, nbytes, torch.int32, device)
        return own.tensor, own

    import torch.distributed._symmetric_memory as symm_mem

    buf = symm_mem.empty(nbytes // 4, dtype=torch.int32, device=device)
    if method == "rebind":
        hf.rebind_to_fabric(buf)
    return buf, None


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("role", choices=("serve", "connect"))
    p.add_argument("--gpu", type=int, default=0, help="local device index")
    p.add_argument("--host", default="127.0.0.1", help="server address (connect side)")
    p.add_argument("--port", type=int, default=55600)
    p.add_argument("--method", choices=("own", "shim", "rebind"), default="own")
    p.add_argument("--buffer-mib", type=int, default=64)
    p.add_argument("--verify-mib", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    me = SERVER_RANK if args.role == "serve" else CLIENT_RANK

    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda", args.gpu)
    hf.load()

    ok, detail = hf.fabric_supported(args.gpu)
    if not ok:
        print(f"[{args.role}] fatal: fabric unusable on gpu {args.gpu} -- {detail}")
        return 2

    nbytes = args.buffer_mib * MIB
    n_ver = min(args.verify_mib, args.buffer_mib) * MIB // 4

    buf, owner = make_buffer(args.method, args.gpu, device, nbytes)
    buf[:n_ver] = sentinel(n_ver, me, device)
    torch.cuda.synchronize()

    win = hf.SymmFabricWindow(buf, args.gpu)
    handle, size, offset = win.descriptor()
    print(
        f"[{args.role}] pid {os.getpid()} gpu {args.gpu} method={args.method}: "
        f"buf {buf.data_ptr():#x}, exported {len(handle)}-byte fabric handle "
        f"{handle[:8].hex()}...",
        flush=True,
    )

    # ---- meet over TCP and swap descriptors; nothing else is shared ----
    if args.role == "serve":
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", args.port))
        srv.listen(1)
        print(f"[serve] waiting on :{args.port}", flush=True)
        sock, addr = srv.accept()
        srv.close()
        print(f"[serve] peer connected from {addr[0]}", flush=True)
    else:
        sock = socket.create_connection((args.host, args.port), timeout=60)
        print(f"[connect] connected to {args.host}:{args.port}", flush=True)

    send_all(sock, WIRE.pack(MAGIC, size, offset, handle))
    magic, peer_size, peer_offset, peer_handle = WIRE.unpack(recv_all(sock, WIRE.size))
    if magic != MAGIC:
        raise RuntimeError(f"bad magic {magic!r}")

    descriptors = [None, None]
    descriptors[me] = (handle, size, offset)
    descriptors[1 - me] = (peer_handle, peer_size, peer_offset)
    win.import_peers(descriptors, me)
    print(f"[{args.role}] imported peer window -> {win.peer_ptr[1 - me]:#x}", flush=True)

    # ---- verify in both directions ----
    errors = 0
    peer_buf = win.get_buffer(1 - me, (n_ver,), torch.int32)

    if args.role == "connect":
        bad = int((peer_buf != sentinel(n_ver, SERVER_RANK, device)).sum().item())
        errors += bad != 0
        print(f"[connect] read server's buffer: {'OK' if not bad else f'MISMATCH ({bad})'}", flush=True)

        # write our pattern into the server's memory, straight over the fabric
        peer_buf.copy_(sentinel(n_ver, CLIENT_RANK, device))
        torch.cuda.synchronize()
        send_all(sock, b"W")
        print("[connect] wrote our pattern into the server's buffer", flush=True)
        recv_all(sock, 1)
    else:
        bad = int((peer_buf != sentinel(n_ver, CLIENT_RANK, device)).sum().item())
        errors += bad != 0
        print(f"[serve] read client's buffer: {'OK' if not bad else f'MISMATCH ({bad})'}", flush=True)

        recv_all(sock, 1)  # client finished writing into our buffer
        got = buf[:n_ver]
        bad = int((got != sentinel(n_ver, CLIENT_RANK, device)).sum().item())
        errors += bad != 0
        print(
            f"[serve] our buffer now holds the client's pattern: "
            f"{'OK' if not bad else f'MISMATCH ({bad})'}",
            flush=True,
        )
        send_all(sock, b"D")

    sock.close()
    win.close()
    if owner is not None:
        owner.close()

    print(f"[{args.role}] {'SUCCESS' if errors == 0 else f'FAILED ({errors})'}", flush=True)
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
