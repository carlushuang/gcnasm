#!/usr/bin/env python3
"""What CUDA torch does today: symmetric memory end to end in pure python.

The parent example exists because torch on ROCm cannot hand you a fabric-exportable
symm_mem buffer -- get_fabric_access() is compiled out -- so it needs a C++ helper. On
CUDA that helper is unnecessary: torch negotiates the handle type itself and
symm_mem.rendezvous() gives you peer buffers directly. This script is the control case,
and it is pure python/torch -- ctypes appears only in the diagnostic, never in the data
path.

    Phase 0  which shareable handle type did torch pick, and why (replicates torch's
             own get_fabric_access() gate: cuMemCreate/export/import with FABRIC)
    Phase 1  rendezvous, then read every peer's buffer with hdl.get_buffer() + torch ops
    Phase 2  cross-GPU bandwidth with tensor.copy_(), no custom kernel

    ./run.sh
    ./run.sh --buffer-mib 512 --sizes 1,16,64,256,512
"""

import argparse
import ctypes
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

MIB = 1 << 20

# CUmemAllocationHandleType
CU_HANDLE_POSIX_FD = 0x1
CU_HANDLE_FABRIC = 0x8
HANDLE_NAMES = {CU_HANDLE_POSIX_FD: "POSIX_FD", CU_HANDLE_FABRIC: "FABRIC"}


def human(nbytes):
    if nbytes >= (1 << 30):
        return f"{nbytes / (1 << 30):.2f}GB"
    if nbytes >= MIB:
        return f"{nbytes >> 20}MB"
    return f"{nbytes >> 10}KB"


def sentinel(n, rank, device):
    """Per-rank pattern so a mismatch tells you which buffer you actually read."""
    return torch.arange(n, device=device, dtype=torch.int32) + rank * 100000


class _Loc(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _Flags(ctypes.Structure):
    _fields_ = [("compressionType", ctypes.c_ubyte), ("rdma", ctypes.c_ubyte), ("usage", ctypes.c_ushort)]


class _Prop(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", _Loc),
        ("win32", ctypes.c_void_p),
        ("allocFlags", _Flags),
    ]


class Driver:
    """Thin ctypes view of the CUDA driver, for the diagnostic only."""

    def __init__(self):
        self.cu = ctypes.CDLL("libcuda.so.1")

    def err(self, rc):
        name = ctypes.c_char_p()
        self.cu.cuGetErrorName(ctypes.c_int(rc), ctypes.byref(name))
        return f"{rc}({name.value.decode() if name.value else '?'})"

    def handle_types_of(self, ptr):
        """Which handle types can the allocation behind `ptr` actually export?"""
        h = ctypes.c_void_p()
        rc = self.cu.cuMemRetainAllocationHandle(ctypes.byref(h), ctypes.c_void_p(ptr))
        if rc != 0:
            return None, {}, f"not VMM-backed: {self.err(rc)}"

        prop = _Prop()
        self.cu.cuMemGetAllocationPropertiesFromHandle(ctypes.byref(prop), h)
        requested = prop.requestedHandleTypes

        results = {}
        buf = (ctypes.c_ubyte * 64)()
        for mask in (CU_HANDLE_FABRIC, CU_HANDLE_POSIX_FD):
            rc = self.cu.cuMemExportToShareableHandle(
                ctypes.byref(buf), h, ctypes.c_int(mask), ctypes.c_ulonglong(0)
            )
            results[mask] = rc
        self.cu.cuMemRelease(h)
        return requested, results, None

    def fabric_gate(self, dev):
        """Replicate torch's isFabricSupported(): granularity, create, export, import."""
        prop = _Prop()
        prop.type = 1  # CU_MEM_ALLOCATION_TYPE_PINNED
        prop.requestedHandleTypes = CU_HANDLE_FABRIC
        prop.location.type = 1  # CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = dev

        gran = ctypes.c_size_t()
        rc = self.cu.cuMemGetAllocationGranularity(ctypes.byref(gran), ctypes.byref(prop), ctypes.c_int(1))
        if rc != 0:
            return f"granularity failed: {self.err(rc)}"

        h = ctypes.c_void_p()
        rc = self.cu.cuMemCreate(ctypes.byref(h), gran, ctypes.byref(prop), ctypes.c_ulonglong(0))
        if rc != 0:
            return f"cuMemCreate(FABRIC) failed: {self.err(rc)}"

        buf = (ctypes.c_ubyte * 64)()
        rc = self.cu.cuMemExportToShareableHandle(
            ctypes.byref(buf), h, ctypes.c_int(CU_HANDLE_FABRIC), ctypes.c_ulonglong(0)
        )
        if rc != 0:
            self.cu.cuMemRelease(h)
            return f"export failed: {self.err(rc)}"

        h2 = ctypes.c_void_p()
        rc = self.cu.cuMemImportFromShareableHandle(
            ctypes.byref(h2), ctypes.byref(buf), ctypes.c_int(CU_HANDLE_FABRIC)
        )
        if rc != 0:
            self.cu.cuMemRelease(h)
            return f"import failed: {self.err(rc)}"

        self.cu.cuMemRelease(h2)
        self.cu.cuMemRelease(h)
        return None  # fabric fully usable


def report_handle_choice(drv, buf, dev):
    """Phase 0: what torch chose, and if it was not fabric, why not."""
    requested, exports, problem = drv.handle_types_of(buf.data_ptr())
    if problem:
        print(f"[probe] {problem}")
        return None

    chosen = HANDLE_NAMES.get(requested, f"0x{requested:x}")
    detail = "  ".join(
        f"export-{HANDLE_NAMES[m].lower()}={'ok' if rc == 0 else f'FAIL {drv.err(rc)}'}"
        for m, rc in exports.items()
    )
    print(f"[probe] torch symm_mem buffer @ {buf.data_ptr():#x}")
    print(f"[probe]   created with requestedHandleTypes=0x{requested:x} ({chosen})")
    print(f"[probe]   {detail}")

    if requested == CU_HANDLE_FABRIC:
        print("[probe]   -> torch negotiated FABRIC by itself; nothing else is needed")
    else:
        why = drv.fabric_gate(dev)
        print(f"[probe]   -> torch fell back to POSIX-FD because its fabric gate failed:")
        print(f"[probe]      {why}")
        print("[probe]      (fabric handles need the IMEX daemon; POSIX fds work node-local)")
    return requested


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--buffer-mib", type=int, default=256, help="symm_mem tensor per rank")
    p.add_argument("--verify-mib", type=int, default=16, help="bytes checked in the correctness phase")
    p.add_argument("--sizes", type=str, default="1,16,64,256", help="bandwidth sweep sizes, MiB")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--loop", type=int, default=20)
    p.add_argument("--backend", default="gloo", help="process group backend for the rendezvous")
    p.add_argument("--verbose", action="store_true", help="print correctness lines from every rank")
    return p.parse_args()


def main():
    args = parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    dist.init_process_group(args.backend)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    drv = Driver()
    buffer_bytes = args.buffer_mib * MIB
    verify_bytes = min(args.verify_mib, args.buffer_mib) * MIB
    sizes = [s for s in (int(x) * MIB for x in args.sizes.split(",") if x.strip()) if s <= buffer_bytes]

    if rank == 0:
        print(f"torch {torch.__version__}, cuda {torch.version.cuda}")
        print(f"{torch.cuda.get_device_name(local_rank)}  {world} rank(s)  buffer {human(buffer_bytes)}/rank\n")

    # ---- pure torch: allocate the symmetric buffer ----
    numel = buffer_bytes // 4
    t0 = time.perf_counter()
    buf = symm_mem.empty(numel, dtype=torch.int32, device=device)
    torch.cuda.synchronize()
    alloc_ms = (time.perf_counter() - t0) * 1e3

    handle_type = None
    if rank == 0:
        handle_type = report_handle_choice(drv, buf, local_rank)
        print()

    n_ver = verify_bytes // 4
    buf[:n_ver] = sentinel(n_ver, rank, device)
    torch.cuda.synchronize()

    # ---- pure torch: rendezvous. this is the part ROCm cannot do over fabric ----
    t0 = time.perf_counter()
    try:
        hdl = symm_mem.rendezvous(buf, dist.group.WORLD.group_name)
    except RuntimeError as exc:
        if rank == 0 and os.environ.get("TORCH_SYMM_MEM_DISABLE_MULTICAST") != "1":
            print(f"[rendezvous] FAILED: {exc}")
            print("[rendezvous] on GPUs without NVLS multicast (H20 among them) torch errors")
            print("[rendezvous] instead of skipping it -- set TORCH_SYMM_MEM_DISABLE_MULTICAST=1")
            print("[rendezvous] (./run.sh does this for you)")
        raise
    rendezvous_ms = (time.perf_counter() - t0) * 1e3
    if rank == 0:
        mc_off = os.environ.get("TORCH_SYMM_MEM_DISABLE_MULTICAST") == "1"
        print(f"[rendezvous] ok, world_size={hdl.world_size}, {rendezvous_ms:.1f} ms")
        print(f"[rendezvous] multicast: {'disabled via TORCH_SYMM_MEM_DISABLE_MULTICAST' if mc_off else 'enabled'}\n")
    dist.barrier()

    # ---- phase 1: read every peer's buffer through torch, no custom kernel ----
    errors = 0
    for peer in range(world):
        peer_buf = hdl.get_buffer(peer, (n_ver,), torch.int32)
        bad = int((peer_buf != sentinel(n_ver, peer, device)).sum().item())
        errors += bad != 0
        if bad or rank == 0 or args.verbose:
            tag = "local" if peer == rank else "peer "
            status = "OK" if bad == 0 else f"MISMATCH ({bad} elems)"
            print(f"[rank {rank}] torch read {tag} {peer}: {human(verify_bytes)} {status}", flush=True)
    dist.barrier()

    # ---- phase 2: bandwidth with tensor.copy_(), still pure torch ----
    peak = {"local": 0.0, "read": 0.0, "write": 0.0}
    if world >= 2 and sizes:
        peer = (rank + 1) % world
        local_buf = torch.empty(max(sizes) // 4, dtype=torch.int32, device=device)

        if rank == 0:
            print(f"\ncross-GPU bandwidth, {world} ring pairs (rank -> rank+1)")
            print(f"  torch tensor.copy_() throughout;  warmup={args.warmup} loop={args.loop}")
            print("  local: within one GPU (reference ceiling);  read: local <- peer;  write: local -> peer\n")
            print(f"{'size':>10} {'local GB/s':>13} {'read GB/s':>13} {'write GB/s':>13}   (aggregate)")
            print(f"{'----':>10} {'----------':>13} {'---------':>13} {'----------':>13}")

        def timed(dst, src, nbytes):
            for _ in range(args.warmup):
                dst.copy_(src)
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(args.loop):
                dst.copy_(src)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t) * 1e3 / args.loop
            return nbytes / (ms / 1e3) / 1e9

        for nbytes in sizes:
            n = nbytes // 4
            mine = hdl.get_buffer(rank, (n,), torch.int32)
            theirs = hdl.get_buffer(peer, (n,), torch.int32)
            dst = local_buf[:n]

            dist.barrier()
            local = timed(mine, dst, nbytes)
            dist.barrier()
            read = timed(dst, theirs, nbytes)
            dist.barrier()
            write = timed(theirs, dst, nbytes)

            gbps = torch.tensor([local, read, write], dtype=torch.float64)
            dist.all_reduce(gbps, op=dist.ReduceOp.SUM)
            if rank == 0:
                print(
                    f"{human(nbytes):>10} {gbps[0].item():13.1f} "
                    f"{gbps[1].item():13.1f} {gbps[2].item():13.1f}"
                )
            for k, v in zip(("local", "read", "write"), gbps.tolist()):
                peak[k] = max(peak[k], v)

    dist.barrier()
    total_errors = torch.tensor([errors], dtype=torch.int64)
    dist.all_reduce(total_errors, op=dist.ReduceOp.SUM)
    ok = total_errors.item() == 0

    if rank == 0:
        print("\nSUCCESS" if ok else f"\nFAILED ({total_errors.item()} bad reads)")
        print(
            f"[summary] platform=cuda handle_type={HANDLE_NAMES.get(handle_type, handle_type)} "
            f"ranks={world} buffer_mib={args.buffer_mib} alloc_ms={alloc_ms:.1f} "
            f"rendezvous_ms={rendezvous_ms:.1f} local_gbs={peak['local']:.0f} "
            f"read_gbs={peak['read']:.0f} write_gbs={peak['write']:.0f} "
            f"status={'OK' if ok else 'FAIL'}"
        )

    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
