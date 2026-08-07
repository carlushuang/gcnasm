#!/usr/bin/env python3
"""Cross-GPU transfer over HIP fabric handles, on a pure-torch symmetric memory buffer.

The buffer is a plain `torch.distributed._symmetric_memory` tensor -- allocated,
filled and compared with ordinary torch code. The only thing this example adds is the
piece torch lacks on ROCm: making that buffer fabric-capable and exporting it, so peers
can map it. Nothing intercepts or replaces a HIP entry point.

    Phase 0  rebind the torch buffer to fabric-capable backing, and show the handle
             types before and after
    Phase 1  correctness -- read every peer's buffer with pure torch ops
    Phase 2  bandwidth   -- ring read / write sweep across the fabric mapping

    ./run.sh
    ./run.sh --buffer-mib 512 --sizes 1,16,64,256,512
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

import hip_fabric as hf

MIB = 1 << 20


def human(nbytes):
    if nbytes >= (1 << 30):
        return f"{nbytes / (1 << 30):.2f}GB"
    if nbytes >= MIB:
        return f"{nbytes >> 20}MB"
    return f"{nbytes >> 10}KB"


def sentinel(n, rank, device):
    """Per-rank pattern so a mismatch tells you which buffer you actually read."""
    return torch.arange(n, device=device, dtype=torch.int32) + rank * 100000


def handle_types(t):
    """Which shareable handle types does the allocation behind `t` support?"""
    retain, fabric, fd = hf.probe_ptr(t.data_ptr())
    ok = lambda rc: "ok" if rc == 0 else f"FAIL(hip {rc})"  # noqa: E731
    return (
        f"vmm-backed={ok(retain)}  export-fabric={ok(fabric)}  export-posix-fd={ok(fd)}",
        fabric == 0,
    )


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--buffer-mib", type=int, default=256, help="symm_mem tensor per rank")
    p.add_argument("--verify-mib", type=int, default=16, help="bytes checked in the correctness phase")
    p.add_argument("--sizes", type=str, default="1,16,64,256", help="bandwidth sweep sizes, MiB")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--loop", type=int, default=20)
    p.add_argument("--verbose", action="store_true", help="print correctness lines from every rank")
    return p.parse_args()


def main():
    args = parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # gloo is enough: all that is exchanged collectively is 64 bytes of handle per rank
    dist.init_process_group("gloo")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dev = local_rank

    hf.load()
    # CU count comes from torch, not hipDeviceGetAttribute: see fs_fabric_supported()
    num_cu = torch.cuda.get_device_properties(dev).multi_processor_count
    supported, detail = hf.fabric_supported(dev)
    if not supported:
        if rank == 0:
            print(f"[fatal] device {dev}: fabric handle unusable -- {detail}")
        dist.destroy_process_group()
        return 2

    buffer_bytes = args.buffer_mib * MIB
    verify_bytes = min(args.verify_mib, args.buffer_mib) * MIB
    sizes = [int(s) * MIB for s in args.sizes.split(",") if s.strip()]
    sizes = [s for s in sizes if s <= buffer_bytes]

    if rank == 0:
        print(f"torch {torch.__version__}, hip {torch.version.hip}")
        print(f"{world} rank(s), {num_cu} CUs/GPU, symm_mem buffer {human(buffer_bytes)}/rank\n")

    # ---- pure torch: allocate the symmetric buffer ----
    numel = buffer_bytes // 4
    buf = symm_mem.empty(numel, dtype=torch.int32, device=device)
    before, _ = handle_types(buf)

    # ---- our part: give that same buffer fabric-capable backing, in place ----
    # As allocated it is POSIX-fd only, because c10::cuda::get_fabric_access() is inside
    # `#if !defined(USE_ROCM)`. The handle types are frozen at hipMemCreate, so rather
    # than intercept torch's allocator we rebind this one VA range onto fabric memory.
    # It discards the contents, hence before the fill below.
    base, size = hf.rebind_to_fabric(buf)
    after, exportable = handle_types(buf)

    if rank == 0:
        print(f"[probe] torch symm_mem buffer @ {buf.data_ptr():#x}")
        print(f"[probe]   as allocated : {before}")
        print(f"[probe]   after rebind : {after}")
        print(
            f"[probe]   same pointer, alloc {human(size)} @ {base:#x}; "
            f"costs 2x physical until torch frees it"
        )
        print()

    flag = torch.tensor([0 if exportable else 1], dtype=torch.int64)
    dist.all_reduce(flag, op=dist.ReduceOp.SUM)
    if flag.item() != 0:
        if rank == 0:
            print("[fatal] buffer is still not fabric-exportable after rebind")
        dist.destroy_process_group()
        return 2

    # ---- pure torch again: fill the (now fabric-backed) buffer with torch ops ----
    n_ver = verify_bytes // 4
    buf[:n_ver] = sentinel(n_ver, rank, device)
    torch.cuda.synchronize()

    # ---- export over fabric, exchange, import every peer ----
    win = hf.SymmFabricWindow(buf, dev)

    descriptors = [None] * world
    dist.all_gather_object(descriptors, win.descriptor())
    win.import_peers(descriptors, rank)

    if rank == 0 or args.verbose:
        print(
            f"[rank {rank}] dev {dev}: torch buf {buf.data_ptr():#x} "
            f"(alloc {human(win.size)} +{win.offset}) -> peers {[hex(p) for p in win.peer_ptr]}",
            flush=True,
        )
    dist.barrier()

    # ---- phase 1: read every peer's buffer with pure torch ops ----
    errors = 0
    for peer in range(world):
        # a torch tensor aliasing the peer's memory; the comparison is the transfer
        peer_buf = win.get_buffer(peer, (n_ver,), torch.int32)
        bad = int((peer_buf != sentinel(n_ver, peer, device)).sum().item())
        errors += bad != 0
        if bad or rank == 0 or args.verbose:
            tag = "local" if peer == rank else "peer "
            status = "OK" if bad == 0 else f"MISMATCH ({bad} elems)"
            print(f"[rank {rank}] torch read {tag} {peer}: {human(verify_bytes)} {status}", flush=True)
    dist.barrier()

    # ---- phase 2: ring bandwidth over the fabric mapping ----
    if world >= 2 and sizes:
        peer = (rank + 1) % world
        max_bytes = max(sizes)
        local_buf = torch.empty(max_bytes // 4, dtype=torch.int32, device=device)

        if rank == 0:
            print(f"\ncross-GPU bandwidth over fabric, {world} ring pairs (rank -> rank+1)")
            print(f"  same uint4 copy kernel throughout;  warmup={args.warmup} loop={args.loop}")
            print("  local: within one GPU (reference ceiling);  read: local <- peer;  write: local -> peer\n")
            print(f"{'size':>10} {'local GB/s':>13} {'read GB/s':>13} {'write GB/s':>13}   (aggregate)")
            print(f"{'----':>10} {'----------':>13} {'---------':>13} {'----------':>13}")

        def timed(dst, src, nbytes):
            ms = hf.bench(dev, dst, src, nbytes, num_cu, args.warmup, args.loop)
            return nbytes / (ms / 1e3) / 1e9

        for nbytes in sizes:
            # device-local copy with the same kernel, as a reference for the fabric numbers
            dist.barrier()
            local = timed(win.peer_ptr[rank], local_buf.data_ptr(), nbytes)
            dist.barrier()
            read = timed(local_buf.data_ptr(), win.peer_ptr[peer], nbytes)
            dist.barrier()
            write = timed(win.peer_ptr[peer], local_buf.data_ptr(), nbytes)

            gbps = torch.tensor([local, read, write], dtype=torch.float64)
            dist.all_reduce(gbps, op=dist.ReduceOp.SUM)
            if rank == 0:
                print(
                    f"{human(nbytes):>10} {gbps[0].item():13.1f} "
                    f"{gbps[1].item():13.1f} {gbps[2].item():13.1f}"
                )

    # the write phase overwrote peer buffers, so tear down without re-checking
    dist.barrier()
    win.close()

    total_errors = torch.tensor([errors], dtype=torch.int64)
    dist.all_reduce(total_errors, op=dist.ReduceOp.SUM)
    if rank == 0:
        print("\nSUCCESS" if total_errors.item() == 0 else f"\nFAILED ({total_errors.item()} bad reads)")

    dist.destroy_process_group()
    return 0 if total_errors.item() == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
