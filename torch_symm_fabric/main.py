#!/usr/bin/env python3
"""Cross-GPU transfer over HIP fabric handles, three ways to get a fabric-capable buffer.

torch.distributed._symmetric_memory allocates through hipMemCreate and on ROCm always
asks for the POSIX-fd handle type, because c10::cuda::get_fabric_access() is inside
`#if !defined(USE_ROCM)`. The handle types are frozen at creation, so a symm_mem buffer
can never be exported over fabric as allocated. There are three ways around that:

  own     we allocate the fabric window ourselves; torch only gets a tensor view.
          Not a symm_mem tensor. One allocation, no interposition.
  shim    LD_PRELOAD fabric_shim.cpp so torch's own hipMemCreate asks for fabric.
          A real symm_mem tensor, fabric from birth -- but hipMemCreate is overridden
          process-wide. This is what an upstream get_fabric_access() would do.
  rebind  torch allocates, then we remap that VA range onto fabric backing in place.
          A real symm_mem tensor, no interposition -- but it costs 2x physical memory.

Everything after that is shared: export a 64-byte fabric handle, all-gather it, import
every peer, and read/write peer memory. --method selects which one to measure.

    ./run.sh --method rebind
    ./bench_methods.sh              # all three, side by side
"""

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

import hip_fabric as hf

MIB = 1 << 20
METHODS = ("own", "shim", "rebind")


def human(nbytes):
    if nbytes >= (1 << 30):
        return f"{nbytes / (1 << 30):.2f}GB"
    if nbytes >= MIB:
        return f"{nbytes >> 20}MB"
    return f"{nbytes >> 10}KB"


def sentinel(n, rank, device):
    """Per-rank pattern so a mismatch tells you which buffer you actually read."""
    return torch.arange(n, device=device, dtype=torch.int32) + rank * 100000


def handle_types(ptr):
    """Which shareable handle types does the allocation behind `ptr` support?"""
    retain, fabric, fd = hf.probe_ptr(ptr)
    ok = lambda rc: "ok" if rc == 0 else f"FAIL(hip {rc})"  # noqa: E731
    return (
        f"vmm-backed={ok(retain)}  export-fabric={ok(fabric)}  export-posix-fd={ok(fd)}",
        fabric == 0,
    )


def make_buffer(method, dev, device, nbytes):
    """Produce a fabric-capable int32 buffer by the chosen method.

    Returns (tensor, owner, before_desc, after_desc). `owner` is closed at teardown, or
    None when torch owns the allocation.
    """
    numel = nbytes // 4

    if method == "own":
        own = hf.OwnFabricBuffer(dev, nbytes, torch.int32, device)
        desc, _ = handle_types(own.tensor.data_ptr())
        return own.tensor, own, "n/a (torch not involved)", desc

    # both remaining methods start from a real torch symmetric-memory tensor
    buf = symm_mem.empty(numel, dtype=torch.int32, device=device)
    before, _ = handle_types(buf.data_ptr())

    if method == "shim":
        # nothing to do: if libfabric_shim.so is preloaded the allocation is already
        # fabric-capable, and if it is not, the check below reports it
        return buf, None, before, before

    hf.rebind_to_fabric(buf)  # discards contents, hence before any fill
    after, _ = handle_types(buf.data_ptr())
    return buf, None, before, after


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--method", choices=METHODS, default="rebind", help="how to get a fabric buffer")
    p.add_argument("--buffer-mib", type=int, default=256, help="buffer per rank")
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
    sizes = [s for s in (int(x) * MIB for x in args.sizes.split(",") if x.strip()) if s <= buffer_bytes]

    if rank == 0:
        print(f"torch {torch.__version__}, hip {torch.version.hip}")
        print(
            f"method={args.method}  {world} rank(s)  {num_cu} CUs/GPU  "
            f"buffer {human(buffer_bytes)}/rank\n"
        )

    # Warm up torch's symmetric-memory machinery before measuring, so first-use lazy
    # initialisation is not charged to the buffer.
    warm = symm_mem.empty(MIB // 4, dtype=torch.int32, device=device)
    del warm
    torch.cuda.synchronize()

    # ---- build the buffer, timing it and pricing its physical footprint ----
    free_before, _ = hf.mem_info()
    t0 = time.perf_counter()
    buf, owner, before_desc, after_desc = make_buffer(args.method, dev, device, buffer_bytes)
    torch.cuda.synchronize()
    setup_ms = (time.perf_counter() - t0) * 1e3
    free_after, _ = hf.mem_info()
    phys_mib = (free_before - free_after) >> 20

    desc, exportable = handle_types(buf.data_ptr())
    if rank == 0:
        print(f"[probe] buffer @ {buf.data_ptr():#x}")
        print(f"[probe]   before : {before_desc}")
        print(f"[probe]   after  : {after_desc}")
        print(f"[probe]   setup {setup_ms:.1f} ms, physical {phys_mib} MiB for a {human(buffer_bytes)} buffer")
        if not exportable and args.method == "shim":
            print(
                "[probe] not fabric-capable: libfabric_shim.so is not preloaded.\n"
                "[probe] use ./run.sh --method shim, which sets LD_PRELOAD for you."
            )
        print()

    flag = torch.tensor([0 if exportable else 1], dtype=torch.int64)
    dist.all_reduce(flag, op=dist.ReduceOp.SUM)
    if flag.item() != 0:
        if rank == 0:
            print(f"[fatal] buffer is not fabric-exportable with method={args.method}")
        dist.destroy_process_group()
        return 2

    # ---- fill with torch ops (after any rebind, which discards contents) ----
    n_ver = verify_bytes // 4
    buf[:n_ver] = sentinel(n_ver, rank, device)
    torch.cuda.synchronize()

    # ---- shared path: export over fabric, exchange, import every peer ----
    win = hf.SymmFabricWindow(buf, dev)
    descriptors = [None] * world
    dist.all_gather_object(descriptors, win.descriptor())
    win.import_peers(descriptors, rank)

    if rank == 0 or args.verbose:
        print(
            f"[rank {rank}] dev {dev}: buf {buf.data_ptr():#x} "
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
    peak = {"local": 0.0, "read": 0.0, "write": 0.0}
    if world >= 2 and sizes:
        peer = (rank + 1) % world
        local_buf = torch.empty(max(sizes) // 4, dtype=torch.int32, device=device)

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
            peak["local"] = max(peak["local"], gbps[0].item())
            peak["read"] = max(peak["read"], gbps[1].item())
            peak["write"] = max(peak["write"], gbps[2].item())

    # the write phase overwrote peer buffers, so tear down without re-checking
    dist.barrier()
    win.close()
    if owner is not None:
        owner.close()

    total_errors = torch.tensor([errors], dtype=torch.int64)
    dist.all_reduce(total_errors, op=dist.ReduceOp.SUM)
    ok = total_errors.item() == 0

    if rank == 0:
        print("\nSUCCESS" if ok else f"\nFAILED ({total_errors.item()} bad reads)")
        # one machine-readable line, collated by bench_methods.sh
        print(
            f"[summary] method={args.method} ranks={world} buffer_mib={args.buffer_mib} "
            f"setup_ms={setup_ms:.1f} phys_mib={phys_mib} "
            f"local_gbs={peak['local']:.0f} read_gbs={peak['read']:.0f} "
            f"write_gbs={peak['write']:.0f} status={'OK' if ok else 'FAIL'}"
        )

    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
