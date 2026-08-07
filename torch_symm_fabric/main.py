#!/usr/bin/env python3
"""Cross-GPU transfer over HIP fabric handles, on a pure-torch symmetric memory buffer.

The buffer is a plain `torch.distributed._symmetric_memory` tensor -- allocated,
filled and compared with ordinary torch code. The only thing this example adds is the
piece torch lacks on ROCm: exporting that buffer as a fabric handle so peers can map it.

    Phase 0  probe the handle types torch's allocation actually supports
    Phase 1  correctness -- read every peer's buffer with pure torch ops
    Phase 2  bandwidth   -- ring read / write sweep across the fabric mapping

Run through ./run.sh, which preloads libfabric_shim.so; without it torch allocates
POSIX-fd-only memory and the export in phase 0 fails with a pointer to the fix.
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


def report_handle_types(t, rank):
    """Print which shareable handle types torch's own allocation supports."""
    retain, fabric, fd = hf.probe_ptr(t.data_ptr())
    ok = lambda rc: "ok" if rc == 0 else f"FAIL(hip {rc})"  # noqa: E731
    print(
        f"[probe] torch symm_mem buffer @ {t.data_ptr():#x}: "
        f"vmm-backed={ok(retain)}  export-fabric={ok(fabric)}  export-posix-fd={ok(fd)}"
    )
    if fabric != 0:
        print(
            "[probe] torch allocated this POSIX-fd-only, so it cannot cross a fabric.\n"
            "[probe] c10::cuda::get_fabric_access() is inside `#if !defined(USE_ROCM)`, and the\n"
            "[probe] handle types of a VMM allocation are frozen at hipMemCreate time.\n"
            "[probe] Preload libfabric_shim.so to flip that one bit -- use ./run.sh."
        )
    else:
        print("[probe] fabric-exportable (libfabric_shim.so is active)")
    return fabric == 0


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

    # ---- pure torch: allocate the symmetric buffer and fill it with torch ops ----
    numel = buffer_bytes // 4
    buf = symm_mem.empty(numel, dtype=torch.int32, device=device)
    n_ver = verify_bytes // 4
    buf[:n_ver] = sentinel(n_ver, rank, device)
    torch.cuda.synchronize()

    if rank == 0:
        exportable = report_handle_types(buf, rank)
        print()
    else:
        exportable = hf.probe_ptr(buf.data_ptr())[1] == 0

    flag = torch.tensor([0 if exportable else 1], dtype=torch.int64)
    dist.all_reduce(flag, op=dist.ReduceOp.SUM)
    if flag.item() != 0:
        dist.destroy_process_group()
        return 2

    # ---- our part: export over fabric, exchange, import every peer ----
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
