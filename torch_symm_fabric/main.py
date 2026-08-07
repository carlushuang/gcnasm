#!/usr/bin/env python3
"""Cross-GPU transfer over a HIP fabric-handle symmetric window, driven from torch.

Each rank owns one GPU and builds the local half of a symmetric window with the HIP
VMM API (hipMemCreate with requestedHandleTypes = hipMemHandleTypeFabric), wraps it
zero-copy as a torch tensor, all-gathers the 64-byte fabric handle, and imports every
peer's window. After that a peer's buffer is just a device pointer, so an ordinary
kernel can read or write it.

    Phase 0  probe what torch.distributed._symmetric_memory hands you on this build
    Phase 1  correctness -- every rank reads every peer's window and checks the sentinel
    Phase 2  bandwidth   -- ring read / write sweep across the fabric mapping

    ./run.sh                        # all visible GPUs
    ./run.sh --window-mib 512 --sizes 1,16,64,256
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist

import hip_fabric as hf

MIB = 1 << 20


def human(nbytes):
    if nbytes >= (1 << 30):
        return f"{nbytes / (1 << 30):.2f}GB"
    if nbytes >= MIB:
        return f"{nbytes >> 20}MB"
    return f"{nbytes >> 10}KB"


def sentinel(n, rank, device):
    """Per-rank pattern so a mismatch tells you which window you actually read."""
    return torch.arange(n, device=device, dtype=torch.int32) + rank * 100000


def probe_torch_symm_mem(device):
    """Report whether a torch symm_mem buffer is VMM-backed and fabric-exportable.

    On ROCm builds c10::cuda::get_fabric_access() is compiled out (`#if !defined(USE_ROCM)`),
    so torch always allocates symmetric memory with the POSIX-fd handle type and the fabric
    export below fails. That is exactly why this example allocates its own window.
    """
    try:
        import torch.distributed._symmetric_memory as symm_mem
    except ImportError:
        print("[probe] torch.distributed._symmetric_memory unavailable")
        return

    try:
        t = symm_mem.empty(MIB, dtype=torch.uint8, device=device)
    except Exception as exc:  # allocation can fail if no symm-mem backend is present
        print(f"[probe] symm_mem.empty failed: {type(exc).__name__}: {exc}")
        return

    retain, fabric, fd = hf.probe_ptr(t.data_ptr())
    ok = lambda rc: "ok" if rc == 0 else f"FAIL(hip {rc})"  # noqa: E731
    print(
        f"[probe] torch symm_mem buffer @ {t.data_ptr():#x}: "
        f"vmm-backed={ok(retain)}  export-fabric={ok(fabric)}  export-posix-fd={ok(fd)}"
    )
    if fabric != 0:
        print(
            "[probe] -> torch cannot export this buffer over fabric on ROCm; "
            "this example allocates its own fabric-capable window instead"
        )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--window-mib", type=int, default=256, help="symmetric window size per rank")
    p.add_argument("--verify-mib", type=int, default=16, help="bytes checked in the correctness phase")
    p.add_argument("--sizes", type=str, default="1,16,64,256", help="bandwidth sweep sizes, MiB")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--loop", type=int, default=20)
    p.add_argument("--no-probe", action="store_true", help="skip the torch symm_mem probe")
    p.add_argument("--verbose", action="store_true", help="print correctness lines from every rank")
    return p.parse_args()


def main():
    args = parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # gloo is enough: the only thing exchanged collectively is 64 bytes of handle per rank
    dist.init_process_group("gloo")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dev = local_rank

    hf.load()
    # CU count comes from torch, not hipDeviceGetAttribute: see fs_fabric_supported()
    num_cu = torch.cuda.get_device_properties(dev).multi_processor_count
    supported, detail = hf.fabric_supported(dev)
    if not supported:
        print(f"[fatal] rank {rank} dev {dev}: fabric handle unusable -- {detail}", flush=True)
        dist.destroy_process_group()
        return 2

    window_bytes = args.window_mib * MIB
    verify_bytes = min(args.verify_mib, args.window_mib) * MIB
    sizes = [int(s) * MIB for s in args.sizes.split(",") if s.strip()]
    sizes = [s for s in sizes if s <= window_bytes]

    if rank == 0:
        print(f"torch {torch.__version__}, hip {torch.version.hip}")
        print(f"{world} rank(s), {num_cu} CUs/GPU, window {human(window_bytes)}/rank\n")
        if not args.no_probe:
            probe_torch_symm_mem(device)
            print()

    # ---- build the symmetric window and exchange fabric handles ----
    win = hf.FabricWindow(dev, window_bytes)
    buf = hf.tensor_from_ptr(win.local_ptr, win.total // 4, torch.int32, device)

    n_ver = verify_bytes // 4
    buf[:n_ver] = sentinel(n_ver, rank, device)
    torch.cuda.synchronize()

    gathered = [None] * world
    dist.all_gather_object(gathered, (win.handle, win.total))
    win.import_peers([g[0] for g in gathered], [g[1] for g in gathered], rank)

    print(
        f"[rank {rank}] dev {dev}: local {win.local_ptr:#x} "
        f"({human(win.total)}) -> peers {[hex(p) for p in win.peer_ptr]}",
        flush=True,
    )
    dist.barrier()

    # ---- phase 1: read every peer's window and check its sentinel ----
    errors = 0
    staging = torch.empty(n_ver, dtype=torch.int32, device=device)
    for peer in range(world):
        hf.copy(dev, staging.data_ptr(), win.peer_ptr[peer], verify_bytes, num_cu)
        bad = int((staging != sentinel(n_ver, peer, device)).sum().item())
        errors += bad != 0
        # only rank 0 narrates the happy path; a mismatch always speaks up
        if bad or rank == 0 or args.verbose:
            tag = "local" if peer == rank else "peer "
            status = "OK" if bad == 0 else f"MISMATCH ({bad} elems)"
            print(f"[rank {rank}] read {tag} {peer}: {human(verify_bytes)} {status}", flush=True)
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
            return nbytes / (hf.bench(dev, dst, src, nbytes, num_cu, args.warmup, args.loop) / 1e3) / 1e9

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

    # the write phase overwrote peer windows, so tear down without re-checking
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
