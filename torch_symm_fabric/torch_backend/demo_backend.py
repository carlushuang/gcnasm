#!/usr/bin/env python3
"""Drive the FABRIC backend entirely through torch's own symm_mem entry points.

    torchrun --nnodes=1 --nproc_per_node=4 demo_backend.py

Nothing here calls our code directly after the import: symm_mem.empty and
symm_mem.rendezvous dispatch into the registered allocator.
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

import fabric_backend  # noqa: F401  -- the import is what registers "FABRIC"

MIB = 1 << 20


def sentinel(n, rank, device):
    return torch.arange(n, device=device, dtype=torch.int32) + rank * 100000


def main():
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    lr = int(os.environ.get("LOCAL_RANK", "0"))

    dist.init_process_group("gloo")
    torch.cuda.set_device(lr)
    device = torch.device("cuda", lr)

    symm_mem.set_backend("FABRIC")
    # registers (rank, world_size, store) for the group name so rendezvous can find it
    symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)
    if rank == 0:
        print(f"torch {torch.__version__}")
        print(f"backend: {symm_mem.get_backend(device)}\n")

    n = 16 * MIB // 4

    # torch's own API from here on -- our allocator is behind it
    buf = symm_mem.empty(n, dtype=torch.int32, device=device)
    buf[:] = sentinel(n, rank, device)
    torch.cuda.synchronize()

    hdl = symm_mem.rendezvous(buf, dist.group.WORLD.group_name)
    if rank == 0:
        print(f"rendezvous: world_size={hdl.world_size} rank={hdl.rank} "
              f"buffer_size={hdl.buffer_size >> 20}MB")
        base, stride = fabric_backend.flat_layout(buf)
        print(f"flat layout: base={base:#x} stride={stride >> 20}MB")
        print(f"buffer_ptrs: {[hex(p) for p in hdl.buffer_ptrs]}")
        strides = [hdl.buffer_ptrs[i + 1] - hdl.buffer_ptrs[i] for i in range(world - 1)]
        print(f"uniform stride: {len(set(strides)) <= 1}\n")

    errors = 0
    for peer in range(world):
        t = hdl.get_buffer(peer, (n,), torch.int32)
        bad = int((t != sentinel(n, peer, device)).sum().item())
        errors += bad != 0
        if rank == 0:
            print(f"  get_buffer({peer}): {'OK' if not bad else f'MISMATCH ({bad})'}")

    dist.barrier()
    if rank == 0:
        print("\nSUCCESS" if errors == 0 else f"\nFAILED ({errors})")
    dist.destroy_process_group()
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
