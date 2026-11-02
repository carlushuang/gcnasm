#!/usr/bin/env python3
"""Minimal fabric_mem usage: allocate, rendezvous, read peers, run a kernel over the span.

    torchrun --nnodes=1 --nproc_per_node=4 demo_fabric_mem.py

Cross-node, with no launcher in common, swap the exchange (see --tcp):

    node A> python3 demo_fabric_mem.py --tcp --rank 0 --world 2
    node B> python3 demo_fabric_mem.py --tcp --rank 1 --world 2 --host <A>
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fabric_mem as fsm  # noqa: E402
import hip_fabric as hf  # noqa: E402

MIB = 1 << 20


def sentinel(n, rank, device):
    return torch.arange(n, device=device, dtype=torch.int32) + rank * 100000


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mib", type=int, default=64, help="buffer per rank")
    p.add_argument("--tcp", action="store_true", help="exchange over TCP instead of torch.distributed")
    p.add_argument("--rank", type=int, default=None)
    p.add_argument("--world", type=int, default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=55600)
    p.add_argument("--gpu", type=int, default=None, help="local device (TCP mode)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.tcp:
        rank, world = args.rank, args.world
        dev = args.gpu if args.gpu is not None else rank
        exchange = fsm.tcp(rank, world, host=args.host, port=args.port)
        group_name = None
    else:
        import torch.distributed as dist

        dist.init_process_group("gloo")
        rank, world = dist.get_rank(), dist.get_world_size()
        dev = int(os.environ.get("LOCAL_RANK", "0"))
        exchange = None
        group_name = dist.group.WORLD.group_name

    torch.cuda.set_device(dev)
    device = torch.device("cuda", dev)
    n = args.mib * MIB // 4

    # 1. allocate -- a fabric-capable buffer that is still an ordinary torch tensor
    buf = fsm.empty(n, dtype=torch.int32, device=device)
    buf[:] = sentinel(n, rank, device)
    torch.cuda.synchronize()

    # 2. rendezvous -- export, exchange, import, map every rank into one flat span
    hdl = fsm.rendezvous(buf, group_name, exchange=exchange)

    if rank == 0:
        print(f"world={hdl.world_size} buffer={args.mib}MB/rank")
        print(f"flat span base={hdl.flat_base:#x} stride={hdl.stride >> 20}MB")

    # 3. read a peer with torch -- the comparison is the transfer
    errors = 0
    for peer in range(hdl.world_size):
        t = hdl.get_buffer(peer, (n,), torch.int32)
        bad = int((t != sentinel(n, peer, device)).sum().item())
        errors += bad != 0
        if rank == 0:
            print(f"  torch read rank {peer}: {'OK' if not bad else f'MISMATCH ({bad})'}")

    # 4. read every peer from one kernel, addressing by base + r*stride
    d = hdl.device_desc()
    gathered = torch.empty(hdl.world_size * n, dtype=torch.int32, device=device)
    hf.gather_flat(
        dev, gathered.data_ptr(), d.base, d.stride, 0, d.world,
        n * 4, torch.cuda.get_device_properties(dev).multi_processor_count,
    )
    expected = torch.cat([sentinel(n, p, device) for p in range(hdl.world_size)])
    bad = int((gathered != expected).sum().item())
    errors += bad != 0
    if rank == 0:
        print(f"  kernel gather via base+r*stride: {'OK' if not bad else f'MISMATCH ({bad})'}")
        print("SUCCESS" if errors == 0 else f"FAILED ({errors})")

    hdl.close()
    if not args.tcp:
        import torch.distributed as dist

        dist.destroy_process_group()
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
