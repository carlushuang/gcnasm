"""End-to-end MegaMoEV2 (aiter FlyDSL, a8w4, intra-node EP8) on MI355X: stock vs IOD-local variants.

torchrun --nproc_per_node 8 e2e_mega_moe.py   (env: TOKENS=64,512,4096 VARIANTS=stock,s2q,s2q+s2w,s2q+s2x)
A variant is a "+"-joined set of mega_moe_iod flags (s2q: Stage2 per-IOD queue, s2w/s2x: W2 on the owning/other IOD,
s1q: Stage1 per-IOD queue, s1w/s1x: W1 on the owning/other IOD, s1a: per-IOD copies of the dispatched tokens).
All variants of one token count are built side by side and timed in interleaved rounds (ROUNDS x ITERS
barrier-aligned single forwards, then SUSTAIN seconds of back-to-back forwards for power). Outputs are compared
bitwise with stock on every rank on two alternating inputs, before and after timing (a variant that silently skipped
work would return the previous input's result).
"""
import glob
import os
import threading
import time

import numpy as np
import torch
import torch.distributed as dist
import mori.shmem as ms

MODEL, INTER, EXPERTS, TOPK = 7168, 2048, 256, 8
TOKENS = [int(v) for v in os.environ.get("TOKENS", "64,512,4096").split(",")]
VARIANTS = os.environ.get("VARIANTS", "stock,s2q,s2q+s2w,s2q+s2x").split(",")
ROUNDS = int(os.environ.get("ROUNDS", "5"))
ITERS = int(os.environ.get("ITERS", "20"))
SUSTAIN = float(os.environ.get("SUSTAIN", "1.0"))
SEED = int(os.environ.get("SEED", "7"))
assert VARIANTS[0] == "stock"


def setup():
    rank, world, local = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl", rank=rank, world_size=world, device_id=dev)
    import torch._C._distributed_c10d as c10d

    c10d._register_process_group("default", dist.group.WORLD)
    ms.shmem_torch_process_group_init("default")
    return rank, world, dev


def hwmon_dir(dev):
    p = torch.cuda.get_device_properties(dev)
    bdf = f"{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0"
    return glob.glob(f"/sys/bus/pci/devices/{bdf}/hwmon/hwmon*")[0]


def read_hw(hw, name):
    try:
        with open(os.path.join(hw, name)) as f:
            return float(f.read())
    except Exception:
        return float("nan")


class PhaseTimer:
    """CUDA events around the Stage1 and Stage2 kernel launches of one MegaMoEV2 instance."""

    def __init__(self, moe):
        self.on, self.s1, self.s2 = False, [], []
        for name, sink in (("_run_fused_stage1", self.s1), ("_fused_stage2_call", self.s2)):
            orig = getattr(moe, name)

            def wrapped(*a, _orig=orig, _sink=sink, **k):
                if not self.on:
                    return _orig(*a, **k)
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                r = _orig(*a, **k)
                e1.record()
                _sink.append((e0, e1))
                return r

            setattr(moe, name, wrapped)

    def collect(self):
        torch.cuda.synchronize()
        out = ([a.elapsed_time(b) for a, b in self.s1], [a.elapsed_time(b) for a, b in self.s2])
        self.s1.clear()
        self.s2.clear()
        return out


def make_moe(variant, rank, world, mtpr, weights):
    w1, w1s, w2, w2s = weights
    if variant == "stock":
        from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2

        extra = {}
    else:
        from aiter.ops.flydsl.kernels.mega_moe_iod import MegaMoEV2

        extra = dict(iod_mode=variant)
    return MegaMoEV2(rank=rank, world_size=world, model_dim=MODEL, inter_dim=INTER, experts=EXPERTS, topk=TOPK,
                     quant="a8w4", w1=w1, w1_scale=w1s, w2=w2, w2_scale=w2s, max_tok_per_rank=mtpr, **extra)


def main():
    rank, world, dev = setup()
    epr = EXPERTS // world
    hw = hwmon_dir(dev)
    g = torch.Generator(device=dev).manual_seed(SEED + 1000 * rank)
    w1 = torch.randint(0, 256, (epr * 2 * INTER * MODEL // 2,), dtype=torch.uint8, device=dev, generator=g)
    w1s = torch.randint(118, 128, (epr * 2 * INTER * MODEL // 32,), dtype=torch.uint8, device=dev, generator=g)
    w2 = torch.randint(0, 256, (epr * MODEL * INTER // 2,), dtype=torch.uint8, device=dev, generator=g)
    w2s = torch.randint(118, 128, (epr * MODEL * INTER // 32,), dtype=torch.uint8, device=dev, generator=g)
    weights = (w1, w1s, w2, w2s)

    def gather(vals):
        t = torch.tensor(vals, dtype=torch.float64, device=dev)
        parts = [torch.zeros_like(t) for _ in range(world)]
        dist.all_gather(parts, t)
        return torch.stack(parts).cpu().numpy()

    for tokens in TOKENS:
        mtpr = 1 << (tokens - 1).bit_length()
        inputs = []
        for _ in range(2):
            x = (torch.randn(tokens, MODEL, device=dev, generator=g) * 0.5).to(torch.bfloat16)
            ids = torch.rand(tokens, EXPERTS, device=dev, generator=g).topk(TOPK, dim=1).indices.to(torch.int32)
            wts = torch.softmax(torch.randn(tokens, TOPK, device=dev, generator=g), dim=1).contiguous()
            inputs.append((x, wts, ids.contiguous()))

        def run_pattern(moe):
            outs, nonfinite = [], 0
            for i in (0, 1, 0, 1):
                torch.cuda.synchronize()
                ms.shmem_barrier_all()
                out = moe.forward(*inputs[i])
                torch.cuda.synchronize()
                ms.shmem_barrier_all()
                outs.append(out[:tokens].contiguous().view(torch.uint8).clone())
                nonfinite += int((~torch.isfinite(out[:tokens].float())).sum().item())
            return outs, nonfinite

        moes, timers, checks, refs = {}, {}, {}, None
        for variant in VARIANTS:
            if rank == 0:
                print(f"[tokens/rank={tokens}] building {variant}", flush=True)
            t0 = time.time()
            moe = make_moe(variant, rank, world, mtpr, weights)
            outs, nonfinite = run_pattern(moe)
            if refs is None:
                refs = outs
                assert not torch.equal(outs[0], outs[1]), "the two inputs must give different outputs"
            checks[variant] = [all(torch.equal(o, r) for o, r in zip(outs, refs)), nonfinite]
            moes[variant], timers[variant] = moe, PhaseTimer(moe)
            if rank == 0:
                print(f"  built + checked in {time.time() - t0:.1f} s; {getattr(moe, 'iod_info', {})}", flush=True)
            if variant == "stock" and rank == 0:
                cfg = moe._select_config(tokens)
                print(f"  stage1 {cfg.stage1}\n  stage2 {cfg.stage2}", flush=True)

        x, wts, ids = inputs[0]
        stats = {v: dict(single=[], s1=[], s2=[], b2b=[], pw=[], fq=[]) for v in VARIANTS}
        for rnd in range(ROUNDS):
            order = VARIANTS if rnd % 2 == 0 else VARIANTS[::-1]
            for variant in order:
                moe, tm, st = moes[variant], timers[variant], stats[variant]
                for _ in range(3):
                    moe.forward(x, wts, ids)
                torch.cuda.synchronize()
                lat = []
                tm.on = True
                for _ in range(ITERS):
                    dist.barrier()
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                    moe.forward(x, wts, ids)
                    e1.record()
                    torch.cuda.synchronize()
                    lat.append(e0.elapsed_time(e1))
                tm.on = False
                s1, s2 = tm.collect()
                stop, pw, fq = [False], [], []

                def sampler():
                    while not stop[0]:
                        pw.append(read_hw(hw, "power1_input") / 1e6)
                        fq.append(read_hw(hw, "freq1_input") / 1e6)
                        time.sleep(0.02)

                # every rank must issue the same number of (collective) forwards: size the burst from the slowest rank
                lat_max = gather([np.median(lat)])[:, 0].max()
                n = 8 * max(1, int(SUSTAIN * 1e3 / (8 * lat_max)))
                dist.barrier()
                th = threading.Thread(target=sampler)
                th.start()
                t0 = time.time()
                for i in range(n):
                    moe.forward(x, wts, ids)
                    if i % 8 == 7:
                        torch.cuda.synchronize()
                torch.cuda.synchronize()
                b2b = (time.time() - t0) * 1e3 / n
                stop[0] = True
                th.join()
                a = gather([np.median(lat), np.median(s1), np.median(s2), b2b, np.mean(pw[len(pw) // 4:]),
                            np.median(fq)])
                st["single"].append(a[:, 0].max())
                st["s1"].append(a[:, 1].max())
                st["s2"].append(a[:, 2].max())
                st["b2b"].append(a[:, 3].mean())
                st["pw"].append(a[:, 4].mean())
                st["fq"].append(a[:, 5].mean())

        for variant in VARIANTS:
            outs2, nonfinite2 = run_pattern(moes[variant])
            checks[variant][0] = checks[variant][0] and all(torch.equal(o, r) for o, r in zip(outs2, refs))
            checks[variant][1] += nonfinite2
            ctr = getattr(moes[variant], "_iod_ctr", None)
            checks[variant].append(0 if ctr is None else int(ctr.abs().sum().item()))
        chk = {v: gather(checks[v]) for v in VARIANTS}
        if rank == 0:
            base = {k: float(np.median(stats["stock"][k])) for k in ("single", "s1", "s2", "b2b")}
            print(f"\n### tokens/rank={tokens} (mtpr={mtpr}), {ROUNDS} interleaved rounds x {ITERS} forwards; "
                  f"ms = median over rounds of the max-rank median; spread = [min,max] over rounds", flush=True)
            print(f"{'variant':16s} {'forward ms':>11s} {'spread':>15s} {'vs stock':>8s} {'stage1 ms':>9s} "
                  f"{'stage2 ms':>9s} {'s2 vs stock':>11s} {'b2b ms':>7s} {'W/GPU':>6s} {'sclk':>5s}  checks")
            for v in VARIANTS:
                st = stats[v]
                single, s1, s2 = (float(np.median(st[k])) for k in ("single", "s1", "s2"))
                c = chk[v]
                print(f"{v:16s} {single:11.3f} [{min(st['single']):6.3f},{max(st['single']):6.3f}] "
                      f"{100 * (single / base['single'] - 1):+7.1f}% {s1:9.3f} {s2:9.3f} "
                      f"{100 * (s2 / base['s2'] - 1):+10.1f}% {np.median(st['b2b']):7.3f} "
                      f"{np.median(st['pw']):6.0f} {np.median(st['fq']):5.0f}  "
                      f"bitwise {int(c[:, 0].sum())}/{world}, non-finite {int(c[:, 1].sum())}, "
                      f"dirty counters {int((c[:, 2] != 0).sum())}", flush=True)
        for v in list(moes):
            del moes[v]
        torch.cuda.synchronize()
        ms.shmem_barrier_all()
    dist.barrier()
    try:
        ms.shmem_finalize()
    except Exception:
        pass
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
