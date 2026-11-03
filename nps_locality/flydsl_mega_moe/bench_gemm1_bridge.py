"""Bridge between the standalone GEMM1 benchmark and the fused MegaMoEV2 Stage 1 on MI355X (SPX + NPS1).

Same shapes and method as bench_gemm1_iod.py, but the kernel configuration comes from the environment
(TILE_N, NUM_WAVES, GRID_MULT, ASYNC_A, AMAJOR; defaults = the production Stage 1 configuration) and the operands use
the fused kernel's layouts: W1 in the per-IOD granule stream ("stream"), activations as two copies per 8 KiB granule
pair ("insert13"). Ownership is the stock grid-stride loop (N-tile n on XCD n % 8, IOD (n % 8) // 4).

stock         : stock layouts
B local       : W1 stream layout on the owning IOD
A local       : activation copy on the owning IOD
A+B local     : both
A 50/50       : activations read from the copies in the stock 50/50 IOD mix (paired layout, no locality)
A 50/50 + B   : W1 local, activations read from the copies in the stock 50/50 IOD mix
A+B anti      : both on the other IOD
"""
import glob
import os
import threading
import time

import numpy as np
import torch
import flydsl.expr as fx
from aiter.ops.flydsl.kernels.mega_moe_iod.gemm1 import gemm1_kernel
from aiter.ops.flydsl.kernels.mega_moe_iod.iod_place import place_rows_insert13, place_w1

DEV = int(os.environ.get("DEV", "0"))
torch.cuda.set_device(DEV)
MODEL, INTER, E = 7168, 2048, 32
TILE_N = int(os.environ.get("TILE_N", "512"))
NUM_WAVES = int(os.environ.get("NUM_WAVES", "8"))
GRID_MULT = int(os.environ.get("GRID_MULT", "1"))
ASYNC_A = os.environ.get("ASYNC_A", "1") == "1"
AMAJOR = os.environ.get("AMAJOR", "1") == "1"
SBM = int(os.environ.get("SBM", "32"))  # production Stage 1 (EP8): 32 at 64 tokens/rank, 64 at 512, 128 at 4096
N_TILES = 2 * INTER // TILE_N
assert N_TILES % 8 == 0 or N_TILES == 8
TPE_LIST = [int(v) for v in os.environ.get("TPE", "1,4,32").split(",")]
SECONDS, ROUNDS = float(os.environ.get("SECONDS", "3")), int(os.environ.get("ROUNDS", "3"))
VARIANTS = os.environ.get("VARIANTS", "stock,B local,A local,A+B local,A 50/50 + B,A+B anti").split(",")


def hwmon_dir():
    p = torch.cuda.get_device_properties(DEV)
    bdf = f"{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0"
    return glob.glob(f"/sys/bus/pci/devices/{bdf}/hwmon/hwmon*")[0]


HW = hwmon_dir()


def read_hw(name):
    try:
        with open(os.path.join(HW, name)) as f:
            return float(f.read())
    except Exception:
        return float("nan")


w_log = torch.randint(0, 256, (E * 2 * INTER * MODEL // 2,), dtype=torch.uint8, device="cuda")
scale_w = torch.randint(118, 128, (E * 2 * INTER * MODEL // 32,), dtype=torch.uint8, device="cuda")
w_loc, ph_loc, _ = place_w1(w_log, E, MODEL, INTER, TILE_N, anti=False, owner="rr")
w_anti, ph_anti, _ = place_w1(w_log, E, MODEL, INTER, TILE_N, anti=True, owner="rr")


def as_rows(buf):
    rows = buf.numel() // MODEL
    return buf[: rows * MODEL].view(torch.float8_e4m3fn).view(rows, MODEL)


def make_inputs(tpe):
    m_tiles = E * tpe
    rows = m_tiles * SBM
    x = torch.randn(rows, MODEL, device="cuda").clamp(-6, 6).to(torch.float8_e4m3fn)
    scale_x = torch.randint(118, 128, (rows, MODEL // 32), dtype=torch.uint8, device="cuda")
    trb = torch.arange(m_tiles, dtype=torch.int32, device="cuda") * SBM
    eids = torch.arange(m_tiles, dtype=torch.int32, device="cuda") // tpe
    out = torch.zeros(rows, INTER, dtype=torch.float8_e4m3fn, device="cuda")
    pcols = ((INTER // 32) + 7) // 8 * 8
    out_scale = torch.zeros(((rows + 255) // 256) * 256 * pcols + INTER + 8192, dtype=torch.uint8, device="cuda")
    x_buf, ph_a, _ = place_rows_insert13(x)
    x_ins = as_rows(x_buf)
    cfgs = {
        "stock": (w_log, None, 0, x, None, 0),
        "B local": (w_loc, ph_loc, 2, x, None, 0),
        "A local": (w_log, None, 0, x_ins, ph_a, 2),
        "A 50/50": (w_log, None, 0, x_ins, ph_a, 3),
        "A+B local": (w_loc, ph_loc, 2, x_ins, ph_a, 2),
        "A 50/50 + B": (w_loc, ph_loc, 2, x_ins, ph_a, 3),
        "A+B anti": (w_anti, ph_anti, 2, x_ins, ph_a, 4),
    }
    return rows, scale_x, trb, eids, out, out_scale, {k: cfgs[k] for k in VARIANTS}, x_buf


def launch(cfg, inp):
    rows, scale_x, trb, eids, out, out_scale, cfgs, _ = inp
    w, phase, mode, xa, phase_a, a_mode = cfgs[cfg]
    gemm1_kernel(out, xa, w, scale_x, scale_w, trb, eids, out_scale, rows,
                 fx.Stream(torch.cuda.current_stream().cuda_stream),
                 model_dim=MODEL, inter_dim=INTER, tile_n=TILE_N, num_waves=NUM_WAVES, sort_block_m=SBM,
                 grid_mult=GRID_MULT, async_a_copy=ASYNC_A, mfma_amajor=AMAJOR, a_dtype="fp8", out_dtype="fp8",
                 phase=phase, placed_mode=mode, phase_a=phase_a, a_placed_mode=a_mode)


def bench(cfg, inp, flop):
    for _ in range(3):
        launch(cfg, inp)
    torch.cuda.synchronize()
    stop, pw, fq = [False], [], []

    def sampler():
        while not stop[0]:
            pw.append(read_hw("power1_input") / 1e6)
            fq.append(read_hw("freq1_input") / 1e6)
            time.sleep(0.025)

    th = threading.Thread(target=sampler)
    th.start()
    t0, n = time.time(), 0
    while True:
        for _ in range(16):
            launch(cfg, inp)
        n += 16
        torch.cuda.synchronize()
        el = time.time() - t0
        if el >= SECONDS:
            break
    stop[0] = True
    th.join()
    k = len(pw) // 5
    return flop * n / el / 1e12, float(np.mean(pw[k:])), float(np.median(fq[k:]))


print(f"dev {DEV}: sort_block_m={SBM} tile_n={TILE_N} num_waves={NUM_WAVES} grid_mult={GRID_MULT} async_a={int(ASYNC_A)} "
      f"amajor={int(AMAJOR)}; MODEL={MODEL} INTER={INTER} E={E}; power cap {read_hw('power1_cap') / 1e6:.0f} W",
      flush=True)
for tpe in TPE_LIST:
    inp = make_inputs(tpe)
    rows = inp[0]
    flop = 2.0 * rows * (2 * INTER) * MODEL
    names = list(inp[6])
    ref, checks = None, []
    for cfg in names:
        inp[4].zero_()
        inp[5].zero_()
        launch(cfg, inp)
        torch.cuda.synchronize()
        res = (inp[4].view(torch.uint8).clone(), inp[5].clone())
        if ref is None:
            ref = res
        else:
            checks.append(f"{cfg}: {'ok' if torch.equal(res[0], ref[0]) and torch.equal(res[1], ref[1]) else 'MISMATCH'}")
    acc = {c: [] for c in names}
    for r in range(ROUNDS):
        for i in range(len(names)):
            c = names[(i + r) % len(names)]
            acc[c].append(bench(c, inp, flop))
    print(f"### {tpe} m-tile(s) per expert: {rows} rows, {flop / 1e9:.0f} GFLOP; bitwise vs stock: {'; '.join(checks)}")
    base = None
    for c in names:
        tf = [v[0] for v in acc[c]]
        m = float(np.mean(tf))
        base = base or m
        print(f"  {c:14s} {m:8.1f} TFLOP/s [{min(tf):6.0f},{max(tf):6.0f}] {np.mean([v[1] for v in acc[c]]):6.0f} W "
              f"{np.mean([v[2] for v in acc[c]]):6.0f} MHz {(m / base - 1) * 100:+6.1f}%", flush=True)
    del inp
    torch.cuda.empty_cache()
