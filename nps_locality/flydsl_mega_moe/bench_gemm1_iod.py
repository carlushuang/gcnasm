"""MegaMoEV2 standalone GEMM1 (aiter FlyDSL, a8w4) with stock vs IOD-placed W1 on MI355X (SPX + NPS1).

stock      : W1 in the stock shuffle_weight_w4 layout
local      : W1 tile (e, n) placed in the 8 KiB granules homed on the IOD of the XCD that owns N-tile n
anti-local : same layout, but placed on the other IOD (control)
The stock grid-stride loop gives XCD x the N-tiles n % 8 == x (round-robin dispatch), so no scheduling change.
"""
import ctypes
import glob
import os
import threading
import time

import numpy as np
import torch
import flydsl.expr as fx
from aiter.ops.flydsl.kernels.mega_moe_iod.gemm1 import gemm1_kernel

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = int(os.environ.get("DEV", "0"))
torch.cuda.set_device(DEV)
MODEL, INTER, E = (int(os.environ.get(k, v)) for k, v in (("MODEL", 7168), ("INTER", 2048), ("E", 32)))
TILE_N, SBM, GRID_MULT = 256, 32, 4
N_TILES = 2 * INTER // TILE_N
assert N_TILES % 8 == 0 and MODEL % 1024 == 0
TPE_LIST = [int(v) for v in os.environ.get("TPE", "1,4,32").split(",")]  # 32-row m-tiles per expert
SECONDS, ROUNDS = float(os.environ.get("SECONDS", "3")), int(os.environ.get("ROUNDS", "3"))

lib = ctypes.CDLL(os.path.join(HERE, "libiodprobe.so"))
lib.iod_page_phases.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]


def page_phases(buf):
    torch.cuda.synchronize()
    out = np.zeros(buf.numel() >> 21, dtype=np.uint8)
    weak = ctypes.c_int(0)
    rc = lib.iod_page_phases(ctypes.c_void_p(buf.data_ptr()), ctypes.c_size_t(buf.numel()), out.ctypes.data,
                             ctypes.byref(weak))
    assert rc == 0, f"probe failed rc={rc}"
    return out, weak.value


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


# ---- weights: logical W1 bytes (shuffle_weight_w4 layout of E x 2*INTER x MODEL fp4), E8M0 scales
G_BYTES = 8 * MODEL                 # one 16-row group
GPG = MODEL // 1024                 # 8 KiB granules per group
GPT = (TILE_N // 16) * GPG          # granules per 256-row tile (112 for MODEL = 7168)
assert GPT <= 128
NPAGES = E * N_TILES // 2           # one 2 MiB page per (expert, N-tile pair): one tile per IOD
w_log = torch.randint(0, 256, (E * 2 * INTER * MODEL // 2,), dtype=torch.uint8, device="cuda")
scale_w = torch.randint(118, 128, (E * 2 * INTER * MODEL // 32,), dtype=torch.uint8, device="cuda")
w_loc = torch.zeros(NPAGES << 21, dtype=torch.uint8, device="cuda")
w_anti = torch.zeros(NPAGES << 21, dtype=torch.uint8, device="cuda")
t0 = time.time()
ph_loc, weak_loc = page_phases(w_loc)
ph_anti_raw, weak_anti = page_phases(w_anti)
ph_anti = ph_anti_raw ^ 1  # place every tile on the other IOD
print(f"W1 {w_log.numel() / 2**20:.0f} MiB logical, placed {NPAGES} x 2 MiB pages; phases probed in "
      f"{time.time() - t0:.2f} s (weak pages: {weak_loc}, {weak_anti})", flush=True)


def build_placed(dst, ph):
    n = torch.arange(N_TILES)
    d = (n % 8) // 4
    local = (n // 8) * 4 + (n % 4)
    page = torch.arange(E)[:, None] * (N_TILES // 2) + local[None, :]
    phase = torch.from_numpy(ph.astype(np.int64))[page]
    j = torch.arange(GPT)
    slot = page[:, :, None] * 256 + 2 * j[None, None, :] + (d[None, :, None] ^ phase[:, :, None])
    dst.view(NPAGES * 256, 8192).index_copy_(0, slot.reshape(-1).cuda(), w_log.view(E * N_TILES * GPT, 8192))


build_placed(w_loc, ph_loc)
build_placed(w_anti, ph_anti)
CFGS = {
    "stock": (w_log, None, 0),
    "local (IOD-placed W1)": (w_loc, torch.from_numpy(ph_loc.astype(np.int32)).cuda(), 1),
    "anti-local": (w_anti, torch.from_numpy(ph_anti.astype(np.int32)).cuda(), 1),
}


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
    return rows, x, scale_x, trb, eids, out, out_scale


def launch(cfg, inp):
    rows, x, scale_x, trb, eids, out, out_scale = inp
    w, phase, mode = CFGS[cfg]
    gemm1_kernel(out, x, w, scale_x, scale_w, trb, eids, out_scale, rows,
                 fx.Stream(torch.cuda.current_stream().cuda_stream),
                 model_dim=MODEL, inter_dim=INTER, tile_n=TILE_N, sort_block_m=SBM, grid_mult=GRID_MULT,
                 a_dtype="fp8", out_dtype="fp8", phase=phase, placed_mode=mode)


def bench(cfg, inp, flop):
    for _ in range(3):
        launch(cfg, inp)
    torch.cuda.synchronize()
    stop, pw, fq, tj = False, [], [], []

    def sampler():
        while not stop:
            pw.append(read_hw("power1_input") / 1e6)
            fq.append(read_hw("freq1_input") / 1e6)
            tj.append(read_hw("temp2_input") / 1e3)
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
    stop = True
    th.join()
    k = len(pw) // 5
    return flop * n / el / 1e12, float(np.mean(pw[k:])), float(np.median(fq[k:])), float(np.mean(tj[k:]))


print(f"dev {DEV}: MODEL={MODEL} INTER={INTER} E={E} tile_n={TILE_N} sort_block_m={SBM}; power cap "
      f"{read_hw('power1_cap') / 1e6:.0f} W", flush=True)
for tpe in TPE_LIST:
    inp = make_inputs(tpe)
    rows = inp[0]
    flop = 2.0 * rows * (2 * INTER) * MODEL
    ref = None
    for cfg in CFGS:
        inp[5].zero_()
        inp[6].zero_()
        launch(cfg, inp)
        torch.cuda.synchronize()
        res = (inp[5].view(torch.uint8).clone(), inp[6].clone())
        if ref is None:
            ref = res
        else:
            same = torch.equal(res[0], ref[0]) and torch.equal(res[1], ref[1])
            print(f"  [tpe={tpe}] {cfg} vs stock: {'bitwise identical' if same else 'MISMATCH'}", flush=True)
    names = list(CFGS)
    acc = {c: [] for c in names}
    for r in range(ROUNDS):
        for i in range(len(names)):
            c = names[(i + r) % len(names)]
            acc[c].append(bench(c, inp, flop))
    print(f"\n### {tpe} m-tile(s) per expert: {rows} rows, {flop / 1e9:.0f} GFLOP per launch")
    print(f"{'config':26s} {'TFLOP/s':>9s} {'[min,max]':>16s} {'power W':>8s} {'sclk MHz':>9s} {'Tj C':>6s} {'vs stock':>9s}")
    base = None
    for c in names:
        tf = [v[0] for v in acc[c]]
        m = float(np.mean(tf))
        base = base or m
        print(f"{c:26s} {m:9.1f} [{min(tf):6.0f},{max(tf):6.0f}] {np.mean([v[1] for v in acc[c]]):8.0f} "
              f"{np.mean([v[2] for v in acc[c]]):9.0f} {np.mean([v[3] for v in acc[c]]):6.1f} {(m / base - 1) * 100:+8.1f}%",
              flush=True)
