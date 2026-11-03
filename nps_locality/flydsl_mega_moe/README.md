# flydsl_mega_moe: IOD locality for FlyDSL MegaMoEV2 on MI355X, from GEMM1 to the 8-GPU pipeline

This directory applies the parent directory's SPX + NPS1 locality rule to a production MoE kernel, aiter's FlyDSL MegaMoEV2 (A8W4: FP8 activations × MXFP4 weights). MegaMoEV2 is the closest AMD counterpart of DeepGEMM's Mega MoE: it fuses token dispatch, GEMM1 + SwiGLU, GEMM2 and combine.

- **Part 1** places the operands of the standalone GEMM1.
- **Part 2** carries the placement through the fused pipeline on 8 GPUs (Stage 1 with dispatch, Stage 2, per-IOD token copies) and measures it end to end.

Tested with aiter `6fe1bfc8f` (`aiter/ops/flydsl/kernels/mega_moe`), flydsl 0.4.1 and torch 2.10 on ROCm 7.2, in a `rocm/atom-dev` container on an 8×MI355X box, on 2026-11-03.

## Part 1: standalone GEMM1

### What changes

`make_mega_moe_iod.py` copies `aiter/ops/flydsl/kernels/mega_moe` into a new package, `mega_moe_iod`, and patches `gemm_util.py` and `gemm1.py`. The original package is not touched.

- **No scheduling change.** The stock grid-stride loop already gives XCD `x` the N-tiles `n ≡ x (mod 8)`: workgroups are dispatched round-robin over the XCDs, `n_tiles % 8 == 0`, and the grid is a multiple of 8. So the block computing tile `(m, n)` runs on IOD `(n mod 8) / 4`, and both operands are placed for that IOD.
- **IOD-placed W1 (`BWeightLoader`).**
  - Each 2 MiB page holds one expert's IOD-0 and IOD-1 W1 tiles, interleaved per 8 KiB granule using the probed page phase. 112 of each tile's 128 slots are used.
  - `begin_tile()` does one page-phase lookup per tile. Each load then addresses `page·2 MiB + (iod ^ phase)·8 KiB + (granule << 14) + …`.
- **IOD-local activations (`ATileLoader`).**
  - There is one copy of the activations per IOD.
  - A 32-row × 7168 B M-tile is exactly 28 granules, so each page holds 4 tiles per IOD.
  - `for_tile()` points the tile's buffer descriptor at `page·2 MiB + slot·512 KiB + (iod ^ phase)·8 KiB`. Every in-tile offset gets the IOD bit inserted at position 13 (`o + (o & ~8191)`).
  - The block reads the copy on its own IOD.
- **Page phases.** `iodprobe.cpp` (built as `libiodprobe.so` and called through `ctypes` on torch tensors) probes them with the same method as `../spx_granule_map.cpp`, in 0.04 s for 512 MiB.
- **Kernel arguments.** `compile_gemm1` / `gemm1_kernel` take `placed_mode` + `phase` (W1) and `a_placed_mode` + `phase_a` (activations).
- **Anti-local control.** The same layouts built with flipped phases.

All five variants produce bitwise identical outputs, both the FP8 values and the E8M0 scales, for every shape.

### Results

The shape is DeepSeek-V3-like: model dim 7168, intermediate 2048, 32 local experts, so W1 is 448 MiB. Each launch processes 32-row M-tiles (1, 4 or 32 per expert) with tile_n 256. Results are 3 interleaved rounds of 3 s on HIP device 0, with round-to-round spread ≤ 0.5%. Raw outputs are in `results/`; `v1_*` are the earlier W1-only runs.

| M-tiles per expert | Cap | stock | B local | A local | **A+B local** | A+B anti |
|---|---|---|---|---|---|---|
| 1 (decode, 60 GFLOP) | 1400 W | 634 TFLOP/s, 1300 W | −3.4%, 1201 W | +0.4%, 1294 W | −0.2%, **1230 W (−70 W)** | −36.2% |
| 4 (241 GFLOP) | 1400 W | 1225 TFLOP/s | +5.3% | +7.4% | **+15.2% (1411 TFLOP/s)** | −19.0% |
| 32 (prefill, 1.92 TFLOP) | 1400 W | 1612 TFLOP/s | +2.7% | +3.6% | **+9.5% (1766 TFLOP/s)** | −30.8% |
| 1 | 1000 W | 588 TFLOP/s | +1.2% | −1.3% | **+5.2%** | −31.3% |
| 4 | 1000 W | 1032 TFLOP/s | +5.2% | +7.2% | **+14.9%** | −6.4% |
| 32 | 1000 W | 1334 TFLOP/s | +2.0% | +3.3% | **+8.9%** | −30.7% |

### Reading the results

- **Both operands must be local.** Placing W1 alone or the activations alone gives +3–7% on the heavier shapes. Together they give **+15.2%** (4 M-tiles per expert) and **+9.5%** (32) at the default 1400 W cap, and about the same at 1000 W.
- **The gain comes from fewer memory stalls, not from clock.** A+B local runs at a *lower* sclk than stock (about 2.0 vs 2.25 GHz at 1400 W) while doing more work.
- **Decode.**
  - At 1400 W A+B local keeps throughput (−0.2%) with 70 W less.
  - At 1000 W, where power is the limit, it is 5.2% faster.
  - W1 alone draws even less power in decode (−99 W) but loses 3.4% throughput.
- **Crossing the IOD–IOD link is expensive for this kernel.** The anti-local control loses 19–36%.
- **Compared with DeepGEMM.** DeepGEMM localizes only the weights. On MI355X the activations matter as much: at 32 M-tiles per expert each XCD re-reads every activation tile.

## Part 2: the fused MegaMoEV2 end to end on 8 GPUs

The same placement rule, applied to the production pipeline: fused Stage 1 (dispatch + GEMM1 + SwiGLU), Stage 2 (GEMM2 + weighted P2P scatter) and combine, A8W4 with intra-node EP8 on 8×MI355X (mori shmem). The v3 and v4 sections of `make_mega_moe_iod.py` patch the same `mega_moe_iod` copy, and `MegaMoEV2(..., iod_mode=...)` takes a "+"-joined set of flags:

| Flag | Patched | What it does |
|---|---|---|
| `s1q` | `mega_moe_stage1.py` | XCC_ID-based ownership in the fused Stage 1. Each CTA reads `s_getreg_b32 hwreg(20)` once and claims tiles from the work-head shard of its IOD (`XCC_ID >> 2`); IOD g takes N-tiles [4g, 4g + 4) of every M slot. Works for both the sharded and the ready-order schedule, and the prepare kernel already zeroes the shards. |
| `s1w` / `s1x` | `gemm_util.py` (`BWeightLoader`) | W1 in a per-IOD granule stream. A 512-row tile is 224 granules and can straddle 3 pages, whose phases are loaded once per tile. `s1x` puts every tile on the other IOD (control). |
| `s1a` | `dispatch.py`, `gemm_util.py` (`ATileLoader`), `mega_moe_v2.py` | Per-IOD token copies. The receive buffer becomes a symmetric buffer of twice the size. The producer CTAs on all 8 source GPUs write every 16 B of a row to both granules of its pair, `o + (o & ~8191)` and `+ 8192`, so they need no phase tables. GEMM1 reads the copy homed on its IOD, `+ ((iod ^ phase[o >> 20]) << 13)`, on both the register and the LDS-DMA A paths. Controls: `s1z` reads the same copies in the stock 50/50 IOD mix, `s1y` reads the other IOD's copy. |
| `s2q` | `mega_moe_stage2.py` | Replaces the four stock Stage 2 schedules with one persistent per-IOD queue (4 blocks per CU; `s2gN` sets N). IOD g owns N-blocks [g·nnb/2, (g + 1)·nnb/2) of every M-block. The next ticket is claimed before the current unit runs, and the last block to retire zeroes the counters. |
| `s2w` / `s2x` | `gemm2.py` | W2 placed per 16-row group, so the layout does not depend on BN (128 or 256): group r of an expert goes to IOD r // (N / 32). `s2x` is the other-IOD control. |

`e2e_mega_moe.py` builds all variants side by side on 8 ranks: DeepSeek-V3 shape (256 experts, top-8, model dim 7168, intermediate 2048), random routing, weights and activations.

- **Correctness checks.** Every rank's output is compared bitwise with stock on two alternating inputs, before and after timing. A variant that silently skipped work would return the previous input's output. The queue counters must also be back to zero.
- **Timing.** All variants run in 5 interleaved rounds.
  - Forward time is a barrier-aligned single forward on the slowest rank, median over rounds; round-to-round spread is ≤ 1–2%.
  - Stage 1 and Stage 2 times come from CUDA events around the two kernel launches.
- **Power caps.** `run_e2e.sh` can cap all 8 GPUs and restores the caps on exit.

All 63 combinations of variant, batch size and cap are bitwise identical to stock on all 8 ranks. Raw outputs are `results/e2e_nominal.txt`, `results/e2e_cap900.txt` and `results/e2e_s2_grid_sweep.txt`.

### Results

Kernel times in µs (slowest rank, median of 5 rounds; lower is better). "Queue" is the scheduling-only control: the per-IOD queue with the stock layouts. For scale, stock Stage 1 at 4096 tokens per rank runs at about 1.7 PFLOP/s (1.92 TFLOP per rank) and Stage 2 at about 0.9 PFLOP/s. At the default 1400 W cap the pipeline draws 740 W (64 tokens per rank), 990 W (512) and 1150 W (4096); the 900 W cap binds at 512 and 4096.

Stage 1 (dispatch + GEMM1):

| Tokens/rank | Cap | stock | queue `s1q` | W1 local `+s1w` | W1 other IOD `+s1x` | W1 local + token copies `+s1w+s1a` | copies read 50/50 `+s1w+s1z` | copies read other IOD `+s1w+s1y` |
|---|---|---|---|---|---|---|---|---|
| 64 | 1400 W | 199 | 187 | 184 | 223 | 183 | | |
| 512 | 1400 W | 302 | 314 | 294 | 361 | 336 | | |
| 4096 | 1400 W | 1135 | 1140 | 1139 | 1211 | 1363 | | |
| 64 | 900 W | 190 | 185 | 184 | 218 | 186 | 185 | 187 |
| 512 | 900 W | 306 | 302 | 307 | 369 | 343 | 337 | 351 |
| 4096 | 900 W | 1338 | 1343 | 1340 | 1394 | 1472 | 1445 | 1490 |

Stage 2 (GEMM2 + P2P scatter):

| Tokens/rank | Cap | stock | queue `s2q` | W2 local `+s2w` | W2 other IOD `+s2x` |
|---|---|---|---|---|---|
| 64 | 1400 W | 61 | 75 | 74 | 99 |
| 512 | 1400 W | 199 | 215 | 208 | 329 |
| 4096 | 1400 W | 1076 | 1102 | 1152 | 1225 |
| 64 | 900 W | 62 | 78 | 76 | 99 |
| 512 | 900 W | 198 | 214 | 208 | 328 |
| 4096 | 900 W | 1125 | 1135 | 1205 | 1278 |

Whole forward (dispatch, both stages and combine), relative to stock:

| Tokens/rank | Cap | stock | `s1q+s1w` | `s1q+s1w+s1a` | `s2q+s2w` | `s1q+s1x` | `s2q+s2x` |
|---|---|---|---|---|---|---|---|
| 64 | 1400 W | 0.380 ms | **−3.1%** | −3.7% | +1.6% | +6.5% | +7.3% |
| 512 | 1400 W | 0.641 ms | **−1.1%** | +4.9% | +3.9% | +9.4% | +21.4% |
| 4096 | 1400 W | 2.439 ms | +0.2% | +9.5% | +3.4% | +3.6% | +7.4% |
| 64 | 900 W | 0.376 ms | **−2.0%** | −2.2% | +3.1% | +7.3% | +10.0% |
| 512 | 900 W | 0.645 ms | −0.5% | +5.4% | +1.9% | +9.4% | +20.0% |
| 4096 | 900 W | 2.715 ms | −0.2% | +4.5% | +1.3% | +2.2% | +4.2% |

### Reading the end-to-end results

- **XCC_ID-based ownership is free in Stage 1.** The stock Stage 1 already claims tiles from atomic work queues. Making the queue per IOD (`s1q`) costs between −6% and +4% against stock, which is within the variation of the stock schedule.
- **In the fused pipeline, the stock 50/50 granule interleave is already close to local.**
  - At the default cap, W1 placement makes Stage 1 1.6% (64 tokens per rank) and 6.4% (512) faster than the queue-only control, and changes nothing at 4096. Under the 900 W cap it stays within ±2%.
  - The other-IOD controls cost +4 to +22% in Stage 1 and +11 to +53% in Stage 2 against the queue-only controls, so the traffic is IOD-sensitive. The stock layout simply already sends only half of it across.
- **Per-IOD token copies are not worth their cost.**
  - Writing every dispatched row twice costs more than the local reads save, and the cost grows with the batch: Stage 1 is −0.5%, +14% and +20% against W1-local alone at 64, 512 and 4096 tokens per rank.
  - The controls separate the two effects at 4096 tokens under 900 W. The double write alone (`s1z`) costs +8%, and local reads of the copies (`s1a`) are not faster than 50/50 reads (+2%).
- **Neither W2 placement nor the per-IOD queue pays off in Stage 2.**
  - The queue replaces four tuned static schedules with a dynamic one, at +23%, +8% and +2% of Stage 2 time (64, 512, 4096 tokens per rank).
  - On top of the queue, local W2 is 1–3% faster at 64 and 512 tokens, but 4.5% (1400 W) and 6% (900 W) slower at 4096, even though it lowers power (sclk 1993 vs 1951 MHz under 900 W). This is the same placed-layout cost the opus GEMM showed at 1400 W.
- **Why the standalone GEMM1 gains (+9.5–15%) do not carry over is not isolated yet.** The fused kernel differs in several ways:
  - Tile shape: tile_n 512 with 8 waves.
  - A reaches LDS by DMA, and GEMM1 overlaps the dispatch traffic.
  - W1 is read through the stream layout, whose tiles straddle pages.

  At the default cap the pipeline is not power-limited, so the mechanism measured before, lower fabric power turning into higher sclk, has nothing to act on. Under the 900 W cap it is power-limited, but W1-local still only breaks even.

## Run

```sh
# inside a container with aiter (FlyDSL kernels) + flydsl 0.4.x; this directory reachable from the container
hipcc -O3 --offload-arch=gfx950 -shared -fPIC iodprobe.cpp -o libiodprobe.so
python3 make_mega_moe_iod.py /path/to/aiter/aiter/ops/flydsl/kernels
TPE=1,4,32 SECONDS=3 ROUNDS=3 python3 bench_gemm1_iod.py
# on the host, with sudo and idle GPUs: same benchmark under a 1000 W cap, cap restored on exit
CAP=1000 DEV=0 BDF=0000:75:00.0 SMI=3 ./run_capped.sh

# Part 2, on the host with all 8 GPUs idle: end to end, every variant side by side (IOD_PROBE_LIB = libiodprobe.so path)
VARIANTS=stock,s1q,s1q+s1w,s1q+s1x,s1q+s1w+s1a,s2q+s2g4,s2q+s2g4+s2w,s2q+s2g4+s2x TOKENS=64,512,4096 ./run_e2e.sh
# the same with all 8 GPUs capped at 900 W, caps restored on exit
CAP=900 VARIANTS=stock,s1q+s1w,s1q+s1w+s1a,s1q+s1w+s1z,s1q+s1w+s1y TOKENS=64,512,4096 ./run_e2e.sh
```

## Limitations and next steps

- **The standalone and fused results disagree, and the gap is not explained yet.** A natural next step is to run the fused layouts (W1 stream, A insert13) and tile shape (tile_n 512, 8 waves, LDS-DMA A) in the standalone benchmark. That would separate the layout and kernel-shape effects from the dispatch overlap and power regime of the pipeline.
- **The Stage 2 per-IOD queue is costly at small batches** (+23% of Stage 2 time at 64 tokens per rank). A cheaper schedule would hand out units without a per-unit atomic, for example by binding the static stock schedules to `XCC_ID`. That would make the W2 measurement independent of the queue overhead.
- **Scales use the stock layout.** The W1, W2 and activation scales were not placed.
- **Limited coverage.** Only 64, 512 and 4096 tokens per rank with uniform random routing, the DeepSeek-V3 shape, and caps of 1400 W and 900 W were measured.
