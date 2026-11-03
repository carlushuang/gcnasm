# flydsl_mega_moe: IOD-local weights and activations for FlyDSL MegaMoEV2 GEMM1 on MI355X

This directory applies the parent directory's SPX + NPS1 locality rule to a production MoE kernel: aiter's FlyDSL MegaMoEV2 standalone GEMM1 (A8W4: FP8 activations × MXFP4 weights, SwiGLU + MXFP8 quant epilogue). MegaMoEV2 is the closest AMD counterpart of DeepGEMM's Mega MoE: one kernel that fuses token dispatch, GEMM1, GEMM2 and combine.

Tested with aiter `6fe1bfc8f` (`aiter/ops/flydsl/kernels/mega_moe`), flydsl 0.4.1 and torch 2.10 on ROCm 7.2, in a `rocm/atom-dev` container on an 8×MI355X box, on 2026-11-03.

## What changes

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

## Results

The shape is DeepSeek-V3-like: model dim 7168, intermediate 2048, 32 local experts, so W1 is 448 MiB. Each launch processes 32-row M-tiles (1, 4 or 32 per expert) with tile_n 256. Results are 3 interleaved rounds of 3 s on HIP device 0, with round-to-round spread ≤ 0.5%. Raw outputs are in `results/`; `v1_*` are the earlier W1-only runs.

| M-tiles per expert | Cap | stock | B local | A local | **A+B local** | A+B anti |
|---|---|---|---|---|---|---|
| 1 (decode, 60 GFLOP) | 1400 W | 634 TFLOP/s, 1300 W | −3.4%, 1201 W | +0.4%, 1294 W | −0.2%, **1230 W (−70 W)** | −36.2% |
| 4 (241 GFLOP) | 1400 W | 1225 TFLOP/s | +5.3% | +7.4% | **+15.2% (1411 TFLOP/s)** | −19.0% |
| 32 (prefill, 1.92 TFLOP) | 1400 W | 1612 TFLOP/s | +2.7% | +3.6% | **+9.5% (1766 TFLOP/s)** | −30.8% |
| 1 | 1000 W | 588 TFLOP/s | +1.2% | −1.3% | **+5.2%** | −31.3% |
| 4 | 1000 W | 1032 TFLOP/s | +5.2% | +7.2% | **+14.9%** | −6.4% |
| 32 | 1000 W | 1334 TFLOP/s | +2.0% | +3.3% | **+8.9%** | −30.7% |

## Reading the results

- **Both operands must be local.** Placing W1 alone or the activations alone gives +3–7% on the heavier shapes. Together they give **+15.2%** (4 M-tiles per expert) and **+9.5%** (32) at the default 1400 W cap, and about the same at 1000 W.
- **The gain comes from fewer memory stalls, not from clock.** A+B local runs at a *lower* sclk than stock (about 2.0 vs 2.25 GHz at 1400 W) while doing more work.
- **Decode.**
  - At 1400 W A+B local keeps throughput (−0.2%) with 70 W less.
  - At 1000 W, where power is the limit, it is 5.2% faster.
  - W1 alone draws even less power in decode (−99 W) but loses 3.4% throughput.
- **Crossing the IOD–IOD link is expensive for this kernel.** The anti-local control loses 19–36%.
- **Compared with DeepGEMM.** DeepGEMM localizes only the weights. On MI355X the activations matter as much: at 32 M-tiles per expert each XCD re-reads every activation tile.

## Run

```sh
# inside a container with aiter (FlyDSL kernels) + flydsl 0.4.x; this directory reachable from the container
hipcc -O3 --offload-arch=gfx950 -shared -fPIC iodprobe.cpp -o libiodprobe.so
python3 make_mega_moe_iod.py /path/to/aiter/aiter/ops/flydsl/kernels
TPE=1,4,32 SECONDS=3 ROUNDS=3 python3 bench_gemm1_iod.py
# on the host, with sudo and idle GPUs: same benchmark under a 1000 W cap, cap restored on exit
CAP=1000 DEV=0 BDF=0000:75:00.0 SMI=3 ./run_capped.sh
```

## Limitations and next steps

- **The activation copies are built by the benchmark.** In MegaMoEV2 the dispatch would have to write each token twice, once per IOD, into the placed layout. That doubles dispatch writes, but they go to local granules.
- **Implicit tile ownership.** Ownership relies on round-robin dispatch, `n_tiles % 8 == 0` and a grid that is a multiple of 8. The fused stage 1 assigns roles by arrival ticket, so it would need `XCC_ID`-based ownership.
- **Not yet covered.** The W1 scales and activation scales still use the stock layout. Stage 2 (GEMM2 / W2) and the fused end-to-end MegaMoEV2 on 8 GPUs are not done yet.
