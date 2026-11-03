# flydsl_mega_moe: IOD-placed weights for FlyDSL MegaMoEV2 GEMM1 on MI355X

This directory applies the parent directory's SPX + NPS1 locality rule to a production MoE kernel: aiter's FlyDSL MegaMoEV2 standalone GEMM1 (A8W4: FP8 activations × MXFP4 weights, SwiGLU + MXFP8 quant epilogue). MegaMoEV2 is the closest AMD counterpart of DeepGEMM's Mega MoE: one kernel that fuses token dispatch, GEMM1, GEMM2 and combine.

Tested with aiter `6fe1bfc8f` (`aiter/ops/flydsl/kernels/mega_moe`), flydsl 0.4.1 and torch 2.10 on ROCm 7.2, in a `rocm/atom-dev` container on an 8×MI355X box, on 2026-11-03.

## What changes

`make_mega_moe_iod.py` copies `aiter/ops/flydsl/kernels/mega_moe` into a new package, `mega_moe_iod`, and patches two files. The original package is not touched.

- **`gemm_util.py`.** `BWeightLoader` gets a placed mode.
  - `begin_tile()` is called once per tile. It finds the tile's 2 MiB page and IOD bit from a page-phase table.
  - Each W1 load then addresses `page·2 MiB + (iod ^ phase)·8 KiB + (granule << 14) + …` instead of the stock linear offset. That is one table lookup per tile plus shifts per load.
- **`gemm1.py`.** `do_tile` calls `begin_tile`, and `compile_gemm1` / `gemm1_kernel` take `placed_mode` and the page-phase table.
- **No scheduling change.**
  - The stock grid-stride loop already gives XCD `x` the N-tiles `n ≡ x (mod 8)`: workgroups are dispatched round-robin over the XCDs, `n_tiles % 8 == 0`, and the grid is a multiple of 8. So W1 tile `(e, n)` is placed on IOD `(n mod 8) / 4`, the IOD that reads it.
  - Each 2 MiB page holds one expert's IOD-0 tile and IOD-1 tile, interleaved per 8 KiB granule using the probed page phase. 112 of each tile's 128 granule slots are used (12.5% padding).
- **Page phases.** `iodprobe.cpp` (built as `libiodprobe.so` and called through `ctypes` on torch tensors) probes them with the same method as `../spx_granule_map.cpp`, in 0.04 s for 512 MiB.
- **Anti-local control.** The same layout built with flipped phases, so every tile sits on the other IOD.

The three variants produce bitwise identical outputs, both the FP8 values and the E8M0 scales, for every shape.

## Results

The shape is DeepSeek-V3-like: model dim 7168, intermediate 2048, 32 local experts, so W1 is 448 MiB. Each launch processes 32-row M-tiles (1, 4 or 32 per expert) with tile_n 256. Results are 3 interleaved rounds of 3 s on HIP device 0, with round-to-round spread ≤ 0.3%. Raw outputs are in `results/`.

| M-tiles per expert | Cap | stock | local (IOD-placed W1) | anti-local |
|---|---|---|---|---|
| 1 (decode, 60 GFLOP) | 1400 W | 633 TFLOP/s, 1288 W | 623 TFLOP/s (−1.5%), **1215 W (−73 W, +4.4% TFLOP/J)** | −32.9% |
| 4 (241 GFLOP) | 1400 W | 1232 TFLOP/s, 1376 W | **1310 TFLOP/s (+6.3%)**, 1367 W | −8.1% |
| 32 (prefill, 1.92 TFLOP) | 1400 W | 1616 TFLOP/s, 1384 W | **1656 TFLOP/s (+2.5%)**, 1372 W | −4.3% |
| 1 | 1000 W | 589 TFLOP/s, sclk 1006 MHz | **605 TFLOP/s (+2.7%)**, sclk 1416 MHz | −28.5% |
| 4 | 1000 W | 1040 TFLOP/s | **1093 TFLOP/s (+5.2%)** | −4.8% |
| 32 | 1000 W | 1332 TFLOP/s | **1358 TFLOP/s (+2.0%)** | −2.4% |

## Reading the results

- **The placement reaches the hardware.** Anti-local, where every W1 read crosses the IOD–IOD link, loses up to 33% in decode: W1 streams at about 3.3 TB/s instead of about 4.8 TB/s.
- **Decode.** Local placement keeps throughput within 1.5% of stock and draws 73 W less at 1400 W. At 1000 W the freed power raises sclk by 41%, but decode is memory-bound, so throughput only rises 2.7%.
- **Heavier batches gain throughput at the same or lower sclk:** +6.3% with 4 M-tiles per expert and +2.5% with 32. The gain comes from fewer memory stalls rather than a higher clock.
- **The gains are smaller than the opus GEMM's +9–11% (`../fp8_gemm_locality`), for two reasons:**
  - FP4 weights move half as many bytes per FLOP as FP8.
  - Only W1 is local. The activations (dispatched tokens) are still read half from the other IOD, and at 32 M-tiles per expert each XCD re-reads every activation tile, so activation traffic dominates.

## Run

```sh
# inside a container with aiter (FlyDSL kernels) + flydsl 0.4.x; this directory reachable from the container
hipcc -O3 --offload-arch=gfx950 -shared -fPIC iodprobe.cpp -o libiodprobe.so
python3 make_mega_moe_iod.py /path/to/aiter/aiter/ops/flydsl/kernels
TPE=1,4,32 SECONDS=3 ROUNDS=3 python3 bench_gemm1_iod.py
# on the host, with sudo and idle GPUs: same benchmark under a 1000 W cap, cap restored on exit
CAP=1000 DEV=0 BDF=0000:75:00.0 SMI=3 ./run_capped.sh
```

## Limitations

- **Only W1 is placed.** The activations and the W1 scales still use the stock layout.
- **Implicit tile ownership.** Ownership relies on round-robin dispatch, `n_tiles % 8 == 0` and a grid that is a multiple of 8. The fused stage 1 assigns roles by arrival ticket, so it would need `XCC_ID`-based ownership.
- **Stage 2 is untouched.** GEMM2 (W2) and the fused end-to-end MegaMoEV2 on 8 GPUs are not done yet.
