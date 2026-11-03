# fp8_gemm_locality: an IOD-locality-aware FP8 GEMM on MI355X (SPX + NPS1)

This is the parent directory's findings applied to a real FP8 GEMM. One full-GPU kernel keeps both its weight (B) reads and its activation (A) reads on the IOD that runs the workgroup. The approach follows DeepGEMM's locality-aware Mega MoE ([deepseek-ai/DeepGEMM#462](https://github.com/deepseek-ai/DeepGEMM/pull/462)), which reports lower power and higher clocks on B200 under heavy load.

`fp8_gemm_locality.cc` is `opus_gemm/fp8_gemm/gemm_a8w8_8wave.cc` with the same 256×256×128 tile, the same 8-wave main loop and the same fp32 C output. It makes three changes:

1. **Tile claim per IOD.** Each workgroup reads `XCC_ID >> 2` and claims a tile from the queue of its IOD. N-half `d` belongs to IOD `d`. A workgroup steals from the other queue only if its own is empty, which never happened here (dispatch is round-robin across XCCs, so both groups get the same number of workgroups).
2. **IOD-placed B.**
   - With `K = 8192`, one fp8 row is exactly one 8 KiB granule, and the NPS1 rule from the parent directory is `home = addr[13] ^ phase(2 MiB page)`.
   - So placing a row on an IOD is a row permutation. Local row `j` of a page sits in granule `2 * j + (d ^ phase[page])`.
   - A 128-row half-tile is therefore just a different base address (`page * 2 MiB + (d ^ phase) * 8 KiB`) and row stride (16 KiB). Only the two B offsets and the B row stride change in the kernel.
   - The page phases are probed at startup, which takes 0.05 s for 512 MiB.
3. **Per-IOD copy of A.** A is placed the same way, once per IOD, and each workgroup reads its own IOD's copy.

C keeps its logical layout. Every configuration is checked **bitwise** against the naive one, and naive is checked against a CPU reference on 4096 sampled outputs.

## Results

These were measured on MI355X with ROCm 7.2.0. B is 512 MiB, larger than the 256 MiB Infinity Cache, so B streams from HBM. The settings are `N = 65536`, `K = 8192`, and 3 interleaved rounds of 3 s with the start config rotated each round. Every config uses the same physical B pages (the B buffer is re-uploaded per layout), and round-to-round spread is under 0.3%. Raw outputs are in `results/`.

The configurations compared:

- **naive:** row-major A and B, one tile queue.
- **sched only:** per-IOD queues, row-major B.
- **layout only:** IOD-placed B, one queue.
- **A-local only:** per-IOD A copy, everything else naive.
- **local B:** IOD-placed B plus per-IOD queues.
- **local A+B:** local B plus the per-IOD A copy.
- **anti-local:** local B, but each IOD takes the other IOD's half.

### Power-limited: cap lowered to 1000 W and 800 W (`run_capped.sh`)

| Cap | M | naive | local B | local A+B | anti-local |
|---|---|---|---|---|---|
| 1000 W | 4096 | 1598 TFLOP/s, 955 MHz | +8.2%, 1052 MHz | **+8.8%**, 1054 MHz | −4.9% |
| 1000 W | 1024 | 1589 TFLOP/s, 975 MHz | +8.0%, 1119 MHz | **+8.9%**, 1135 MHz | −4.8% |
| 800 W | 4096 | 1123 TFLOP/s, 667 MHz | +10.1%, 736 MHz | **+10.9%**, 744 MHz | −5.0% |
| 800 W | 1024 | 1129 TFLOP/s, 678 MHz | +9.7%, 747 MHz | **+10.9%**, 755 MHz | −5.2% |

Board power sits exactly at the cap in every row, so throughput per watt improves by the same percentage as throughput.

### At the default 1400 W cap

| M | naive | sched only | layout only | A-local only | local B | local A+B | anti-local |
|---|---|---|---|---|---|---|---|
| 256 | 1775 TFLOP/s, 2097 MHz | +0.6% | −3.7% | −1.0% | −3.0%, 2356 MHz | −4.4%, 2355 MHz | −10.4% |
| 1024 | 1916 TFLOP/s, 2002 MHz | −0.3% | −2.8% | −0.4% | −3.8%, 2276 MHz | −0.7%, 2186 MHz | −11.0% |
| 4096 | 1987 TFLOP/s, 1931 MHz | −0.6% | −3.4% | −0.1% | −1.8%, 2070 MHz | **+3.8%**, 1965 MHz (+4.3% GFLOP/J) | −12.1% |

### Reading the results

- **The DeepGEMM mechanism works on MI355X.** When board power is the binding limit, sclk is the bottleneck: it falls from about 2.0 GHz at 1400 W to about 0.95 GHz at 1000 W and about 0.67 GHz at 800 W. Keeping traffic on its home IOD frees power that would otherwise move data across the IOD–IOD link. The SMU turns it into 10–16% more sclk, which gives +9–11% FP8 GEMM throughput at the same power.
- **Both operands must be local.** Placing B alone, or replicating A alone, does not help at 1400 W. A and B tiles have the same size (256 × 8 KiB), so A traffic matters as much as B traffic.
- **Anti-local is the cost of crossing.** It is 5% slower when power-limited, and 10–12% slower at 1400 W, where the IOD–IOD link (about 2 TB/s per direction) becomes the bottleneck. Naive interleaving sends half of all traffic across that link.
- **At 1400 W the picture is mixed.**
  - Only the compute-heavier shape gains (M = 4096: +3.8%).
  - The placed layout costs about 3% on its own ("layout only"), which outweighs the clock gain at M ≤ 1024.
  - The power cap is not the binding limit there: naive at M = 256 runs about 60 W under the cap at about 2.1 GHz. So the higher sclk of the local configs at 1400 W partly reflects more stall cycles, not only freed power.
  - The clean test of the mechanism is the power-limited table above.

## B-tile access microbenchmark (`tile_stride_bw.cpp`)

This reads 256 rows × 128 B per k-step with 8 lanes × 16 B per row, which is the GEMM's B load pattern, without MFMA:

| Layout | HBM, 4 GiB of distinct tiles | L2-resident (noisy) |
|---|---|---|
| row-major (8 KiB row stride) | 5.76–5.83 TB/s | 28.6–30.8 TB/s |
| IOD-placed-like (16 KiB row stride) | 5.76–5.79 TB/s | 30.8–32.7 TB/s |
| IOD-placed-like + k-chunk XOR swizzle | **6.23–6.24 TB/s** | 27.6–28.4 TB/s |
| row-major + k-chunk XOR swizzle | **6.23–6.24 TB/s** | 32.2–32.8 TB/s |

- **The layout cost is not raw bandwidth.** The IOD-placed layout streams from HBM as fast as row-major, so its ~3% cost inside the GEMM is a latency or activity effect rather than a bandwidth one.
- **Swizzling the k-chunk helps any layout.** XOR-swizzling the order of the 128 B k-chunks per row (`chunk = k ^ (row & 63)`) streams 7–8% faster from HBM. Without it, all 256 rows of a tile hit the same column offset at once and pile onto the same HBM channels. This is not applied to the GEMM yet.

## Build and run

```sh
OPUS_INCLUDE_DIR=/path/to/aiter/csrc/include sh rebuild.sh
./build/fp8_gemm_locality -m 1024 -n 65536 -s 3 -r 3 --all    # 1400 W decomposition (7 configs)
sudo -v && DEV=0 BDF=0000:75:00.0 SMI=3 ./run_capped.sh        # 1000 W / 800 W, restores the cap on exit
./build/tile_stride_bw 0
```

`run_capped.sh` needs the rocm-smi index that matches the HIP device's PCI address (`rocm-smi --showbus`). It refuses to run while any KFD process is present.

## Limitations and next steps

- **K = 8192 only.** That makes one row equal one granule. A general K needs a granule-blocked weight layout, for example 64 rows × 128 B per 8 KiB granule, and a matching B loader.
- **No swizzle yet.** The k-chunk swizzle above should be worth several percent for every config, local or not.
- **Placed-layout cost.** The ~3% at 1400 W is not explained yet. It is not HBM or L2 bandwidth (microbenchmark above), and not the per-IOD queues ("sched only" ≈ naive).
- **Scope.** All of this is a single GEMM. A MoE with per-expert weights would place each expert's N-halves the same way, and the tokens or activations would need a per-IOD copy, as A does here.
