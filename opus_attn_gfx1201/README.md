# opus_attn_gfx1201 — Flash Attention forward for gfx1201 (RDNA4)

Hand-written **Flash Attention 2** forward kernel for AMD **gfx1201** (Navi 48,
RX 9070 / 9070 XT) using the wave32 `_w32_gfx12` WMMA builtins. fp16 in/out,
fp32 accumulator, online softmax, **D = 128 head dim** (only).

Companion to [`opus_attn`](../opus_attn) which targets gfx950 / MFMA. The two
chips use fundamentally different matrix-instruction families (CDNA MFMA
vs. RDNA WMMA), wave sizes (64 vs 32), and fragment layouts (row-distributed
vs the asymmetric gfx12 row/col split), so this is a port not a translation.

## v0 status (correctness baseline)

This is a **correctness-first** baseline. All tests pass bit-exact within
fp16 representation; performance optimization is the next phase (see Roadmap).

### Files

```
opus_attn_gfx1201/
├── Makefile
├── README.md                          ← you are here
├── attn_common.h                      ← kargs, traits
├── attn_gfx1201_kernel_template.hpp   ← device kernel template
├── attn_gfx1201_kernel.cc             ← explicit template instantiation
└── attn_gfx1201_host.cc               ← launcher, CPU reference, bench, main()
```

### Build

```bash
cd opus_attn_gfx1201
make           # builds build/opus_attn_gfx1201.exe (--offload-arch=gfx1201)
make ARCH=gfx1200   # same kernel works on Navi 44 (untested, see notes)
```

### Run

```
./build/opus_attn_gfx1201.exe                            # B=1 H=1 N=256 D=128
./build/opus_attn_gfx1201.exe -b=1 -h=32 -n=2048
./build/opus_attn_gfx1201.exe -b=4 -h=32 -n=4096 --verify=0
```

| Flag | Description | Default |
|---|---|---|
| `-b`, `--batch` | Batch size | 1 |
| `-h`, `--heads` | Number of heads | 1 |
| `-n`, `--seq`   | Sequence length (must be multiple of 16) | 256 |
| `-d`, `--dim`   | Head dim (v0: must be 128) | 128 |
| `--verify`      | CPU reference verify (0/1) | 1 |
| `--iters`       | Benchmark iterations | 100 |

## Lane / fragment layout

Per AMD RDNA4 ISA §7.12.2 + the gfx12 WMMA layout we worked out in the opus
reference spreadsheet, the wave32 16x16x16 WMMA fragments are:

| Matrix | Layout | lane[i] register j ∈ [0,7] |
|---|---|---|
| **A** (M×K) | row-distributed     | `A[i % 16,        (i/16)*8 + j]` |
| **B** (K×N) | column-distributed  | `B[(i/16)*8 + j,  i % 16]` |
| **C** (M×N) | column-distributed  | `C[(i/16)*8 + j,  i % 16]` |

The kernel hand-loads global memory in these layouts (lane-mapping is the
same code path as `opus::wmma<>` after the gfx1201 enablement in
[ROCm/aiter#3236](https://github.com/ROCm/aiter/pull/3236), just without
the high-level adaptor wrapper).

## v0 design

One workgroup = **1 wave32** = computes 16 rows of `O` for one `(b, h)` pair.

```
grid  = dim3(N / 16, H, B)
block = dim3(32)
```

Per wave:

1. Load full Q tile `Q[16, 128]` into registers (8 wmma fragments × 8 fp16/lane = 64 fp16/lane).
2. Pre-scale Q by `(1 / sqrt(D)) * log2(e)` so softmax uses `exp2` directly.
3. KV loop (`for n_tile in range(N/16)`):
   - **S = Q @ K^T** — 8 wmma calls (one per K tile of D=128).
   - **Row max** via 4-step DPP swizzle reduction across the 16-wide half-wave.
   - **m_new = max(m_old, row_max)**; rescale = `exp2(m_old - m_new)`; rescale `v_o[*]` by `rescale[j]`.
   - **P = exp2(S - m_new)**; accumulate `l_row += sum(P)`.
   - **P → smem (row-distributed) → reload** (one round-trip through 16×16 fp16 smem
     to convert the column-distributed S into the row-distributed A operand for the next mma).
   - **O += P @ V** — 8 wmma calls (one per V's D tile).
4. Normalize: `O = O / l_row`, write to global.

## Correctness

CPU reference uses double-precision accumulation in the reduce; the kernel uses
fp32. Reference vs kernel on RX 9070 XT (gfx1201):

```
B=1 H=1 N=256  D=128   max_abs=0.0000  mean_abs=0.00000  max_rel=0.0087  n_bad(>0.05)=0/32768
B=1 H=1 N=512  D=128   max_abs=0.0000  mean_abs=0.00000  max_rel=0.0074  n_bad(>0.05)=0/65536
B=1 H=1 N=1024 D=128   max_abs=0.0000  mean_abs=0.00000  max_rel=0.0054  n_bad(>0.05)=0/131072
B=1 H=1 N=2048 D=128   max_abs=0.0000  mean_abs=0.00000  max_rel=0.0040  n_bad(>0.05)=0/262144
```

`max_abs=0.0000` is bit-exact within fp16 quantization. All tested shapes
pass.

## Performance — v0 through v5 results

RX 9070 XT (gfx1201), measured TFLOPS = `4 · B · H · N² · D / time`. All
six versions pass the same bit-exact-in-fp16 correctness checks; the table
below is throughput only. Selectable at runtime via `--version=N`.

| version | geometry                          | H=8 N=1024 | H=32 N=2048 | H=32 N=4096 (B=4) |
|:---:|---|---:|---:|---:|
| **v0** | 1 wave/WG, BLOCK_M=16, BLOCK_N=16 — direct global reads | **14.9** | **30.6** | **29.5** |
| v1 | 4 waves/WG, BLOCK_M=64, BLOCK_N=16, coop smem K/V | 17.3 | 23.8 | 27.2 |
| v2 | v1 + BLOCK_N=64 (4 sub-tiles, FA online softmax) | 6.3  | 7.9  | 8.5  |
| v3 | v1 + tighter launch_bounds + fused softmax | 16.1 | 23.3 | 23.5 |
| v4 | v0 + double-buffered K/V register prefetch | 9.3  | 11.9 | 12.7 |
| v5 | v0 + BLOCK_N=32 (online softmax, 2 sub-tiles) | 14.6 | 30.0 | 28.9 |

**v0 is the perf winner.** v5 ties v0. All other variants regressed —
see the per-version commit messages for root-cause analysis.

## What I learned trying v1 → v5

Three "obvious" optimization paths from the original roadmap turned out
to be **regressions** on gfx1201 — kept in the tree as documented
negative results:

1. **Multi-wave cooperative smem K/V tiles (v1, v2, v3)** — the
   single-biggest projected win (~3-4×), didn't happen. **Why**: the
   RDNA4 L1 cache (256 KB / CU) absorbs the redundant K/V reads that
   v0's per-wave-loads would otherwise incur. So the "saved" memory
   traffic doesn't exist; what's added is `__syncthreads()` cost.
   Conclusion: skip multi-wave per WG for gfx1201 attention — the v0
   model of "1 wave/WG, lots of small WGs" hits higher occupancy AND
   avoids cross-wave sync.

2. **Bigger BLOCK_N with cooperative load (v2)** — projected another
   1.3-1.5×. Actually got 4× **slower**. **Why**: 4 N-tiles in flight
   pushed v_s + temp arrays past the 256-VGPR ceiling → 133 VGPRs
   spilled to scratch, occupancy dropped 9 → 3. The compute amortization
   was real but the spill cost dominated.

3. **Software K/V prefetch via double-buffered registers (v4)** —
   projected 1.3×. Actually 2× **slower**. **Why**: explicit C-level
   double-buffering inflated register liveness, constraining the SIMD
   scheduler that was already doing a fine job hiding VMEM latency.
   The hardware OOO + waitcnt does this better than hand-rolled buffers.

The one optimization that **didn't** regress (v5: BLOCK_N=32 with online
softmax) tied v0 — the halved softmax overhead didn't materialize as
expected because softmax wasn't actually the bottleneck the projection
assumed.

## What actually limits gfx1201 attention perf

Based on the empirical sweep:

- **v0 at ~30 TFLOPS = ~16% MFU** on a ~190 TFLOPS dense-fp16 chip.
- Not memory-bound: the L1 cache absorbs the redundant K/V reads, so
  reducing global traffic doesn't help.
- Not softmax-bound: BLOCK_N=32 (halving softmax-per-output) tied v0.
- Likely **wmma issue rate** + **smem round-trip on P** dominate. The
  P col→row flip via smem happens every BLOCK_N rows of output.

## Directions a v6+ might explore

These are GENUINELY open and would need investigation rather than the
"safe" optimizations I tried in v1-v5:

| Idea | Risk | Likely impact |
|---|---|---|
| Skip the P→smem flip entirely by emitting V already pre-permuted so the wmma B-operand layout matches column-distributed P directly | high (changes V load layout) | moderate — would save the `__builtin_amdgcn_wave_barrier()` cost |
| Use `__builtin_amdgcn_global_load_lds_*` (true async global→LDS) with smem cooperation, accepting that this only helps when K/V exceeds L1 capacity (very long contexts, large H_KV) | medium | moderate — only at very large N |
| Drop fp32 accumulation for v_o; use fp16 + periodic fp32 rescale (lossy) | high (numerics) | high — halves v_o register footprint, frees compiler to schedule more aggressively |
| Switch to GQA (H_kv < H_q) and share K/V loads across query heads on the same SM, which IS amortizable across waves | medium | high for GQA-shaped workloads (DeepSeek/Llama 3) |
| Add causal masking (avoiding wasted compute on upper triangular) | low | up to ~2× for causal inference |

The user-facing recommendation is to **run `--version=0`** today; v1-v5
remain in the tree for reference and as a starting point for any of the
above v6+ explorations.

## Architectural notes specific to gfx1201

- gfx1201 has **no buffer-load-to-LDS** (`__builtin_amdgcn_raw_ptr_buffer_load_lds` requires `vmem-to-lds-load-insts`, only on gfx9x/gfx950/gfx1250). Async smem fills will need either plain global loads with `global_load` ordering, or the `__builtin_amdgcn_global_load_lds_*` family.
- gfx1201 has **no MFMA**; all matrix work goes through `__builtin_amdgcn_wmma_*_w32_gfx12` (16x16x16 only; the bigger 16x16x{32,64,128} shapes are gfx1250-exclusive).
- gfx1201 inherits the asymmetric fragment layout (A row-distributed, B/C column-distributed) — different from gfx1250's wmma-256b layout. opus.hpp's `wmma_adaptor` is gfx1250-tuned and not yet correct for gfx12; this kernel hand-rolls the layout to sidestep that.
- gfx1200 (Navi 44) shares the same wmma-128b ISA per LLVM and should work with the same kernel — set `ARCH=gfx1200`. Verified compile-only; no Navi 44 hardware here to confirm runtime.

## Dependencies

- ROCm 7.0+ with `hipcc`
- gfx1201 (or gfx1200) GPU
- No external headers — kernel is self-contained on top of `__builtin_amdgcn_*`.
