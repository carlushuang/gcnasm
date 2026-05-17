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
├── attn_gfx1201_kernel_v{0..10}_template.hpp   ← device kernel templates
├── attn_gfx1201_kernel_v{0..10}.cc             ← explicit template instantiations
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

## Performance — v0 through v10 results

RX 9070 XT (gfx1201), measured TFLOPS = `4 · B · H · N² · D / time`. All
eleven versions pass the same bit-exact-in-fp16 correctness checks; the table
below is throughput only. Selectable at runtime via `--version=N`.

| version | geometry                          | H=8 N=1024 | H=32 N=2048 | H=32 N=4096 (B=4) |
|:---:|---|---:|---:|---:|
| v0 | 1 wave/WG, BLOCK_M=16, BLOCK_N=16 — direct global reads | 14.9 | 30.3 | 29.2 |
| v1 | 4 waves/WG, BLOCK_M=64, BLOCK_N=16, coop smem K/V | 17.3 | 23.8 | 27.2 |
| v2 | v1 + BLOCK_N=64 (4 sub-tiles, FA online softmax) | 6.3  | 7.9  | 8.5  |
| v3 | v1 + tighter launch_bounds + fused softmax | 16.1 | 23.3 | 23.5 |
| v4 | v0 + double-buffered K/V register prefetch | 9.3  | 11.9 | 12.7 |
| v5 | v0 + BLOCK_N=32 (online softmax, 2 sub-tiles) | 14.6 | 30.0 | 28.9 |
| **v6** | v0 + **contiguous V load + smem transpose** | **19.9** | 32.3 | 34.3 |
| v7 | v6 + batched V load (all 8 D-tiles at once, 1 barrier) | 22.5 | 31.1 | 30.6 |
| v8 | v6 + BLOCK_N=32 | 19.1 | 34.9 | 33.2 |
| **v9** | v6 + **V pre-transposed in DRAM** (no in-kernel V flip) | 18.9 | **36.3** | **37.0** |
| v10 | v9 + BLOCK_N=32 | 19.2 | 36.3 | 36.4 |

**v9 wins** at the production-sized shapes (≥H=32, N≥2048): **37.0 TFLOPS = ~19.5% MFU**
on the 9070 XT's ~190 TFLOPS dense fp16 peak. **v6 wins at small shapes** because it
doesn't require a pre-transpose pass.

### What v6 vs v0 actually changed

v0's V load is strided per lane (`v_col[j * stride_n]`), which forces the
compiler to emit 8 `global_load_u16`/`d16_hi` pairs per V tile per lane —
80+ small loads per outer iter. v6 loads V along its contiguous D axis (1
`global_load_b128` per lane), then transposes the tile in smem to recover
the B-fragment layout. Net: 10× fewer V load instructions, +15% TFLOPS.

### What v9 vs v6 actually changed

v9 assumes V is already stored in DRAM as `[B, H, D, N]` (transposed from
the standard `[B, H, N, D]`). The B-fragment access pattern is then
naturally contiguous along N — no in-kernel transpose needed at all. Saves
the smem store + wave_barrier + strided smem reads per V tile vs v6. Net:
+8–14% TFLOPS over v6. The host runs a one-time transpose kernel before
the benchmark loop; in production this is amortizable since the V projection
output can be emitted directly in the transposed layout.

This matches CK's `MakeShuffledVRegBlockDescriptor` pattern in spirit — CK
shuffles in registers after a row-major DRAM load; v9 takes the simpler
path of doing the transpose once in DRAM and skipping the per-tile shuffle.

## What I learned trying v1 → v10

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

The one v1-v5 optimization that **didn't** regress (v5: BLOCK_N=32 with online
softmax) tied v0. That projection ("halve softmax overhead → faster") was
wrong because softmax was not the bottleneck — turned out the **V load
pattern** was.

### What I learned from v6-v10 (after disasm-driven profiling)

Disassembling v0 with `llvm-objdump` revealed the actual issue:

```
v0 V load: 42× global_load_u16  +  40× global_load_d16_hi_b16   (per outer iter)
v0 K load:  7× global_load_b128 +   9× global_load_b96          (per outer iter)
```

K is loaded contiguously along D (vectorizes to `b128`/`b96`), but V is read
strided by `stride_n = D = 128 fp16` per lane to construct the wmma
B-fragment layout — the compiler is forced to emit 16-bit-granularity loads.
Per outer iter: 80+ V loads vs 16 K loads. **V load was the bandwidth
bottleneck the whole time**, hidden under "we're wmma-bound" intuition.

Fixes:
- **v6**: load V along its contiguous axis, transpose in smem → **+15%** over v0.
- **v9**: pre-transpose V in DRAM so the B-layout load is naturally contiguous,
  drop the smem flip entirely → **+27%** over v0 (~19.5% MFU).
- **v10** (v9 + BLOCK_N=32): tied v9 — softmax still not the bottleneck.

## What still limits gfx1201 attention perf at ~19.5% MFU

- v9 at 37 TFLOPS / 190 TFLOPS dense = ~19.5% MFU.
- The S → P transpose via smem still happens once per outer iter (1 store +
  barrier + 8 strided reads). That accounts for ~10 small ds-ops per iter.
- The fp32 v_o accumulator pins 64 VGPRs/lane (32 dwords); the compiler has
  to keep them live across the entire V tile loop.
- wmma issue rate: at 16 wmmas/outer iter × 128 outer × 16 cycles/wmma =
  32K cycles of pure wmma per WG; we're running at ~2× that. The "other 50%"
  is global load latency + smem transpose + softmax dependency chain.

## Directions a v11+ might explore

| Idea | Risk | Likely impact |
|---|---|---|
| Causal masking (skip upper triangular) | low | up to ~2× for causal inference |
| GQA (share K/V across query heads) | medium | high for GQA-shaped workloads (Llama 3, DeepSeek) |
| fp16 v_o accumulator with periodic fp32 rescale | high (numerics) | high — halves v_o footprint, frees scheduler |
| Drop the S→P smem flip by computing the wmma in transposed form (compute O^T = V^T @ P^T directly, write transposed) | high (layout rework) | moderate — saves the last smem barrier |
| `__builtin_amdgcn_global_load_lds_*` (true async global→LDS) for K | medium | only helps when K exceeds L1 (very long contexts) |

The user-facing recommendation is **`--version=9`** for production (with V
pre-transposed) or **`--version=6`** if pre-transposing V is impractical.
v0–v5 remain in the tree as documented negative-result references.

## Architectural notes specific to gfx1201

- gfx1201 has **no buffer-load-to-LDS** (`__builtin_amdgcn_raw_ptr_buffer_load_lds` requires `vmem-to-lds-load-insts`, only on gfx9x/gfx950/gfx1250). Async smem fills will need either plain global loads with `global_load` ordering, or the `__builtin_amdgcn_global_load_lds_*` family.
- gfx1201 has **no MFMA**; all matrix work goes through `__builtin_amdgcn_wmma_*_w32_gfx12` (16x16x16 only; the bigger 16x16x{32,64,128} shapes are gfx1250-exclusive).
- gfx1201 inherits the asymmetric fragment layout (A row-distributed, B/C column-distributed) — different from gfx1250's wmma-256b layout. opus.hpp's `wmma_adaptor` is gfx1250-tuned and not yet correct for gfx12; this kernel hand-rolls the layout to sidestep that.
- gfx1200 (Navi 44) shares the same wmma-128b ISA per LLVM and should work with the same kernel — set `ARCH=gfx1200`. Verified compile-only; no Navi 44 hardware here to confirm runtime.

## Dependencies

- ROCm 7.0+ with `hipcc`
- gfx1201 (or gfx1200) GPU
- No external headers — kernel is self-contained on top of `__builtin_amdgcn_*`.
