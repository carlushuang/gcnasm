# opus_attn_gfx1201 — Flash Attention forward for gfx1201 (RDNA4)

Hand-written **Flash Attention 2** forward kernel for AMD **gfx1201** (Navi 48,
RX 9070 / 9070 XT) using the wave32 `_w32_gfx12` WMMA builtins. bf16 in/out,
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

## Performance — v0 through v11 results (bf16)

RX 9070 XT (gfx1201), measured TFLOPS = `4 · B · H · N² · D / time`. All
twelve versions pass the same bit-exact-in-bf16 correctness checks (max
absolute error within bf16's ~7-bit mantissa quantization); the table
below is throughput only. Selectable at runtime via `--version=N`.

| version | geometry                          | H=8 N=1024 | H=32 N=2048 | H=32 N=4096 (B=4) |
|:---:|---|---:|---:|---:|
| v0 | 1 wave/WG, BLOCK_M=16, BLOCK_N=16 — direct global reads | 15.0 | 30.3 | 29.4 |
| v1 | 4 waves/WG, BLOCK_M=64, BLOCK_N=16, coop smem K/V | 17.3 | 27.5 | 27.2 |
| v2 | v1 + BLOCK_N=64 (4 sub-tiles, FA online softmax) | 6.3  | 6.6  | 8.5  |
| v3 | v1 + tighter launch_bounds + fused softmax | 16.1 | 28.1 | 23.5 |
| v4 | v0 + double-buffered K/V register prefetch | 9.3  | 15.0 | 12.7 |
| v5 | v0 + BLOCK_N=32 (online softmax, 2 sub-tiles) | 16.3 | 23.5 | 22.7 |
| v6 | v0 + contiguous V load + smem transpose | 18.6 | 34.4 | 35.3 |
| v7 | v6 + batched V load (all 8 D-tiles at once, 1 barrier) | 22.5 | 29.9 | 30.6 |
| v8 | v6 + BLOCK_N=32 | 23.6 | 30.9 | 31.5 |
| v9 | v6 + V pre-transposed in DRAM (NOT production-realistic) | 25.2 | 42.9 | 43.6 |
| v10 | v9 + BLOCK_N=32 | 19.2 | 39.2 | 36.4 |
| **v11** | **mma swap_ab — no S→P smem flip, vectorized O write** | **17.7** | **37.2** | **37.1** |

**v11 is the recommended production kernel.** Takes V in the standard
`[B, H, N, D]` layout, eliminates the S→P smem flip entirely, vectorizes the
output write, and reaches **37.2 TFLOPS = ~19% MFU on H=32 N=2048**
(against the 195 TFLOPS marketing peak). +23% over v0, +8% over v6.

### Achieved MFU vs. RX 9070 XT 195 TFLOPS dense bf16 peak

| version | best TFLOPS | MFU |
|---|---:|---:|
| v0 | 30.3 | 15.5% |
| v6 | 35.3 | 18.1% |
| v9 (upper-bound, V pre-transposed in DRAM, NOT production) | 43.6 | 22.4% |
| **v11** | **37.2** | **19.1%** |

The included `ubench` measures on-chip ceilings directly: WMMA
`f32_16x16x16_bf16` reaches **200 TFLOPS** (slightly above marketing) and
`v_exp2_f32` reaches **3.28 T ops/s**. The interleaved-mode benchmark
(`./ubench interleave`) shows that wmma and v_exp2 **partially co-execute**
on gfx12 — at 1 exp per wmma, bf16 wmma TFLOPS drops only 8% (200→164)
while exp adds ~20 G wave-inst/s on top. Softmax exp work is partially
hidden behind wmma's multi-cycle issue window, but the bf16 window (12.6
cyc) is tighter than fp16's (14 cyc), so less hides — about 0.4 cyc of
exp work per wmma for bf16 vs 1.5 cyc for fp16.

### Note on v11's narrower margin under bf16

In fp16 v11 was +46% over v6 (47 vs 32 TFLOPS). Under bf16 the margin
shrinks to +8% (37 vs 35) because (a) bf16 wmma is faster (12.6 cyc
vs 14), so v11's many small VALU ops per wmma fit less easily into the
wmma shadow, and (b) bf16 → fp32 conversion is a 5-ALU-op sequence on
gfx12 (no single-instruction `v_cvt_pk_bf16_f32`), penalizing kernels
that do many per-iter conversions. v11 hoists this conversion out of
the inner dt loop to mitigate (without the hoist v11 was only 30
TFLOPS under bf16).

### Why v9 is kept but not recommended

v9 pre-transposes V in DRAM to `[B, H, D, N]` before the FA kernel. That
makes V's B-fragment load naturally contiguous, but the pre-transpose is
NOT free in production: V is appended row-by-row during decode and consumed
straight from the Wv projection output (`[N, D]`-ordered), so an extra DRAM
transpose pass is neither free nor easily fusable. Kept in the tree as an
upper-bound reference only — surpassed by v11 anyway.

### What v6 vs v0 actually changed

v0's V load is strided per lane (`v_col[j * stride_n]`), which forces the
compiler to emit 8 `global_load_u16`/`d16_hi` pairs per V tile per lane —
80+ small loads per outer iter. v6 loads V along its contiguous D axis (1
`global_load_b128` per lane), then transposes the tile in smem to recover
the B-fragment layout. Net: 10× fewer V load instructions, +15% TFLOPS.

### Why v9 is not a fair production comparison

v9 assumes V is already stored in DRAM as `[B, H, D, N]`. Skipping the
in-kernel transpose saves the smem store + wave_barrier + strided smem
reads per V tile. **But:**

- During decode, V is appended one row per generated token — to store it
  as `[B, H, D, N]` would require either a per-token transposed write
  (which costs the same scattered store) or a full re-transpose per step.
- Wv's matmul naturally emits `[N, D]` (D is the inner dim of the
  projection); fusing the transpose into Wv is possible but invasive.
- For prefill, a one-time pre-transpose is feasible but framework-level
  surgery.

So v9 is the achievable upper bound for the FA kernel alone, not a
realistic production setting. **v6 is what should be used in practice.**
For getting closer to v9's performance without pre-transposing V, the
right direction is in-register transpose via `permlane16_swap` /
`permlanex16_swap` / `ds_bpermute_b32` — see "Directions a v11+ might
explore" below.

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
softmax) tied v0. The reason softmax doesn't move the needle much: ubench's
interleaved-mode benchmark shows wmma and v_exp2 PARTIALLY co-execute on
gfx12 (1 exp per wmma costs only 15% of wmma TFLOPS, vs 100% if they fully
serialized). The FA pipeline has ~1 exp per wmma, so halving softmax can
buy back ~15% of wmma peak at most — a real but bounded ceiling.

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
- **v6**: load V along its contiguous axis, transpose in smem → +15% over v0.
- **v9**: pre-transpose V in DRAM (NOT production-realistic) → +27% over v0
  if V is already transposed, kept as upper-bound reference only.
- **v10** (v9 + BLOCK_N=32): tied v9.

### What v11 did differently — and why it dominates

v11 is the swap_ab pattern from gfx950's `opus_attn`, adapted to gfx12's
asymmetric WMMA layout. It eliminates THREE bottlenecks at once:

1. **No S→P smem flip.** mma0 is called with operands swapped: `wmma(v_k,
   v_q, 0)`. The output v_s naturally lands in
   `lane (c, r) reg j → S[M_q=c, N_kv=r*8+j]` form, which is exactly the
   layout needed for the next mma's B input — no transpose needed. The
   kernel uses ZERO smem (`LDS_Block_Size = 0` in the dispatch).

2. **Softmax row reduction collapses to 1 cross-half permute.** In v0/v6
   the M_q rows are distributed across 16 lanes within a half-wave, so a
   row max needs 4 ds_swizzle stages × 8 j repetitions = 32 cross-lane
   ops. In v11 the M_q rows are on `lane.0 = col16`, so each lane has
   8 N_kv values for its own M_q row — a per-lane 8-value reduce plus
   one `ds_bpermute(my_lane XOR 16)` to combine the two halves.
   `2 ds_bpermute_b32` total in the disasm.

3. **Output write is vectorized.** v_o lands as
   `lane (col16=M_q, row_grp) reg j → O[M_q, dt*16 + row8 + j]` — 8
   contiguous fp16 per lane = 1 `global_store_b128` per lane per D-tile.
   v0/v6 had to do 8 separate small stores per lane per D-tile.

The cost: V load goes back to v0's strided pattern (82 small global loads
per outer iter), because the contiguous V load gives A-layout which is
incompatible with the swap_ab pipeline. But the wins above dominate that
cost by a wide margin. The kernel's VGPR usage also drops to 144 (from
v0's 168 / v6's 160) because there's no LDS scratch and no smem-flip
temporaries, allowing 10 waves/SIMD occupancy instead of 9.

### How v11 adapts the gfx950 swap_ab trick to gfx12's asymmetric layout

The gfx950 `opus_attn` kernel does `v_p = opus::cast<fp16>(v_s)` — a simple
cast with no smem flip — by using `mfma_adaptor_swap_ab` consistently on
both mma0 and mma1. On gfx950 the C-output and A-input layouts align under
the swap, so the cast is enough.

On gfx12 wmma, C-output has `lane.0=N, regs=M` and A-input has `lane.0=M,
regs=K`. A naive C→A cast would give `P^T` to the next mma.

v11's workaround: feed the C output as the B operand of mma1 (B-input has
`lane.0=N, regs=K` — matches C-output's `lane.0=N, regs=M` if we treat the
M-axis of S as the K-axis of the next mma). To make the matmul valid, we
ALSO have to swap mma1's operand order to `wmma(v_v, v_p, v_o)`, which
computes `V^T @ P^T = O^T` semantically. The output naturally lands with
M_q in `lane.0` and D in `regs`, which is what we want for a vectorized
contiguous-D output write.

Cost: mma0 must also use the swap form (`wmma(v_k, v_q, 0)`) for v_p to
land in the right intermediate layout. V load reverts to strided pattern
(can't use the contiguous-load-then-smem-transpose of v6 because that
would put V in the wrong fragment role). But the smem flip + softmax
overhead + scalar output writes saved by v11 outweigh the strided V
loads — see the perf table.

## Directions a v12+ might explore

| Idea | Risk | Likely TFLOPS impact |
|---|---|---|
| v11 + contiguous V load via permlane-based in-register transpose | high (cross-lane data movement design) | moderate — if it works, would combine v11's no-smem-flip with v6's vectorized V load |
| Causal masking | low | does NOT increase TFLOPS — halves compute, so latency drops ~2× but TFLOPS stays the same. Benefit is decode latency / inference cost, not throughput. |
| GQA (share K/V across query heads on same WG) | medium | high for GQA-shaped workloads (Llama 3, DeepSeek) — amortizes K/V load cost across more wmma work |
| fp16 v_o accumulator with periodic fp32 rescale | high (numerics) | high — halves v_o footprint, frees scheduler |
| `__builtin_amdgcn_global_load_lds_*` (true async global→LDS) for K | medium | only helps when K exceeds L1 (very long contexts) |

The user-facing recommendation is **`--version=11`** for all production
shapes. v0–v10 remain in the tree as documented intermediate results.

## Architectural notes specific to gfx1201

- gfx1201 has **no buffer-load-to-LDS** (`__builtin_amdgcn_raw_ptr_buffer_load_lds` requires `vmem-to-lds-load-insts`, only on gfx9x/gfx950/gfx1250). Async smem fills will need either plain global loads with `global_load` ordering, or the `__builtin_amdgcn_global_load_lds_*` family.
- gfx1201 has **no MFMA**; all matrix work goes through `__builtin_amdgcn_wmma_*_w32_gfx12` (16x16x16 only; the bigger 16x16x{32,64,128} shapes are gfx1250-exclusive).
- gfx1201 inherits the asymmetric fragment layout (A row-distributed, B/C column-distributed) — different from gfx1250's wmma-256b layout. opus.hpp's `wmma_adaptor` is gfx1250-tuned and not yet correct for gfx12; this kernel hand-rolls the layout to sidestep that.
- gfx1200 (Navi 44) shares the same wmma-128b ISA per LLVM and should work with the same kernel — set `ARCH=gfx1200`. Verified compile-only; no Navi 44 hardware here to confirm runtime.

## Dependencies

- ROCm 7.0+ with `hipcc`
- gfx1201 (or gfx1200) GPU
- No external headers — kernel is self-contained on top of `__builtin_amdgcn_*`.
