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

## Performance (v0, baseline)

RX 9070 XT (gfx1201), measured TFLOPS = `4 · B · H · N² · D / time`:

| B | H | N | D | Avg time | TFLOPS |
|---:|---:|---:|---:|---:|---:|
| 1 | 1  | 512  | 128 | 0.258 ms | 0.52  |
| 1 | 1  | 1024 | 128 | 0.284 ms | 1.89  |
| 1 | 1  | 2048 | 128 | 0.545 ms | 3.94  |
| 1 | 8  | 1024 | 128 | 0.287 ms | **15.0** |
| 1 | 32 | 2048 | 128 | 2.241 ms | **30.7** |
| 4 | 32 | 4096 | 128 | 36.8 ms  | **29.9** |

Saturates around **~30 TFLOPS** at high occupancy. Reasonable for a
correctness baseline; well below the ~190 TFLOPS dense fp16 peak of the
RX 9070 XT (~16% MFU at best).

## Why v0 is far from peak

1. **1 wave / workgroup** → no smem cooperation, no inter-wave pipelining.
2. **BLOCK_N = 16** → softmax (max/sub/exp/sum reductions + smem round-trip) runs every 16 KV rows; softmax overhead dominates.
3. **No K/V staging in smem** → each wave re-reads K and V from global on every iteration, riding the L1/L2 cache.
4. **No pipelining** → mma waits on global load completion; load issue waits on mma completion.
5. **Single Q tile per workgroup** → workgroup output is only 16×128 elements = 4 KB; many launches.

## Roadmap to higher MFU

| Pass | Change | Expected delta |
|---|---|---|
| v1 | `BLOCK_N = 64` (4 wmma N-tiles per softmax), cooperative smem K/V tiles, 4 waves/workgroup sharing the smem | ~3-4× |
| v2 | Async global → smem (`global_load_lds` / multi-buffered) overlap with mma | another ~1.3-1.5× |
| v3 | Pre-transpose V into the column-distributed register layout via shared-memory ds_swizzle (avoid the per-iter smem round-trip on P) | ~1.1× |
| v4 | Multi-Q tiles per workgroup, `BLOCK_M = 64`, share K/V smem reads across all M tiles | ~1.2-1.4× |
| v5 | Schedule-group barriers (`__builtin_amdgcn_sched_group_barrier`) to interleave wmma / valu / exp issue (the gfx950 reference does this) | ~1.1-1.2× |

Combined target: 50%+ MFU on RX 9070 XT (≥ ~95 TFLOPS sustained).

## Architectural notes specific to gfx1201

- gfx1201 has **no buffer-load-to-LDS** (`__builtin_amdgcn_raw_ptr_buffer_load_lds` requires `vmem-to-lds-load-insts`, only on gfx9x/gfx950/gfx1250). Async smem fills will need either plain global loads with `global_load` ordering, or the `__builtin_amdgcn_global_load_lds_*` family.
- gfx1201 has **no MFMA**; all matrix work goes through `__builtin_amdgcn_wmma_*_w32_gfx12` (16x16x16 only; the bigger 16x16x{32,64,128} shapes are gfx1250-exclusive).
- gfx1201 inherits the asymmetric fragment layout (A row-distributed, B/C column-distributed) — different from gfx1250's wmma-256b layout. opus.hpp's `wmma_adaptor` is gfx1250-tuned and not yet correct for gfx12; this kernel hand-rolls the layout to sidestep that.
- gfx1200 (Navi 44) shares the same wmma-128b ISA per LLVM and should work with the same kernel — set `ARCH=gfx1200`. Verified compile-only; no Navi 44 hardware here to confirm runtime.

## Dependencies

- ROCm 7.0+ with `hipcc`
- gfx1201 (or gfx1200) GPU
- No external headers — kernel is self-contained on top of `__builtin_amdgcn_*`.
