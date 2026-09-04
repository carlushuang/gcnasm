# MXFP8 Scaled-MFMA GEMM Kernel for AMD GPU

A batched MXFP8 × MXFP8 → FP32 GEMM kernel built on the [OPUS](https://github.com/ROCm/aiter/tree/main/csrc/include/opus) template library, targeting AMD gfx950 (MI355X).

## What the kernel does

The kernel computes a batched matrix multiply `C = A · B^T` per batch, where:

- `A` is `fp8 e4m3` of shape `[batch, M, K]`
- `B` is `fp8 e4m3` of shape `[batch, N, K]`
- `C` is `fp32` of shape `[batch, M, N]`

The multiply itself is `V_MFMA_SCALE_F32_16X16X128_F8F6F4`: the scale factors are consumed by the MFMA instruction, not applied afterwards in VALU. Each MFMA reads one `e8m0` scale byte for A and one for B out of a packed dword, selected by the `op_sel` operands.

### Microscaling (MX) scheme

Scale factors follow the OCP MX convention: one `e8m0` exponent per 32 contiguous K elements.

| Scale tensor | Logical shape (per batch) | Group granularity | Meaning |
|---|---|---|---|
| `SFA` (A scale) | `[M, K/32]` | `GROUP_M × GROUP_K = 1 × 32` | one e8m0 scale per 32-element K run of a row of `A` |
| `SFB` (B scale) | `[N, K/32]` | `GROUP_N × GROUP_K = 1 × 32` | one e8m0 scale per 32-element K run of a row of `B` |

- `GROUP_M = 1`, `GROUP_N = 1`, `GROUP_K = 32`
- All scale factors are `e8m0` (`uint8`), one byte each

#### SFA and SFB share a single code path

`SFA` is `[M, K/32]` and `SFB` is `[N, K/32]` — the same shape, differing only in which of `M` or `N` names the rows. The kernel exploits this: the global→LDS scale producer is written once and parameterised by a base pointer and a stride, with wave 0 driving the `SFA` copy and wave 1 the `SFB` copy through the same instructions. Both legs must therefore use the same LDS stage stride, which a `static_assert` in the kernel enforces.

#### Scale reordering

The scale tensors are not fed to the kernel in their logical `[M, K/32]` / `[N, K/32]` layout. They are repacked once, before the launch, into a consumer-major image.

The reason is that the scale bytes a single MFMA needs are scattered in the logical layout. One `V_MFMA_SCALE_F32_16X16X128_F8F6F4` consumes `SCALE_KGROUPS_PER_MFMA = W_K / GROUP_K = 4` scale bytes per lane, and those four are 32 K-elements apart in a row — while the 16 lanes covering the MFMA's rows are `K/32` bytes apart from each other. Fetching that directly would be a strided gather in the producer and a strided `ds_read` in the consumer.

So the repack writes each `(tile, k-tile)` block out in exactly the order the consumer reads it:

| Tensor | Packed order | Tile size |
|---|---|---|
| `SFA` | `[consumer_wave_m][r][q][m_call]` | `packed_sfa_tile_elem = B_M * NUM_KGROUPS` |
| `SFB` | `[half_n][consumer_wave_n][r][q][n_call]` | `packed_sfb_tile_elem = B_N * NUM_KGROUPS` |

Here `r` is the row within the MFMA's `W_M`/`W_N`, `q` the K-group within one MFMA, and `m_call`/`n_call` the repeat index. The innermost two axes are exactly what one lane wants contiguously, and `q` sitting just outside `m_call` is what lets the consumer's `ds_read` pick up a whole dword of four scale bytes for the `op_sel` operands.

Two things fall out of this. The packed tile is a byte-for-byte image of the consumer-facing LDS tile, so the global→LDS producer is a flat linear copy — it reads `VEC_GLOBAL_SCALE = 16` bytes per lane fully coalesced and writes them straight down, with no address arithmetic that depends on the MFMA layout. And because each `(tile, k-tile)` block is self-contained and contiguous, `stride_sfa` / `stride_sfb` collapse to a single element count per tile (`packed_sfa_tile_elem` / `packed_sfb_tile_elem`), which is what keeps the producer's pointer walk to one add per stage.

The repack itself runs on the GPU (`gemm_a8w8_mxfp8_scale_repack.hpp`), where the scales already live: it is a 4×4 byte transpose, one wave per `(tile, k-tile)` block, four strided dword loads and one coalesced `dwordx4` store. On GPU 4, both tensors together cost 10.4 µs at 8192³. `pack_sfa_consumer_major` / `pack_sfb_consumer_major` in the host file are the CPU version of the same permutation, kept as the oracle `--device-repack` checks against.

### Kernel configuration

Default traits (`gemm_a8w8_mxfp8_scale_traits<>`):

| Parameter | Value |
|---|---|
| BLOCK_M × BLOCK_N × BLOCK_K | 256 × 256 × 128 |
| GROUP_M × GROUP_N × GROUP_K | 1 × 1 × 32 |
| Warps per block / block size | 8 / 512 |
| MFMA tile (W_M × W_N × W_K) | 16 × 16 × 128 |
| Warp tiling (T_M × T_N × T_K) | 4 × 2 × 1 |
| Output tiles per workgroup | 4 or 1, chosen at launch |
| A/B global-load vector (fp8) | 16 elems |
| C global-store vector (fp32) | 4 elems |
| Scale global-load vector (e8m0) | 16 bytes |

## Files

```
mxfp8_gemm_16x16x128_scale/
├── Makefile
├── rebuild.sh
├── bench.sh
├── gemm_a8w8_mxfp8_scale_common.h            # kargs + traits
├── gemm_a8w8_mxfp8_scale_kernel_template.hpp # kernel body
├── gemm_a8w8_mxfp8_scale_repack.hpp          # device-side scale repack (--device-repack)
├── gemm_a8w8_mxfp8_scale_kernel.cc           # device TU: the two instantiations
└── gemm_a8w8_mxfp8_scale_host.cc             # host launcher / benchmark / CPU reference
```

The kernel body is laid out as `// Prologue` → `// Main Loop` → `// Epilogue`. The main loop deliberately stops one K tile short (`tile + 1 < loops`); that last tile is consumed in the epilogue from LDS with no further global loads, which is what makes room for the output handoff.

## Prerequisites

- ROCm with hipcc (for the HIP runtime and the binutils used by `make check`)
- gfx950 GPU target (e.g. MI355X)
- OPUS headers from [aiter](https://github.com/ROCm/aiter): set `OPUS_INCLUDE_DIR` to `<aiter_root>/csrc/include`
- OpenMP (for CPU reference and random init)
- **clang 23 or newer.** clang 20/21/22 refuse the `#pragma unroll 4` on the main K loop (they report `loop not unrolled` and emit 64 MFMA); clang 23 unrolls it to 192 MFMA, which is worth roughly **+13%**. Set `CLANG23_ROOT` to such a build, or override `HIPCC=/opt/rocm/bin/hipcc` and accept the slower code — `make check` will fail in that case, by design.

## Build

```bash
cd opus_gemm/mxfp8_gemm_16x16x128_scale
export OPUS_INCLUDE_DIR=/path/to/aiter/csrc/include
make -j
```

Or use the convenience script (clean build + gate + a default 8192³ run):

```bash
./rebuild.sh
```

### Build gates

Two properties of the generated code have regressed silently in the past, so they are checked mechanically against the **linked executable** (never the `.o` — the linker is what decides the code that actually runs):

```bash
make check   # >=192 v_mfma_scale_f32_16x16x128, and 0 s_setprio
             # (reports 384: 192 per instantiation, two of them)
make regs    # VGPR / SGPR / spill / LDS
```

## Run

```bash
./build/gemm_a8w8_mxfp8_scale.exe                                    # defaults: b=8, m=256, n=512, k=256
./build/gemm_a8w8_mxfp8_scale.exe -m 8192 -n 8192 -k 8192 -b 1       # timing
./build/gemm_a8w8_mxfp8_scale.exe -m 8192 -n 8192 -k 8192 -b 1 -v 1  # validate vs CPU
```

Or through make, which runs `check` first:

```bash
make benchmark            # 8192^3, -w 200 -i 100
make verify               # 8192^3, -v 1
make benchmark GPU=2      # pin to one card
make benchmark SHAPE="-m 4096 -n 4096 -k 4096 -b 1"
```

### Command-line options

| Flag | Description | Default |
|---|---|---|
| `-b`, `--b` | Batch size | 8 |
| `-m`, `--m` | M dimension | 256 |
| `-n`, `--n` | N dimension | 512 |
| `-k`, `--k` | K dimension | 256 |
| `-v`, `--verify` | CPU reference verification (0=off, 1=on) | 0 |
| `-w`, `--warmup` | Warmup iterations | 200 |
| `-i`, `--iterations` | Timed iterations | 100 |
| `--timeline` | Per-kernel timeline of consecutive launches (requires `-v 0`) | off |
| `--device-repack` | Repack the scales on the GPU and check the result against the host packer | off |

All flags accept both `-m 4096` and `-m=4096` syntax. `M / N / K` must be multiples of `BLOCK_M / BLOCK_N / BLOCK_K`, and `K` a multiple of `GROUP_K = 32`.

## Benchmarking

```bash
./bench.sh              # 5 runs on the default card
```

## Performance

Measured on MI355X, `batch = 1`, square problem size `M = N = K`, `-w 200 -i 100`:

| M=N=K | Tiles per WG | Grid | Avg Time (ms) | TFlops |
|---:|:--:|---|---:|---:|
| 1024  | 1 | (16,   1, 1) | 0.0173 |  124.46 |
| 2048  | 1 | (64,   1, 1) | 0.0258 |  667.15 |
| 4096  | 1 | (256,  1, 1) | 0.0563 | 2440.74 |
| 8192  | 4 | (256,  1, 1) | 0.3663 | 3004.94 |
| 16384 | 4 | (1024, 1, 1) | 3.1919 | 2755.78 |

