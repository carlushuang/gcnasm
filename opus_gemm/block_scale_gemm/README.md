# FP8 Block-Scale GEMM Kernel for AMD GPU

A batched FP8 × FP8 → FP32 block-scale GEMM kernel built on the [OPUS](https://github.com/ROCm/aiter/tree/main/csrc/include/opus) template library, targeting AMD gfx950 (MI355X).

## What the kernel does

The kernel computes a batched matrix multiply `C = A · B^T` per batch, where:

- `A` is `fp8 e4m3` of shape `[batch, M, K]`
- `B` is `fp8 e4m3` of shape `[batch, N, K]`
- `C` is `fp32` of shape `[batch, M, N]`

The K-reduction is accumulated in `fp32` on MFMA, and every K-group's partial sum is multiplied by the product of the corresponding `SFA` / `SFB` fp32 scale factors before being added to the final accumulator.

### Block-scale scheme

Scale factors are shared across fixed-size groups along `(M, N, K)`:

| Scale tensor | Shape (per batch) | Group granularity | Meaning |
|---|---|---|---|
| `SFA` (A scale) | `[num_groups_k, num_groups_m]` | `GROUP_M × GROUP_K = 1 × 128` | one fp32 scale per `1 × 128` tile of `A` |
| `SFB` (B scale) | `[num_groups_n, num_groups_k]` | `GROUP_N × GROUP_K = 128 × 128` | one fp32 scale per `128 × 128` tile of `B` |

- `GROUP_M = 1`, `GROUP_N = 128`, `GROUP_K = 128`
- `num_groups_m = M / GROUP_M`, `num_groups_n = N / GROUP_N`, `num_groups_k = K / GROUP_K`
- All scale factors are stored as `fp32`

#### `SFA` is transposed for cache locality

### Kernel configuration

Default traits (`gemm_a8w8_blockscale_traits<>`):

| Parameter | Value |
|---|---|
| BLOCK_M × BLOCK_N × BLOCK_K | 256 × 256 × 128 |
| GROUP_M × GROUP_N × GROUP_K | 1 × 128 × 128 |
| Warps per block / block size | 8 / 512 |
| MFMA tile (W_M × W_N × W_K) | 16 × 16 × 128 |
| Warp tiling (T_M × T_N × T_K) | 4 × 2 × 1 |
| A/B global-load vector (fp8) | 16 elems |
| C global-store vector (fp32) | 4 elems |

## Files

```
block_scale_gemm/
├── Makefile
├── rebuild.sh
├── gemm_a8w8_blockscale_common.h              # kargs + traits
├── gemm_a8w8_blockscale_kernel_template.hpp   # kernel body
├── gemm_a8w8_blockscale_kernel.cc             # device-only TU + host stub
└── gemm_a8w8_blockscale_host.cc               # host launcher / benchmark / CPU reference
```

## Prerequisites

- ROCm with hipcc
- gfx950 GPU target (e.g. MI355X)
- OPUS headers from [aiter](https://github.com/ROCm/aiter): set `OPUS_INCLUDE_DIR` to `<aiter_root>/csrc/include`
- OpenMP (for CPU reference and random init)

## Build

```bash
cd opus_gemm/block_scale_gemm
export OPUS_INCLUDE_DIR=/path/to/aiter/csrc/include
make -j
```

Or use the convenience script (build + a default run):

```bash
./rebuild.sh
```

## Run

```bash
./build/gemm_a8w8_blockscale.exe                         # defaults: b=8, m=256, n=512, k=256
./build/gemm_a8w8_blockscale.exe -b 1 -m 4096 -n 4096 -k 4096
./build/gemm_a8w8_blockscale.exe -b 1 -m 1024 -n 1024 -k 1024 -v 1   # validate vs CPU
```

### Command-line options

| Flag | Description | Default |
|---|---|---|
| `-b`, `--b` | Batch size | 8 |
| `-m`, `--m` | M dimension | 256 |
| `-n`, `--n` | N dimension | 512 |
| `-k`, `--k` | K dimension | 256 |
| `-v`, `--verify` | CPU reference verification (0=off, 1=on) | 0 |

All flags accept both `-m 4096` and `-m=4096` syntax. `M / N / K` must be multiples of `GROUP_M / GROUP_N / GROUP_K` respectively.

## Kernel resource usage

Reported by `-Rpass-analysis=kernel-resource-usage` on gfx950:

| VGPR | SGPR | Wave | Occupancy | LDS (bytes) |
|:---:|:---:|:---:|:---:|:---:|
| 256 | 71 | 8 | 2 | 135168 |

## Performance

Measured on MI355X, `batch = 1`, square problem size `M = N = K`:

| M=N=K | Grid | Avg Time (ms) | TFlops |
|---:|---|---:|---:|
| 1024  | (16,   1, 1) | 0.0181 |  118.96 |
| 2048  | (64,   1, 1) | 0.0279 |  616.77 |
| 4096  | (256,  1, 1) | 0.0614 | 2237.77 |
| 8192  | (1024, 1, 1) | 0.4057 | 2710.24 |
| 16384 | (4096, 1, 1) | 3.2737 | 2686.87 |
