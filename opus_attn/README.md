# opus_attn — GQA Flash Attention Kernel for gfx950

Hand-written Grouped-Query Attention (GQA) kernel using the [OPUS](https://github.com/ROCm/aiter) template library and MFMA 32x32x16 bf16 instructions on AMD gfx950 (MI355).

## Features

- Flash Attention with online softmax (no materialized NxN attention matrix)
- Causal and non-causal variants (parallel compilation via `make -j`)
- Double-buffered K/V tiles in shared memory
- Software-pipelined global→shared→register data movement
- Fine-grained scheduling barriers for MFMA/VALU/EXP interleaving
- `__HIP_DEVICE_COMPILE__` guard for fast host pass (~580ms saved)
- CPU reference implementation for validation

## Files

```
opus_attn/
├── Makefile                              # Parallel build (make -j)
├── rebuild.sh                            # Build + benchmark both variants
├── gqa_common.h                          # Shared types: bf16_t, opus_gqa_kargs, opus_gqa_traits
├── gqa_gfx950_kernel_template.hpp        # Kernel implementation (included by variant .cc files)
├── gqa_gfx950_kernel_causal.cc           # Causal kernel instantiation
├── gqa_gfx950_kernel_noncausal.cc        # Non-causal kernel instantiation
├── gqa_gfx950_host.cc                    # Host launcher, benchmark, validation, main()
├── hip_minimal.h                         # Local HIP minimal header
└── monolithic/                           # Original single-file build (for reference)
    ├── gqa_gfx950.cc
    └── rebuild.sh
```

## Prerequisites

- ROCm with hipcc (tested with ROCm 7.1.1)
- gfx950 GPU target
- OPUS headers from [aiter](https://github.com/ROCm/aiter): set `OPUS_INCLUDE_DIR` to `<aiter_root>/csrc/include`
- OpenMP support (for CPU reference and random init)

## Build

```bash
cd opus_attn
export OPUS_INCLUDE_DIR=/path/to/aiter/csrc/include   # default: /home/carhuang/repo/aiter/csrc/include
make -j        # parallel build: causal + non-causal + host + link
```

Or use the convenience script (builds + runs benchmarks):

```bash
./rebuild.sh
```

### Monolithic build (for reference)

```bash
cd monolithic
./rebuild.sh
```

## Run

```bash
./build/gqa_attn.exe                          # causal, N=1024 (default)
./build/gqa_attn.exe --no-causal              # non-causal
./build/gqa_attn.exe -n=16384                 # causal, N=16384
./build/gqa_attn.exe --no-causal -n=16384     # non-causal, N=16384
```

### Command-line options

| Flag | Description | Default |
|------|-------------|---------|
| `-b`, `--batch` | Batch size | 16 |
| `-h`, `--heads` | Number of query heads | 64 |
| `--hkv` | Number of KV heads | 8 |
| `-n`, `--seq` | Sequence length | 1024 |
| `-d`, `--dim` | Head dimension (must be 128) | 128 |
| `--causal` | Enable causal masking | (default) |
| `--no-causal` | Disable causal masking | |
| `-v`, `--verify` | CPU reference verification (0=off, 1=on) | 0 |

All flags support both `-n 16384` and `-n=16384` syntax.

## Kernel configuration

The kernel is parameterized via `opus_gqa_traits<Q_TILE, KV_TILE, D_TILE, NUM_WARPS, CAUSAL>`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Q_TILE_SIZE | 32 | Query tile size per warp |
| KV_TILE_SIZE | 64 | KV tile size in shared memory |
| D_TILE_SIZE | 128 | Head dimension (fixed) |
| NUM_WARPS | 8 | Warps per workgroup (512 threads) |
| CAUSAL | `true`/`false` | Causal masking (two separate kernel binaries) |
| MFMA | 32x32x16 bf16 | Matrix multiply instruction |
| Shared memory | 68096 bytes | Double-buffered K + V tiles |

## Compile time

Measured on MI355X with ROCm 7.1.1 and [optimized opus.hpp](https://github.com/ROCm/aiter/pull/2701):

| Build mode | Time | Notes |
|------------|------|-------|
| `make -j` (parallel) | **~1.35s** | Both kernel variants + host compiled in parallel |
| `make` (sequential) | ~4.2s | Causal + non-causal + host sequentially |
| Monolithic (`monolithic/rebuild.sh`) | ~2.9s | Single file, host+device |

### Compile-time techniques applied

- **`__HIP_DEVICE_COMPILE__` guard**: kernel .cc files skip the full kernel body on the host pass, providing only an empty stub for `__device_stub__` generation (~580ms saved per kernel file)
- **`-D__HIPCC_RTC__`**: applied to kernel .cc files to skip the implicit `__clang_hip_runtime_wrapper.h` on the host pass (~250ms saved per kernel file)
- **Parallel build**: causal, non-causal, and host compile simultaneously via `make -j`

### Per-file breakdown

| File | Time | VGPRs | Spill | Occ |
|------|------|-------|-------|-----|
| `gqa_gfx950_kernel_causal.cc` | ~1.3s | 244 | 0 | 2 |
| `gqa_gfx950_kernel_noncausal.cc` | ~1.3s | 238 | 0 | 2 |
| `gqa_gfx950_host.cc` | ~0.9s | — | — | — |
| Link | ~0.03s | — | — | — |

## Performance

B=16, H=64, H_KV=8, D=128, measured on MI355X:

| N | Causal TFlops | Causal Time | Non-causal TFlops | Non-causal Time |
|---:|---:|---:|---:|---:|
| 1024 | 648 | 0.42 ms | 939 | 0.59 ms |
| 2048 | 920 | 1.20 ms | 1124 | 1.96 ms |
| 4096 | 1085 | 4.05 ms | 1212 | 7.26 ms |
| 8192 | 1178 | 14.94 ms | 1253 | 28.09 ms |
| 16384 | 1233 | 57.07 ms | 1278 | 110.14 ms |
