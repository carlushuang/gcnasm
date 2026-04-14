# opus_attn — GQA Flash Attention Kernel for gfx950

Hand-written Grouped-Query Attention (GQA) kernel using the [OPUS](https://github.com/ROCm/aiter) template library and MFMA 32x32x16 bf16 instructions on AMD gfx950 (MI355).

## Features

- Flash Attention with online softmax (no materialized NxN attention matrix)
- Causal mask support for autoregressive attention
- Double-buffered K/V tiles in shared memory
- Software-pipelined global→shared→register data movement
- Fine-grained scheduling barriers for MFMA/VALU/EXP interleaving
- CPU reference implementation for validation

## Files

| File | Description |
|------|-------------|
| `gqa_gfx950.cc` | Monolithic single-TU build (device + host) |
| `gqa_gfx950_kernel.cc` | Device-only: kernel + device helpers (split build) |
| `gqa_gfx950_host.cc` | Host-only: benchmark, validation, main() (split build) |
| `gqa_common.h` | Shared types: `bf16_t`, `opus_gqa_kargs`, `opus_gqa_traits` |
| `rebuild.sh` | Monolithic build script |
| `rebuild_split.sh` | Split build script (separate device/host compilation + link) |
| `compare_compile_modes.sh` | Compile-time comparison script (device-only vs monolithic) |

## Prerequisites

- ROCm with hipcc (tested with ROCm 6.x)
- gfx950 GPU target
- OPUS headers from [aiter](https://github.com/ROCm/aiter): `aiter/csrc/include` (contains `opus/opus.hpp` and `opus/hip_minimal.hpp`)
- OpenMP support (for CPU reference and random init)

## Build

### Monolithic build (recommended)

```bash
cd opus_attn
export OPUS_INCLUDE_DIR=/path/to/aiter/csrc/include
./rebuild.sh
```

Builds `gqa_gfx950.cc` as a single translation unit and prints the total compile time.

### Split build (experimental)

```bash
cd opus_attn
export OPUS_INCLUDE_DIR=/path/to/aiter/csrc/include
./rebuild_split.sh
```

Builds `gqa_gfx950_kernel.cc` and `gqa_gfx950_host.cc` separately, then links them. The script prints device compile time, host compile time, link time, and total time.

### Compile-mode comparison

```bash
cd opus_attn
export OPUS_INCLUDE_DIR=/path/to/aiter/csrc/include
RUN_BINARY=0 ./compare_compile_modes.sh
```

Measures two compile paths:

- device-only compilation of `gqa_gfx950_kernel.cc` via `--cuda-device-only`
- full monolithic compilation of `gqa_gfx950.cc`

This script is for comparison/experiments. The device-only timing isolates device compilation cost, but it is not the same as the full split-build end-to-end time.

## Run

```bash
cd build
./gqa_attn.exe
```

### Command-line options

| Flag | Description | Default |
|------|-------------|---------|
| `-b`, `--batch` | Batch size | 16 |
| `-h`, `--heads` | Number of query heads | 64 |
| `--hkv` | Number of KV heads | 8 |
| `-n`, `--seq` | Sequence length | 1024 |
| `-d`, `--dim` | Head dimension (must be 128) | 128 |

Example:

```bash
./gqa_attn.exe -b 16 -h 64 --hkv 8 -n 16384 -d 128
```

### Expected output

```
GQA Attention: B=16, H=64, H_KV=8, GROUP_SIZE=8, N=16384, D=128, CAUSAL=1
GQA kernel launch config: grid=(64,64,16), block=512 (NUM_WARPS=8), smem=68096 bytes (K/V tiles)

GQA Causal Kernel Performance: avg_time=56.205 ms, 1251.99 TFlops
```

## Kernel configuration

The kernel is parameterized via `opus_gqa_traits<Q_TILE, KV_TILE, D_TILE, NUM_WARPS, CAUSAL>`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Q_TILE_SIZE | 32 | Query tile size per warp |
| KV_TILE_SIZE | 64 | KV tile size in shared memory |
| D_TILE_SIZE | 128 | Head dimension (fixed) |
| NUM_WARPS | 8 | Warps per workgroup (512 threads) |
| CAUSAL | `true` | Whether causal masking is enabled |
| MFMA | 32x32x16 bf16 | Matrix multiply instruction |
| Shared memory | 68096 bytes | Double-buffered K + V tiles |

## Compile-time notes

Measured on MI355X with ROCm 7.1.1 and [optimized opus.hpp](https://github.com/ROCm/aiter/pull/2701):

The `compare_compile_modes.sh` script reproduces the device-only vs monolithic comparison below. The `--cuda-device-only` result only measures device-side compilation, not the full split-build pipeline.

| Build | Time | VGPRs | Spill | Occupancy |
|-------|------|-------|-------|-----------|
| Device-only (`--cuda-device-only`) | ~1.47s | 244 | 0 | 2 |
| Full monolithic (`gqa_gfx950.cc`) | ~2.71s | 244 | 0 | 2 |

Device-only breakdown: Frontend ~0.9s, Backend ~0.6s.

### Performance

| Sequence Length (N) | Performance (Causal) | Performance (Non-Causal) | Time (Causal) | Time (Non-Causal) |
|:---:|:---:|:---:|:---:|:---:|
| 1024 | 645.00 TFlops | 952.74 TFlops | 0.426 ms | 0.577 ms |
| 2048 | 915.58 TFlops | 1165.66 TFlops | 1.201 ms | 1.887 ms |
| 4096 | 1116.15 TFlops | 1251.85 TFlops | 3.940 ms | 7.026 ms |
| 8192 | 1208.93 TFlops | 1264.44 TFlops | 14.552 ms | 27.826 ms |
| 16384 | 1250.83 TFlops | 1284.76 TFlops | 56.258 ms | 109.543 ms |
