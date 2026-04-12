# opus_attn — GQA Flash Attention Kernel for gfx950

Hand-written Grouped-Query Attention (GQA) kernel using the [OPUS](https://github.com/ROCm/aiter) template library and MFMA 32x32x16 bf16 instructions on AMD gfx950 (MI350).

## Features

- Flash Attention with online softmax (no materialized NxN attention matrix)
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
| `hip_minimal.h` | Minimal HIP device-side header (replaces `hip/hip_runtime.h` for device-only compilation) |
| `rebuild.sh` | Monolithic build script |
| `rebuild_split.sh` | Split build script (separate device/host compilation + link) |
| `rebuild_split_v2.sh` | Compile-time comparison script (device-only vs monolithic) |

## Prerequisites

- ROCm with hipcc (tested with ROCm 6.x)
- gfx950 GPU target
- OPUS headers from [aiter](https://github.com/ROCm/aiter): `aiter/csrc/include/opus/opus.hpp`
- OpenMP support (for CPU reference and random init)

## Build

### Monolithic build (recommended)

```bash
cd opus_attn

# Set the path to aiter's include directory containing opus/opus.hpp
export OPUS_INCLUDE_DIR=/path/to/aiter/csrc/include

# Edit rebuild.sh to set OPUS_INCLUDE_DIR, then:
./rebuild.sh
```

Or build directly:

```bash
mkdir -p build && cd build
hipcc ../gqa_gfx950.cc \
  -I/path/to/aiter/csrc/include \
  -std=c++20 -fopenmp -O3 -Wall \
  --offload-arch=gfx950 -ffast-math \
  -Rpass-analysis=kernel-resource-usage \
  -o gqa_attn.exe
```

### Split build (experimental)

Compiles device and host as separate TUs. Useful for compile-time analysis but currently suffers from VGPR spills due to thin-LTO in separate compilation.

```bash
cd opus_attn
# Edit rebuild_split.sh to set OPUS_INCLUDE_DIR, then:
./rebuild_split.sh
```

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
GQA Attention: B=16, H=64, H_KV=8, GROUP_SIZE=8, N=16384, D=128
GQA kernel launch config: grid=(64,64,16), block=512 (NUM_WARPS=8), smem=68096 bytes (K/V tiles)

GQA Kernel Performance: avg_time=111.695 ms, 1260.02 TFlops
```

## Kernel configuration

The kernel is parameterized via `opus_gqa_traits<Q_TILE, KV_TILE, D_TILE, NUM_WARPS>`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Q_TILE_SIZE | 32 | Query tile size per warp |
| KV_TILE_SIZE | 64 | KV tile size in shared memory |
| D_TILE_SIZE | 128 | Head dimension (fixed) |
| NUM_WARPS | 8 | Warps per workgroup (512 threads) |
| MFMA | 32x32x16 bf16 | Matrix multiply instruction |
| Shared memory | 68096 bytes | Double-buffered K + V tiles |

## Compile-time notes

Measured on MI355X with ROCm 7.1.1 and [optimized opus.hpp](https://github.com/ROCm/aiter/pull/2701):

| Build | Time | VGPRs | Spill | Occupancy |
|-------|------|-------|-------|-----------|
| Device-only (`--cuda-device-only`) | ~1.47s | 251 | 0 | 2 |
| Full monolithic (`gqa_gfx950.cc`) | ~2.86s | 251 | 0 | 2 |

Device-only breakdown: Frontend ~0.9s, Backend ~0.6s.

### Performance

| N | TFlops |
|---|--------|
| 1024 | ~922 |
| 16384 | ~1268 |
