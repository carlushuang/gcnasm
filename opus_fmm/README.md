# opus_fmm — Flat Matrix Multiply Kernels

Hand-written GEMM kernels using the [OPUS](https://github.com/ROCm/aiter) template library and MFMA instructions on AMD gfx942/gfx950.

## Files

| File | Description |
|------|-------------|
| `flatmm_a8w8_gfx950.cc` | FP8 (A=fp8, W=fp8) flat GEMM for gfx950 |
| `flatmm_a16w16_gfx950.cc` | BF16 (A=bf16, W=bf16) flat GEMM for gfx950 |
| `flatmm_a16w16_gfx942.cc` | BF16 (A=bf16, W=bf16) flat GEMM for gfx942 |
| `rebuild.sh` | Build script |

## Build

```bash
cd opus_fmm

# Set OPUS_INCLUDE_DIR in rebuild.sh or build directly:
hipcc flatmm_a8w8_gfx950.cc \
  -I/path/to/aiter/csrc/include \
  -fPIC -std=c++17 -fopenmp -O3 -Wall \
  --offload-arch=gfx950 \
  -o build/flatmm.exe
```

## Compile time and resource usage

Measured on MI355X with ROCm 7.1.1 and [optimized opus.hpp](https://github.com/ROCm/aiter/pull/2701):

| Kernel | Arch | Build time | VGPRs | Spill | Occupancy |
|--------|------|-----------|-------|-------|-----------|
| flatmm_a8w8 | gfx950 | ~2.44s | 248 | 0 | 2 |
| flatmm_a16w16 | gfx950 | ~2.17s | 234 | 0 | 2 |
| flatmm_a16w16 | gfx942 | ~1.80s | 216 | 0 | 2 |

## Run

```bash
./build/flatmm.exe
```

### Expected output (flatmm_a8w8_gfx950)

```
[Overall] ✓ ALL BATCHES VALID

Kernel Performance: avg_time=0.0260 ms, 20.64 TFlops
```

### Expected output (flatmm_a16w16_gfx950)

```
[Overall] ✓ ALL BATCHES VALID

Kernel Performance: avg_time=0.0120 ms, 22.39 TFlops
```
