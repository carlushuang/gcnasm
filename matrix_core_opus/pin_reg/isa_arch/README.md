# Per-arch opus tiled-MMA pinned ISA (gfx942 / gfx950 / gfx1201)

Dumped from `../matrix_core_arch.cpp` — one `__global__` per target, each built on
`opus::make_tiled_mma` (the high-level opus path, same as the
`op_tests/opus/device` tiled tests) with the register plan applied via the
`amdgpu_pin_{vgpr,agpr}` attributes:

- gfx942/gfx950: `make_tiled_mma<..>(.., opus::mfma_adaptor_swap_ab{})`, `pin_agpr` A/B.
- gfx1201: `make_tiled_mma(opus::make_wmma<..>(.., opus::wmma_adaptor_swap_ab{}), ..)`, `pin_vgpr` A/B.

16x16x16 f16, one wave, BLOCK == WAVE (single tile per thread).

| arch | family | pinned instruction |
|------|--------|--------------------|
| gfx942  | CDNA3 (MFMA) | `v_mfma_f32_16x16x16_f16 v[0:3], a[8:9], a[0:1]` — A/B born in AGPR, C in VGPR |
| gfx950  | CDNA3 (MFMA) | `v_mfma_f32_16x16x16_f16 v[0:3], a[8:9], a[0:1]` — A/B born in AGPR, C in VGPR |
| gfx1201 | RDNA4 (WMMA) | `v_wmma_f32_16x16x16_f16 v[0:7], v[12:15], v[8:11]` — no AGPR file, A/B/C in VGPR |

(A/B are swapped by `*_adaptor_swap_ab`.) On CDNA the A/B fragment is loaded
straight into the pinned AGPRs (`buffer_load_dwordx2 a[0:1]` / `a[8:9]`), 0
`v_accvgpr`; on RDNA4 into the pinned VGPRs (`buffer_load_b128 v[8:11]` /
`v[12:15]`).

Regenerate:
```bash
clang++ -x hip --cuda-device-only -S -O3 --offload-arch=<arch> -nogpulib \
  --rocm-path=/opt/rocm -I<aiter/csrc/include> -std=c++20 \
  matrix_core_arch.cpp -o matrix_core.<arch>.s
```
with a pin-enabled clang (carlushuang/llvm-project PR #1) and an opus that has the
WMMA tiled_mma path (aiter `csrc/include/opus`).

## Note on the larger tiled kernel (`../matrix_core.cc`, block_v2)

`block_v2` (BLOCK_M=192) makes each thread's A/B fragment wide (>=256-bit,
assembled from several `buffer_load_dwordx2`); that wide multi-load AGPR fold did
not reproduce in this build environment. These single-tile (BLOCK == WAVE) opus
tiled_mma kernels use a one-load fragment, so the AGPR/VGPR pins fold directly and
deterministically into the requested registers across all three arches.
