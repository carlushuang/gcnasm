# Per-arch matrix-core pinned ISA (gfx942 / gfx950 / gfx1201)

Dumped from `../matrix_core_arch.cpp`, which has one `__global__` per target
(`matrix_core_gfx942`, `matrix_core_gfx950`, `matrix_core_gfx1201`), each a
16x16x16 f16 matrix-core tile with the opus register plan applied via the
`amdgpu_pin_{vgpr,agpr}` attributes.

| arch | family | pinned instruction |
|------|--------|--------------------|
| gfx942  | CDNA3 (MFMA) | `v_mfma_f32_16x16x16_f16 v[0:3], a[0:1], a[2:3]`  — A/B born in AGPR, C in VGPR |
| gfx950  | CDNA3 (MFMA) | `v_mfma_f32_16x16x16_f16 v[0:3], a[0:1], a[2:3]`  — A/B born in AGPR, C in VGPR |
| gfx1201 | RDNA4 (WMMA) | `v_wmma_f32_16x16x16_f16 v[20:27], v[8:11], v[12:15]` — no AGPR file, A/B/C in VGPR |

Loaded straight into the pinned registers (`global_load ... a[..]` / `v[..]`),
no `v_accvgpr`.

Run-verified on real hardware (pinned code object executes correctly):
**gfx942 PASS**, **gfx1201 PASS** (pinned kernel bit-identical to the unpinned
reference). gfx950 is ISA here (CDNA3, identical MFMA to gfx942).

Regenerate:
```bash
clang++ -x hip --cuda-device-only -S -O3 --offload-arch=<arch> \
  -nogpulib --rocm-path=/opt/rocm matrix_core_arch.cpp -o matrix_core.<arch>.s
```
with a pin-enabled clang (carlushuang/llvm-project PR #1).

## Note on the full opus tiled kernel (`../matrix_core.cc`, `block_v2`)

`block_v2` uses opus `make_tiled_mma`, whose A/B fragments are wide (>=256-bit,
assembled from several `buffer_load_dwordx2`). Two constraints made the full tiled
kernel unsuitable for a clean per-arch pinned dump here:

1. opus `tiled_mma` dispatches only to MFMA, so it is CDNA-only — it cannot be
   compiled for gfx1201 (RDNA4/WMMA).
2. The wide multi-load AGPR fold for `block_v2`'s fragments did not reproduce in
   this build environment (A/B stayed in VGPR), independent of the pin pass
   version — an environment difference from the original MI355X validation.

These single-tile per-arch kernels use the foldable matrix-core form so the
AGPR/VGPR pins take effect deterministically and identically across gfx942 /
gfx950 / gfx1201.
