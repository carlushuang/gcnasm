# Opus C++ pinned-GEMM ISA dumps (gfx942 / gfx950 / gfx1201)

Dumped from `../gemm_pin_multiarch.cpp` (a 16x16x16 f16 matrix-core GEMM tile with
the A/B inputs and the accumulator register-pinned via the `amdgpu_pin_*`
attributes). Each `.s` contains both the `gemm_plain` (compiler default) and the
`gemm_pinned` kernel so the placement contrast is visible.

Pinned kernel matrix-core instruction per arch:

| arch | family | pinned instruction |
|------|--------|--------------------|
| gfx942  | CDNA3 (MFMA) | `v_mfma_f32_16x16x16_f16 v[0:3], a[0:1], a[2:3]` (A/B→AGPR, C→VGPR) |
| gfx950  | CDNA3 (MFMA) | `v_mfma_f32_16x16x16_f16 v[0:3], a[0:1], a[2:3]` (A/B→AGPR, C→VGPR) |
| gfx1201 | RDNA4 (WMMA) | `v_wmma_f32_16x16x16_f16 v[0:7], v[8:11], v[12:15]` (A/B→VGPR, C→VGPR) |

RDNA4 has no AGPR file, so A/B are pinned to VGPRs there; the CDNA parts place
A/B directly in the requested AGPRs (`global_load ... a[..]`), no `v_accvgpr`.

Run-verified (pinned kernel bit-identical to plain on real hardware): **gfx942
PASS**, **gfx1201 PASS**. gfx950 is ISA here (previously validated on MI355X).

Regenerate:
```bash
clang++ -x hip --cuda-device-only -S -O3 --offload-arch=<arch> \
  -nogpulib --rocm-path=/opt/rocm gemm_pin_multiarch.cpp -o gemm_pin.<arch>.s
```
using a pin-enabled clang (carlushuang/llvm-project PR #1).
