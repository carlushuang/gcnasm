# FlyDSL pinned-GEMM ISA dumps (gfx942 / gfx950)

Dumped from the FlyDSL example (`../gemm_pin_rocdl.py`): a 16x16x16 f16
`rocdl.mfma` GEMM whose A/B inputs are pinned to AGPR via `llvm.amdgcn.pin.agpr`
(see `../flydsl_patch/pin_mfma.py`). The FlyDSL-emitted device LLVM IR was fed to
the pin-enabled `llc` for each arch.

| arch | family | pinned instruction |
|------|--------|--------------------|
| gfx942 | CDNA3 (MFMA) | `v_mfma_f32_16x16x16_f16 v[0:3], a[0:1], a[64:65]` (A→a[0:1], B→a[64:65], C→VGPR) |
| gfx950 | CDNA3 (MFMA) | `v_mfma_f32_16x16x16_f16 v[0:3], a[0:1], a[64:65]` (A→a[0:1], B→a[64:65], C→VGPR) |

A/B are loaded straight into the pinned AGPRs (`buffer_load_dwordx2 a[0:1] /
a[64:65]`), 0 `v_accvgpr`.

gfx1201 (RDNA4) is not covered here: the FlyDSL example uses `rocdl.mfma`, which
is CDNA-only. An RDNA WMMA FlyDSL kernel would be needed; the opus C++ example
(`../../pin_reg/`) covers gfx1201 via WMMA.

Regenerate:
```bash
llc -mtriple=amdgcn-amd-amdhsa -mcpu=<arch> -O3 <flydsl-emitted-ir>.ll -o gemm_pin.<arch>.s
```
using the pin-enabled `llc` from FlyDSL's LLVM (see `../flydsl-llvm-pin.patch`).
