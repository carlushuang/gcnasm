# matrix_core_opus/pin_reg — split register plan (accumulator VGPR, inputs AGPR)

`matrix_core_kernel_block_v2` (BLOCK_M=256, BLOCK_N=192) pinning the register
plan with declaration attributes only:

```cpp
__attribute__((amdgpu_pin_vgpr(0)))  ... v_c;   // accumulator -> VGPR
__attribute__((amdgpu_pin_agpr(0)))  auto v_a = ...;   // A input -> AGPR
__attribute__((amdgpu_pin_agpr(64))) auto v_b = ...;   // B input -> AGPR
```

The pin drives occupancy (no `__launch_bounds__`) and forces the *mixed* MFMA
form `v_mfma v[D], a[A], a[B]`: the accumulator stays in VGPRs while the inputs
use AGPRs. gfx950 has separate 256 VGPR + 256 AGPR files, so this frees VGPRs
for the accumulator without costing occupancy.

## Result (MI355X, gfx950, pin-enabled clang)
- baseline (no pins): occupancy 4, 247 VGPR spill.
- split pins: occupancy 2, 128 VGPR + 116 AGPR, 0 spill, all 48 MFMAs write v[D]
  with AGPR inputs, valid.

Requires a pin-enabled clang (`amdgpu_pin_vgpr`/`amdgpu_pin_agpr` + the
`llvm.amdgcn.pin.*` intrinsics are not upstream). Reuses `../half.hpp`.
