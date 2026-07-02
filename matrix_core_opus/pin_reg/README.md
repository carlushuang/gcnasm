# matrix_core_opus/pin_reg — register pinning via a declaration attribute

`matrix_core_kernel_block_v2` (BLOCK_M=256, BLOCK_N=192) pinning the MFMA
accumulator to VGPRs with a single declaration attribute instead of hand-written
asm or per-assignment builtins:

```cpp
__attribute__((amdgpu_pin_vgpr(0))) typename decltype(mma)::vtype_c v_c;
```

The compiler auto-pins every value stored to `v_c` (chunking the 192-VGPR
accumulator internally), and the pin drives the occupancy target so it lands in
VGPRs with zero spill — no `__launch_bounds__` needed.

## Result (MI355X, gfx950, pin-enabled clang)
- baseline: compiler picks occupancy 4 (128-VGPR budget) -> v_c (192) spills 247 VGPRs.
- with `amdgpu_pin_vgpr(0)`: occupancy 2, 232 VGPRs, 0 spill, all 48 MFMAs write v[, valid.

Requires a pin-enabled clang (`amdgpu_pin_vgpr`/`amdgpu_pin_agpr` attributes and the
`llvm.amdgcn.pin.*` intrinsics are not in upstream ROCm yet). Reuses `../half.hpp`.
