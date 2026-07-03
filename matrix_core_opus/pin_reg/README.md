# matrix_core_opus/pin_reg — split register plan (accumulator VGPR, inputs AGPR)

`matrix_core_kernel_block_v2` (BLOCK_M=192, BLOCK_N=128) pins the register plan
with declaration attributes only:

```cpp
                                     ... v_c;         // accumulator -> VGPR (natural)
__attribute__((amdgpu_pin_agpr(0)))  auto v_a = ...;  // A input -> AGPR a[0:..]
__attribute__((amdgpu_pin_agpr(64))) auto v_b = ...;  // B input -> AGPR a[64:..]
```

This produces the *mixed* MFMA form `v_mfma v[D], a[A], a[B]`: the accumulator
stays in VGPRs (`v[D]`/`v[C]`) while the A/B inputs live in AGPRs and are loaded
there directly (`buffer_load a[..]`), with no `v_accvgpr` shuffles in the loop.
gfx950 has separate 256 VGPR + 256 AGPR files, so moving A/B to AGPR frees VGPRs
for the accumulator.

## `__launch_bounds__(256, 1)` is required

Moving A/B into AGPR *frees* VGPRs, which lets the compiler raise occupancy; the
smaller per-wave VGPR budget then no longer holds the (VGPR) accumulator and it
spills / rotates through AGPRs (`v_accvgpr`). Capping occupancy with
`__launch_bounds__(256, 1)` keeps the accumulator resident. (An experimental
`-mllvm -amdgpu-pin-acc-vgpr-margin=N` makes the pin drive occupancy itself, but
it currently perturbs the hard-pinned physreg live ranges — use `__launch_bounds__`.)

## Result (MI355X, gfx950, pin-enabled clang), verified

192x128 tile, m=768 n=384 k=64, `-verify-machineinstrs` clean, runs `valid`:
- baseline (no pins): occupancy 4, 128 VGPR, 0 AGPR.
- split pins + `__launch_bounds__(256,1)`: occupancy 2, 192 VGPR + 72 AGPR,
  **0 spill**, **0 `v_accvgpr` in the loop**, all 24 MFMAs are
  `v_mfma_f32_16x16x16_f16 v[C], a[A], a[B]`, A/B loaded directly into AGPR
  (`v_a -> a[0:..]`, `v_b -> a[64:..]`).

The accumulator must fit the per-wave VGPR budget at the chosen occupancy; a tile
whose accumulator exceeds the VGPR file cannot keep C in VGPR (use a smaller tile).

Requires a pin-enabled clang (`amdgpu_pin_vgpr`/`amdgpu_pin_agpr` + the
`llvm.amdgcn.pin.*` intrinsics are not upstream). Reuses `../half.hpp`.

## Reproduced + hardware-validated on gfx950 and gfx942

`matrix_core.gfx950.s` / `matrix_core.gfx942.s` are the dumped ISA for
`matrix_core_kernel_block_v2` (192x128) with A/B pinned to AGPR and the
accumulator in VGPR: 24/24 `v_mfma_f32_16x16x16_f16 v[C], a[A], a[B]`, A/B born
in AGPR (`buffer_load_dwordx2 a[..]`), 0 `v_accvgpr`. Run-validated on hardware
(pinned result vs CPU GEMM, nrms 2.4e-4): **gfx950 (MI355) VALID, gfx942 VALID**.

Build recipe (pin-enabled clang from carlushuang/llvm-project PR #1, on a ROCm
matching the clang's version so device libs load — do NOT use `-nogpulib`, whose
different scheduling splits the K loop across blocks and defeats the AGPR fold):
```
clang++ -x hip --cuda-device-only -S -O3 --offload-arch=gfx950 \
  --rocm-path=/opt/rocm -I<aiter>/csrc/include -std=c++17 matrix_core.cc
```
