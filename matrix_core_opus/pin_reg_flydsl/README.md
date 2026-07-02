# pin_reg_flydsl — register pinning in FlyDSL (A/B -> AGPR, C -> VGPR)

FlyDSL (MLIR Python DSL) analog of `../pin_reg`. A/B MFMA input fragments are
pinned to the AGPR file and the accumulator stays in VGPR — `v_mfma v[C], a[A],
a[B]` with the inputs loaded directly into AGPR, no inline asm.

Files:
- `pin.py` — `pin_agpr(value, regno)` / `pin_vgpr(value, regno)`; emit
  `llvm.amdgcn.pin.{agpr,vgpr}` via `llvm.call_intrinsic` (the path FlyDSL
  already uses for `llvm.amdgcn.s.setreg`).
- `gemm_pin.py` — tiled MMA GEMM (from FlyDSL `examples/03-tiledMma.py`) with the
  pins on the A/B fragments.
- `flydsl-llvm-pin.patch` — the pin patch **rebased onto FlyDSL's pinned LLVM**
  (`ROCm/llvm-project @ 7f77ca0dbda...`, from FlyDSL `thirdparty/llvm-hash.txt`).
  `git apply`-clean on that commit; 11 LLVM files, no clang.
- `verify_pin_mfma.mlir` / `verify_pin_mfma.gfx950.s` — the FlyDSL emission
  pattern reduced to MLIR, and its verified ISA (below).

## Verified end-to-end on FlyDSL's LLVM

Built `mlir-translate` + `llc` from `ROCm/llvm-project @ 7f77ca0db` with the patch
applied, then:

```bash
mlir-translate --mlir-to-llvmir verify_pin_mfma.mlir -o pin.ll
llc -mcpu=gfx950 -O3 pin.ll -o verify_pin_mfma.gfx950.s
```

`verify_pin_mfma.mlir` pins the A/B fragments with
`llvm.call_intrinsic "llvm.amdgcn.pin.agpr"` and feeds `rocdl.mfma`. Result ISA:

```
global_load_dwordx2 a[0:1], v1, s[0:1]      ; A born in AGPR at pin 0
global_load_dwordx2 a[8:9], v1, s[2:3]      ; B born in AGPR at pin 8
v_mfma_f32_16x16x16_f16 v[0:3], a[0:1], a[8:9], 0   ; v[C], a[A], a[B]
```

2 AGPR loads, **0 v_accvgpr**: `call_intrinsic("llvm.amdgcn.pin.agpr")` →
`mlir-translate` (resolves the intrinsic from the patched LLVM) → `llc`
(SIPreColorPins + SIFoldOperands) places A/B directly in AGPR at the pinned
registers and the MFMA reads them; the accumulator is VGPR. This is the FlyDSL
mechanism proven on FlyDSL's own LLVM.

## Workflow (aiter / FlyDSL container)

```bash
# 1. FlyDSL's LLVM (per thirdparty/llvm-hash.txt), patched
git clone https://github.com/ROCm/llvm-project.git
cd llvm-project && git checkout 7f77ca0dbda4abbf9af06537b2c475f20ccd6007
git apply /path/to/flydsl-llvm-pin.patch

# 2. build MLIR (+python bindings) as FlyDSL's scripts/build_llvm.sh does, or:
cmake -S llvm -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="mlir;clang;lld" -DLLVM_TARGETS_TO_BUILD="X86;AMDGPU" \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON -DLLVM_INSTALL_UTILS=ON \
  -DCMAKE_INSTALL_PREFIX=$PWD/mlir_install
ninja -C build -j$(nproc) install       # SIPreColorPins runs in gpu-module-to-binary

# 3. FlyDSL against the patched MLIR, then run
export MLIR_PATH=$PWD/mlir_install       # FlyDSL build.sh honors this
pip install flydsl   # or FlyDSL scripts/build.sh
python gemm_pin.py
```

## Deltas from the upstreamed carlushuang patch (LLVM churn)

The public patch (carlushuang/llvm-project#1) targets `roc-7.1.1` (27682a1).
FlyDSL's commit (7f77ca0db) is newer; the rebase needed three real adjustments,
already folded into `flydsl-llvm-pin.patch`:

1. `getMFMASrcCVDstVGPROp(uint16_t)` -> `(uint32_t)` — tablegen widened the
   InstrMapping opcode type.
2. The `amdgpu-no-agpr` inference was replaced by the `amdgpu-agpr-alloc`
   attribute (`AAAMDGPUMinAGPRAlloc`); `pin_agpr` now adds a
   `case Intrinsic::amdgcn_pin_agpr` in `CheckForMinAGPRAllocs` (requires
   `regno + numRegs` AGPRs) instead of `CheckForNoAGPRs` returning false.
3. `getOccupancyWithNumVGPRs` gained a `DynamicVGPRBlockSize` argument.

(Also `rocdl.mfma` MLIR syntax in this LLVM uses literal immargs and a 3-operand
type signature — reflected in `verify_pin_mfma.mlir`.)

## Known open point (in-container)

FlyDSL fragments are register-backed tensors, not single SSA values. `pin.py`
pins a fragment's underlying value; if a fragment lowers to several register
values, pin each (`for i, v in enumerate(frag.__extract_to_ir_values__()):
pin_agpr(v, base + i*width)`). Validate the exact hook when running `gemm_pin.py`.

## Caveats (same as the C++ path)
- The accumulator must fit the per-wave VGPR budget for the mixed form; else cap
  occupancy via the FlyDSL launch config / `amdgpu-waves-per-eu`.
- A pin whose value is a sub-slice of a shared load (one `ds_read2` feeding two
  fragments) is a no-op in the backend — already handled by the patch.
