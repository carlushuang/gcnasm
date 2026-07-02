# pin_reg_flydsl — register pinning in FlyDSL (A/B -> AGPR, C -> VGPR)

FlyDSL (MLIR Python DSL) analog of `../pin_reg` (the C++/HIP version). The A/B
MFMA input fragments are pinned to the AGPR file and the accumulator stays in
VGPR, producing `v_mfma v[C], a[A], a[B]` with the inputs loaded directly into
AGPR — no inline asm.

Files:
- `pin.py` — `pin_agpr(value, regno)` / `pin_vgpr(value, regno)` helpers; they
  emit `llvm.amdgcn.pin.{agpr,vgpr}` via `llvm.call_intrinsic` (the same path
  FlyDSL already uses for `llvm.amdgcn.s.setreg`).
- `gemm_pin.py` — tiled MMA GEMM (from FlyDSL `examples/03-tiledMma.py`) with the
  pins applied to the A/B fragments.
- `flydsl-llvm-pin.patch` — the LLVM half of the pin patch (git-apply-able).

## Why a patched LLVM is required

FlyDSL does AMDGPU codegen **in-process** (`gpu-module-to-binary{format=fatbin}`)
against the LLVM it was built with. Register pinning is an LLVM intrinsic plus a
target codegen pass, so that LLVM must carry the patch:
- the `llvm.amdgcn.pin.*` intrinsics (else `call_intrinsic` rejects the name), and
- the `SIPreColorPins` pass + `SIFoldOperands` AGPR-load fold (they run
  automatically inside `gpu-module-to-binary`).

The patch touches only AMDGPU/LLVM files (no clang), so it is independent of the
FlyDSL front end.

## Workflow (inside a recent aiter / FlyDSL container)

FlyDSL's LLVM is `AlexAUT/llvm-project @ ee8c4b0f5db` (per the FlyDSL playbook).

```bash
# 1. patch FlyDSL's LLVM
git clone https://github.com/AlexAUT/llvm-project.git
cd llvm-project && git checkout ee8c4b0f5db
git apply /path/to/flydsl-llvm-pin.patch          # or: git apply --3way
#   SIPreColorPins.cpp is a new file (clean); the edited AMDGPU files may need a
#   small context fixup if this LLVM has drifted from ROCm 27682a1 -- resolve and
#   record the delta to sync back to carlushuang/llvm-project.

# 2. build it (mlir + lld, as FlyDSL needs)
cmake ../llvm -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="mlir;clang;lld" \
  -DLLVM_TARGETS_TO_BUILD="X86;AMDGPU" \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DCMAKE_INSTALL_PREFIX=$HOME/llvm-pin-install \
  -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_INSTALL_UTILS=ON \
  -DPython3_EXECUTABLE=$(which python3)
ninja -j$(nproc) && ninja install

# 3. get FlyDSL and rebuild it against the patched LLVM
pip install flydsl        # or build from source (FlyDSL build guide, Issue #22)
#   Point FlyDSL's LLVM at the patched install and rebuild libFlyPythonCAPI +
#   the _mlir bindings so call_intrinsic resolves llvm.amdgcn.pin.*:
#     -DLLVM_DIR / -DMLIR_DIR = $HOME/llvm-pin-install/lib/cmake/{llvm,mlir}
#   (also symlink the patched ld.lld into /opt/rocm/llvm/bin, per the playbook).

# 4. run the example
export LD_LIBRARY_PATH=<flydsl>/_mlir/_mlir_libs:$LD_LIBRARY_PATH
python gemm_pin.py                                 # "Result correct: True"
```

## Verifying the ISA

Dump the kernel's assembly (FlyDSL can emit the module; or inspect the fatbin)
and check the inner MMA:

```
buffer_load_... a[...]                     # A/B loaded directly into AGPR
v_mfma_f32_16x16x4_f32 v[C], a[A], a[B]    # inputs AGPR, accumulator VGPR
# no v_accvgpr shuffles in the loop
```

## Known open point (expected in-container iteration)

FlyDSL fragments are register-backed tensors, not single SSA values. `pin.py`
pins a fragment's underlying value; if a fragment lowers to several register
values, pin each (`for i, v in enumerate(frag.__extract_to_ir_values__()):
pin_agpr(v, base + i*width)`). The exact hook (fragment value vs per-register,
and whether it survives the copy→mma dataflow) is the thing to validate/adjust
in-container. Any backend change needed to make it land cleanly is a fix to the
LLVM patch — sync it back to `carlushuang/llvm-project` (branch
`carhuang/amdgpu_pin_reg`, PR #1).

## Caveats (same as the C++ path)

- The accumulator must fit the per-wave VGPR budget for the mixed form; if not,
  cap occupancy via the FlyDSL launch config / `amdgpu-waves-per-eu` (the analog
  of `__launch_bounds__`).
- A pin whose value is a sub-slice of a shared load (one `ds_read2` feeding two
  fragments) is handled as a no-op by the backend — already fixed in the patch.
