# pin_reg_flydsl — register pinning in FlyDSL (A/B -> AGPR, C -> VGPR)

FlyDSL (MLIR Python DSL) analog of `../pin_reg`. The A/B MFMA input fragments are
pinned to the AGPR file and the accumulator stays in VGPR — `v_mfma v[C], a[A],
a[B]` with the inputs loaded directly into AGPR, no inline asm — the same
instruction the C++ opus kernel emits.

## Achieved end-to-end on gfx950

Verified in a `rocm/atom` pytorch container (torch 2.10 / rocm7.2.2, gfx950). The
FlyDSL kernel's executed ISA matches the C++ opus impl:

```
buffer_load_dwordx2 a[0:1], ...                        # A -> AGPR
buffer_load_dwordx2 a[64:65], ...                      # B -> AGPR
v_mfma_f32_16x16x16_f16 v[0:3], a[0:1], a[64:65], 0    # v[C], a[A], a[B]
Result correct: True
```

```bash
FLYDSL_DUMP_IR=1 FLYDSL_PIN_MFMA_AGPR=1 FLYDSL_PIN_CODEGEN_LLC=1 \
FLYDSL_PATCHED_LLC=<flydsl-llvm>/build/bin/llc  python gemm_pin_rocdl.py
```

Works for the low-level `gemm_pin_rocdl.py` (clean `v[C], a[A], a[B]`) and the
high-level `examples/03-tiledMma.py` (accumulator VGPR + AGPR inputs; some
operands fall back to VGPR because all MFMAs share fixed AGPR bases 0/64 — a
per-MFMA base assignment would fill them). Correct in both.

## Files

- `pin.py` — `pin_agpr(value, regno)` / `pin_vgpr(value, regno)`; emit
  `llvm.amdgcn.pin.{agpr,vgpr}` via `llvm.call_intrinsic` (the path FlyDSL
  already uses for `llvm.amdgcn.s.setreg`).
- `gemm_pin_rocdl.py` — low-level `rocdl.mfma` GEMM (16x16x16 f16) with `pin_agpr`
  on the A/B vectors; the one that produces the clean opus instruction above.
- `gemm_pin.py` — small tiled MMA GEMM (from FlyDSL `examples/03-tiledMma.py`).
- `gemm_pin_large.py` — complex GEMM (128x128x64, double-buffered LDS K-loop,
  4-wave, 16x16x16 f16 MFMA), modeled on FlyDSL `examples/04-preshuffle_gemm.py`;
  the opus-scale analog. A/B fragments pinned to AGPR, accumulator in VGPR.
- `flydsl_patch/` — the two FlyDSL source changes (see below).
- `flydsl-llvm-pin.patch` — the LLVM pin patch **rebased onto FlyDSL's pinned
  LLVM** (`ROCm/llvm-project @ 7f77ca0dbda...`, from FlyDSL
  `thirdparty/llvm-hash.txt`). `git apply`-clean on that commit; 11 LLVM files,
  no clang.

## Two FlyDSL source changes (`flydsl_patch/`)

Both in `python/flydsl/compiler/`:

1. `pin_mfma.py::pin_all_mfma_agpr` — walks the lowered module and wraps every
   `rocdl.mfma` src0/src1 with `llvm.amdgcn.pin.agpr` just before device codegen.
   (Emitting the pin from FlyDSL *python* into a kernel does not survive
   trace/lowering — high-level `tiled_mma` fragments are register-space
   `fly.memref`s, not SSA values, and `promote_regmem_to_vectorssa` /
   `convert_fly_to_rocdl` reconstruct the MFMA dataflow — so the pin must be
   applied on the *lowered* `rocdl.mfma`.)
2. `pin_mfma.py::codegen_via_llc` + a hook in `jit_function.py` — codegen the
   device module with the patched **`llc`+`lld`** instead of the in-process
   `gpu-module-to-binary` serializer. Required: the serializer's `optimizeLlvm`
   step *drops* the pin (confirmed by disassembling its bitcode: pin count 0),
   whereas `llc` / `opt -O3 -> llc` on the identical IR keep it and place A/B in
   AGPR. `codegen_via_llc` loads the resulting raw HSA code object via a minimal
   `gpu.binary` (HIP's `hipModuleLoadData` accepts it).

Enabled by env vars: `FLYDSL_PIN_MFMA_AGPR=1` (`FLYDSL_PIN_A_AGPR`/
`FLYDSL_PIN_B_AGPR` set the bases), `FLYDSL_PIN_CODEGEN_LLC=1`,
`FLYDSL_PATCHED_LLC=<flydsl-llvm>/build/bin/llc`. The hook currently lives in
FlyDSL's dump path (`FLYDSL_DUMP_IR=1`); for production the same call belongs
before the `gpu-module-to-binary` fragment in the non-dump `_run_pipeline`.

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
ninja -C build -j$(nproc) install

# 3. FlyDSL against the patched MLIR, apply flydsl_patch/, then run
export MLIR_PATH=$PWD/mlir_install       # FlyDSL build.sh honors this
pip install flydsl                       # or FlyDSL scripts/build.sh
FLYDSL_DUMP_IR=1 FLYDSL_PIN_MFMA_AGPR=1 FLYDSL_PIN_CODEGEN_LLC=1 \
FLYDSL_PATCHED_LLC=$PWD/build/bin/llc  python gemm_pin_rocdl.py
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
type signature.)

## Caveats (same as the C++ path)
- The accumulator must fit the per-wave VGPR budget for the mixed form; else cap
  occupancy via the FlyDSL launch config / `amdgpu-waves-per-eu`.
- A pin whose value is a sub-slice of a shared load (one `ds_read2` feeding two
  fragments) is a no-op in the backend — already handled by the patch.
