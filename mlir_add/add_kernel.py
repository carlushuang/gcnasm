#!/usr/bin/env python3
"""
Elementwise vector add kernel: C[i] = A[i] + B[i]

Built entirely with MLIR Python bindings using gpu, memref, arith, scf,
and rocdl dialects.  The script:

  1. Constructs the MLIR module programmatically (Python API)
  2. Lowers through pass pipeline (PassManager API)
  3. Translates to LLVM IR (mlir-translate-20)
  4. Compiles to AMDGPU assembly (llc-20)

Usage:
    python3 add_kernel.py [chip]          # default: gfx942
    python3 add_kernel.py gfx90a
"""

import re
import shutil
import subprocess
import sys

from mlir.ir import (
    Context,
    F32Type,
    FunctionType,
    IndexType,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    Location,
    MemRefType,
    Module,
    ShapedType,
    StringAttr,
    TypeAttr,
    UnitAttr,
)
from mlir.dialects import arith, gpu, memref, rocdl, scf
from mlir.passmanager import PassManager


BLOCK_SIZE = 256


# ---------------------------------------------------------------------------
# 1.  Build high-level IR
# ---------------------------------------------------------------------------
def build_vector_add_module() -> Module:
    ctx = Context()
    with ctx, Location.unknown():
        module = Module.create()

        i32 = IntegerType.get_signless(32)
        idx = IndexType.get()
        f32 = F32Type.get()
        dyn_f32 = MemRefType.get([ShapedType.get_dynamic_size()], f32)

        kernel_type = FunctionType.get(
            inputs=[dyn_f32, dyn_f32, dyn_f32, idx],
            results=[],
        )

        with InsertionPoint(module.body):
            gpu_mod = gpu.GPUModuleOp("add_module")
            block = gpu_mod.bodyRegion.blocks.append()

            with InsertionPoint(block):
                kf = gpu.GPUFuncOp(TypeAttr.get(kernel_type))
                kf.operation.attributes["sym_name"] = StringAttr.get(
                    "vector_add"
                )
                kf.operation.attributes["gpu.kernel"] = UnitAttr.get()

                entry = kf.body.blocks.append(*kernel_type.inputs)
                with InsertionPoint(entry):
                    A, B, C, N = entry.arguments

                    # gid = blockIdx.x * BLOCK_SIZE + threadIdx.x
                    tid = rocdl.workitem_id_x(i32)
                    bid = rocdl.workgroup_id_x(i32)
                    bs = arith.ConstantOp(
                        i32, IntegerAttr.get(i32, BLOCK_SIZE)
                    ).result
                    off = arith.MulIOp(bid, bs).result
                    gid_i32 = arith.AddIOp(off, tid).result
                    gid = arith.IndexCastOp(idx, gid_i32).result

                    # Bounds-checked element-wise add
                    cond = arith.CmpIOp(
                        arith.CmpIPredicate.ult, gid, N
                    ).result
                    if_op = scf.IfOp(cond)
                    with InsertionPoint(if_op.then_block):
                        a = memref.LoadOp(A, [gid]).result
                        b = memref.LoadOp(B, [gid]).result
                        s = arith.AddFOp(a, b).result
                        memref.StoreOp(s, C, [gid])
                        scf.YieldOp([])

                    gpu.ReturnOp([])

        module.operation.attributes["gpu.container_module"] = UnitAttr.get()

        assert module.operation.verify()
        return module


# ---------------------------------------------------------------------------
# 2.  Lower to LLVM dialect
# ---------------------------------------------------------------------------
def lower_to_llvm(module: Module, chip: str = "gfx942") -> str:
    """Run pass pipeline and return the plain-module MLIR (gpu.module stripped)."""
    with module.context:
        pm = PassManager.parse(
            "builtin.module("
            "  gpu.module("
            f"   convert-gpu-to-rocdl{{chipset={chip}}},"
            "    convert-scf-to-cf,"
            "    expand-strided-metadata,"
            "    convert-arith-to-llvm,"
            "    convert-index-to-llvm,"
            "    reconcile-unrealized-casts"
            "  ),"
            "  finalize-memref-to-llvm,"
            "  convert-cf-to-llvm,"
            "  reconcile-unrealized-casts"
            ")"
        )
        pm.run(module.operation)

    # Strip the gpu.module wrapper so mlir-translate sees a plain module.
    lowered = str(module)
    m = re.search(
        r"gpu\.module @\w+ attributes \{([^}]+)\} \{(.+)\}\s*\}$",
        lowered,
        re.DOTALL,
    )
    if not m:
        raise RuntimeError("failed to extract gpu.module body")
    return f"module attributes {{{m.group(1)}}} {{\n{m.group(2).strip()}\n}}"


# ---------------------------------------------------------------------------
# 3 & 4.  Translate to LLVM IR and compile to AMDGPU asm
# ---------------------------------------------------------------------------
MLIR_TRANSLATE = shutil.which("mlir-translate-20") or shutil.which("mlir-translate")
LLC = shutil.which("llc-20") or shutil.which("llc")


def _pipe(tool, args, text):
    if tool is None:
        return None
    r = subprocess.run(
        [tool] + args, input=text, capture_output=True, text=True
    )
    if r.returncode != 0:
        print(f"[{tool}] {r.stderr.strip()}", file=sys.stderr)
        return None
    return r.stdout


def translate_to_llvmir(mlir_text: str) -> str | None:
    return _pipe(MLIR_TRANSLATE, ["--mlir-to-llvmir"], mlir_text)


def compile_to_asm(llvm_ir: str, chip: str = "gfx942") -> str | None:
    return _pipe(LLC, ["-march=amdgcn", f"-mcpu={chip}", "-"], llvm_ir)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    chip = sys.argv[1] if len(sys.argv) > 1 else "gfx942"

    # --- high-level IR ---
    module = build_vector_add_module()
    banner = lambda t: print(f"\n{'=' * 60}\n{t}\n{'=' * 60}")

    banner(f"HIGH-LEVEL MLIR  (gpu + memref + arith + scf + rocdl)")
    print(module)

    # --- lower ---
    lowered = lower_to_llvm(module, chip)
    banner("LOWERED MLIR  (llvm + rocdl)")
    print(lowered)

    # --- LLVM IR ---
    llvm_ir = translate_to_llvmir(lowered)
    if llvm_ir:
        banner("LLVM IR")
        print(llvm_ir)

    # --- AMDGPU asm ---
    if llvm_ir:
        asm = compile_to_asm(llvm_ir, chip)
        if asm:
            banner(f"AMDGPU ASSEMBLY  ({chip})")
            print(asm)


if __name__ == "__main__":
    main()
