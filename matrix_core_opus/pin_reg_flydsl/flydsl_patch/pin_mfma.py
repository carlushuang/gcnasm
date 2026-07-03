# SPDX-License-Identifier: Apache-2.0
"""Pin rocdl.mfma A/B operands into the AGPR file (opt-in).

Runs just before gpu-module-to-binary, on the lowered LLVM/rocdl module: wraps
each rocdl.mfma src0/src1 with llvm.amdgcn.pin.agpr (bitcast to i32-based widths,
which are the selectable pin overloads). The AMDGPU backend (SIPreColorPins)
then loads A/B directly into AGPR and, keeping the accumulator in VGPR, emits
v_mfma v[C], a[A], a[B]. Enabled by env FLYDSL_PIN_MFMA_AGPR=1 (bases overridable
via FLYDSL_PIN_A_AGPR / FLYDSL_PIN_B_AGPR).

This lives at the LLVM stage on purpose: a raw llvm.call_intrinsic inserted into
the fly kernel body does not survive the fly->rocdl lowering, and tiled_mma
fragments are register-space fly.memrefs (not SSA values) at trace time.
"""
import os
from .._mlir import ir
from .._mlir.dialects import llvm as _llvm


def _cast(TypeCls, t):
    try:
        return TypeCls(t)
    except (ValueError, TypeError):
        return None


def _elt_bits(t):
    it = _cast(ir.IntegerType, t)
    if it:
        return it.width
    for T, w in ((ir.F16Type, 16), (ir.BF16Type, 16), (ir.F32Type, 32)):
        if _cast(T, t):
            return w
    raise TypeError(f"pin: unknown element width for {t}")


def _dwords(ty):
    vt = _cast(ir.VectorType, ty)
    if vt is None:
        return _elt_bits(ty) // 32
    return vt.shape[0] * _elt_bits(vt.element_type) // 32


def _pinnable_direct(ty):
    """True if `ty` has a direct llvm.amdgcn.pin.* selection pattern (no bitcast
    needed): i32/f32 scalars and v{2,4,8,16}i32 / v{4,8,16}f32 / v8f16."""
    vt = _cast(ir.VectorType, ty)
    if vt is None:
        return _cast(ir.IntegerType, ty) is not None and ir.IntegerType(ty).width == 32 \
               or _cast(ir.F32Type, ty) is not None
    n = vt.shape[0]
    et = vt.element_type
    if _cast(ir.IntegerType, et) and ir.IntegerType(et).width == 32:
        return n in (2, 4, 8, 16)
    if _cast(ir.F32Type, et):
        return n in (4, 8, 16)
    if _cast(ir.F16Type, et) or _cast(ir.BF16Type, et):
        return n == 8
    return False


def _pin_operand(mfma, idx, regno):
    v = mfma.operands[idx]
    ty = v.type
    i32 = ir.IntegerType.get_signless(32)
    with ir.InsertionPoint(mfma), mfma.location:
        if _pinnable_direct(ty):
            back = _llvm.call_intrinsic(
                ty, "llvm.amdgcn.pin.agpr",
                [v, _llvm.mlir_constant(ir.IntegerAttr.get(i32, regno))], [], [])
        else:
            dw = _dwords(ty)
            if dw < 1:
                return
            i32ty = i32 if dw == 1 else ir.VectorType.get([dw], i32)
            vi = _llvm.bitcast(i32ty, v)
            reg = _llvm.mlir_constant(ir.IntegerAttr.get(i32, regno))
            pinned = _llvm.call_intrinsic(i32ty, "llvm.amdgcn.pin.agpr", [vi, reg], [], [])
            back = _llvm.bitcast(ty, pinned)
    mfma.operands[idx] = back

def pin_all_mfma_agpr(module):
    a_base = int(os.environ.get("FLYDSL_PIN_A_AGPR", "0"))
    b_base = int(os.environ.get("FLYDSL_PIN_B_AGPR", "64"))
    found = []

    def rec(op):
        for region in op.regions:
            for block in region.blocks:
                for o in block.operations:
                    if o.operation.name.startswith("rocdl.mfma"):
                        found.append(o.operation)
                    rec(o.operation)

    rec(module.operation)
    for o in found:
        _pin_operand(o, 0, a_base)
        _pin_operand(o, 1, b_base)
    return len(found)


# ---------------------------------------------------------------------------
# Codegen the device module via llc+lld (which honors the pin) instead of the
# in-process gpu-module-to-binary serializer (whose optimizeLlvm drops the pin).
# Enabled by FLYDSL_PIN_CODEGEN_LLC=1; tools from FLYDSL_PATCHED_LLC (patched llc)
# and ROCm's ld.lld. Builds a gpu.binary with the raw code object and splices it.
# ---------------------------------------------------------------------------
import subprocess, tempfile, os as _os
from pathlib import Path


def _single(module, name):
    for op in module.body.operations:
        if op.operation.name == name:
            return op
    raise RuntimeError(f"no top-level {name}")


def codegen_via_llc(module, llir, chip="gfx950"):
    llc = _os.environ.get("FLYDSL_PATCHED_LLC",
                          "/home/carhuang/llvm-pin/flydsl-llvm/build/bin/llc")
    lld = _os.environ.get("FLYDSL_LLD", "/opt/rocm/llvm/bin/ld.lld")
    gpu_mod = _single(module, "gpu.module")
    sym = ir.SymbolTable.get_symbol_name(gpu_mod.operation)  # e.g. @kernels
    name = str(sym).lstrip("@").strip('"')
    with tempfile.TemporaryDirectory(prefix="flydsl_llc_") as d:
        d = Path(d)
        (d / "dev.ll").write_text(llir)
        subprocess.run([llc, "-O3", "-mtriple=amdgcn-amd-amdhsa", f"-mcpu={chip}",
                        "-filetype=obj", str(d / "dev.ll"), "-o", str(d / "dev.o")],
                       check=True, capture_output=True, text=True)
        subprocess.run([lld, "-shared", str(d / "dev.o"), "-o", str(d / "dev.hsaco")],
                       check=True, capture_output=True, text=True)
        obj = (d / "dev.hsaco").read_bytes()
        if _os.environ.get("FLYDSL_PIN_DUMP_HSACO"):
            objdump = _os.path.join(_os.path.dirname(llc), "llvm-objdump")
            try:
                dis = subprocess.run([objdump, "-d", f"--mcpu={chip}", str(d / "dev.hsaco")],
                                     capture_output=True, text=True).stdout
                for ln in dis.splitlines():
                    if "v_mfma" in ln or "buffer_load" in ln or "v_accvgpr" in ln:
                        print("[flydsl.compile][ISA]", ln.strip().split("//")[0].strip())
            except Exception as _e:
                print("objdump failed:", _e)
    esc = "".join("\\%02X" % b for b in obj)
    text = ('module attributes {gpu.container_module} { gpu.binary @%s '
            '[#gpu.object<#rocdl.target<chip = "%s">, "%s">] }' % (name, chip, esc))
    ext = ir.Module.parse(text, module.context)
    ext_bin = _single(ext, "gpu.binary")
    ir.InsertionPoint(gpu_mod).insert(ext_bin.operation.clone())
    gpu_mod.operation.erase()
