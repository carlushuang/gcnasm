# SPDX-License-Identifier: MIT
"""Register-pinning hints for FlyDSL kernels (no inline asm).

These mirror the C++ `amdgpu_pin_vgpr` / `amdgpu_pin_agpr` attribute: they lower
to the `llvm.amdgcn.pin.{vgpr,agpr}` intrinsics, which the AMDGPU backend
(SIPreColorPins + the SIFoldOperands AGPR-load fold) turns into

    - a value born in / held in the requested register file, and
    - for an MFMA input pinned to AGPR, the mixed  v_mfma v[C], a[A], a[B]  form
      (the accumulator stays in VGPR).

Requires a FlyDSL built against an LLVM that carries the pin patch
(flydsl-llvm-pin.patch); otherwise `llvm.call_intrinsic` rejects the unknown
intrinsic name. See README.md.

The result MUST be consumed (fed to the MMA / stored back). An unused pin is
dead-code-eliminated, exactly like the C++ builtin.
"""

from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir import ir
import flydsl.expr as fx


def _unwrap(x):
    return x.value if hasattr(x, "value") and isinstance(x.value, ir.Value) else x


def _pin_value(v, regno, agpr):
    """Pin one LLVM-typed SSA vector value; returns the pinned value.

    The pin intrinsic is only selectable for i32-based widths (i32/v2i32/...), so
    bitcast to <dwords x i32> around the pin (a float/half base would need
    v2f32/v4f16, which have no pattern)."""
    from flydsl._mlir.ir import IntegerType, VectorType
    n = fx.arith.unwrap(fx.arith.constant(int(regno), type=fx.typing.T.i32))
    name = "llvm.amdgcn.pin.agpr" if agpr else "llvm.amdgcn.pin.vgpr"
    ty = v.type
    vt = _cast(VectorType, ty)
    bits = vt.shape[0] * _bitwidth(vt.element_type) if vt else _bitwidth(ty)
    dwords = bits // 32
    i32 = IntegerType.get_signless(32)
    i32ty = i32 if dwords == 1 else VectorType.get([dwords], i32)
    vi = _llvm.bitcast(i32ty, v)
    pinned = _llvm.call_intrinsic(i32ty, name, [vi, n], [], [])
    return _llvm.bitcast(ty, pinned)


def _cast(TypeCls, t):
    try:
        return TypeCls(t)
    except (ValueError, TypeError):
        return None


def _bitwidth(t):
    from flydsl._mlir.ir import IntegerType, F16Type, F32Type, BF16Type
    it = _cast(IntegerType, t)
    if it:
        return it.width
    for T, w in ((F16Type, 16), (BF16Type, 16), (F32Type, 32)):
        if _cast(T, t):
            return w
    raise TypeError(f"pin: unknown element type width for {t}")


def _pin(frag_or_value, regno, agpr):
    # Bare SSA value: pin directly.
    if isinstance(frag_or_value, ir.Value):
        return _pin_value(frag_or_value, regno, agpr)
    # FlyDSL fragment = a register-space memref/tile, not an SSA value. Load its
    # register vector, pin it, and store it back, then hand the same fragment to
    # the MMA. load->pin->store is value-identical but routes the tile's contents
    # through the pin so the backend places them in the requested register file.
    frag = frag_or_value
    v = _unwrap(frag.load())
    frag.store(_pin_value(v, regno, agpr))
    return frag


def pin_agpr(value, regno):
    """Pin `value` to the AGPR file starting at AGPR `regno`."""
    return _pin(value, regno, agpr=True)


def pin_vgpr(value, regno):
    """Pin `value` to the VGPR file starting at VGPR `regno`."""
    return _pin(value, regno, agpr=False)
