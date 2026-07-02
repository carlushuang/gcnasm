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


def _to_value(x):
    """Best-effort extraction of the underlying MLIR ir.Value from a FlyDSL SSA
    wrapper / fragment. FlyDSL exposes either a bare ir.Value, a `.value`, or
    `__extract_to_ir_values__()`."""
    if isinstance(x, ir.Value):
        return x
    if hasattr(x, "value") and isinstance(x.value, ir.Value):
        return x.value
    if hasattr(x, "__extract_to_ir_values__"):
        vs = x.__extract_to_ir_values__()
        if len(vs) == 1:
            return vs[0]
        raise ValueError("pin: multi-value fragment; pin each register value")
    raise TypeError(f"pin: cannot get an ir.Value from {type(x)}")


def _pin(value, regno, agpr):
    v = _to_value(value)
    n = fx.arith.unwrap(fx.arith.constant(int(regno), type=fx.typing.T.i32))
    name = "llvm.amdgcn.pin.agpr" if agpr else "llvm.amdgcn.pin.vgpr"
    # call_intrinsic mangles the overload from v.type; if the binding needs an
    # explicit suffix, pass e.g. name + ".v4i32".
    return _llvm.call_intrinsic(v.type, name, [v, n], [], [])


def pin_agpr(value, regno):
    """Pin `value` to the AGPR file starting at AGPR `regno`."""
    return _pin(value, regno, agpr=True)


def pin_vgpr(value, regno):
    """Pin `value` to the VGPR file starting at VGPR `regno`."""
    return _pin(value, regno, agpr=False)
