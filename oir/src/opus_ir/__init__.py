"""opus-ir: a typed, hardware-explicit schedule IR whose vocabulary is opus.hpp (ROCm/aiter)."""

from .builder import KernelBuilder, kernel
from .diagnostics import Diagnostic, Severity, VerificationError, VerificationResult
from .expr import Expr, HwId, block_x, block_y, block_z, cdiv, const, emax, emin, hw, lane, sym, var, wave
from .ir import (
    SCHEMA_VERSION,
    Access,
    Kernel,
    Lds,
    Mma,
    MmaKind,
    Multicast,
    NamedBarrier,
    Op,
    Param,
    Pipeline,
    Reg,
    Role,
    RoleKind,
    Semaphore,
    Tdm,
)
from .layout import PY, Layout, MmaLayout, Operand
from .passes.memory import plan_lds
from .targets import Counter, Feature, MmaInst, Target, all_targets, get_target
from .types import DType, Symbol, TensorType
from .verifier import verify

thread_x = hw(HwId.THREAD_X)
P, Y = PY.P, PY.Y

__all__ = [name for name in dir() if not name.startswith("_")]
