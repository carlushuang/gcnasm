"""Integer index expressions for p-coordinates, scalar offsets, trip counts, and grid sizes.

Expressions are deliberately small: integer arithmetic over constants, kernel symbols, loop
variables, and the opus hardware-id wrappers. There is no layout algebra here; layouts live in
``layout.py`` and use these expressions only for coordinates.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, FrozenSet, Optional, Tuple, Union

from .types import Symbol


class HwId(str, Enum):
    """Hardware ids, each one opus wrapper. ``lane`` and ``thread_*`` are lane-divergent."""

    LANE = "lane"  # opus::lane_id()
    WAVE = "wave"  # thread_id_x() / warp_size (waveid_in_workgroup() on gfx1250)
    THREAD_X = "thread_x"  # opus::thread_id_x()
    THREAD_Y = "thread_y"
    THREAD_Z = "thread_z"
    BLOCK_X = "block_x"  # opus::block_id_x()
    BLOCK_Y = "block_y"
    BLOCK_Z = "block_z"
    CLUSTER_X = "cluster_x"  # opus::cluster_id_x(), gfx1250
    CLUSTER_Y = "cluster_y"
    CLUSTER_Z = "cluster_z"
    CLUSTER_WG_X = "cluster_wg_x"  # opus::cluster_workgroup_id_x(), gfx1250
    CLUSTER_WG_Y = "cluster_wg_y"
    CLUSTER_WG_Z = "cluster_wg_z"

    @property
    def divergent(self) -> bool:
        return self in {HwId.LANE, HwId.THREAD_X, HwId.THREAD_Y, HwId.THREAD_Z}

    @property
    def cluster(self) -> bool:
        return self.value.startswith("cluster")


class BinOp(str, Enum):
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"  # floor division; operands are non-negative by construction
    MOD = "mod"
    CDIV = "cdiv"  # ceil division
    MIN = "min"
    MAX = "max"


ExprLike = Union["Expr", int, Symbol]


@dataclass(frozen=True)
class Expr:
    """One node: a constant, a symbol, a loop variable, a hardware id, or a binary op."""

    kind: str  # const | sym | var | hw | bin
    value: Any = None
    op: Optional[BinOp] = None
    args: Tuple[Expr, ...] = ()

    @staticmethod
    def of(value: ExprLike) -> Expr:
        if isinstance(value, Expr):
            return value
        if isinstance(value, bool):
            raise TypeError("bool is not an index expression")
        if isinstance(value, int):
            return Expr("const", value)
        if isinstance(value, Symbol):
            return Expr("sym", value.name)
        raise TypeError("cannot convert %r to an index expression" % (value,))

    def _bin(self, op: BinOp, other: ExprLike, swap: bool = False) -> Expr:
        lhs, rhs = self, Expr.of(other)
        if swap:
            lhs, rhs = rhs, lhs
        return Expr("bin", op=op, args=(lhs, rhs))

    def __add__(self, o: ExprLike) -> Expr:
        return self._bin(BinOp.ADD, o)

    def __radd__(self, o: ExprLike) -> Expr:
        return self._bin(BinOp.ADD, o, True)

    def __sub__(self, o: ExprLike) -> Expr:
        return self._bin(BinOp.SUB, o)

    def __rsub__(self, o: ExprLike) -> Expr:
        return self._bin(BinOp.SUB, o, True)

    def __mul__(self, o: ExprLike) -> Expr:
        return self._bin(BinOp.MUL, o)

    def __rmul__(self, o: ExprLike) -> Expr:
        return self._bin(BinOp.MUL, o, True)

    def __floordiv__(self, o: ExprLike) -> Expr:
        return self._bin(BinOp.DIV, o)

    def __rfloordiv__(self, o: ExprLike) -> Expr:
        return self._bin(BinOp.DIV, o, True)

    def __mod__(self, o: ExprLike) -> Expr:
        return self._bin(BinOp.MOD, o)

    def __rmod__(self, o: ExprLike) -> Expr:
        return self._bin(BinOp.MOD, o, True)

    def walk(self) -> Iterable[Expr]:
        yield self
        for arg in self.args:
            yield from arg.walk()

    def hw_ids(self) -> FrozenSet[HwId]:
        return frozenset(HwId(node.value) for node in self.walk() if node.kind == "hw")

    def symbols(self) -> FrozenSet[str]:
        return frozenset(node.value for node in self.walk() if node.kind == "sym")

    def loop_vars(self) -> FrozenSet[str]:
        return frozenset(node.value for node in self.walk() if node.kind == "var")

    @property
    def wave_uniform(self) -> bool:
        """True when every lane of a wave computes the same value (legal as an opus ``s_os``)."""
        return not any(hw.divergent for hw in self.hw_ids())

    def evaluate(self, env: Dict[str, int]) -> Optional[int]:
        """Evaluate with ``env`` binding symbol, var, and hw names; None if anything is unbound."""
        if self.kind == "const":
            return int(self.value)
        if self.kind in ("sym", "var", "hw"):
            return env.get(self.value)
        lhs, rhs = (arg.evaluate(env) for arg in self.args)
        if lhs is None or rhs is None:
            return None
        if self.op in (BinOp.DIV, BinOp.MOD, BinOp.CDIV) and rhs == 0:
            return None
        return {
            BinOp.ADD: lambda: lhs + rhs,
            BinOp.SUB: lambda: lhs - rhs,
            BinOp.MUL: lambda: lhs * rhs,
            BinOp.DIV: lambda: lhs // rhs,
            BinOp.MOD: lambda: lhs % rhs,
            BinOp.CDIV: lambda: -(-lhs // rhs),
            BinOp.MIN: lambda: min(lhs, rhs),
            BinOp.MAX: lambda: max(lhs, rhs),
        }[self.op]()

    def to_json(self) -> Any:
        if self.kind == "const":
            return self.value
        if self.kind == "bin":
            return [self.op.value] + [arg.to_json() for arg in self.args]
        return {self.kind: self.value}

    @classmethod
    def from_json(cls, value: Any) -> Expr:
        if isinstance(value, int) and not isinstance(value, bool):
            return cls("const", value)
        if isinstance(value, list) and value:
            return cls("bin", op=BinOp(value[0]), args=tuple(cls.from_json(v) for v in value[1:]))
        if isinstance(value, dict) and len(value) == 1:
            ((kind, name),) = value.items()
            if kind == "hw":
                HwId(name)
            if kind in ("sym", "var", "hw"):
                return cls(kind, name)
        raise ValueError("unknown expression encoding: %r" % (value,))

    def __str__(self) -> str:
        if self.kind == "const":
            return str(self.value)
        if self.kind == "bin":
            return "%s(%s)" % (self.op.value, ", ".join(str(arg) for arg in self.args))
        return "%s:%s" % (self.kind, self.value)


def const(value: int) -> Expr:
    return Expr("const", value)


def sym(symbol: Union[Symbol, str]) -> Expr:
    return Expr("sym", symbol if isinstance(symbol, str) else symbol.name)


def var(name: str) -> Expr:
    return Expr("var", name)


def hw(hw_id: Union[HwId, str]) -> Expr:
    return Expr("hw", HwId(hw_id).value)


def cdiv(a: ExprLike, b: ExprLike) -> Expr:
    return Expr.of(a)._bin(BinOp.CDIV, b)


def emin(a: ExprLike, b: ExprLike) -> Expr:
    return Expr.of(a)._bin(BinOp.MIN, b)


def emax(a: ExprLike, b: ExprLike) -> Expr:
    return Expr.of(a)._bin(BinOp.MAX, b)


lane = hw(HwId.LANE)
wave = hw(HwId.WAVE)
block_x = hw(HwId.BLOCK_X)
block_y = hw(HwId.BLOCK_Y)
block_z = hw(HwId.BLOCK_Z)
