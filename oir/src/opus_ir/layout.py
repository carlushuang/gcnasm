"""opus layouts: strided index descriptors with an optional x/p/y distribution view.

This mirrors opus exactly and adds nothing: ``offset = sum(index[i] * stride[i])`` over a
flattened shape, where each x-dim (tensor axis) is split into an ordered group of ``p`` dims
(distributed across lanes/waves) and ``y`` dims (held inside one lane). There is no composition,
complement, or swizzle algebra; that is a deliberate opus non-goal.

Lowering of a ``Layout``:

* all-``y`` with no p-coord -> ``opus::make_layout(shape, stride)``
* otherwise -> ``opus::make_layout(shape, opus::unfold_x_stride(dim, shape, x_stride),
  opus::unfold_p_coord(dim, p_coord))`` with ``dim`` spelled as nested ``p_dim``/``y_dim`` tuples.

Lowering of an ``MmaLayout``: ``opus::partition_layout_{a,b,c}(mma, x_stride[, p_coord])``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from functools import reduce
from operator import mul
from typing import Any, Dict, Optional, Tuple

from .expr import Expr, ExprLike


class PY(str, Enum):
    P = "p"  # opus::p_dim -- across threads
    Y = "y"  # opus::y_dim -- within a thread


@dataclass(frozen=True)
class Layout:
    name: str
    shape: Tuple[int, ...]  # flattened, opus order; every entry is static
    dims: Tuple[Tuple[PY, ...], ...]  # one group per x-dim, covering ``shape`` in order
    x_stride: Tuple[Expr, ...]  # one stride per x-dim, in elements
    p_coord: Tuple[Expr, ...] = ()  # one coordinate per ``p`` entry, in order

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(int(s) for s in self.shape))
        object.__setattr__(self, "dims", tuple(tuple(PY(d) for d in g) for g in self.dims))
        object.__setattr__(self, "x_stride", tuple(Expr.of(s) for s in self.x_stride))
        object.__setattr__(self, "p_coord", tuple(Expr.of(c) for c in self.p_coord))

    @classmethod
    def packed(cls, name: str, shape: Sequence[int]) -> Layout:
        """``opus::make_layout(shape...)``: row-major, single-thread view."""
        strides = [reduce(mul, shape[i + 1 :], 1) for i in range(len(shape))]
        return cls(name, tuple(shape), tuple((PY.Y,) for _ in shape), tuple(strides))

    @classmethod
    def strided(cls, name: str, shape: Sequence[int], stride: Sequence[ExprLike]) -> Layout:
        """``opus::make_layout(shape, stride)``."""
        return cls(name, tuple(shape), tuple((PY.Y,) for _ in shape), tuple(stride))

    @property
    def flat_dims(self) -> Tuple[PY, ...]:
        return tuple(d for g in self.dims for d in g)

    @property
    def y_shape(self) -> Tuple[int, ...]:
        return tuple(s for s, d in zip(self.shape, self.flat_dims) if d is PY.Y)

    @property
    def p_shape(self) -> Tuple[int, ...]:
        return tuple(s for s, d in zip(self.shape, self.flat_dims) if d is PY.P)

    @property
    def x_shape(self) -> Tuple[int, ...]:
        out, i = [], 0
        for group in self.dims:
            out.append(reduce(mul, self.shape[i : i + len(group)], 1))
            i += len(group)
        return tuple(out)

    @property
    def elems_per_thread(self) -> int:
        return reduce(mul, self.y_shape, 1)

    @property
    def threads_spanned(self) -> int:
        return reduce(mul, self.p_shape, 1)

    @property
    def distributed(self) -> bool:
        return bool(self.p_coord) or PY.P in self.flat_dims

    @property
    def innermost_y(self) -> Optional[int]:
        """Extent of the last flattened dim when it is a ``y`` dim (the vectorizable run)."""
        if self.flat_dims and self.flat_dims[-1] is PY.Y:
            return self.shape[-1]
        return None

    def unfolded_strides(self, env: Dict[str, int]) -> Optional[Tuple[int, ...]]:
        """Per flattened dim stride (opus ``unfold_x_stride``); None if a stride is not computable."""
        out, i = [], 0
        for group, xs in zip(self.dims, self.x_stride):
            base = xs.evaluate(env)
            if base is None:
                return None
            extents = self.shape[i : i + len(group)]
            out += [base * reduce(mul, extents[j + 1 :], 1) for j in range(len(group))]
            i += len(group)
        return tuple(out)

    def offset(self, y_index: Sequence[int], env: Dict[str, int]) -> Optional[int]:
        """Element offset of y-coordinate ``y_index`` for the lane described by ``env``."""
        strides = self.unfolded_strides(env)
        coords = [c.evaluate(env) for c in self.p_coord]
        if strides is None or any(c is None for c in coords):
            return None
        p_it, y_it, total = iter(coords), iter(y_index), 0
        for d, stride in zip(self.flat_dims, strides):
            total += (next(p_it) if d is PY.P else next(y_it)) * stride
        return total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "layout",
            "name": self.name,
            "shape": list(self.shape),
            "dims": [[d.value for d in g] for g in self.dims],
            "x_stride": [s.to_json() for s in self.x_stride],
            "p_coord": [c.to_json() for c in self.p_coord],
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> Layout:
        return cls(
            name=value["name"],
            shape=tuple(value["shape"]),
            dims=tuple(tuple(g) for g in value["dims"]),
            x_stride=tuple(Expr.from_json(s) for s in value["x_stride"]),
            p_coord=tuple(Expr.from_json(c) for c in value.get("p_coord", ())),
        )


class Operand(str, Enum):
    A = "a"
    B = "b"
    C = "c"


@dataclass(frozen=True)
class MmaLayout:
    """The per-lane operand layout an opus mma adaptor derives (``partition_layout_{a,b,c}``).

    Shape, dims and p-coord come from the adaptor; only the x-stride of the backing tensor (and
    optionally an explicit p-coord) is committed here. ``packed=True`` selects
    ``partition_layout_{a,b,c}_packed`` (contiguous register/LDS image, no x-stride).
    """

    name: str
    mma: str
    operand: Operand
    x_stride: Tuple[Expr, ...] = ()
    p_coord: Tuple[Expr, ...] = ()
    packed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "operand", Operand(self.operand))
        object.__setattr__(self, "x_stride", tuple(Expr.of(s) for s in self.x_stride))
        object.__setattr__(self, "p_coord", tuple(Expr.of(c) for c in self.p_coord))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "mma_layout",
            "name": self.name,
            "mma": self.mma,
            "operand": self.operand.value,
            "x_stride": [s.to_json() for s in self.x_stride],
            "p_coord": [c.to_json() for c in self.p_coord],
            "packed": self.packed,
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> MmaLayout:
        return cls(
            name=value["name"],
            mma=value["mma"],
            operand=Operand(value["operand"]),
            x_stride=tuple(Expr.from_json(s) for s in value.get("x_stride", ())),
            p_coord=tuple(Expr.from_json(c) for c in value.get("p_coord", ())),
            packed=bool(value.get("packed", False)),
        )


def layout_from_dict(value: Dict[str, Any]):
    if value.get("kind") == "mma_layout":
        return MmaLayout.from_dict(value)
    return Layout.from_dict(value)
