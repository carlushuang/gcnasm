"""Scalar, symbolic dimension, and tensor types, spelled exactly as opus spells them."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from functools import reduce
from operator import mul
from typing import Any, Dict, Optional, Tuple, Union


class DType(str, Enum):
    """One entry per opus ``REGISTER_DTYPE`` / ``OPUS_DEFINE_{F,D}PACKS`` type.

    The value is the opus base name; the C++ type is ``opus::<value>_t``.
    """

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"  # E4M3; FNUZ on gfx942, OCP elsewhere (see Target.fp8_flavor)
    BF8 = "bf8"  # E5M2; FNUZ on gfx942, OCP elsewhere
    FP4 = "fp4"  # E2M1, packed two per byte
    E8M0 = "e8m0"  # MX block scale, exponent only
    I64 = "i64"
    U64 = "u64"
    I32 = "i32"
    U32 = "u32"
    I16 = "i16"
    U16 = "u16"
    I8 = "i8"
    U8 = "u8"
    INT4 = "int4"  # packed two per byte
    UINT4 = "uint4"  # packed two per byte

    @property
    def bits(self) -> int:
        return _BITS[self]

    @property
    def packed(self) -> bool:
        """Sub-byte opus ``packs`` types; never byte-addressable as a scalar."""
        return self.bits < 8

    @property
    def is_float(self) -> bool:
        return self in _FLOATS

    @property
    def cpp(self) -> str:
        return "opus::%s_t" % self.value

    def nbytes(self, count: int) -> int:
        """Storage bytes of ``count`` elements; packed types round up to whole bytes."""
        return (count * self.bits + 7) // 8


_BITS = {
    DType.FP32: 32,
    DType.FP16: 16,
    DType.BF16: 16,
    DType.FP8: 8,
    DType.BF8: 8,
    DType.FP4: 4,
    DType.E8M0: 8,
    DType.I64: 64,
    DType.U64: 64,
    DType.I32: 32,
    DType.U32: 32,
    DType.I16: 16,
    DType.U16: 16,
    DType.I8: 8,
    DType.U8: 8,
    DType.INT4: 4,
    DType.UINT4: 4,
}
_FLOATS = {DType.FP32, DType.FP16, DType.BF16, DType.FP8, DType.BF8, DType.FP4, DType.E8M0}


@dataclass(frozen=True)
class Symbol:
    """A bounded runtime integer (a kernel argument).

    Bounds are a correctness contract, not a hint: verification uses them to prove LDS capacity,
    loop trip counts, and instruction divisibility.
    """

    name: str
    minimum: int = 1
    maximum: Optional[int] = None
    multiple_of: int = 1

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError("symbol name must be an identifier: %r" % self.name)
        if self.minimum < 0:
            raise ValueError("symbol minimum must be non-negative")
        if self.maximum is not None and self.maximum < self.minimum:
            raise ValueError("symbol maximum must be >= minimum")
        if self.multiple_of <= 0:
            raise ValueError("symbol multiple_of must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "symbol",
            "name": self.name,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "multiple_of": self.multiple_of,
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> Symbol:
        return cls(
            name=str(value["name"]),
            minimum=int(value.get("minimum", 1)),
            maximum=None if value.get("maximum") is None else int(value["maximum"]),
            multiple_of=int(value.get("multiple_of", 1)),
        )


Dim = Union[int, Symbol]


def dim_to_json(dim: Dim) -> Any:
    return dim if isinstance(dim, int) else dim.to_dict()


def dim_from_json(value: Any) -> Dim:
    if isinstance(value, int):
        return value
    if isinstance(value, dict) and value.get("kind") == "symbol":
        return Symbol.from_dict(value)
    raise ValueError("unknown dimension encoding: %r" % (value,))


def dim_max(dim: Dim) -> Optional[int]:
    return dim if isinstance(dim, int) else dim.maximum


def dim_multiple_of(dim: Dim, divisor: int) -> bool:
    if isinstance(dim, int):
        return dim % divisor == 0
    return dim.multiple_of % divisor == 0


def static_product(dims: Iterable[Dim]) -> Optional[int]:
    values = list(dims)
    if any(not isinstance(dim, int) for dim in values):
        return None
    return reduce(mul, values, 1)


@dataclass(frozen=True)
class TensorType:
    dtype: DType
    shape: Tuple[Dim, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", DType(self.dtype))
        object.__setattr__(self, "shape", tuple(self.shape))
        if not self.shape:
            raise ValueError("tensor rank must be at least one")
        for dim in self.shape:
            if not isinstance(dim, (int, Symbol)):
                raise TypeError("tensor dimensions must be int or Symbol")
            if isinstance(dim, int) and dim <= 0:
                raise ValueError("static tensor dimensions must be positive")

    @property
    def rank(self) -> int:
        return len(self.shape)

    def static_numel(self) -> Optional[int]:
        return static_product(self.shape)

    def maximum_numel(self) -> Optional[int]:
        extents = [dim_max(dim) for dim in self.shape]
        if any(extent is None for extent in extents):
            return None
        return reduce(mul, extents, 1)

    def symbols(self) -> Iterable[Symbol]:
        return (dim for dim in self.shape if isinstance(dim, Symbol))

    def to_dict(self) -> Dict[str, Any]:
        return {"dtype": self.dtype.value, "shape": [dim_to_json(dim) for dim in self.shape]}

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> TensorType:
        return cls(DType(value["dtype"]), tuple(dim_from_json(dim) for dim in value["shape"]))
