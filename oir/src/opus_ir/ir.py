"""Canonical, JSON-serializable opus schedule IR.

Every resource and every op names the single opus construct it lowers to (see docs/ir_spec.md).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from functools import reduce
from operator import mul
from typing import Any, Dict, Optional, Tuple, Union

from .expr import Expr
from .layout import Layout, MmaLayout, layout_from_dict
from .targets import Target, get_target
from .types import DType, Symbol, TensorType

SCHEMA_VERSION = 1


class Access(str, Enum):
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"


class RoleKind(str, Enum):
    GENERIC = "generic"
    PRODUCER = "producer"
    CONSUMER = "consumer"
    MMA = "mma"
    EPILOGUE = "epilogue"


class MmaKind(str, Enum):
    MFMA = "mfma"  # opus::make_mfma / make_tiled_mma over mfma
    WMMA = "wmma"  # opus::make_wmma / make_tiled_mma over wmma


class Multicast(str, Enum):
    NONE = "none"  # tdm_traits::multicast<> (no peers)
    ALONG_X = "x"  # tdm_traits::peers_along_x()
    ALONG_Y = "y"  # tdm_traits::peers_along_y()


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, Expr):
        return value.to_json()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def _tuple(value: Any) -> Any:
    return tuple(_tuple(v) for v in value) if isinstance(value, list) else value


@dataclass(frozen=True)
class Param:
    """A global tensor argument -> ``opus::make_gmem(ptr[, size_bytes])``.

    ``oob=True`` passes the buffer size so out-of-bounds lanes read 0 / drop stores in hardware
    (buffer-resource range checking); it is the opus way to express tail handling.
    """

    name: str
    type: TensorType
    access: Access = Access.READ
    oob: bool = True
    alignment: int = 16

    def __post_init__(self) -> None:
        object.__setattr__(self, "access", Access(self.access))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type.to_dict(),
            "access": self.access.value,
            "oob": self.oob,
            "alignment": self.alignment,
        }

    @classmethod
    def from_dict(cls, v: Dict[str, Any]) -> Param:
        return cls(v["name"], TensorType.from_dict(v["type"]), v["access"], v["oob"], v["alignment"])


@dataclass(frozen=True)
class Lds:
    """A shared-memory region -> ``opus::make_smem(ptr)`` at a planned byte offset.

    ``shape`` is one stage; ``stages`` copies are laid out back to back. ``pad`` appends that
    many elements to each innermost row (the opus bank-conflict recipe) and is part of the stage
    footprint.
    """

    name: str
    dtype: DType
    shape: Tuple[int, ...]
    stages: int = 1
    pad: int = 0
    offset_bytes: Optional[int] = None
    alignment: int = 16
    alias_of: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", DType(self.dtype))
        object.__setattr__(self, "shape", tuple(int(s) for s in self.shape))

    @property
    def stage_elems(self) -> int:
        rows = reduce(mul, self.shape[:-1], 1)
        return rows * (self.shape[-1] + self.pad)

    @property
    def stage_bytes(self) -> int:
        return self.dtype.nbytes(self.stage_elems)

    @property
    def total_bytes(self) -> int:
        return self.stage_bytes * self.stages

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype.value,
            "shape": list(self.shape),
            "stages": self.stages,
            "pad": self.pad,
            "offset_bytes": self.offset_bytes,
            "alignment": self.alignment,
            "alias_of": self.alias_of,
        }

    @classmethod
    def from_dict(cls, v: Dict[str, Any]) -> Lds:
        return cls(**{k: _tuple(x) for k, x in v.items()})


@dataclass(frozen=True)
class Reg:
    """A per-lane register tile -> ``opus::vector_t<T, elems>`` (or ``array`` for packed types)."""

    name: str
    dtype: DType
    elems: int
    zero_init: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", DType(self.dtype))

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "dtype": self.dtype.value, "elems": self.elems, "zero_init": self.zero_init}

    @classmethod
    def from_dict(cls, v: Dict[str, Any]) -> Reg:
        return cls(**v)


@dataclass(frozen=True)
class Mma:
    """A warp or block mma -> ``opus::make_mfma`` / ``make_wmma`` / ``make_tiled_mma``.

    ``wave`` is the single instruction shape (W_M, W_N, W_K); ``expand`` is how many times each
    wave repeats it along M/N/K (opus E_*); ``tile`` is the wave grid (opus T_*).
    """

    name: str
    kind: MmaKind
    a: DType
    b: DType
    c: DType
    wave: Tuple[int, int, int]
    expand: Tuple[int, int, int] = (1, 1, 1)
    tile: Tuple[int, int, int] = (1, 1, 1)
    swap_ab: bool = False
    scaled: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", MmaKind(self.kind))
        for f in ("a", "b", "c"):
            object.__setattr__(self, f, DType(getattr(self, f)))
        for f in ("wave", "expand", "tile"):
            object.__setattr__(self, f, tuple(int(x) for x in getattr(self, f)))

    @property
    def tiled(self) -> bool:
        return self.expand != (1, 1, 1) or self.tile != (1, 1, 1)

    @property
    def waves(self) -> int:
        return reduce(mul, self.tile, 1)

    @property
    def block_tile(self) -> Tuple[int, int, int]:
        return tuple(w * e * t for w, e, t in zip(self.wave, self.expand, self.tile))

    def lane_elems(self, warp_size: int) -> Dict[str, int]:
        """Per-lane register elements for A, B, C (opus ``vtype_a/b/c`` times the expansion)."""
        wm, wn, wk = self.wave
        em, en, ek = self.expand
        return {
            "a": wm * wk // warp_size * em * ek,
            "b": wn * wk // warp_size * en * ek,
            "c": wm * wn // warp_size * em * en,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {f: _plain(getattr(self, f)) for f in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, v: Dict[str, Any]) -> Mma:
        return cls(**{k: _tuple(x) for k, x in v.items()})


@dataclass(frozen=True)
class Tdm:
    """A gfx1250 tensor-DMA window -> ``opus::tdm<T, TileShape, Policies...>`` + ``make_tdm``.

    The window walks ``param`` in ``tile``-shaped steps and lands in ``lds``. Policies are the
    opus tdm_traits tags: padding (LDS write side, load direction only), multicast, gather, cache.
    """

    name: str
    param: str
    lds: str
    tile: Tuple[int, ...]
    pad_interval: int = 0  # elements; 0 = no padding
    pad_amount: int = 0
    multicast: Multicast = Multicast.NONE
    gather_index_size: int = 0  # 0 = no gather
    cache_policy: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tile", tuple(int(x) for x in self.tile))
        object.__setattr__(self, "multicast", Multicast(self.multicast))

    def to_dict(self) -> Dict[str, Any]:
        return {f: _plain(getattr(self, f)) for f in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, v: Dict[str, Any]) -> Tdm:
        return cls(**{k: _tuple(x) for k, x in v.items()})


@dataclass(frozen=True)
class Role:
    """A named, fixed set of waves (wave = ``thread_id_x() / warp_size``)."""

    name: str
    waves: Tuple[int, ...]
    kind: RoleKind = RoleKind.GENERIC

    def __post_init__(self) -> None:
        object.__setattr__(self, "waves", tuple(int(w) for w in self.waves))
        object.__setattr__(self, "kind", RoleKind(self.kind))

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "waves": list(self.waves), "kind": self.kind.value}

    @classmethod
    def from_dict(cls, v: Dict[str, Any]) -> Role:
        return cls(v["name"], tuple(v["waves"]), v["kind"])


@dataclass(frozen=True)
class Pipeline:
    name: str
    stages: int

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "stages": self.stages}

    @classmethod
    def from_dict(cls, v: Dict[str, Any]) -> Pipeline:
        return cls(**v)


@dataclass(frozen=True)
class NamedBarrier:
    """gfx1250 named workgroup barrier -> ``s_barrier_init_ptr`` / ``s_barrier_join_ptr``."""

    name: str
    members: Tuple[str, ...]  # roles

    def __post_init__(self) -> None:
        object.__setattr__(self, "members", tuple(self.members))

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "members": list(self.members)}

    @classmethod
    def from_dict(cls, v: Dict[str, Any]) -> NamedBarrier:
        return cls(v["name"], tuple(v["members"]))


@dataclass(frozen=True)
class Semaphore:
    """Cross-workgroup counter in global memory -> ``opus::workgroup_barrier(u32*)``."""

    name: str
    param: str

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "param": self.param}

    @classmethod
    def from_dict(cls, v: Dict[str, Any]) -> Semaphore:
        return cls(**v)


AnyLayout = Union[Layout, MmaLayout]


@dataclass(frozen=True)
class Op:
    """One opus call. ``attrs`` is stored as canonical JSON so the node is hashable and exact."""

    opcode: str
    inputs: Tuple[str, ...] = ()
    outputs: Tuple[str, ...] = ()
    role: Optional[str] = None
    attrs_json: str = "{}"
    body: Tuple[Op, ...] = ()

    @classmethod
    def create(
        cls,
        opcode: str,
        inputs: Sequence[str] = (),
        outputs: Sequence[str] = (),
        role: Optional[str] = None,
        attrs: Optional[Mapping[str, Any]] = None,
        body: Sequence[Op] = (),
    ) -> Op:
        canonical = json.dumps(_plain(dict(attrs or {})), sort_keys=True, separators=(",", ":"))
        return cls(opcode, tuple(inputs), tuple(outputs), role, canonical, tuple(body))

    @property
    def attrs(self) -> Dict[str, Any]:
        return json.loads(self.attrs_json)

    def expr(self, key: str) -> Optional[Expr]:
        value = self.attrs.get(key)
        return None if value is None else Expr.from_json(value)

    def walk(self) -> Iterable[Op]:
        yield self
        for child in self.body:
            yield from child.walk()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "opcode": self.opcode,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "role": self.role,
            "attrs": self.attrs,
            "body": [child.to_dict() for child in self.body],
        }

    @classmethod
    def from_dict(cls, v: Dict[str, Any]) -> Op:
        return cls.create(
            v["opcode"],
            v.get("inputs", ()),
            v.get("outputs", ()),
            v.get("role"),
            v.get("attrs", {}),
            [cls.from_dict(c) for c in v.get("body", ())],
        )


_COLLECTIONS = (
    ("params", Param),
    ("lds", Lds),
    ("regs", Reg),
    ("mmas", Mma),
    ("tdms", Tdm),
    ("roles", Role),
    ("pipelines", Pipeline),
    ("barriers", NamedBarrier),
    ("semaphores", Semaphore),
)


@dataclass(frozen=True)
class Kernel:
    name: str
    target: str
    block_size: int
    grid: Tuple[Expr, Expr, Expr]
    params: Tuple[Param, ...]
    symbols: Tuple[Symbol, ...] = ()
    lds: Tuple[Lds, ...] = ()
    regs: Tuple[Reg, ...] = ()
    mmas: Tuple[Mma, ...] = ()
    layouts: Tuple[AnyLayout, ...] = ()
    tdms: Tuple[Tdm, ...] = ()
    roles: Tuple[Role, ...] = ()
    pipelines: Tuple[Pipeline, ...] = ()
    barriers: Tuple[NamedBarrier, ...] = ()
    semaphores: Tuple[Semaphore, ...] = ()
    body: Tuple[Op, ...] = ()
    cluster: Optional[Tuple[int, int, int]] = None
    metadata_json: str = "{}"
    schema_version: int = field(default=SCHEMA_VERSION)

    @property
    def target_info(self) -> Target:
        return get_target(self.target)

    @property
    def num_waves(self) -> int:
        return self.block_size // self.target_info.warp_size

    @property
    def metadata(self) -> Dict[str, Any]:
        return json.loads(self.metadata_json)

    def all_ops(self) -> Iterable[Op]:
        for op in self.body:
            yield from op.walk()

    def resources(self) -> Dict[str, Any]:
        """Every named resource; names share one namespace."""
        table: Dict[str, Any] = {}
        for attr, _ in _COLLECTIONS:
            for item in getattr(self, attr):
                table[item.name] = item
        for item in self.layouts:
            table[item.name] = item
        return table

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "name": self.name,
            "target": self.target,
            "block_size": self.block_size,
            "grid": [g.to_json() for g in self.grid],
            "symbols": [s.to_dict() for s in self.symbols],
            "cluster": None if self.cluster is None else list(self.cluster),
            "layouts": [item.to_dict() for item in self.layouts],
            "body": [op.to_dict() for op in self.body],
            "metadata": self.metadata,
        }
        for attr, _ in _COLLECTIONS:
            out[attr] = [item.to_dict() for item in getattr(self, attr)]
        return out

    @classmethod
    def from_dict(cls, v: Dict[str, Any]) -> Kernel:
        version = int(v.get("schema_version", 0))
        if version != SCHEMA_VERSION:
            raise ValueError("unsupported IR schema version %d, expected %d" % (version, SCHEMA_VERSION))
        kwargs: Dict[str, Any] = {attr: tuple(kind.from_dict(item) for item in v.get(attr, ())) for attr, kind in _COLLECTIONS}
        return cls(
            name=v["name"],
            target=v["target"],
            block_size=int(v["block_size"]),
            grid=tuple(Expr.from_json(g) for g in v["grid"]),
            symbols=tuple(Symbol.from_dict(s) for s in v.get("symbols", ())),
            layouts=tuple(layout_from_dict(item) for item in v.get("layouts", ())),
            body=tuple(Op.from_dict(op) for op in v.get("body", ())),
            cluster=None if v.get("cluster") is None else tuple(v["cluster"]),
            metadata_json=json.dumps(v.get("metadata", {}), sort_keys=True, separators=(",", ":")),
            **kwargs,
        )

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
