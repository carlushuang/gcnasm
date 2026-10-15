"""Exact AMDGPU target contracts, transcribed from the arch guards in opus.hpp.

A target never silently degrades to another. Every table here cites the opus construct it
mirrors, so a change to opus.hpp has exactly one place to be reflected.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, FrozenSet, Optional, Tuple

from .types import DType


class Feature(str, Enum):
    MFMA = "mfma"  # opus::mfma
    MFMA_SCALE = "mfma_scale"  # mfma operator()(a, b, c, scale_a, scale_b), gfx950
    WMMA = "wmma"  # opus::wmma
    WMMA_SCALE = "wmma_scale"  # wmma scaled overloads, gfx1250
    BUFFER_LOAD_LDS = "buffer_load_lds"  # gmem::async_load on CDNA (buffer_load ... lds)
    GLOBAL_LOAD_ASYNC_LDS = "global_load_async_lds"  # gmem::async_load on gfx1250
    DS_LOAD_TR = "ds_load_tr"  # smem::tr_load, gfx950 / gfx1250
    TDM = "tdm"  # opus::tdm tensor-DMA window, gfx1250
    CLUSTER = "cluster"  # cluster ids + sync_cluster(), gfx1250
    NAMED_BARRIER = "named_barrier"  # s_barrier_init_ptr / join, gfx1250
    ATOMIC_FADD_F32 = "atomic_fadd_f32"
    ATOMIC_PK_ADD_F16 = "atomic_pk_add_f16"
    ATOMIC_PK_ADD_BF16 = "atomic_pk_add_bf16"
    ATOMIC_FMINMAX_F32 = "atomic_fminmax_f32"


class Counter(str, Enum):
    """Hardware wait counters, each one opus wait wrapper."""

    VM = "vm"  # gfx9 s_waitcnt vmcnt   -> opus::s_waitcnt_vmcnt
    LGKM = "lgkm"  # gfx9 s_waitcnt lgkmcnt -> opus::s_waitcnt_lgkmcnt
    LOAD = "load"  # gfx1250 -> opus::s_wait_loadcnt
    STORE = "store"  # gfx1250 -> opus::s_wait_storecnt
    DS = "ds"  # gfx1250 -> opus::s_wait_dscnt
    KM = "km"  # gfx1250 -> opus::s_wait_kmcnt
    ASYNC = "async"  # gfx1250 -> opus::s_wait_asynccnt
    TENSOR = "tensor"  # gfx1250 -> opus::s_wait_tensorcnt


class MemClass(str, Enum):
    """What kind of memory traffic an op issues; the target maps it to a counter."""

    GMEM_LOAD = "gmem_load"
    GMEM_STORE = "gmem_store"
    GMEM_ASYNC_LDS = "gmem_async_lds"
    LDS = "lds"
    TENSOR_DMA = "tensor_dma"


@dataclass(frozen=True)
class MmaInst:
    """One row of an opus mfma/wmma dispatch table."""

    kind: str  # mfma | wmma
    a: DType
    b: DType
    c: DType
    m: int
    n: int
    k: int
    scaled: bool = False
    chained_k: Optional[int] = None  # opus *_STEP_K_ dispatch: issues k/chained_k instructions

    @property
    def key(self) -> Tuple[str, str, str, str, int, int, int, bool]:
        return (self.kind, self.a.value, self.b.value, self.c.value, self.m, self.n, self.k, self.scaled)


@dataclass(frozen=True, eq=False)
class Target:
    name: str  # exact --offload-arch
    warp_size: int  # opus::get_warp_size()
    lds_bytes: int  # opus::get_smem_size()
    features: FrozenSet[Feature]
    counters: Dict[Counter, int]  # counter -> maximum encodable outstanding count
    counter_map: Dict[MemClass, Counter]
    async_lds_bytes: FrozenSet[int]  # legal per-lane sizes for gmem::async_load
    mma: Tuple[MmaInst, ...]
    fp8_flavor: str  # "fnuz" | "ocp", numeric_limits<fp8_t>
    max_threads_per_block: int = 1024
    tdm_in_flight: int = 0  # opus tdm ring-depth note: tensor ops in flight per wave
    notes: Tuple[str, ...] = ()

    def supports(self, feature: Feature) -> bool:
        return feature in self.features

    def counter_for(self, mem: MemClass) -> Optional[Counter]:
        return self.counter_map.get(mem)

    def find_mma(self, kind: str, a: DType, b: DType, c: DType, m: int, n: int, k: int, scaled: bool) -> Optional[MmaInst]:
        key = (kind, a.value, b.value, c.value, m, n, k, scaled)
        return next((inst for inst in self.mma if inst.key == key), None)

    @property
    def hipcc_arch_flag(self) -> str:
        return "--offload-arch=%s" % self.name


F32, F16, BF16 = DType.FP32, DType.FP16, DType.BF16
FP8, BF8, FP4 = DType.FP8, DType.BF8, DType.FP4


def _mfma(a, b, c, m, n, k, scaled=False, chained_k=None):
    return MmaInst("mfma", a, b, c, m, n, k, scaled, chained_k)


def _wmma(a, b, c, m, n, k, scaled=False, chained_k=None):
    return MmaInst("wmma", a, b, c, m, n, k, scaled, chained_k)


# struct mfma: "#if gfx942 || gfx9_4_generic || gfx950" block
_MFMA_CDNA3 = (
    _mfma(F16, F16, F32, 32, 32, 8),
    _mfma(F16, F16, F32, 16, 16, 16),
    _mfma(BF16, BF16, F32, 32, 32, 8),
    _mfma(BF16, BF16, F32, 16, 16, 16),
    _mfma(FP8, FP8, F32, 32, 32, 16),
    _mfma(FP8, FP8, F32, 16, 16, 32),
    _mfma(BF8, BF8, F32, 32, 32, 16),
    _mfma(BF8, BF8, F32, 16, 16, 32),
    _mfma(F32, F32, F32, 32, 32, 2),
    _mfma(F32, F32, F32, 16, 16, 4),
)
# struct mfma: "#if gfx942 || gfx9_4_generic" DISPATCH_MFMA_STEP_K_ (two chained instructions)
_MFMA_GFX942_CHAINED = (
    _mfma(F16, F16, F32, 32, 32, 16, chained_k=8),
    _mfma(F16, F16, F32, 16, 16, 32, chained_k=16),
    _mfma(BF16, BF16, F32, 32, 32, 16, chained_k=8),
    _mfma(BF16, BF16, F32, 16, 16, 32, chained_k=16),
)
# struct mfma: "#if gfx950" native double-K plus DISPATCH_MFMA_SCALE_ (f8f6f4)
_MFMA_GFX950 = (
    _mfma(F16, F16, F32, 32, 32, 16),
    _mfma(F16, F16, F32, 16, 16, 32),
    _mfma(BF16, BF16, F32, 32, 32, 16),
    _mfma(BF16, BF16, F32, 16, 16, 32),
) + tuple(
    _mfma(a, b, F32, m, n, k, scaled=True) for a, b in ((FP8, FP8), (FP8, FP4), (FP4, FP4)) for m, n, k in ((32, 32, 64), (16, 16, 128))
)
# struct wmma: "#if gfx1250" (wmma-256b, wave32)
_WMMA_GFX1250 = (
    (
        _wmma(F16, F16, F32, 16, 16, 32),
        _wmma(F16, F16, F16, 16, 16, 32),
        _wmma(BF16, BF16, F32, 16, 16, 32),
        _wmma(BF16, BF16, BF16, 16, 16, 32),
        _wmma(F32, F32, F32, 16, 16, 4),
    )
    + tuple(_wmma(a, b, c, 16, 16, k) for k in (64, 128) for c in (F32, F16) for a in (FP8, BF8) for b in (FP8, BF8))
    + tuple(_wmma(a, b, F32, 16, 16, 128, scaled=True) for a in (FP8, BF8, FP4) for b in (FP8, BF8, FP4))
    + (_wmma(FP4, FP4, F32, 32, 16, 128, scaled=True),)
)
# struct wmma: "#if gfx1201 || gfx1200" (wmma-128b, _w32_gfx12 builtins, wave32 default)
_WMMA_GFX12 = (
    tuple(_wmma(a, a, c, 16, 16, 16) for a in (F16, BF16) for c in (F32, a))
    + tuple(_wmma(a, b, F32, 16, 16, 16) for a in (FP8, BF8) for b in (FP8, BF8))
    + tuple(_wmma(a, a, c, 16, 16, 32, chained_k=16) for a in (F16, BF16) for c in (F32, a))
)

_GFX9_COUNTERS = {Counter.VM: 63, Counter.LGKM: 15}  # s_waitcnt field widths in opus
_GFX9_MAP = {
    MemClass.GMEM_LOAD: Counter.VM,
    MemClass.GMEM_STORE: Counter.VM,
    MemClass.GMEM_ASYNC_LDS: Counter.VM,
    MemClass.LDS: Counter.LGKM,
}
_GFX1250_COUNTERS = {
    Counter.LOAD: 63,
    Counter.STORE: 63,
    Counter.DS: 63,
    Counter.KM: 31,
    Counter.ASYNC: 63,
    Counter.TENSOR: 63,
}
_GFX1250_MAP = {
    MemClass.GMEM_LOAD: Counter.LOAD,
    MemClass.GMEM_STORE: Counter.STORE,
    MemClass.GMEM_ASYNC_LDS: Counter.ASYNC,
    MemClass.LDS: Counter.DS,
    MemClass.TENSOR_DMA: Counter.TENSOR,
}
_CDNA_ATOMICS = frozenset({Feature.ATOMIC_FADD_F32, Feature.ATOMIC_PK_ADD_F16})
_GFX12_NOTE = (
    "opus has no gfx12 waitcnt wrapper (its s_waitcnt uses the gfx9 encoding); wait ops are rejected",
    "gmem::async_load on gfx12 falls back to a synchronous load + ds_write",
    "wave64 (-mwavefrontsize64) wmma variants are not modeled",
)

_TARGETS: Dict[str, Target] = {
    "gfx942": Target(
        name="gfx942",
        warp_size=64,
        lds_bytes=65536,
        features=frozenset({Feature.MFMA, Feature.BUFFER_LOAD_LDS}) | _CDNA_ATOMICS,
        counters=_GFX9_COUNTERS,
        counter_map=_GFX9_MAP,
        async_lds_bytes=frozenset({1, 2, 4}),
        mma=_MFMA_CDNA3 + _MFMA_GFX942_CHAINED,
        fp8_flavor="fnuz",
    ),
    "gfx950": Target(
        name="gfx950",
        warp_size=64,
        lds_bytes=163840,
        features=frozenset(
            {
                Feature.MFMA,
                Feature.MFMA_SCALE,
                Feature.BUFFER_LOAD_LDS,
                Feature.DS_LOAD_TR,
                Feature.ATOMIC_PK_ADD_BF16,
                Feature.ATOMIC_FMINMAX_F32,
            }
        )
        | _CDNA_ATOMICS,
        counters=_GFX9_COUNTERS,
        counter_map=_GFX9_MAP,
        async_lds_bytes=frozenset({1, 2, 4, 12, 16}),
        mma=_MFMA_CDNA3 + _MFMA_GFX950,
        fp8_flavor="ocp",
    ),
    "gfx1250": Target(
        name="gfx1250",
        warp_size=32,
        lds_bytes=327680,
        features=frozenset(
            {
                Feature.WMMA,
                Feature.WMMA_SCALE,
                Feature.GLOBAL_LOAD_ASYNC_LDS,
                Feature.DS_LOAD_TR,
                Feature.TDM,
                Feature.CLUSTER,
                Feature.NAMED_BARRIER,
                Feature.ATOMIC_PK_ADD_BF16,
                Feature.ATOMIC_FMINMAX_F32,
            }
        )
        | _CDNA_ATOMICS,
        counters=_GFX1250_COUNTERS,
        counter_map=_GFX1250_MAP,
        async_lds_bytes=frozenset({1, 2, 4, 8, 16}),
        mma=_WMMA_GFX1250,
        fp8_flavor="ocp",
        tdm_in_flight=3,
    ),
}
for _name in ("gfx1200", "gfx1201"):
    _TARGETS[_name] = Target(
        name=_name,
        warp_size=32,
        lds_bytes=65536,
        features=frozenset({Feature.WMMA, Feature.ATOMIC_FADD_F32}),
        counters={},
        counter_map={},
        async_lds_bytes=frozenset({1, 2, 4, 8, 16}),
        mma=_WMMA_GFX12,
        fp8_flavor="ocp",
        notes=_GFX12_NOTE,
    )


def get_target(name: str) -> Target:
    try:
        return _TARGETS[name.lower()]
    except KeyError as exc:
        raise ValueError("unsupported exact target %r; supported: %s" % (name, ", ".join(sorted(_TARGETS)))) from exc


def all_targets() -> Dict[str, Target]:
    return dict(_TARGETS)
