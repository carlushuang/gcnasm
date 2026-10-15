"""Ergonomic Python builder for canonical opus schedule IR."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .expr import Expr, ExprLike, var
from .ir import (
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
from .targets import Counter, get_target
from .types import Dim, DType, Symbol, TensorType


def _name(value: Any) -> str:
    if isinstance(value, str):
        return value
    name = getattr(value, "name", None)
    if not isinstance(name, str):
        raise TypeError("expected a named IR resource, got %r" % (value,))
    return name


def _ej(value: ExprLike) -> Any:
    return Expr.of(value).to_json()


class KernelBuilder:
    """Mutable construction facade; :meth:`build` returns an immutable :class:`Kernel`."""

    def __init__(
        self,
        name: str,
        target: str,
        block_size: int,
        grid: Sequence[ExprLike] = (1, 1, 1),
        cluster: Optional[Tuple[int, int, int]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not name.isidentifier():
            raise ValueError("kernel name must be an identifier: %r" % name)
        get_target(target)
        self.name, self.target, self.block_size = name, target, block_size
        self.grid = grid
        self.cluster = cluster
        self.metadata = dict(metadata or {})
        self._symbols: Dict[str, Symbol] = {}
        self._res: Dict[str, Dict[str, Any]] = {
            k: {} for k in ("params", "lds", "regs", "mmas", "layouts", "tdms", "roles", "pipelines", "barriers", "semaphores")
        }
        self._stack: List[List[Op]] = [[]]
        self._role: Optional[str] = None
        self._built = False

    @property
    def grid(self) -> Tuple[Expr, Expr, Expr]:
        return self._grid

    @grid.setter
    def grid(self, value: Sequence[ExprLike]) -> None:
        """Workgroup counts (x, y, z) as host expressions of symbols; settable after symbols exist."""
        value = tuple(value)
        if not 1 <= len(value) <= 3:
            raise ValueError("grid has 1 to 3 dimensions")
        self._grid = tuple(Expr.of(g) for g in value + (1,) * (3 - len(value)))

    # -- declarations ------------------------------------------------------------------------

    def _add(self, collection: str, item: Any) -> Any:
        if self._built:
            raise RuntimeError("this KernelBuilder has already been finalized")
        if not item.name.isidentifier():
            raise ValueError("IR names must be identifiers: %r" % item.name)
        if any(item.name in c for c in self._res.values()):
            raise ValueError("duplicate IR name: %s" % item.name)
        self._res[collection][item.name] = item
        return item

    def symbol(self, name: str, minimum: int = 1, maximum: Optional[int] = None, multiple_of: int = 1) -> Symbol:
        candidate = Symbol(name, minimum, maximum, multiple_of)
        previous = self._symbols.setdefault(name, candidate)
        if previous != candidate:
            raise ValueError("symbol %s was redeclared with a different contract" % name)
        return candidate

    def param(
        self,
        name: str,
        shape: Sequence[Dim],
        dtype: DType,
        access: Access = Access.READ,
        oob: bool = True,
        alignment: int = 16,
    ) -> Param:
        return self._add("params", Param(name, TensorType(dtype, tuple(shape)), access, oob, alignment))

    def lds(
        self,
        name: str,
        shape: Sequence[int],
        dtype: DType,
        stages: int = 1,
        pad: int = 0,
        offset_bytes: Optional[int] = None,
        alignment: int = 16,
        alias_of: Any = None,
    ) -> Lds:
        alias = None if alias_of is None else _name(alias_of)
        return self._add("lds", Lds(name, dtype, tuple(shape), stages, pad, offset_bytes, alignment, alias))

    def reg(self, name: str, dtype: DType, elems: int, zero_init: bool = False) -> Reg:
        return self._add("regs", Reg(name, dtype, elems, zero_init))

    def mma(
        self,
        name: str,
        kind: Union[MmaKind, str],
        a: DType,
        b: DType,
        c: DType,
        wave: Tuple[int, int, int],
        expand: Tuple[int, int, int] = (1, 1, 1),
        tile: Tuple[int, int, int] = (1, 1, 1),
        swap_ab: bool = False,
        scaled: bool = False,
    ) -> Mma:
        return self._add("mmas", Mma(name, kind, a, b, c, wave, expand, tile, swap_ab, scaled))

    def layout(
        self,
        name: str,
        shape: Sequence[int],
        dims: Optional[Sequence[Sequence[Union[PY, str]]]] = None,
        x_stride: Optional[Sequence[ExprLike]] = None,
        p_coord: Sequence[ExprLike] = (),
    ) -> Layout:
        """``dims=None`` is ``make_layout(shape[, stride])``; otherwise an x/p/y distributed view."""
        if dims is None:
            item = Layout.packed(name, shape) if x_stride is None else Layout.strided(name, shape, x_stride)
        else:
            if x_stride is None:
                raise ValueError("a distributed layout must commit its x_stride")
            item = Layout(name, tuple(shape), tuple(tuple(g) for g in dims), tuple(x_stride), tuple(p_coord))
        return self._add("layouts", item)

    def mma_layout(
        self,
        name: str,
        mma: Union[Mma, str],
        operand: Union[Operand, str],
        x_stride: Sequence[ExprLike] = (),
        p_coord: Sequence[ExprLike] = (),
        packed: bool = False,
    ) -> MmaLayout:
        return self._add("layouts", MmaLayout(name, _name(mma), operand, tuple(x_stride), tuple(p_coord), packed))

    def tdm(
        self,
        name: str,
        param: Union[Param, str],
        lds: Union[Lds, str],
        tile: Sequence[int],
        pad_interval: int = 0,
        pad_amount: int = 0,
        multicast: Union[Multicast, str] = Multicast.NONE,
        gather_index_size: int = 0,
        cache_policy: Optional[int] = None,
    ) -> Tdm:
        return self._add(
            "tdms",
            Tdm(
                name,
                _name(param),
                _name(lds),
                tuple(tile),
                pad_interval,
                pad_amount,
                multicast,
                gather_index_size,
                cache_policy,
            ),
        )

    def role(self, name: str, waves: Sequence[int], kind: RoleKind = RoleKind.GENERIC) -> Role:
        return self._add("roles", Role(name, tuple(waves), kind))

    def pipeline(self, name: str, stages: int) -> Pipeline:
        return self._add("pipelines", Pipeline(name, stages))

    def named_barrier(self, name: str, members: Sequence[Union[Role, str]]) -> NamedBarrier:
        return self._add("barriers", NamedBarrier(name, tuple(_name(m) for m in members)))

    def semaphore(self, name: str, param: Union[Param, str]) -> Semaphore:
        return self._add("semaphores", Semaphore(name, _name(param)))

    # -- structure ---------------------------------------------------------------------------

    def emit(
        self,
        opcode: str,
        inputs: Sequence[Any] = (),
        outputs: Sequence[Any] = (),
        attrs: Optional[Mapping[str, Any]] = None,
        role: Any = None,
        body: Sequence[Op] = (),
    ) -> Op:
        if self._built:
            raise RuntimeError("this KernelBuilder has already been finalized")
        role_name = self._role if role is None else _name(role)
        op = Op.create(opcode, [_name(i) for i in inputs], [_name(o) for o in outputs], role_name, attrs, body)
        self._stack[-1].append(op)
        return op

    @contextmanager
    def on(self, role: Union[Role, str]) -> Iterator[None]:
        """Ops emitted inside run only on ``role``'s waves."""
        previous, self._role = self._role, _name(role)
        try:
            yield
        finally:
            self._role = previous

    @contextmanager
    def loop(
        self,
        name: str,
        trip: ExprLike,
        pipeline: Optional[Union[Pipeline, str]] = None,
        unroll: int = 1,
    ) -> Iterator[Expr]:
        """A counted loop ``for name in [0, trip)``; yields the loop variable expression."""
        self._stack.append([])
        try:
            yield var(name)
        finally:
            body = self._stack.pop()
            attrs = {"var": name, "trip": _ej(trip), "unroll": unroll}
            if pipeline is not None:
                attrs["pipeline"] = _name(pipeline)
            self.emit("loop", attrs=attrs, body=body)

    # -- memory movement ---------------------------------------------------------------------

    def gmem_load(self, src: Any, layout: Any, dst: Any, vec: int = 1, offset: ExprLike = 0) -> Op:
        """``dst = g_src.load<vec>(layout, offset)``"""
        return self.emit("gmem_load", [src], [dst], {"layout": _name(layout), "vec": vec, "offset": _ej(offset)})

    def gmem_store(self, src: Any, dst: Any, layout: Any, vec: int = 1, offset: ExprLike = 0) -> Op:
        """``g_dst.store<vec>(src, layout, offset)``"""
        return self.emit("gmem_store", [src], [dst], {"layout": _name(layout), "vec": vec, "offset": _ej(offset)})

    def gmem_atomic_add(self, src: Any, dst: Any, layout: Any, vec: int = 1, offset: ExprLike = 0) -> Op:
        """``g_dst.atomic_add<vec>(src, layout, offset)``"""
        attrs = {"layout": _name(layout), "vec": vec, "offset": _ej(offset)}
        return self.emit("gmem_atomic_add", [src], [dst], attrs)

    def async_load(
        self,
        src: Any,
        layout_g: Any,
        dst: Any,
        layout_s: Any,
        vec: int = 1,
        offset: ExprLike = 0,
        stage: ExprLike = 0,
    ) -> Op:
        """``g_src.async_load<vec>(lds_base(dst, stage), layout_g, layout_s, offset)`` (global -> LDS)"""
        attrs = {
            "layout_g": _name(layout_g),
            "layout_s": _name(layout_s),
            "vec": vec,
            "offset": _ej(offset),
            "stage": _ej(stage),
        }
        return self.emit("gmem_async_load", [src], [dst], attrs)

    def smem_load(self, src: Any, layout: Any, dst: Any, vec: int = 1, stage: ExprLike = 0, transpose: bool = False) -> Op:
        """``dst = s_src.load<vec>(layout)`` or ``tr_load<vec>`` when ``transpose``"""
        opcode = "smem_tr_load" if transpose else "smem_load"
        return self.emit(opcode, [src], [dst], {"layout": _name(layout), "vec": vec, "stage": _ej(stage)})

    def smem_store(self, src: Any, dst: Any, layout: Any, vec: int = 1, stage: ExprLike = 0) -> Op:
        """``s_dst.store<vec>(src, layout)``"""
        return self.emit("smem_store", [src], [dst], {"layout": _name(layout), "vec": vec, "stage": _ej(stage)})

    def tdm_load(self, window: Any, origin: Sequence[ExprLike], stage: ExprLike = 0) -> Op:
        """``opus::async_load(window)`` after positioning the window at ``origin`` (elements)"""
        attrs = {"origin": [_ej(o) for o in origin], "stage": _ej(stage)}
        return self.emit("tdm_load", [window], [], attrs)

    def tdm_store(self, window: Any, origin: Sequence[ExprLike], stage: ExprLike = 0) -> Op:
        """``opus::async_store(window)``: LDS stage -> global tile at ``origin``"""
        attrs = {"origin": [_ej(o) for o in origin], "stage": _ej(stage)}
        return self.emit("tdm_store", [window], [], attrs)

    # -- compute -----------------------------------------------------------------------------

    def mma_op(
        self,
        mma: Any,
        a: Any,
        b: Any,
        c: Any,
        accumulate: bool = True,
        scale_a: Any = None,
        scale_b: Any = None,
    ) -> Op:
        """``c = mma(a, b, c)`` (or ``mma(a, b)`` when not accumulating); scaled forms take E8M0 scales."""
        inputs = [mma, a, b] + ([c] if accumulate else [])
        attrs: Dict[str, Any] = {"accumulate": accumulate}
        if scale_a is not None or scale_b is not None:
            inputs += [scale_a, scale_b]
            attrs["scaled"] = True
        return self.emit("mma", inputs, [c], attrs)

    def cast(self, src: Any, dst: Any) -> Op:
        """``dst = opus::cast<dst_t>(src)``"""
        return self.emit("cast", [src], [dst])

    def elementwise(self, kind: str, inputs: Sequence[Any], dst: Any, scalar: Optional[float] = None) -> Op:
        attrs: Dict[str, Any] = {"kind": kind}
        if scalar is not None:
            attrs["scalar"] = scalar
        return self.emit("elementwise", inputs, [dst], attrs)

    # -- synchronization ---------------------------------------------------------------------

    def wait(self, counter: Union[Counter, str], outstanding: int = 0) -> Op:
        """Block this wave until at most ``outstanding`` ops are in flight on ``counter``."""
        return self.emit("wait", attrs={"counter": Counter(counter).value, "outstanding": outstanding})

    def sync_threads(self) -> Op:
        """``opus::sync_threads()`` (s_barrier): every wave of the workgroup must reach it."""
        return self.emit("sync_threads")

    def sync_cluster(self) -> Op:
        """``opus::sync_cluster()``, gfx1250."""
        return self.emit("sync_cluster")

    def barrier_arrive(self, barrier: Any) -> Op:
        return self.emit("barrier_arrive", [barrier])

    def barrier_wait(self, barrier: Any) -> Op:
        return self.emit("barrier_wait", [barrier])

    def sem_wait(self, sem: Any, cmp: str, value: ExprLike, slot: int = 0) -> Op:
        """``workgroup_barrier::wait_eq / wait_lt(value, slot)``"""
        return self.emit("sem_wait", [sem], [], {"cmp": cmp, "value": _ej(value), "slot": slot})

    def sem_inc(self, sem: Any, slot: int = 0) -> Op:
        """``workgroup_barrier::inc(slot)``"""
        return self.emit("sem_inc", [], [sem], {"slot": slot})

    # -- finalize ----------------------------------------------------------------------------

    def build(self) -> Kernel:
        if len(self._stack) != 1:
            raise RuntimeError("unterminated loop region")
        self._built = True
        r = self._res
        return Kernel(
            name=self.name,
            target=self.target,
            block_size=self.block_size,
            grid=self.grid,
            params=tuple(r["params"].values()),
            symbols=tuple(self._symbols.values()),
            lds=tuple(r["lds"].values()),
            regs=tuple(r["regs"].values()),
            mmas=tuple(r["mmas"].values()),
            layouts=tuple(r["layouts"].values()),
            tdms=tuple(r["tdms"].values()),
            roles=tuple(r["roles"].values()),
            pipelines=tuple(r["pipelines"].values()),
            barriers=tuple(r["barriers"].values()),
            semaphores=tuple(r["semaphores"].values()),
            body=tuple(self._stack[0]),
            cluster=self.cluster,
            metadata_json=Op.create("_", attrs=self.metadata).attrs_json,
        )


def kernel(
    name: str,
    target: str,
    block_size: int,
    grid: Sequence[ExprLike] = (1, 1, 1),
    cluster: Optional[Tuple[int, int, int]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Callable[[Callable[[KernelBuilder], None]], Kernel]:
    """Build an immutable :class:`Kernel` from a declarative function at import time."""

    def decorator(function: Callable[[KernelBuilder], None]) -> Kernel:
        builder = KernelBuilder(name, target, block_size, grid, cluster, metadata)
        if function(builder) is not None:
            raise TypeError("kernel functions declare into the builder and must return None")
        return builder.build()

    return decorator
