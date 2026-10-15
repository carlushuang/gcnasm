"""Static verification: IR structure, opus/target admission, typing, and LDS synchronization.

Diagnostic codes are a stable interface. The verifier is deliberately incomplete (like Cake's):
it can report false positives on hand-partitioned LDS access and cannot prove absence of races.
On-GPU execution remains ground truth.
"""

from __future__ import annotations

from collections import Counter as Multiset
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from .diagnostics import Severity, VerificationResult
from .expr import Expr
from .ir import Access, Kernel, Lds, Mma, MmaKind, Multicast, NamedBarrier, Op, Param, Reg, Semaphore, Tdm
from .layout import PY, Layout, MmaLayout
from .passes.memory import plan_lds
from .targets import Counter, Feature, MemClass, Target, get_target
from .types import DType, Symbol

E, W, INFO = Severity.ERROR, Severity.WARNING, Severity.INFO

# opcode -> (input kinds, output kinds); None = checked by hand
_SIGNATURES: Dict[str, Optional[Tuple[Tuple[str, ...], Tuple[str, ...]]]] = {
    "gmem_load": (("param",), ("reg",)),
    "gmem_store": (("reg",), ("param",)),
    "gmem_atomic_add": (("reg",), ("param",)),
    "gmem_async_load": (("param",), ("lds",)),
    "smem_load": (("lds",), ("reg",)),
    "smem_tr_load": (("lds",), ("reg",)),
    "smem_store": (("reg",), ("lds",)),
    "tdm_load": (("tdm",), ()),
    "tdm_store": (("tdm",), ()),
    "mma": None,
    "cast": (("reg",), ("reg",)),
    "elementwise": None,
    "wait": ((), ()),
    "sync_threads": ((), ()),
    "sync_cluster": ((), ()),
    "barrier_arrive": (("barrier",), ()),
    "barrier_wait": (("barrier",), ()),
    "sem_wait": (("semaphore",), ()),
    "sem_inc": ((), ("semaphore",)),
    "loop": ((), ()),
}
_ELEMENTWISE_ARITY = {"add": 2, "sub": 2, "mul": 2, "max": 2, "min": 2, "copy": 1, "scale": 1, "fill": 0}
_VEC_BYTES = frozenset({1, 2, 4, 8, 16})
_TR_LOAD_BITS = frozenset({16, 8, 4})  # smem::_tr_load: ds_load_tr16_b128 / tr8_b64 / tr4_b64


def _kind(resource: Any) -> str:
    return {
        Param: "param",
        Lds: "lds",
        Reg: "reg",
        Mma: "mma",
        Tdm: "tdm",
        NamedBarrier: "barrier",
        Semaphore: "semaphore",
        Layout: "layout",
        MmaLayout: "layout",
    }.get(type(resource), type(resource).__name__.lower())


@dataclass
class _Ctx:
    kernel: Kernel
    target: Target
    result: VerificationResult
    res: Dict[str, Any]
    symbols: Dict[str, Symbol]
    all_waves: FrozenSet[int]

    def add(self, code: str, sev: Severity, msg: str, path: str = "kernel", notes: Tuple[str, ...] = ()) -> None:
        self.result.add(code, sev, msg, path, notes)

    def waves_of(self, role: Optional[str]) -> FrozenSet[int]:
        if role is None:
            return self.all_waves
        item = self.res.get(role)
        return frozenset(getattr(item, "waves", ())) & self.all_waves


def verify(kernel: Kernel) -> VerificationResult:
    result = VerificationResult()
    try:
        target = get_target(kernel.target)
    except ValueError as exc:
        result.add("TARGET000", E, str(exc))
        return result
    waves = kernel.block_size // target.warp_size if kernel.block_size > 0 else 0
    ctx = _Ctx(kernel, target, result, kernel.resources(), {s.name: s for s in kernel.symbols}, frozenset(range(waves)))
    _verify_structure(ctx)
    _verify_lds(ctx)
    for item in kernel.layouts:
        _verify_layout(ctx, item)
    for item in kernel.mmas:
        _verify_mma_resource(ctx, item)
    for item in kernel.tdms:
        _verify_tdm(ctx, item)
    for i, op in enumerate(kernel.body):
        _verify_op(ctx, op, "body[%d]" % i, {}, None)
    if result.ok:
        _simulate(ctx)
    return result


# -- structure ---------------------------------------------------------------------------------


def _verify_structure(ctx: _Ctx) -> None:
    k, t = ctx.kernel, ctx.target
    if not k.name.isidentifier():
        ctx.add("IR001", E, "kernel name is not an identifier")
    names = [p.name for p in k.params] + [s.name for s in k.symbols]
    for attr in ("lds", "regs", "mmas", "layouts", "tdms", "roles", "pipelines", "barriers", "semaphores"):
        names += [item.name for item in getattr(k, attr)]
    for name, count in Multiset(names).items():
        if count > 1:
            ctx.add("IR002", E, "duplicate IR name %s" % name)
    for p in k.params:
        for s in p.type.symbols():
            declared = ctx.symbols.get(s.name)
            if declared is None:
                ctx.add("IR003", E, "symbol %s is not declared" % s.name, "param.%s" % p.name)
            elif declared != s:
                ctx.add("IR003", E, "symbol %s has inconsistent bounds" % s.name, "param.%s" % p.name)
        if p.alignment <= 0 or p.alignment & (p.alignment - 1):
            ctx.add("TYPE001", E, "alignment must be a positive power of two", "param.%s" % p.name)
        if p.type.dtype in (DType.FP8, DType.BF8) and t.fp8_flavor == "fnuz":
            ctx.add("TARGET008", INFO, "%s on %s is FNUZ (NaN=0x80, no inf)" % (p.type.dtype.value, t.name), "param.%s" % p.name)
    if not k.params:
        ctx.add("IR004", E, "kernel must declare at least one parameter")
    if not k.body:
        ctx.add("IR005", E, "kernel body is empty")
    if k.block_size <= 0 or k.block_size > t.max_threads_per_block:
        ctx.add("SCHED001", E, "block_size=%d outside [1, %d]" % (k.block_size, t.max_threads_per_block))
    elif k.block_size % t.warp_size:
        ctx.add("SCHED001", E, "block_size=%d is not a multiple of warp_size=%d" % (k.block_size, t.warp_size))
    for axis, g in zip("xyz", k.grid):
        if g.hw_ids() or g.loop_vars():
            ctx.add("SCHED005", E, "grid.%s must be a host expression of symbols and constants" % axis)
        _check_symbols(ctx, g, "grid.%s" % axis)
    if k.cluster is not None and not t.supports(Feature.CLUSTER):
        ctx.add("TARGET009", E, "cluster launch requires %s" % Feature.CLUSTER.value)
    for role in k.roles:
        bad = [w for w in role.waves if w not in ctx.all_waves]
        if bad or not role.waves:
            ctx.add("SCHED002", E, "role waves %s outside [0, %d)" % (bad or "[]", len(ctx.all_waves)), "role.%s" % role.name)
        if len(set(role.waves)) != len(role.waves):
            ctx.add("SCHED002", E, "role lists a wave twice", "role.%s" % role.name)
    if len(k.barriers) > 15:
        ctx.add("TARGET005", E, "%d named barriers; DECLARE_NAMED_BARRIERS provides 15" % len(k.barriers))
    for b in k.barriers:
        if not ctx.target.supports(Feature.NAMED_BARRIER):
            ctx.add("TARGET005", E, "named barriers require %s" % Feature.NAMED_BARRIER.value, "barrier.%s" % b.name)
        for m in b.members:
            if _kind(ctx.res.get(m)) != "role":
                ctx.add("IR007", E, "barrier member %s is not a role" % m, "barrier.%s" % b.name)
    for s in k.semaphores:
        p = ctx.res.get(s.param)
        if not isinstance(p, Param) or p.type.dtype != DType.U32 or p.access is not Access.READ_WRITE:
            ctx.add("TYPE007", E, "semaphore needs a read_write u32 param", "semaphore.%s" % s.name)


def _check_symbols(ctx: _Ctx, expr: Expr, path: str) -> None:
    for name in expr.symbols():
        if name not in ctx.symbols:
            ctx.add("IR003", E, "expression uses undeclared symbol %s" % name, path)
    for hw in expr.hw_ids():
        if hw.cluster and not ctx.target.supports(Feature.CLUSTER):
            ctx.add("TARGET009", E, "%s requires %s" % (hw.value, Feature.CLUSTER.value), path)


def _check_expr(ctx: _Ctx, expr: Expr, path: str, scope: Dict[str, Any], uniform: bool) -> None:
    _check_symbols(ctx, expr, path)
    for v in expr.loop_vars():
        if v not in scope:
            ctx.add("SCHED006", E, "loop variable %s is not in scope" % v, path)
    if uniform and not expr.wave_uniform:
        ctx.add("SCHED009", E, "expression %s must be wave-uniform (it lowers to an SGPR)" % expr, path)


# -- resources ---------------------------------------------------------------------------------


def _verify_lds(ctx: _Ctx) -> None:
    for buf in ctx.kernel.lds:
        path = "lds.%s" % buf.name
        if buf.stages <= 0 or any(s <= 0 for s in buf.shape) or buf.pad < 0:
            ctx.add("MEM001", E, "stages and extents must be positive, pad non-negative", path)
        if buf.offset_bytes is not None and buf.offset_bytes % buf.alignment:
            ctx.add("MEM002", E, "offset %d violates alignment %d" % (buf.offset_bytes, buf.alignment), path)
        if buf.alias_of is not None and not isinstance(ctx.res.get(buf.alias_of), Lds):
            ctx.add("MEM005", E, "alias_of %s is not an LDS region" % buf.alias_of, path)
        if buf.dtype.packed and buf.shape[-1] % 2:
            ctx.add("TYPE006", E, "packed %s rows must hold whole bytes" % buf.dtype.value, path)
    plan = plan_lds(ctx.kernel)
    for left, right in plan.overlaps:
        ctx.add("MEM003", E, "LDS regions %s and %s overlap without alias_of" % (left, right))
    if plan.total_bytes > ctx.target.lds_bytes:
        ctx.add("MEM004", E, "LDS use %d bytes exceeds %s capacity %d" % (plan.total_bytes, ctx.target.name, ctx.target.lds_bytes))


def _verify_layout(ctx: _Ctx, item: Any) -> None:
    path = "layout.%s" % item.name
    for e in item.x_stride + item.p_coord:
        _check_symbols(ctx, e, path)
        if e.loop_vars():
            ctx.add("SCHED006", E, "layouts are loop-invariant; move iteration into op offset/stage", path)
    if isinstance(item, MmaLayout):
        if not isinstance(ctx.res.get(item.mma), Mma):
            ctx.add("LAYOUT006", E, "mma %s is not an mma resource" % item.mma, path)
        if not item.packed and len(item.x_stride) != 2:
            ctx.add("LAYOUT007", E, "an mma operand layout needs a 2-D x_stride (or packed=True)", path)
        if item.packed and item.x_stride:
            ctx.add("LAYOUT007", E, "packed mma layouts take no x_stride", path)
        return
    if sum(len(g) for g in item.dims) != len(item.shape) or any(not g for g in item.dims):
        ctx.add("LAYOUT001", E, "dims groups must be non-empty and cover shape exactly", path)
        return
    if len(item.x_stride) != len(item.dims):
        ctx.add("LAYOUT003", E, "need one x_stride per x-dim (%d), got %d" % (len(item.dims), len(item.x_stride)), path)
    if any(s <= 0 for s in item.shape):
        ctx.add("LAYOUT001", E, "layout extents must be positive", path)
    n_p = sum(1 for d in item.flat_dims if d is PY.P)
    if n_p != len(item.p_coord):
        ctx.add("LAYOUT002", E, "%d p dims but %d p-coords" % (n_p, len(item.p_coord)), path)
    if n_p and all(c.wave_uniform for c in item.p_coord):
        ctx.add("LAYOUT004", W, "p-coords never use lane ids, so every lane of a wave hits the same address", path)
    if item.threads_spanned > ctx.kernel.block_size > 0:
        ctx.add("LAYOUT005", E, "p dims span %d threads but block has %d" % (item.threads_spanned, ctx.kernel.block_size), path)


def _verify_mma_resource(ctx: _Ctx, m: Mma) -> None:
    path, t = "mma.%s" % m.name, ctx.target
    need = {MmaKind.MFMA: Feature.MFMA, MmaKind.WMMA: Feature.WMMA}[m.kind]
    if not t.supports(need):
        ctx.add("TARGET001", E, "%s is not available on %s" % (m.kind.value, t.name), path)
        return
    if m.scaled and not t.supports({MmaKind.MFMA: Feature.MFMA_SCALE, MmaKind.WMMA: Feature.WMMA_SCALE}[m.kind]):
        ctx.add("TARGET001", E, "scaled %s is not available on %s" % (m.kind.value, t.name), path)
        return
    inst = t.find_mma(m.kind.value, m.a, m.b, m.c, *m.wave, m.scaled)
    if inst is None:
        same = sorted("%dx%dx%d" % (i.m, i.n, i.k) for i in t.mma if (i.a, i.b, i.c, i.scaled) == (m.a, m.b, m.c, m.scaled))
        ctx.add(
            "TARGET001",
            E,
            "no opus %s %s/%s->%s %dx%dx%d%s on %s"
            % (m.kind.value, m.a.value, m.b.value, m.c.value, *m.wave, " scaled" if m.scaled else "", t.name),
            path,
            notes=("available shapes for these dtypes: %s" % (", ".join(same) or "none"),),
        )
    elif inst.chained_k:
        ctx.add(
            "TARGET010",
            INFO,
            "lowers to %d chained %dx%dx%d instructions" % (m.wave[2] // inst.chained_k, m.wave[0], m.wave[1], inst.chained_k),
            path,
        )
    if any(x <= 0 for x in m.expand + m.tile):
        ctx.add("MMA002", E, "expand/tile factors must be positive", path)
    if m.waves > len(ctx.all_waves):
        ctx.add("MMA001", E, "mma tile needs %d waves, block has %d" % (m.waves, len(ctx.all_waves)), path)
    if (m.wave[0] * m.wave[2]) % t.warp_size or (m.wave[1] * m.wave[2]) % t.warp_size:
        ctx.add("MMA002", E, "wave shape does not divide across %d lanes" % t.warp_size, path)


def _verify_tdm(ctx: _Ctx, d: Tdm) -> None:
    path = "tdm.%s" % d.name
    if not ctx.target.supports(Feature.TDM):
        ctx.add("TARGET004", E, "tdm requires %s" % Feature.TDM.value, path)
        return
    p, buf = ctx.res.get(d.param), ctx.res.get(d.lds)
    if not isinstance(p, Param) or not isinstance(buf, Lds):
        ctx.add("IR007", E, "tdm needs a param and an LDS region", path)
        return
    if p.type.dtype != buf.dtype:
        ctx.add("TYPE002", E, "tdm moves %s into %s LDS" % (p.type.dtype.value, buf.dtype.value), path)
    if tuple(buf.shape) != tuple(d.tile):
        ctx.add("TDM001", E, "LDS stage shape %s != tdm tile %s" % (list(buf.shape), list(d.tile)), path)
    if p.type.rank != len(d.tile):
        ctx.add("TDM001", E, "tdm tile rank %d != tensor rank %d" % (len(d.tile), p.type.rank), path)
    if (d.pad_interval == 0) != (d.pad_amount == 0):
        ctx.add("TDM002", E, "pad_interval and pad_amount must be both set or both 0", path)
    elif d.pad_interval and (d.pad_interval != buf.shape[-1] or d.pad_amount != buf.pad):
        ctx.add(
            "TDM002",
            W,
            "tdm padding (%d every %d) differs from LDS row pad (%d every %d)" % (d.pad_amount, d.pad_interval, buf.pad, buf.shape[-1]),
            path,
        )
    if d.multicast is not Multicast.NONE and ctx.kernel.cluster is None:
        ctx.add("TDM003", E, "multicast needs a cluster launch (a non-cluster wave may hang)", path)


# -- ops ---------------------------------------------------------------------------------------


def _layout_elems(ctx: _Ctx, layout: Any) -> Optional[int]:
    if isinstance(layout, Layout):
        return layout.elems_per_thread
    if isinstance(layout, MmaLayout):
        m = ctx.res.get(layout.mma)
        if isinstance(m, Mma):
            return m.lane_elems(ctx.target.warp_size)[layout.operand.value]
    return None


def _check_vec(ctx: _Ctx, op: Op, dtype: DType, vec: int, layout: Any, path: str, legal=_VEC_BYTES) -> None:
    nbytes = dtype.nbytes(vec)
    if vec <= 0 or nbytes not in legal or (dtype.packed and (vec * dtype.bits) % 8):
        ctx.add("TYPE003", E, "vec=%d of %s is %s bytes per lane; legal: %s" % (vec, dtype.value, nbytes, sorted(legal)), path)
    if isinstance(layout, Layout) and vec > 1:
        run = layout.innermost_y
        if run is None or run % vec:
            ctx.add("TYPE003", E, "vec=%d must divide the innermost y extent (%s) of layout %s" % (vec, run, layout.name), path)


def _verify_op(ctx: _Ctx, op: Op, path: str, scope: Dict[str, Any], loop_pipe: Optional[str]) -> None:
    path = "%s(%s)" % (path, op.opcode)
    if op.opcode not in _SIGNATURES:
        ctx.add("IR006", E, "unknown opcode %s" % op.opcode, path)
        return
    a, res = op.attrs, ctx.res
    if op.role is not None and _kind(res.get(op.role)) != "role":
        ctx.add("IR007", E, "role %s is not declared" % op.role, path)
    for name in op.inputs + op.outputs:
        if name not in res:
            ctx.add("IR002", E, "reference to undeclared %s" % name, path)
            return
    sig = _SIGNATURES[op.opcode]
    if sig is not None:
        kinds = (tuple(_kind(res[n]) for n in op.inputs), tuple(_kind(res[n]) for n in op.outputs))
        if kinds != sig:
            ctx.add("IR007", E, "operands %s -> %s, expected %s -> %s" % (*kinds, *sig), path)
            return
    for key in ("offset", "stage", "trip"):
        if key in a:
            _check_expr(ctx, Expr.from_json(a[key]), path, scope, uniform=True)
    for o in a.get("origin", ()):
        _check_expr(ctx, Expr.from_json(o), path, scope, uniform=True)

    handler = _OP_CHECKS.get(op.opcode)
    if handler is not None:
        handler(ctx, op, path)
    if loop_pipe is not None:
        pipe = res.get(loop_pipe)
        for name in op.inputs + op.outputs:
            buf = res.get(name)
            if isinstance(buf, Tdm):
                buf = res.get(buf.lds)
            if isinstance(buf, Lds) and getattr(pipe, "stages", None) not in (None, buf.stages) and "stage" in a:
                ctx.add("SCHED008", W, "LDS %s has %d stages, pipeline %s has %d" % (buf.name, buf.stages, pipe.name, pipe.stages), path)
    if op.opcode == "loop":
        v = a.get("var")
        if not isinstance(v, str) or not v.isidentifier() or v in scope:
            ctx.add("SCHED006", E, "loop variable %r is invalid or shadows an outer loop" % v, path)
        pipe = a.get("pipeline")
        if pipe is not None and _kind(res.get(pipe)) != "pipeline":
            ctx.add("IR007", E, "pipeline %s is not declared" % pipe, path)
        inner = dict(scope, **{v: True}) if isinstance(v, str) else scope
        for i, child in enumerate(op.body):
            _verify_op(ctx, child, "%s.body[%d]" % (path, i), inner, pipe)
    elif op.body:
        ctx.add("IR008", E, "only loop ops carry a body", path)


def _chk_gmem(ctx: _Ctx, op: Op, path: str) -> None:
    load = op.opcode == "gmem_load"
    p: Param = ctx.res[op.inputs[0] if load else op.outputs[0]]
    r: Reg = ctx.res[op.outputs[0] if load else op.inputs[0]]
    if load and p.access is Access.WRITE:
        ctx.add("ACCESS002", E, "load from write-only param %s" % p.name, path)
    if not load and p.access is Access.READ:
        ctx.add("ACCESS001", E, "store to read-only param %s" % p.name, path)
    if p.type.dtype != r.dtype:
        ctx.add("TYPE002", E, "param %s is %s, register %s is %s" % (p.name, p.type.dtype.value, r.name, r.dtype.value), path)
    layout = ctx.res.get(op.attrs.get("layout"))
    if _kind(layout) != "layout":
        ctx.add("IR007", E, "layout %s is not declared" % op.attrs.get("layout"), path)
        return
    elems = _layout_elems(ctx, layout)
    if elems is not None and elems != r.elems:
        ctx.add("TYPE004", E, "layout %s gives %d elems/lane, register %s holds %d" % (layout.name, elems, r.name, r.elems), path)
    _check_vec(ctx, op, r.dtype, int(op.attrs.get("vec", 1)), layout, path)
    if op.opcode == "gmem_atomic_add":
        need = {DType.FP32: Feature.ATOMIC_FADD_F32, DType.FP16: Feature.ATOMIC_PK_ADD_F16, DType.BF16: Feature.ATOMIC_PK_ADD_BF16}.get(
            r.dtype
        )
        if r.dtype not in (DType.I32,) and (need is None or not ctx.target.supports(need)):
            ctx.add("TARGET007", E, "buffer atomic add of %s is not available on %s" % (r.dtype.value, ctx.target.name), path)


def _chk_async(ctx: _Ctx, op: Op, path: str) -> None:
    p: Param = ctx.res[op.inputs[0]]
    buf: Lds = ctx.res[op.outputs[0]]
    t = ctx.target
    if not (t.supports(Feature.BUFFER_LOAD_LDS) or t.supports(Feature.GLOBAL_LOAD_ASYNC_LDS)):
        ctx.add("TARGET002", W, "gmem::async_load on %s is a synchronous load + ds_write fallback" % t.name, path)
    if p.access is Access.WRITE:
        ctx.add("ACCESS002", E, "load from write-only param %s" % p.name, path)
    if p.type.dtype != buf.dtype:
        ctx.add("TYPE002", E, "param %s is %s, LDS %s is %s" % (p.name, p.type.dtype.value, buf.name, buf.dtype.value), path)
    lg, ls = ctx.res.get(op.attrs.get("layout_g")), ctx.res.get(op.attrs.get("layout_s"))
    if _kind(lg) != "layout" or _kind(ls) != "layout":
        ctx.add("IR007", E, "layout_g/layout_s must be declared layouts", path)
        return
    eg, es = _layout_elems(ctx, lg), _layout_elems(ctx, ls)
    if eg is not None and es is not None and eg != es:
        ctx.add("TYPE004", E, "global layout moves %d elems/lane, LDS layout %d" % (eg, es), path)
    vec = int(op.attrs.get("vec", 1))
    _check_vec(ctx, op, buf.dtype, vec, lg, path, legal=t.async_lds_bytes)
    if t.supports(Feature.BUFFER_LOAD_LDS) and isinstance(ls, Layout):
        _check_lane_contiguous(ctx, ls, vec, path)


def _check_lane_contiguous(ctx: _Ctx, layout: Layout, vec: int, path: str) -> None:
    """CDNA buffer_load...lds writes LDS at M0 + inst_offset + lane * bytes: only a wave-uniform base survives."""
    env = {s.name: s.minimum for s in ctx.kernel.symbols}
    y0 = [0] * len(layout.y_shape)
    offsets = []
    for lane in range(ctx.target.warp_size):
        offsets.append(layout.offset(y0, dict(env, lane=lane, thread_x=lane, wave=0, thread_y=0, thread_z=0)))
    if any(o is None for o in offsets):
        ctx.add("ASYNC001", W, "cannot prove LDS layout %s is lane-contiguous" % layout.name, path)
        return
    bad = next((lane for lane, o in enumerate(offsets) if o != offsets[0] + lane * vec), None)
    if bad is not None:
        ctx.add(
            "ASYNC001",
            E,
            "LDS layout %s is not lane-contiguous (lane %d -> offset %d, hardware writes %d)"
            % (layout.name, bad, offsets[bad], offsets[0] + bad * vec),
            path,
            notes=("%s buffer_load...lds ignores per-lane LDS addresses; lane l lands at base + l*vec" % ctx.target.name,),
        )


def _chk_smem(ctx: _Ctx, op: Op, path: str) -> None:
    load = op.opcode != "smem_store"
    buf: Lds = ctx.res[op.inputs[0] if load else op.outputs[0]]
    r: Reg = ctx.res[op.outputs[0] if load else op.inputs[0]]
    if buf.dtype != r.dtype:
        ctx.add("TYPE002", E, "LDS %s is %s, register %s is %s" % (buf.name, buf.dtype.value, r.name, r.dtype.value), path)
    if op.opcode == "smem_tr_load":
        if not ctx.target.supports(Feature.DS_LOAD_TR):
            ctx.add("TARGET003", E, "smem::tr_load requires %s" % Feature.DS_LOAD_TR.value, path)
        elif buf.dtype.bits not in _TR_LOAD_BITS:
            ctx.add("TARGET003", E, "tr_load supports 16/8/4-bit elements, not %s" % buf.dtype.value, path)
    layout = ctx.res.get(op.attrs.get("layout"))
    if _kind(layout) != "layout":
        ctx.add("IR007", E, "layout %s is not declared" % op.attrs.get("layout"), path)
        return
    elems = _layout_elems(ctx, layout)
    if elems is not None and elems != r.elems:
        ctx.add("TYPE004", E, "layout %s gives %d elems/lane, register %s holds %d" % (layout.name, elems, r.name, r.elems), path)
    _check_vec(ctx, op, r.dtype, int(op.attrs.get("vec", 1)), layout, path)


def _chk_tdm_op(ctx: _Ctx, op: Op, path: str) -> None:
    d: Tdm = ctx.res[op.inputs[0]]
    p = ctx.res.get(d.param)
    if op.opcode == "tdm_store" and isinstance(p, Param) and p.access is Access.READ:
        ctx.add("ACCESS001", E, "tdm store to read-only param %s" % p.name, path)
    if len(op.attrs.get("origin", ())) != len(d.tile):
        ctx.add("TDM001", E, "origin rank must match tile rank %d" % len(d.tile), path)
    if op.opcode == "tdm_store" and d.pad_interval:
        ctx.add("TDM002", E, "tdm padding is load-direction only (stores ignore it)", path)


def _chk_mma(ctx: _Ctx, op: Op, path: str) -> None:
    ins = [ctx.res[n] for n in op.inputs]
    if len(ins) not in (3, 4, 5, 6) or not isinstance(ins[0], Mma) or any(not isinstance(r, Reg) for r in ins[1:]):
        ctx.add("IR007", E, "mma takes (mma, a, b[, c][, scale_a, scale_b]) -> c", path)
        return
    m: Mma = ins[0]
    accumulate = bool(op.attrs.get("accumulate", True))
    scaled = bool(op.attrs.get("scaled", False))
    n_regs = 2 + int(accumulate) + 2 * int(scaled)
    if len(ins) - 1 != n_regs or len(op.outputs) != 1 or not isinstance(ctx.res[op.outputs[0]], Reg):
        ctx.add("IR007", E, "mma operand count does not match accumulate=%s scaled=%s" % (accumulate, scaled), path)
        return
    if scaled != m.scaled:
        ctx.add("MMA003", E, "op scaled=%s but mma %s scaled=%s" % (scaled, m.name, m.scaled), path)
    lane = m.lane_elems(ctx.target.warp_size)
    regs = {"a": ins[1], "b": ins[2], "c": ctx.res[op.outputs[0]]}
    for operand, reg in regs.items():
        want = getattr(m, operand)
        if reg.dtype != want:
            ctx.add(
                "TYPE005", E, "mma %s operand %s is %s, register %s is %s" % (m.name, operand, want.value, reg.name, reg.dtype.value), path
            )
        if reg.elems != lane[operand]:
            ctx.add(
                "TYPE005",
                E,
                "mma %s operand %s needs %d elems/lane, register %s holds %d" % (m.name, operand, lane[operand], reg.name, reg.elems),
                path,
            )
    if accumulate and ins[3].name != op.outputs[0]:
        ctx.add("MMA004", E, "accumulating mma must read and write the same C register", path)
    for s in ins[len(ins) - 2 :] if scaled else ():
        if s.dtype not in (DType.I32, DType.E8M0) or s.elems != 1:
            ctx.add("TYPE005", E, "scale %s must be one i32/e8m0 per lane" % s.name, path)
    waves = len(ctx.waves_of(op.role))
    if m.waves != waves:
        ctx.add("MMA001", E, "mma %s spans %d waves but runs on %d" % (m.name, m.waves, waves), path)


def _chk_cast(ctx: _Ctx, op: Op, path: str) -> None:
    src, dst = ctx.res[op.inputs[0]], ctx.res[op.outputs[0]]
    if src.elems != dst.elems:
        ctx.add("TYPE005", E, "cast %s(%d) -> %s(%d) changes element count" % (src.name, src.elems, dst.name, dst.elems), path)


def _chk_elementwise(ctx: _Ctx, op: Op, path: str) -> None:
    kind = op.attrs.get("kind")
    arity = _ELEMENTWISE_ARITY.get(kind)
    regs = [ctx.res[n] for n in op.inputs + op.outputs]
    if arity is None:
        ctx.add("IR006", E, "unknown elementwise kind %r; known: %s" % (kind, sorted(_ELEMENTWISE_ARITY)), path)
        return
    if len(op.inputs) != arity or len(op.outputs) != 1 or any(not isinstance(r, Reg) for r in regs):
        ctx.add("IR007", E, "elementwise %s takes %d register inputs and one output" % (kind, arity), path)
        return
    if kind in ("scale", "fill") and "scalar" not in op.attrs:
        ctx.add("IR007", E, "elementwise %s needs a scalar" % kind, path)
    if len({(r.dtype, r.elems) for r in regs}) != 1:
        ctx.add("TYPE005", E, "elementwise operands must share dtype and element count (use cast)", path)


def _chk_wait(ctx: _Ctx, op: Op, path: str) -> None:
    try:
        c = Counter(op.attrs.get("counter"))
    except ValueError:
        ctx.add("TARGET006", E, "unknown counter %r" % op.attrs.get("counter"), path)
        return
    limit = ctx.target.counters.get(c)
    if limit is None:
        notes = ctx.target.notes or ("counters on %s: %s" % (ctx.target.name, sorted(x.value for x in ctx.target.counters)),)
        ctx.add("TARGET006", E, "counter %s does not exist on %s" % (c.value, ctx.target.name), path, notes=tuple(notes))
    elif not 0 <= int(op.attrs.get("outstanding", 0)) <= limit:
        ctx.add("SYNC001", E, "outstanding=%s outside [0, %d] for %s" % (op.attrs.get("outstanding"), limit, c.value), path)


def _chk_sync(ctx: _Ctx, op: Op, path: str) -> None:
    if op.opcode == "sync_cluster":
        if not ctx.target.supports(Feature.CLUSTER) or ctx.kernel.cluster is None:
            ctx.add("TARGET009", E, "sync_cluster needs a gfx1250 cluster launch", path)
    if op.role is not None and ctx.waves_of(op.role) != ctx.all_waves:
        ctx.add("SCHED003", E, "%s inside role %s: waves outside the role never arrive (deadlock)" % (op.opcode, op.role), path)


def _chk_barrier(ctx: _Ctx, op: Op, path: str) -> None:
    b: NamedBarrier = ctx.res[op.inputs[0]]
    if op.role is None or op.role not in b.members:
        ctx.add("SYNC005", E, "role %s is not a member of barrier %s" % (op.role, b.name), path)


def _chk_sem(ctx: _Ctx, op: Op, path: str) -> None:
    if op.opcode == "sem_wait":
        if op.attrs.get("cmp") not in ("eq", "lt"):
            ctx.add("IR007", E, "sem_wait cmp must be 'eq' or 'lt'", path)
    if op.role is not None and ctx.waves_of(op.role) != ctx.all_waves:
        ctx.add("SCHED003", E, "workgroup_barrier wait/inc contains s_barrier; it cannot run inside a role", path)


def _chk_loop(ctx: _Ctx, op: Op, path: str) -> None:
    if "trip" not in op.attrs:
        ctx.add("IR007", E, "loop needs a trip count", path)
    if int(op.attrs.get("unroll", 1)) <= 0:
        ctx.add("IR007", E, "unroll must be positive", path)


_OP_CHECKS = {
    "gmem_load": _chk_gmem,
    "gmem_store": _chk_gmem,
    "gmem_atomic_add": _chk_gmem,
    "gmem_async_load": _chk_async,
    "smem_load": _chk_smem,
    "smem_tr_load": _chk_smem,
    "smem_store": _chk_smem,
    "tdm_load": _chk_tdm_op,
    "tdm_store": _chk_tdm_op,
    "mma": _chk_mma,
    "cast": _chk_cast,
    "elementwise": _chk_elementwise,
    "wait": _chk_wait,
    "sync_threads": _chk_sync,
    "sync_cluster": _chk_sync,
    "barrier_arrive": _chk_barrier,
    "barrier_wait": _chk_barrier,
    "sem_wait": _chk_sem,
    "sem_inc": _chk_sem,
    "loop": _chk_loop,
}


# -- synchronization simulation ----------------------------------------------------------------
#
# Executes the schedule in program order for a few concrete trip counts, keeping one FIFO per
# (wave, counter). An LDS stage write is a set of per-wave events; it completes for a wave when
# that wave's ``wait`` drains the event, and becomes visible to other waves only at a
# sync_threads / named-barrier handoff that happens after completion.


@dataclass
class _Write:
    wid: int
    writers: FrozenSet[int]
    pending: Dict[int, int]  # writer wave -> its instructions still in flight
    ordered: bool  # ds_write: the writer itself sees it in program order
    visible: Set[int] = field(default_factory=set)
    readers: Set[int] = field(default_factory=set)


class _Sim:
    def __init__(self, ctx: _Ctx, policy: int) -> None:
        self.ctx, self.policy = ctx, policy
        self.queues: Dict[Tuple[int, Counter], List[Tuple[int, Optional[Tuple[str, int]]]]] = {}
        self.slots: Dict[Tuple[str, int], _Write] = {}
        self.arrivals: Dict[str, List[Tuple[FrozenSet[int], Dict[Tuple[str, int], int], Set[Tuple[Tuple[str, int], int]]]]] = {}
        self.written: Set[str] = {r.name for r in ctx.kernel.regs if r.zero_init}
        self.next_id = 0
        self.env: Dict[str, int] = {s.name: s.minimum for s in ctx.kernel.symbols}
        self.found: List[Tuple[str, Severity, str, str]] = []

    def report(self, code: str, sev: Severity, msg: str, path: str) -> None:
        self.found.append((code, sev, msg, path))

    def run(self, ops: Tuple[Op, ...], prefix: str) -> None:
        for i, op in enumerate(ops):
            path = "%s[%d](%s)" % (prefix, i, op.opcode)
            if op.opcode == "loop":
                self._loop(op, path)
            else:
                self._step(op, path)

    def _loop(self, op: Op, path: str) -> None:
        a = op.attrs
        trip = Expr.from_json(a["trip"])
        tmin = trip.evaluate(self.env)
        hi_env = dict(self.env, **{s.name: s.maximum for s in self.ctx.kernel.symbols if s.maximum is not None})
        tmax = trip.evaluate(hi_env)
        tmin = 1 if tmin is None else max(tmin, 0)
        pipe = self.ctx.res.get(a.get("pipeline"))
        depth = getattr(pipe, "stages", 1) + 2
        choices = [tmin, tmin + 1, max(tmin, depth)]
        n = choices[min(self.policy, len(choices) - 1)]
        if tmax is not None:
            n = min(n, tmax)
        for it in range(n):
            self.env[a["var"]] = it
            self.run(op.body, path + ".body")
        self.env.pop(a["var"], None)

    def _slot(self, buf: str, op: Op, path: str) -> Optional[Tuple[str, int]]:
        lds = self.ctx.res[buf]
        stage = Expr.from_json(op.attrs.get("stage", 0)).evaluate(self.env)
        if stage is None:
            self.report("SCHED007", E, "stage must be computable from loop variables and constants", path)
            return None
        if not 0 <= stage < lds.stages:
            self.report("SCHED007", E, "stage %d outside [0, %d) of LDS %s" % (stage, lds.stages, buf), path)
            return None
        return (buf, stage)

    def _count(self, op: Op, key: str = "layout") -> int:
        """Instructions one layout op issues: opus loops over elems/vec (layout_load_traits::r_elem)."""
        layout = self.ctx.res.get(op.attrs.get(key))
        elems = _layout_elems(self.ctx, layout) if layout is not None else None
        vec = max(1, int(op.attrs.get("vec", 1)))
        return max(1, (elems or vec) // vec)

    def _issue(self, waves: FrozenSet[int], mem: MemClass, slot: Optional[Tuple[str, int]], wid: int, count: int = 1) -> bool:
        """Enqueue ``count`` instructions on each wave's counter; False when the target has no counter."""
        c = self.ctx.target.counter_for(mem)
        if c is None:
            return False
        for w in waves:
            q = self.queues.setdefault((w, c), [])
            q.extend([(wid, slot)] * count)
            if c is Counter.TENSOR and len(q) > self.ctx.target.tdm_in_flight:
                self.report("SYNC006", W, "more than %d tensor DMAs in flight per wave stall issue" % self.ctx.target.tdm_in_flight, "tdm")
        return True

    def _write(self, slot: Tuple[str, int], waves: FrozenSet[int], mem: MemClass, ordered: bool, path: str, count: int = 1) -> None:
        prev = self.slots.get(slot)
        # Any reader other than a lone self-writer may still be reading bytes this write lands on.
        if prev is not None and prev.readers and len(prev.readers | waves) > 1:
            self.report(
                "SYNC004",
                E,
                "overwrites %s stage %d while waves %s may still read it; add sync_threads (or a barrier handoff) after the reads"
                % (slot[0], slot[1], sorted(prev.readers)),
                path,
            )
        self.next_id += 1
        tracked = self._issue(waves, mem, slot, self.next_id, count)
        self.slots[slot] = _Write(self.next_id, waves, {w: count for w in waves} if tracked else {}, ordered or not tracked)

    def _read(self, slot: Tuple[str, int], waves: FrozenSet[int], path: str) -> None:
        st = self.slots.get(slot)
        if st is None:
            self.report("DATA002", W, "reads %s stage %d before any write in this schedule" % slot, path)
            return
        for w in waves:
            own = w in st.writers
            if own and w in st.pending and not st.ordered:
                c = sorted({c.value for (ww, c), q in self.queues.items() if ww == w and any(e[0] == st.wid for e in q)})
                self.report(
                    "SYNC002",
                    E,
                    "reads %s stage %d before its own async write completed; add wait(%s)" % (slot[0], slot[1], "/".join(c)),
                    path,
                )
                break
            if not own or st.writers != frozenset({w}):
                others = st.writers - {w}
                if others and w not in st.visible:
                    hint = "wait on the writers' counter, then sync_threads" if st.pending else "sync_threads after the write"
                    self.report(
                        "SYNC003",
                        E,
                        "reads %s stage %d written by waves %s without a handoff; %s" % (slot[0], slot[1], sorted(others), hint),
                        path,
                    )
                    break
        st.readers |= set(waves)

    def _step(self, op: Op, path: str) -> None:
        ctx, a = self.ctx, op.attrs
        waves = ctx.waves_of(op.role)
        for name in op.inputs:
            r = ctx.res.get(name)
            if isinstance(r, Reg) and name not in self.written:
                self.report("DATA001", E, "register %s is read before it is written (or mark zero_init)" % name, path)
        code = op.opcode
        if code in ("gmem_load",):
            self._issue(waves, MemClass.GMEM_LOAD, None, 0, self._count(op))
        elif code in ("gmem_store", "gmem_atomic_add"):
            self._issue(waves, MemClass.GMEM_STORE, None, 0, self._count(op))
        elif code == "gmem_async_load":
            slot = self._slot(op.outputs[0], op, path)
            if slot:
                self._write(slot, waves, MemClass.GMEM_ASYNC_LDS, False, path, self._count(op, "layout_g"))
        elif code == "smem_store":
            slot = self._slot(op.outputs[0], op, path)
            if slot:
                self._write(slot, waves, MemClass.LDS, True, path, self._count(op))
        elif code in ("smem_load", "smem_tr_load"):
            slot = self._slot(op.inputs[0], op, path)
            if slot:
                self._read(slot, waves, path)
                self._issue(waves, MemClass.LDS, None, 0, self._count(op))
        elif code in ("tdm_load", "tdm_store"):
            d: Tdm = ctx.res[op.inputs[0]]
            slot = self._slot(d.lds, op, path)
            if slot and code == "tdm_load":
                self._write(slot, waves, MemClass.TENSOR_DMA, False, path)
            elif slot:
                self._read(slot, waves, path)
                self._issue(waves, MemClass.TENSOR_DMA, None, 0)
        elif code == "wait":
            self._wait(waves, Counter(a["counter"]), int(a.get("outstanding", 0)))
        elif code in ("sync_threads", "sem_wait", "sem_inc"):
            for st in self.slots.values():
                if not st.pending:
                    st.visible = set(ctx.all_waves)
                st.readers = set()
        elif code == "barrier_arrive":
            snap = {k: st.wid for k, st in self.slots.items() if st.writers & waves and not (st.pending.keys() & waves)}
            reads = {(k, w) for k, st in self.slots.items() for w in st.readers & waves}
            self.arrivals.setdefault(op.inputs[0], []).append((waves, snap, reads))
        elif code == "barrier_wait":
            for _, snap, reads in self.arrivals.pop(op.inputs[0], []):
                for k, wid in snap.items():
                    st = self.slots.get(k)
                    if st is not None and st.wid == wid and not st.pending:
                        st.visible |= set(waves)
                for k, w in reads:
                    if k in self.slots:
                        self.slots[k].readers.discard(w)
        for name in op.outputs:
            self.written.add(name)

    def _wait(self, waves: FrozenSet[int], c: Counter, n: int) -> None:
        for w in waves:
            q = self.queues.get((w, c), [])
            while len(q) > n:
                wid, slot = q.pop(0)
                st = self.slots.get(slot) if slot else None
                if st is not None and st.wid == wid and w in st.pending:
                    st.pending[w] -= 1
                    if not st.pending[w]:
                        del st.pending[w]


def _simulate(ctx: _Ctx) -> None:
    seen: Set[Tuple[str, str, str]] = set()
    for policy in range(3):
        sim = _Sim(ctx, policy)
        sim.run(ctx.kernel.body, "body")
        for code, sev, msg, path in sim.found:
            if (code, path, msg) not in seen:
                seen.add((code, path, msg))
                ctx.add(code, sev, msg, path)
