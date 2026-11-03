"""Create aiter/ops/flydsl/kernels/mega_moe_iod: a copy of mega_moe whose standalone GEMM1 can read W1 from an
IOD-placed layout (MI355X SPX + NPS1). Run inside the container from /app/aiter-test."""
import os
import shutil
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "/app/aiter-test/aiter/ops/flydsl/kernels"
src, dst = os.path.join(root, "mega_moe"), os.path.join(root, "mega_moe_iod")
shutil.rmtree(dst, ignore_errors=True)
shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))


def patch(fn, old, new, count=1):
    p = os.path.join(dst, fn)
    s = open(p).read()
    n = s.count(old)
    assert n == count, f"{fn}: anchor found {n}x, expected {count}:\n{old[:200]}"
    open(p, "w").write(s.replace(old, new))


# ---------------------------------------------------------------- gemm_util.py: placed-address mode for W1
patch("gemm_util.py", """    def __init__(self, *, w_rsrc, num_acc_n, model_dim, cache_modifier=0):
        self._w_rsrc = w_rsrc
        self._num_acc_n = num_acc_n
        self._cache_modifier = int(cache_modifier)
        self._lane = fx.thread_idx.x % 64
        self._stride_nlane = 16
        self._stride_klane = 256
        self._stride_k0 = 1024
        self._stride_n0 = model_dim * 8
""", """    def __init__(self, *, w_rsrc, num_acc_n, model_dim, cache_modifier=0, placed=None):
        self._w_rsrc = w_rsrc
        self._num_acc_n = num_acc_n
        self._cache_modifier = int(cache_modifier)
        self._lane = fx.thread_idx.x % 64
        self._stride_nlane = 16
        self._stride_klane = 256
        self._stride_k0 = 1024
        self._stride_n0 = model_dim * 8
        # IOD-placed W1 (MI355X SPX + NPS1): 8 KiB granules alternate between the two IODs, home = addr[13] ^ phase(page).
        # Tile (e, n) of IOD d = (n % 8) // 4 (the IOD of the XCD that owns N-tile n under the stock grid-stride loop)
        # lives in 2 MiB page e * n_tiles / 2 + local(n); its 16-row group g, granule j sits in slot
        # 2 * (g * granules_per_group + j) + (d ^ phase[page]) of that page.
        self._placed = placed
        assert model_dim % 1024 == 0
        self._gran_per_group = model_dim // 1024
        self._tile_base = None
        self._grp0 = None

    def begin_tile(self, row_base_i32):
        if const_expr(self._placed is None):
            return
        tile_n, n_tiles = self._placed["tile_n"], self._placed["n_tiles"]
        t = row_base_i32 // fx.Int32(tile_n)
        e = t // fx.Int32(n_tiles)
        n = t - e * fx.Int32(n_tiles)
        d = (n & fx.Int32(7)) >> fx.Int32(2)
        local = ((n >> fx.Int32(3)) << fx.Int32(2)) + (n & fx.Int32(3))
        page = e * fx.Int32(n_tiles // 2) + local
        phase = self._placed["phase_rsrc"][page]
        self._tile_base = (page << fx.Int32(21)) + ((d ^ phase) << fx.Int32(13))
        self._grp0 = (row_base_i32 - t * fx.Int32(tile_n)) >> fx.Int32(4)
""")
patch("gemm_util.py", """        lane_row = fx.Int32(self._lane % 16)
        lane_k = fx.Int32(self._lane // 16)
        n_blk = (row_base_i32 // fx.Int32(16)) + fx.Int32(ni)
        k0 = kstep_i32 * fx.Int32(2) + fx.Int32(ksub)
        byte = (
            n_blk * fx.Int32(self._stride_n0)
            + k0 * fx.Int32(self._stride_k0)
            + lane_k * fx.Int32(self._stride_klane)
            + lane_row * fx.Int32(self._stride_nlane)
        )
""", """        lane_row = fx.Int32(self._lane % 16)
        lane_k = fx.Int32(self._lane // 16)
        k0 = kstep_i32 * fx.Int32(2) + fx.Int32(ksub)
        if const_expr(self._placed is not None):
            gran = (self._grp0 + fx.Int32(ni)) * fx.Int32(self._gran_per_group) + (k0 >> fx.Int32(3))
            byte = (
                self._tile_base
                + (gran << fx.Int32(14))
                + ((k0 & fx.Int32(7)) << fx.Int32(10))
                + lane_k * fx.Int32(self._stride_klane)
                + lane_row * fx.Int32(self._stride_nlane)
            )
        else:
            n_blk = (row_base_i32 // fx.Int32(16)) + fx.Int32(ni)
            byte = (
                n_blk * fx.Int32(self._stride_n0)
                + k0 * fx.Int32(self._stride_k0)
                + lane_k * fx.Int32(self._stride_klane)
                + lane_row * fx.Int32(self._stride_nlane)
            )
""")

# ---------------------------------------------------------------- gemm1.py: hook + kernel arg + mode
patch("gemm1.py", """    b_row = sched.gate_base_row(expert) + n_tile_base
""", """    b_row = sched.gate_base_row(expert) + n_tile_base
    b_loader.begin_tile(b_row)
""")
patch("gemm1.py", """    source_rows=0, a_dtype="fp8", out_dtype="fp8", swiglu_limit=0.0, pair_k=False, evec,
    act="silu", situ_beta=1.0, situ_linear_beta=1.0):""", """    source_rows=0, a_dtype="fp8", out_dtype="fp8", swiglu_limit=0.0, pair_k=False, evec,
    act="silu", situ_beta=1.0, situ_linear_beta=1.0, placed=None):""")
patch("gemm1.py", """        model_dim=model_dim,
        cache_modifier=b_cache_modifier,
    )""", """        model_dim=model_dim,
        cache_modifier=b_cache_modifier,
        placed=placed,
    )""")
patch("gemm1.py", """    act: str = "silu", situ_beta: float = 1.0, situ_linear_beta: float = 1.0,
):
    # fmt: on
    \"\"\"Compile standalone group GEMM1 from the fused Stage1 compute body.\"\"\"""", """    act: str = "silu", situ_beta: float = 1.0, situ_linear_beta: float = 1.0, placed_mode: int = 0,
):
    # fmt: on
    \"\"\"Compile standalone group GEMM1 from the fused Stage1 compute body.\"\"\"""")
patch("gemm1.py", """        tile_row_base: fx.Tensor, expert_ids: fx.Tensor, out_scale: fx.Tensor, num_valid: fx.Int32,
        grid_x: fx.Int32,
    ):""", """        tile_row_base: fx.Tensor, expert_ids: fx.Tensor, out_scale: fx.Tensor, phase: fx.Tensor,
        num_valid: fx.Int32, grid_x: fx.Int32,
    ):""")
patch("gemm1.py", """        wave_id = fx.thread_idx.x // 64

        _, run_tile = build_fused_gemm1(""", """        wave_id = fx.thread_idx.x // 64
        placed = None
        if const_expr(placed_mode != 0):
            placed = dict(
                phase_rsrc=ptr_buf_tensor(fx.get_iter(phase), fx.Int32), n_tiles=n_tiles, tile_n=tile_n
            )

        _, run_tile = build_fused_gemm1(""")
patch("gemm1.py", """            act=act, situ_beta=situ_beta, situ_linear_beta=situ_linear_beta,

        )""", """            act=act, situ_beta=situ_beta, situ_linear_beta=situ_linear_beta, placed=placed,
        )""")
patch("gemm1.py", """        tile_row_base: fx.Tensor, expert_ids: fx.Tensor, out_scale: fx.Tensor, num_valid: fx.Int32,
        grid_x: fx.Int32, stream: fx.Stream,
    ):
        kernel(
            out, x, w, scale_x, scale_w, tile_row_base, expert_ids, out_scale, num_valid, grid_x,""", """        tile_row_base: fx.Tensor, expert_ids: fx.Tensor, out_scale: fx.Tensor, phase: fx.Tensor,
        num_valid: fx.Int32, grid_x: fx.Int32, stream: fx.Stream,
    ):
        kernel(
            out, x, w, scale_x, scale_w, tile_row_base, expert_ids, out_scale, phase, num_valid, grid_x,""")
patch("gemm1.py", """    swiglu_limit: float = 0.0, act: str = "silu", situ_beta: float = 1.0, situ_linear_beta: float = 1.0,
):
    # fmt: on
    \"\"\"Run standalone MegaMoEV2 group GEMM1 and return ``(out, out_scale)``.\"\"\"""", """    swiglu_limit: float = 0.0, act: str = "silu", situ_beta: float = 1.0, situ_linear_beta: float = 1.0,
    phase=None, placed_mode: int = 0,
):
    # fmt: on
    \"\"\"Run standalone MegaMoEV2 group GEMM1 and return ``(out, out_scale)``.\"\"\"""")
patch("gemm1.py", """        swiglu_limit=swiglu_limit, act=act, situ_beta=situ_beta, situ_linear_beta=situ_linear_beta,
    )
    _run_compiled(
        launch, out, x, w.view(torch.uint8), scale_x, scale_w.view(torch.uint8),
        tile_row_base, expert_ids, out_scale,
        fx.Int32(num_valid), fx.Int32(grid_x), stream,
    )""", """        swiglu_limit=swiglu_limit, act=act, situ_beta=situ_beta, situ_linear_beta=situ_linear_beta,
        placed_mode=placed_mode,
    )
    if phase is None:
        phase = torch.zeros(1, dtype=torch.int32, device=out.device)
    _run_compiled(
        launch, out, x, w.view(torch.uint8), scale_x, scale_w.view(torch.uint8),
        tile_row_base, expert_ids, out_scale, phase,
        fx.Int32(num_valid), fx.Int32(grid_x), stream,
    )""")

# ================================================================ v2: IOD-placed activations (A) as well
# BWeightLoader: keep the tile geometry and expose the IOD that owns the tile's N-tile (also when B is not placed)
patch("gemm_util.py", """    def __init__(self, *, w_rsrc, num_acc_n, model_dim, cache_modifier=0, placed=None):""",
"""    def __init__(self, *, w_rsrc, num_acc_n, model_dim, cache_modifier=0, placed=None, geom=None):""")
patch("gemm_util.py", """        self._tile_base = None
        self._grp0 = None
""", """        self._tile_base = None
        self._grp0 = None
        self._geom = geom

    def tile_iod(self, row_base_i32):
        # IOD of the XCD that owns this N-tile under the stock grid-stride loop: (n % 8) // 4
        tile_n, n_tiles = self._geom["tile_n"], self._geom["n_tiles"]
        t = row_base_i32 // fx.Int32(tile_n)
        n = t - (t // fx.Int32(n_tiles)) * fx.Int32(n_tiles)
        return (n & fx.Int32(7)) >> fx.Int32(2)
""")
# ATileLoader: per-IOD copy of the activations; a 32-row tile is 28 granules, 4 tiles per IOD per 2 MiB page,
# so the tile base is page * 2 MiB + slot * 512 KiB + (iod ^ phase) * 8 KiB and in-tile offsets get bit 13 inserted
patch("gemm_util.py", """        a_dtype="fp8",
    ):
        self._sort_block_m = sort_block_m""", """        a_dtype="fp8",
        placed=None,
    ):
        self._placed = placed
        self._sort_block_m = sort_block_m""")
patch("gemm_util.py", """    def for_tile(self, tile_row_base_i32):
        \"\"\"Precompute LDS and tile-local global offsets for one M tile.\"\"\"
        if const_expr(self._indexed_input):""", """    def for_tile(self, tile_row_base_i32, m_tile_i32=None, iod_i32=None):
        \"\"\"Precompute LDS and tile-local global offsets for one M tile.\"\"\"
        if const_expr(self._placed is not None):
            page = m_tile_i32 >> fx.Int32(2)
            slot = m_tile_i32 & fx.Int32(3)
            phase = self._placed["phase_rsrc"][page]
            base = (page << fx.Int32(21)) + (slot << fx.Int32(19)) + ((iod_i32 ^ phase) << fx.Int32(13))
            tile_iter = fx.add_offset(fx.get_iter(self._x_tensor), fx.Int64(base))
            input_bytes = 1 << 19
            input_view = fx.Tensor(fx.make_view(tile_iter, fx.make_layout(input_bytes, 1)))
        elif const_expr(self._indexed_input):""")
patch("gemm_util.py", """        for lds_byte, chunk_base in self._chunks:
            group = (chunk_base + koff) // fx.Int32(16)
            regs.append(""", """        for lds_byte, chunk_base in self._chunks:
            off = chunk_base + koff
            if const_expr(self._placed is not None):
                off = off + ((off >> fx.Int32(13)) << fx.Int32(13))
            group = off // fx.Int32(16)
            regs.append(""")
# gemm1.py: pass the tile's m index and IOD to the A loader; geometry + A placement plumbing
patch("gemm1.py", """    b_loader.begin_tile(b_row)
    a_gather.for_tile(tile_input_base)""", """    b_loader.begin_tile(b_row)
    a_gather.for_tile(tile_input_base, m_tile_i32=m_tile, iod_i32=b_loader.tile_iod(b_row))""")
patch("gemm1.py", """    act="silu", situ_beta=1.0, situ_linear_beta=1.0, placed=None):""",
"""    act="silu", situ_beta=1.0, situ_linear_beta=1.0, placed=None, placed_a=None):""")
patch("gemm1.py", """        source_rows=source_rows, a_dtype=a_dtype)""", """        source_rows=source_rows, a_dtype=a_dtype, placed=placed_a)""")
patch("gemm1.py", """        cache_modifier=b_cache_modifier,
        placed=placed,
    )""", """        cache_modifier=b_cache_modifier,
        placed=placed,
        geom=dict(tile_n=tile_n, n_tiles=n_tiles),
    )""")
patch("gemm1.py", """    act: str = "silu", situ_beta: float = 1.0, situ_linear_beta: float = 1.0, placed_mode: int = 0,
):""", """    act: str = "silu", situ_beta: float = 1.0, situ_linear_beta: float = 1.0, placed_mode: int = 0,
    a_placed_mode: int = 0,
):""")
patch("gemm1.py", """        tile_row_base: fx.Tensor, expert_ids: fx.Tensor, out_scale: fx.Tensor, phase: fx.Tensor,
        num_valid: fx.Int32, grid_x: fx.Int32,
    ):""", """        tile_row_base: fx.Tensor, expert_ids: fx.Tensor, out_scale: fx.Tensor, phase: fx.Tensor,
        phase_a: fx.Tensor, num_valid: fx.Int32, grid_x: fx.Int32,
    ):""")
patch("gemm1.py", """                phase_rsrc=ptr_buf_tensor(fx.get_iter(phase), fx.Int32), n_tiles=n_tiles, tile_n=tile_n
            )
""", """                phase_rsrc=ptr_buf_tensor(fx.get_iter(phase), fx.Int32), n_tiles=n_tiles, tile_n=tile_n
            )
        placed_a = None
        if const_expr(a_placed_mode != 0):
            placed_a = dict(phase_rsrc=ptr_buf_tensor(fx.get_iter(phase_a), fx.Int32))
""")
patch("gemm1.py", """            act=act, situ_beta=situ_beta, situ_linear_beta=situ_linear_beta, placed=placed,
        )""", """            act=act, situ_beta=situ_beta, situ_linear_beta=situ_linear_beta, placed=placed,
            placed_a=placed_a,
        )""")
patch("gemm1.py", """        tile_row_base: fx.Tensor, expert_ids: fx.Tensor, out_scale: fx.Tensor, phase: fx.Tensor,
        num_valid: fx.Int32, grid_x: fx.Int32, stream: fx.Stream,
    ):
        kernel(
            out, x, w, scale_x, scale_w, tile_row_base, expert_ids, out_scale, phase, num_valid, grid_x,""",
"""        tile_row_base: fx.Tensor, expert_ids: fx.Tensor, out_scale: fx.Tensor, phase: fx.Tensor,
        phase_a: fx.Tensor, num_valid: fx.Int32, grid_x: fx.Int32, stream: fx.Stream,
    ):
        kernel(
            out, x, w, scale_x, scale_w, tile_row_base, expert_ids, out_scale, phase, phase_a, num_valid, grid_x,""")
patch("gemm1.py", """    phase=None, placed_mode: int = 0,
):""", """    phase=None, placed_mode: int = 0, phase_a=None, a_placed_mode: int = 0,
):""")
patch("gemm1.py", """        placed_mode=placed_mode,
    )
    if phase is None:
        phase = torch.zeros(1, dtype=torch.int32, device=out.device)
    _run_compiled(
        launch, out, x, w.view(torch.uint8), scale_x, scale_w.view(torch.uint8),
        tile_row_base, expert_ids, out_scale, phase,""", """        placed_mode=placed_mode, a_placed_mode=a_placed_mode,
    )
    if phase is None:
        phase = torch.zeros(1, dtype=torch.int32, device=out.device)
    if phase_a is None:
        phase_a = torch.zeros(1, dtype=torch.int32, device=out.device)
    _run_compiled(
        launch, out, x, w.view(torch.uint8), scale_x, scale_w.view(torch.uint8),
        tile_row_base, expert_ids, out_scale, phase, phase_a,""")


IOD_PLACE_PY = """\"\"\"Host-side IOD placement for MI355X SPX + NPS1 (gcnasm nps_locality/flydsl_mega_moe).

In NPS1 the 8 KiB granules of a 2 MiB physical page alternate between the two IODs; the home of granule 0 (the
page phase) is probed per page with libiodprobe.so (IOD_PROBE_LIB).
\"\"\"
import ctypes
import os

import numpy as np
import torch

_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        lib = ctypes.CDLL(os.environ.get("IOD_PROBE_LIB", "@PROBE_LIB@"))
        lib.iod_page_phases.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                                        ctypes.POINTER(ctypes.c_int)]
        _LIB = lib
    return _LIB


def page_phases(buf):
    \"\"\"Home IOD of granule 0 of every 2 MiB page of ``buf`` (uint8, 2 MiB aligned) and the weak-page count.\"\"\"
    assert buf.data_ptr() % (1 << 21) == 0 and buf.numel() % (1 << 21) == 0
    torch.cuda.synchronize()
    out = np.zeros(buf.numel() >> 21, dtype=np.uint8)
    weak = ctypes.c_int(0)
    rc = _lib().iod_page_phases(ctypes.c_void_p(buf.data_ptr()), ctypes.c_size_t(buf.numel()), out.ctypes.data,
                                ctypes.byref(weak))
    assert rc == 0, f"iod_page_phases failed rc={rc}"
    return out, weak.value


def place_w2(w2, epr, model_dim, inter_dim, anti=False):
    \"\"\"IOD-placed copy of the shuffled a8w4 W2 (epr x model_dim x inter_dim fp4) for gemm2_compute_v2(w2_place).

    16-row group r of expert e (gpg = inter_dim / 1024 granules) goes to IOD g = r // (model_dim / 32); within IOD g's
    granule stream it takes slots q .. q + gpg - 1, q = gpg * (e * model_dim / 32 + r - g * model_dim / 32), i.e.
    page q >> 7, physical granule 2 * (q & 127) + (g ^ phase[page]). ``anti`` homes every group on the other IOD.
    \"\"\"
    gpg = inter_dim // 1024
    assert inter_dim % 1024 == 0 and 128 % gpg == 0 and model_dim % 32 == 0
    half = model_dim // 32
    per_iod = epr * half * gpg
    npages = (per_iod + 127) // 128
    src = w2.contiguous().view(torch.uint8).view(epr, 2, half, gpg, 8192)
    dst = torch.zeros(npages << 21, dtype=torch.uint8, device=w2.device)
    ph, weak = page_phases(dst)
    if anti:
        ph = ph ^ 1
    phase = torch.from_numpy(ph.astype(np.int64)).to(w2.device)
    q = torch.arange(per_iod, device=w2.device)
    page = q >> 7
    gran = dst.view(npages * 256, 8192)
    for g in (0, 1):
        idx = page * 256 + 2 * (q & 127) + (g ^ phase[page])
        gran.index_copy_(0, idx, src[:, g].reshape(per_iod, 8192))
    torch.cuda.synchronize()
    return dst, torch.from_numpy(ph.astype(np.int32)).to(w2.device), weak


def phase_table(ph, dev, pad=2):
    \"\"\"int32 device copy of a phase array, padded so lookups of the next pages never leave the table.\"\"\"
    return torch.from_numpy(np.concatenate([ph, np.zeros(pad, ph.dtype)]).astype(np.int32)).to(dev)


def place_w1(w1, epr, model_dim, inter_dim, tile_n, anti=False):
    \"\"\"IOD-placed copy of the shuffled a8w4 W1 for the fused Stage1 with per-IOD queues (BWeightLoader "stream").

    N-tile n of an expert (tile_n rows, gpt = tile_n / 16 * model_dim / 1024 granules) belongs to IOD
    g = n // (n_tiles / 2); tile (e, n) is tile s = e * n_tiles / 2 + n - g * n_tiles / 2 of IOD g's granule stream,
    granule q = s * gpt + j of the stream lives at page q >> 7, physical granule 2 * (q & 127) + (g ^ phase[page]).
    \"\"\"
    gpg = model_dim // 1024
    n_tiles = 2 * inter_dim // tile_n
    half = n_tiles // 2
    gpt = tile_n // 16 * gpg
    assert model_dim % 1024 == 0 and n_tiles % 2 == 0
    per_iod = epr * half * gpt
    npages = (per_iod + 127) // 128
    src = w1.contiguous().view(torch.uint8).view(epr, 2, half * gpt, 8192)
    dst = torch.zeros(npages << 21, dtype=torch.uint8, device=w1.device)
    ph, weak = page_phases(dst)
    if anti:
        ph = ph ^ 1
    phase = torch.from_numpy(ph.astype(np.int64)).to(w1.device)
    q = torch.arange(per_iod, device=w1.device)
    page = q >> 7
    gran = dst.view(npages * 256, 8192)
    for g in (0, 1):
        idx = page * 256 + 2 * (q & 127) + (g ^ phase[page])
        gran.index_copy_(0, idx, src[:, g].reshape(per_iod, 8192))
    torch.cuda.synchronize()
    return dst, phase_table(ph, w1.device), weak


def alloc_placed_rx(op, rank, world, dev):
    \"\"\"Symmetric receive buffer with two copies of every dispatched row (ATileLoader "insert13").

    Logical byte o of the stock receive buffer is stored at o + (o & ~8191) and + 8192, i.e. in both granules of
    a pair; the copy homed on IOD d is ((d ^ phase[o >> 20]) << 13). Returns (raw, view, p2p table, phases, weak).
    \"\"\"
    import mori.shmem as ms
    from mori.shmem import mori_shmem_create_tensor

    from ..flydsl_dispatch_combine_intranode_op import build_p2p_table

    nbytes = op.payload_rows_max * op.row_bytes
    placed = ((2 * nbytes + (1 << 21) - 1) >> 21) << 21
    raw = mori_shmem_create_tensor((placed + (1 << 21),), torch.int8)
    raw.zero_()  # like the stock receive buffer: tile rows past an expert's end are read but never used
    off = (-raw.data_ptr()) % (1 << 21)
    rx = raw[off: off + placed]
    for pe in range(world):
        if ms.shmem_ptr_p2p(rx.data_ptr(), rank, pe) % (1 << 21):
            raise RuntimeError("symmetric heap mapping of a peer is not 2 MiB aligned")
    ph, weak = page_phases(rx.view(torch.uint8))
    return raw, rx, build_p2p_table(rx, rank, world, dev), phase_table(ph, dev), weak
"""

# ================================================================ v3: Stage 2 (GEMM2 + P2P scatter) per-IOD queue + W2
# iod_place.py: page-phase probe + W2 placement (host side)
with open(os.path.join(dst, "iod_place.py"), "w") as f:
    # default probe library: libiodprobe.so next to this script (override with IOD_PROBE_LIB)
    f.write(IOD_PLACE_PY.replace("@PROBE_LIB@", os.path.join(os.path.dirname(os.path.abspath(__file__)), "libiodprobe.so")))

# gemm2.py: IOD-placed W2 addressing
patch("gemm2.py", """    explicit_m_row=None,
    explicit_n_block=None,
    explicit_expert=None,
):""", """    explicit_m_row=None,
    explicit_n_block=None,
    explicit_expert=None,
    w2_place=None,
):""")
patch("gemm2.py", """            (e * N_OUT_rt + n_block_idx * BN + wave * (BN // 4) + j * 16) * KH4,
        )
        for j in range_constexpr(numAccN)
    ]
""", """            (e * N_OUT_rt + n_block_idx * BN + wave * (BN // 4) + j * 16) * KH4,
        )
        for j in range_constexpr(numAccN)
    ]
    w2_base_b = None
    if const_expr(w2_place is not None):
        # IOD-placed W2 (MI355X SPX + NPS1): 16-row group r of local expert e is homed on IOD g = r // (N_OUT / 32),
        # the IOD that owns n-blocks [g * nnb / 2, (g + 1) * nnb / 2). Its gpg 8 KiB granules are slots q .. q + gpg - 1,
        # q = gpg * (e * N_OUT / 32 + r - g * N_OUT / 32), of IOD g's stream: page q >> 7, granule 2 * (q & 127) + (g ^ phase).
        w2_half = N_OUT_rt // fx.Int32(32)
        w2_base_b = []
        for j in range_constexpr(numAccN):
            r = n_block_idx * fx.Int32(BN // 16) + wave * fx.Int32(BN // 64) + fx.Int32(j)
            g = (r >= w2_half).select(fx.Int32(1), fx.Int32(0))
            q = (e * w2_half + r - g * w2_half) * fx.Int32(int(w2_place["gpg"]))
            page = q >> fx.Int32(7)
            ph = w2_place["phase_rsrc"][page]
            base_b = (page << fx.Int32(21)) + ((q & fx.Int32(127)) << fx.Int32(14)) + ((g ^ ph) << fx.Int32(13))
            w2_base_b.append(fx.Int32(rocdl.readfirstlane(T.i32, base_b)))
""")
patch("gemm2.py", """                bq_off_dw = (
                    bq_base_dw[j]
                    + lane_div_16 * fx.Int32(64)
                    + lane_mod_16 * fx.Int32(4)
                    + kt_rt * fx.Int32(kHalves * 256)
                    + fx.Int32(half * 256)
                )
""", """                if const_expr(w2_place is not None):
                    kh = kt_rt * fx.Int32(kHalves) + fx.Int32(half)  # 1 KiB K-chunk of the 16-row group
                    bq_off_dw = (
                        w2_base_b[j]
                        + ((kh >> fx.Int32(3)) << fx.Int32(14))
                        + ((kh & fx.Int32(7)) << fx.Int32(10))
                        + lane_div_16 * fx.Int32(256)
                        + lane_mod_16 * fx.Int32(16)
                    ) >> fx.Int32(2)
                else:
                    bq_off_dw = (
                        bq_base_dw[j]
                        + lane_div_16 * fx.Int32(64)
                        + lane_mod_16 * fx.Int32(4)
                        + kt_rt * fx.Int32(kHalves * 256)
                        + fx.Int32(half * 256)
                    )
""")

# mega_moe_stage2.py: per-IOD queue schedule, W2 placement plumbing, two extra pointer args
patch("mega_moe_stage2.py", """from flydsl.runtime.device import get_rocm_arch
""", """from flydsl.runtime.device import get_rocm_arch
from flydsl._mlir.dialects import llvm as _llvm

from .. import communication_ops_utils as comm_ops
""")
patch("mega_moe_stage2.py", """    runtime_pair_skip: bool = False, scatter_vec: int = 8):
# fmt: on
    \"\"\"Compile fused GEMM2 and weighted cross-rank P2P scatter.\"\"\"""", """    runtime_pair_skip: bool = False, scatter_vec: int = 8, iod_sched: int = 0, w2_placed: int = 0):
# fmt: on
    \"\"\"Compile fused GEMM2 and weighted cross-rank P2P scatter.\"\"\"""")
patch("mega_moe_stage2.py", """    lds_bytes = lds_peer_off + npes * 8
""", """    lds_bytes = lds_peer_off + npes * 8
    lds_ticket_off = lds_bytes
    if iod_sched:
        lds_bytes += 16
""")
patch("mega_moe_stage2.py", """        f"_sv{scatter_vec}_tb2_rsm1"
""", """        f"_sv{scatter_vec}_tb2_rsm1"
        f"_iod{int(iod_sched)}w{int(w2_placed)}"
""")
patch("mega_moe_stage2.py", """        arg_p2p_comb_inp: fx.Int64, i32_max_m_blocks: fx.Int32,
        i32_inter: fx.Int32, i32_hidden: fx.Int32, i32_kpad: fx.Int32, i32_npad: fx.Int32):""",
"""        arg_p2p_comb_inp: fx.Int64, arg_iod_ctr: fx.Int64, arg_w2_phase: fx.Int64, i32_max_m_blocks: fx.Int32,
        i32_inter: fx.Int32, i32_hidden: fx.Int32, i32_kpad: fx.Int32, i32_npad: fx.Int32):""")
patch("mega_moe_stage2.py", """        p2p_table = ptr_buf_tensor(arg_p2p_comb_inp, fx.Int64)
""", """        p2p_table = ptr_buf_tensor(arg_p2p_comb_inp, fx.Int64)
        w2_place = None
        if const_expr(w2_placed):
            w2_place = dict(phase_rsrc=ptr_buf_tensor(arg_w2_phase, fx.Int32), gpg=inter_dim // 1024)
""")
patch("mega_moe_stage2.py", """                expert_offset=_expert_offset)
            p2p_scatter_epilog(""", """                expert_offset=_expert_offset, w2_place=w2_place)
            p2p_scatter_epilog(""")
patch("mega_moe_stage2.py", """        if const_expr(not persist and g2_spart <= 0):
            bound = total_m_blocks * num_n_blocks""", """        if const_expr(iod_sched):
            # Per-IOD work queue (MI355X SPX): a block serves the queue of the IOD its XCD sits on (XCC_ID >> 2).
            # IOD g owns n-blocks [g * nnb / 2, (g + 1) * nnb / 2) of every m-block, so its W2 can be homed on g.
            # The next ticket is claimed before the current unit runs; the last block to retire zeroes the
            # counters (IOD 0 at +0, IOD 1 at +256, retired blocks at +512 bytes) for the next launch.
            xcc = fx.Int32(_llvm.inline_asm(T.i32, [], "s_getreg_b32 $0, hwreg(20, 0, 4)", "=s", has_side_effects=True))
            iod = xcc >> fx.Int32(2)
            half_nb = num_n_blocks // fx.Int32(2)
            work_iod = total_m_blocks * half_nb
            ctr_addr = arg_iod_ctr + fx.Int64(iod) * fx.Int64(256)
            if tx_i32 == fx.Int32(0):
                first = fx.Int32(comm_ops.atomic_add_agent(ctr_addr, fx.Int32(1)))
                fx.ptr_store(first, lds_typed_ptr(fx.Int32(lds_ticket_off), T.i32, align=4))
            fx.barrier()
            w = fx.Int32(rocdl.readfirstlane(
                T.i32, fx.ptr_load(lds_typed_ptr(fx.Int32(lds_ticket_off), T.i32, align=4)).ir_value()))
            while w < work_iod:
                nxt = fx.Int32(0)
                if tx_i32 == fx.Int32(0):
                    nxt = fx.Int32(comm_ops.atomic_add_agent(ctr_addr, fx.Int32(1)))
                m_block = w // half_nb
                n_block = iod * half_nb + (w - m_block * half_nb)
                run_unskipped_unit(m_block * num_n_blocks + n_block, m_block)
                fx.barrier()
                if tx_i32 == fx.Int32(0):
                    fx.ptr_store(nxt, lds_typed_ptr(fx.Int32(lds_ticket_off), T.i32, align=4))
                fx.barrier()
                w = fx.Int32(rocdl.readfirstlane(
                    T.i32, fx.ptr_load(lds_typed_ptr(fx.Int32(lds_ticket_off), T.i32, align=4)).ir_value()))
            if tx_i32 == fx.Int32(0):
                retired = fx.Int32(comm_ops.atomic_add_agent(arg_iod_ctr + fx.Int64(512), fx.Int32(1)))
                if retired == fx.Int32(fx.grid_dim.x) - fx.Int32(1):
                    for off in range_constexpr(3):
                        comm_ops.store_i32_global_agent_release(arg_iod_ctr + fx.Int64(off * 256), fx.Int32(0))
        elif const_expr(not persist and g2_spart <= 0):
            bound = total_m_blocks * num_n_blocks""")
patch("mega_moe_stage2.py", """        arg_p2p_comb_inp: fx.Int64, i32_max_m_blocks: fx.Int32,
        i32_grid_blocks: fx.Int32, i32_inter: fx.Int32, i32_hidden: fx.Int32, i32_kpad: fx.Int32,
        i32_npad: fx.Int32, stream: fx.Stream):
    # fmt: on
        num_n_blocks = i32_hidden // fx.Int32(BN)
        grid_x = i32_grid_blocks * num_n_blocks""", """        arg_p2p_comb_inp: fx.Int64, arg_iod_ctr: fx.Int64, arg_w2_phase: fx.Int64, i32_max_m_blocks: fx.Int32,
        i32_grid_blocks: fx.Int32, i32_inter: fx.Int32, i32_hidden: fx.Int32, i32_kpad: fx.Int32,
        i32_npad: fx.Int32, stream: fx.Stream):
    # fmt: on
        num_n_blocks = i32_hidden // fx.Int32(BN)
        grid_x = i32_grid_blocks if iod_sched else i32_grid_blocks * num_n_blocks""")
patch("mega_moe_stage2.py", """            arg_p2p_comb_inp, i32_max_m_blocks, i32_inter,""", """            arg_p2p_comb_inp, arg_iod_ctr, arg_w2_phase, i32_max_m_blocks, i32_inter,""")
patch("mega_moe_stage2.py", """    runtime_pair_skip=False, scatter_vec=8, a_dtype="fp8"):""",
"""    runtime_pair_skip=False, scatter_vec=8, a_dtype="fp8", iod_sched=0, w2_placed=0, arg_iod_ctr=None,
    arg_w2_phase=None, iod_grid=0):""", count=2)
patch("mega_moe_stage2.py", """        scatter_vec=scatter_vec,
        a_dtype=a_dtype,
    )""", """        scatter_vec=scatter_vec,
        a_dtype=a_dtype,
        iod_sched=int(iod_sched), w2_placed=int(w2_placed),
    )""", count=2)
patch("mega_moe_stage2.py", """    grid_blocks = launch_cu_num if persist else max_m_blocks""",
"""    grid_blocks = int(iod_grid) if iod_sched else (launch_cu_num if persist else max_m_blocks)
    arg_iod_ctr = fx.Int64(0) if arg_iod_ctr is None else arg_iod_ctr
    arg_w2_phase = fx.Int64(0) if arg_w2_phase is None else arg_w2_phase""", count=2)
patch("mega_moe_stage2.py", """        arg_count_matrix, arg_pair_config, arg_parity, arg_p2p, fx.Int32(max_m_blocks),""",
"""        arg_count_matrix, arg_pair_config, arg_parity, arg_p2p, arg_iod_ctr, arg_w2_phase, fx.Int32(max_m_blocks),""",
      count=2)

# mega_moe_v2.py: iod_mode flags, W2 placement, Stage2 plumbing
patch("mega_moe_v2.py", """        act: str = "silu", situ_beta: float = 1.0, situ_linear_beta: float = 1.0):""",
"""        act: str = "silu", situ_beta: float = 1.0, situ_linear_beta: float = 1.0, iod_mode: str = "stock"):""")
patch("mega_moe_v2.py", """        self.w2_scale = w2_scale if w2_scale.is_contiguous() else w2_scale.contiguous()
""", """        self.w2_scale = w2_scale if w2_scale.is_contiguous() else w2_scale.contiguous()
        # IOD locality (MI355X SPX + NPS1), "+"-joined flags: s2q = Stage2 per-IOD queue, s2w / s2x = W2 homed on the
        # owning / the other IOD (both need s2q)
        self._iod = {flag for flag in iod_mode.split("+") if flag and flag != "stock"}
        self._iod_s2_grid_mult = int(os.environ.get("IOD_S2_GRID_MULT", "2"))
        for flag in [f for f in self._iod if f.startswith("s2g")]:  # s2gN: N persistent Stage2 blocks per CU
            self._iod_s2_grid_mult = int(flag[3:])
            self._iod.discard(flag)
        unknown = self._iod - {"s2q", "s2w", "s2x"}
        if unknown:
            raise ValueError(f"unknown iod_mode flags {sorted(unknown)}")
        if self._iod & {"s2w", "s2x"} and "s2q" not in self._iod:
            raise ValueError("W2 placement needs the per-IOD Stage2 queue (s2q)")
        self._iod_ctr = torch.zeros(256, dtype=torch.int32, device=self.dev)
        self._iod_w2_phase = torch.zeros(1, dtype=torch.int32, device=self.dev)
        self.iod_info = {}
        if self._iod & {"s2w", "s2x"}:
            from .iod_place import place_w2

            self.w2, self._iod_w2_phase, weak = place_w2(
                self.w2, self.epr, self.model_dim, self.inter_dim, anti="s2x" in self._iod
            )
            self.iod_info["w2_weak_pages"] = weak
""")
patch("mega_moe_v2.py", """            runtime_pair_skip=runtime_pair_skip,
            scatter_vec=scatter_vec,
            **self._g2_invariants_by_quant[config.p2p_quant],
        )""", """            runtime_pair_skip=runtime_pair_skip,
            scatter_vec=scatter_vec,
            iod_sched=int("s2q" in self._iod),
            w2_placed=int(bool(self._iod & {"s2w", "s2x"})),
            arg_iod_ctr=fx.Int64(self._iod_ctr.data_ptr()),
            arg_w2_phase=fx.Int64(self._iod_w2_phase.data_ptr()),
            iod_grid=int(self._s1_num_cu * self._iod_s2_grid_mult),
            **self._g2_invariants_by_quant[config.p2p_quant],
        )""")


# ================================================================ v4: fused Stage1 (dispatch + GEMM1) per-IOD queue, W1, tokens
# gemm_util.py: W1 "stream" placement, N-tile ownership by halves, A "insert13" placement (per-IOD row copies)
patch("gemm_util.py", """        return (n & fx.Int32(7)) >> fx.Int32(2)

    def begin_tile(self, row_base_i32):
        if const_expr(self._placed is None):
            return
        tile_n, n_tiles = self._placed["tile_n"], self._placed["n_tiles"]""", """        if const_expr(self._geom.get("owner", "rr") == "half"):
            # per-IOD queue: IOD g owns N-tiles [g * n_tiles / 2, (g + 1) * n_tiles / 2)
            return n // fx.Int32(n_tiles // 2)
        return (n & fx.Int32(7)) >> fx.Int32(2)

    def _begin_tile_stream(self, row_base_i32):
        # Fused Stage1 with per-IOD queues: tile (e, n) of IOD g = n // (n_tiles / 2) is tile
        # s = e * n_tiles / 2 + n - g * n_tiles / 2 of IOD g's granule stream (gpt granules each, may straddle pages);
        # the phases of every page it touches are loaded once per tile.
        tile_n, n_tiles = self._placed["tile_n"], self._placed["n_tiles"]
        half = n_tiles // 2
        gpt = (tile_n // 16) * self._gran_per_group
        t = row_base_i32 // fx.Int32(tile_n)
        e = t // fx.Int32(n_tiles)
        n = t - e * fx.Int32(n_tiles)
        g = n // fx.Int32(half)
        q0 = (e * fx.Int32(half) + n - g * fx.Int32(half)) * fx.Int32(gpt)
        p0 = q0 >> fx.Int32(7)
        tbl = self._placed["phase_rsrc"]
        self._s_q0, self._s_p0, self._s_g = q0, p0, g
        self._s_ph = [
            fx.Int32(rocdl.readfirstlane(T.i32, tbl[p0 + fx.Int32(i)]))
            for i in range_constexpr((gpt + 126) // 128 + 1)
        ]
        self._grp0 = (row_base_i32 - t * fx.Int32(tile_n)) >> fx.Int32(4)

    def begin_tile(self, row_base_i32):
        if const_expr(self._placed is None):
            return
        if const_expr(self._placed.get("mode") == "stream"):
            self._begin_tile_stream(row_base_i32)
            return
        tile_n, n_tiles = self._placed["tile_n"], self._placed["n_tiles"]""")
patch("gemm_util.py", """        k0 = kstep_i32 * fx.Int32(2) + fx.Int32(ksub)
        if const_expr(self._placed is not None):
            gran = (self._grp0 + fx.Int32(ni)) * fx.Int32(self._gran_per_group) + (k0 >> fx.Int32(3))""",
"""        k0 = kstep_i32 * fx.Int32(2) + fx.Int32(ksub)
        if const_expr(self._placed is not None and self._placed.get("mode") == "stream"):
            q = self._s_q0 + (self._grp0 + fx.Int32(ni)) * fx.Int32(self._gran_per_group) + (k0 >> fx.Int32(3))
            page = q >> fx.Int32(7)
            dp = page - self._s_p0
            ph = self._s_ph[0]
            for i in range_constexpr(1, len(self._s_ph)):
                ph = (dp == fx.Int32(i)).select(self._s_ph[i], ph)
            byte = (
                (page << fx.Int32(21))
                + ((q & fx.Int32(127)) << fx.Int32(14))
                + ((self._s_g ^ ph) << fx.Int32(13))
                + ((k0 & fx.Int32(7)) << fx.Int32(10))
                + lane_k * fx.Int32(self._stride_klane)
                + lane_row * fx.Int32(self._stride_nlane)
            )
        elif const_expr(self._placed is not None):
            gran = (self._grp0 + fx.Int32(ni)) * fx.Int32(self._gran_per_group) + (k0 >> fx.Int32(3))""")
patch("gemm_util.py", """    def for_tile(self, tile_row_base_i32, m_tile_i32=None, iod_i32=None):
        \"\"\"Precompute LDS and tile-local global offsets for one M tile.\"\"\"
        if const_expr(self._placed is not None):
            page = m_tile_i32 >> fx.Int32(2)""", """    def _insert13(self, x):
        # in-tile logical byte -> byte of this IOD's copy, relative to the tile's granule pair (see for_tile)
        y = self._p_rb + x
        gy = y >> fx.Int32(13)
        if const_expr(self._placed.get("select") == "parity"):
            # control: alternate the two copies per logical granule, i.e. the stock 50 / 50 IOD mix
            c = (self._p_gin + gy) & fx.Int32(1)
        else:
            c = ((self._p_gin + gy) >= fx.Int32(128)).select(self._p_c1, self._p_c0)
            if const_expr(self._placed.get("select") == "anti"):
                c = c ^ fx.Int32(1)  # control: always the copy on the other IOD
        return y + (gy << fx.Int32(13)) + (c << fx.Int32(13))

    def for_tile(self, tile_row_base_i32, m_tile_i32=None, iod_i32=None):
        \"\"\"Precompute LDS and tile-local global offsets for one M tile.\"\"\"
        if const_expr(self._placed is not None and self._placed.get("mode") == "insert13"):
            # Per-IOD copies of the dispatched rows: logical byte o has its copies at o + (o & ~8191) and + 8192,
            # the one homed on IOD d is ((d ^ phase[o >> 20]) << 13). A 32-row tile spans at most two pages.
            o_b = fx.Int64(tile_row_base_i32) * fx.Int64(self._row_bytes)
            g_b = fx.Int32(o_b >> fx.Int64(13))
            self._p_rb = fx.Int32(o_b & fx.Int64(8191))
            self._p_gin = g_b & fx.Int32(127)
            p0 = g_b >> fx.Int32(7)
            tbl = self._placed["phase_rsrc"]
            self._p_c0 = iod_i32 ^ fx.Int32(rocdl.readfirstlane(T.i32, tbl[p0]))
            self._p_c1 = iod_i32 ^ fx.Int32(rocdl.readfirstlane(T.i32, tbl[p0 + fx.Int32(1)]))
            tile_iter = fx.add_offset(fx.get_iter(self._x_tensor), (o_b >> fx.Int64(13)) << fx.Int64(14))
            input_bytes = 2 * (self._sort_block_m * self._row_bytes + 8192)
            input_view = fx.Tensor(fx.make_view(tile_iter, fx.make_layout(input_bytes, 1)))
        elif const_expr(self._placed is not None):
            page = m_tile_i32 >> fx.Int32(2)""")
patch("gemm_util.py", """            off = chunk_base + koff
            if const_expr(self._placed is not None):
                off = off + ((off >> fx.Int32(13)) << fx.Int32(13))""", """            off = chunk_base + koff
            if const_expr(self._placed is not None and self._placed.get("mode") == "insert13"):
                off = self._insert13(off)
            elif const_expr(self._placed is not None):
                off = off + ((off >> fx.Int32(13)) << fx.Int32(13))""")
patch("gemm_util.py", """                    src_byte = (
                        source_row * fx.Int32(self._row_bytes)
                        + koff
                        + logical_chunk * fx.Int32(16)
                    )
""", """                    src_byte = (
                        source_row * fx.Int32(self._row_bytes)
                        + koff
                        + logical_chunk * fx.Int32(16)
                    )
                    if const_expr(self._placed is not None and self._placed.get("mode") == "insert13"):
                        src_byte = self._insert13(src_byte)
""")
# gemm1.py: ownership rule for the A copy selection
patch("gemm1.py", """    act="silu", situ_beta=1.0, situ_linear_beta=1.0, placed=None, placed_a=None):""",
"""    act="silu", situ_beta=1.0, situ_linear_beta=1.0, placed=None, placed_a=None, iod_owner="rr"):""")
patch("gemm1.py", """        geom=dict(tile_n=tile_n, n_tiles=n_tiles),""", """        geom=dict(tile_n=tile_n, n_tiles=n_tiles, owner=iod_owner),""")

# dispatch.py: double-write of every dispatched row into the per-IOD receive layout
patch("dispatch.py", """    indexed_payload=False,
    fast_copy,
):""", """    indexed_payload=False,
    fast_copy,
    placed_rx=False,
):""")
patch("dispatch.py", """            if const_expr(fast_copy):
                _store_token(destination_buffer, token_vals)
            else:""", """            if const_expr(placed_rx):
                # per-IOD copies (s1a): logical byte o of the receive buffer goes to o + (o & ~8191) and + 8192
                o_row = fx.Int64(payload_row) * fx.Int64(fz_nbytes)
                rx_base = token_remote if const_expr(hoist_remote_resources) else token_table[destination]
                placed_buffer = ptr_buf_tensor(
                    rx_base + ((o_row >> fx.Int64(13)) << fx.Int64(14)), fx.Int32, unit_elems=4
                )
                _store_token_placed(placed_buffer, token_vals, fx.Int32(o_row & fx.Int64(8191)))
            elif const_expr(fast_copy):
                _store_token(destination_buffer, token_vals)
            else:""")
patch("dispatch.py", """        def _store_token(destination_buffer, vals):
""", """        def _store_token_placed(destination_buffer, vals, row_rb):
            for j in range_constexpr(len(vals)):
                unit = lane + fx.Int32(j * 64)
                y = row_rb + unit * fx.Int32(16)
                phys = (y + ((y >> fx.Int32(13)) << fx.Int32(13))) >> fx.Int32(4)
                if const_expr(j < full_units):
                    buf_copy_store(destination_buffer, phys, vals[j], fx.Int32, unit_elems=4)
                    buf_copy_store(destination_buffer, phys + fx.Int32(512), vals[j], fx.Int32, unit_elems=4)
                else:
                    if unit < fx.Int32(fz_n_i32 // 4):
                        buf_copy_store(destination_buffer, phys, vals[j], fx.Int32, unit_elems=4)
                        buf_copy_store(destination_buffer, phys + fx.Int32(512), vals[j], fx.Int32, unit_elems=4)

        def _store_token(destination_buffer, vals):
""")

# mega_moe_stage1.py: XCC_ID-based per-IOD queue, placed W1 / A, placed dispatch writes (iod_s1 bitmask 1/2/4)
patch("mega_moe_stage1.py", """from flydsl.runtime.device import get_rocm_arch
""", """from flydsl.runtime.device import get_rocm_arch
from flydsl._mlir.dialects import llvm as _llvm
""")
patch("mega_moe_stage1.py", """    DispatchSlot,
    emit_direct_fixed_slot_finalize,""", """    DISPATCH_TABLE_SIZE,
    DispatchSlot,
    emit_direct_fixed_slot_finalize,""")
patch("mega_moe_stage1.py", """    bounds_check: bool = False,
    _return_kernel_spec: bool = False,
):""", """    bounds_check: bool = False,
    iod_s1: int = 0,
    _return_kernel_spec: bool = False,
):""")
patch("mega_moe_stage1.py", """    FAST_COPY = compact_dispatch and envs.AITER_MEGA_DISPATCH_FAST_COPY
""", """    FAST_COPY = compact_dispatch and envs.AITER_MEGA_DISPATCH_FAST_COPY
    if iod_s1:
        # IOD locality (MI355X SPX + NPS1): 1 = per-IOD queue, 2 = placed W1, 4 = per-IOD copies of the tokens,
        # 8 / 16 = controls that read the copies in the stock 50 / 50 IOD mix / from the other IOD
        assert compact_dispatch and N_TILES % 2 == 0 and iod_s1 & 1
        assert not (iod_s1 & 4) or (FAST_COPY and not indexed_payload and a_dtype == "fp8")
""")
patch("mega_moe_stage1.py", """        f"{'_bc1' if bounds_check else ''}"
    )""", """        f"{'_bc1' if bounds_check else ''}"
        f"_iod{iod_s1}"
    )""")
patch("mega_moe_stage1.py", """        disp_rsrc = ptr_buf_tensor(addr_disp, fx.Int64)
""", """        disp_rsrc = ptr_buf_tensor(addr_disp, fx.Int64)
        iod = fx.Int32(0)
        placed_w1 = None
        placed_a = None
        if const_expr(iod_s1 != 0):
            xcc = fx.Int32(_llvm.inline_asm(T.i32, [], "s_getreg_b32 $0, hwreg(20, 0, 4)", "=s", has_side_effects=True))
            iod = xcc >> fx.Int32(2)
        if const_expr(iod_s1 & 2):
            placed_w1 = dict(mode="stream", tile_n=tile_n, n_tiles=N_TILES,
                             phase_rsrc=ptr_buf_tensor(disp_rsrc[fx.Int32(DISPATCH_TABLE_SIZE)], fx.Int32))
        if const_expr(iod_s1 & 4):
            placed_a = dict(mode="insert13", select="parity" if iod_s1 & 8 else "anti" if iod_s1 & 16 else "iod",
                            phase_rsrc=ptr_buf_tensor(disp_rsrc[fx.Int32(DISPATCH_TABLE_SIZE + 1)], fx.Int32))
""")
patch("mega_moe_stage1.py", """            act=act, situ_beta=situ_beta, situ_linear_beta=situ_linear_beta,
        )

        if tid == fx.Int32(0):""", """            act=act, situ_beta=situ_beta, situ_linear_beta=situ_linear_beta,
            placed=placed_w1, placed_a=placed_a, iod_owner="half" if iod_s1 else "rr",
        )

        if tid == fx.Int32(0):""")
patch("mega_moe_stage1.py", """                if const_expr(compact_dispatch):
                    if use_ready_order:
                        first_work = fx.Int32(""", """                if const_expr(compact_dispatch and iod_s1 != 0):
                    # per-IOD queue (head shard g): IOD g takes N-tiles [g * N_TILES / 2, (g + 1) * N_TILES / 2) of
                    # every M slot (ready-order slot or M tile), so W1 / tokens can be homed on g
                    iod_work = fx.Int32(
                        comm_ops.atomic_add_agent(a_work_head + fx.Int64(iod) * 64, fx.Int32(work_batch))
                    )
                    iod_slot = iod_work // fx.Int32(N_TILES // 2)
                    first_work = (
                        iod_slot * fx.Int32(N_TILES)
                        + iod * fx.Int32(N_TILES // 2)
                        + (iod_work - iod_slot * fx.Int32(N_TILES // 2))
                    )
                elif const_expr(compact_dispatch):
                    if use_ready_order:
                        first_work = fx.Int32(""")
patch("mega_moe_stage1.py", """                    fast_copy=FAST_COPY,
                )""", """                    fast_copy=FAST_COPY,
                    placed_rx=bool(iod_s1 & 4),
                )""")
patch("mega_moe_stage1.py", """    swiglu_limit=0.0, act="silu", situ_beta=1.0, situ_linear_beta=1.0, bounds_check=False):
    launch = compile_mega_moe_stage1(""", """    swiglu_limit=0.0, act="silu", situ_beta=1.0, situ_linear_beta=1.0, bounds_check=False, iod_s1=0):
    launch = compile_mega_moe_stage1(""")
patch("mega_moe_stage1.py", """        bounds_check=bounds_check,
    )
    _run_compiled(""", """        bounds_check=bounds_check, iod_s1=iod_s1,
    )
    _run_compiled(""")

# mega_moe_v2.py: Stage1 flags, placement and receive-buffer setup, non-bundle launch for IOD variants
patch("mega_moe_v2.py", """        unknown = self._iod - {"s2q", "s2w", "s2x"}""",
"""        unknown = self._iod - {"s2q", "s2w", "s2x", "s1q", "s1w", "s1x", "s1a", "s1z", "s1y"}
        if self._iod & {"s1w", "s1x", "s1a", "s1z", "s1y"} and "s1q" not in self._iod:
            raise ValueError("W1 / token placement needs the per-IOD Stage1 queue (s1q)")""")
patch("mega_moe_v2.py", """        self._build_fused_stage1(w1, w1_scale)
""", """        self._build_fused_stage1(w1, w1_scale)
        self._iod_setup_stage1()
""")
patch("mega_moe_v2.py", """    def preload_aot_bundles(self):
""", """    def _iod_setup_stage1(self):
        \"\"\"IOD locality of the fused Stage1 (flags s1q, s1w / s1x, s1a); extends the dispatch table by two slots.\"\"\"
        self._iod_s1 = 0
        if not self._iod & {"s1q", "s1w", "s1x", "s1a", "s1z", "s1y"}:
            return
        if self._s1_fixed_slot or self._s1_op.indexed_payload or self._a_dtype != "fp8":
            raise ValueError("IOD Stage1 variants need the compact route-major fp8 payload")
        # s1a: token copies read on the own IOD; controls s1z (same copies, stock 50 / 50 mix), s1y (other IOD)
        self._iod_s1 = (
            1
            | (2 if self._iod & {"s1w", "s1x"} else 0)
            | (4 if self._iod & {"s1a", "s1z", "s1y"} else 0)
            | (8 if "s1z" in self._iod else 0)
            | (16 if "s1y" in self._iod else 0)
        )
        extra = [self._iod_ctr.data_ptr(), self._iod_ctr.data_ptr()]
        table = self._s1_disp.cpu().tolist()
        if self._iod_s1 & 2:
            from .iod_place import place_w1

            # W1 is placed for the Stage1 tile_n of the largest token bucket; forwards with another tile_n fail
            self._iod_tile_n = self._bundle_plan.entry_for_tokens(self.mtpr).config.stage1.tile_n
            self._s1_w1, self._iod_w1_phase, weak = place_w1(
                self._s1_w1, self.epr, self.model_dim, self.inter_dim, self._iod_tile_n, anti="s1x" in self._iod
            )
            extra[0] = self._iod_w1_phase.data_ptr()
            self.iod_info["w1_weak_pages"] = weak
        if self._iod_s1 & 4:
            from .iod_place import alloc_placed_rx

            self._iod_rx_raw, rx, self._iod_rx_p2p, self._iod_a_phase, weak = alloc_placed_rx(
                self._s1_op, self.rank, self.world_size, self.dev
            )
            rows = rx.numel() // self._a_view_dim
            self._s1_rx_kernel = rx[: rows * self._a_view_dim].view(self._a_torch_dtype).view(rows, self._a_view_dim)
            table[DispatchSlot.P2P_TOKEN] = self._iod_rx_p2p.data_ptr()
            extra[1] = self._iod_a_phase.data_ptr()
            self.iod_info["rx_weak_pages"] = weak
            self.iod_info["rx_placed_MiB"] = rx.numel() >> 20
        self._s1_disp = torch.tensor(table + extra, dtype=torch.int64, device=self.dev)
        torch.cuda.synchronize()
        ms.shmem_barrier_all()

    def preload_aot_bundles(self):
""")
patch("mega_moe_v2.py", """        use_bundle = (
            entry is not None
            and entry.config.stage1 == config
        )""", """        use_bundle = (
            not self._iod_s1
            and entry is not None
            and entry.config.stage1 == config
        )
        if self._iod_s1 & 2 and config.tile_n != self._iod_tile_n:
            raise ValueError(f"W1 was placed for tile_n={self._iod_tile_n}, this forward uses {config.tile_n}")""")
patch("mega_moe_v2.py", """                payload_chunk_rows=config.payload_chunk_rows,
                tile_state_stride=self._s1_tile_state_stride,
            )""", """                payload_chunk_rows=config.payload_chunk_rows,
                tile_state_stride=self._s1_tile_state_stride,
                iod_s1=self._iod_s1,
            )""")

print("mega_moe_iod created at", dst)
