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
print("mega_moe_iod created at", dst)
