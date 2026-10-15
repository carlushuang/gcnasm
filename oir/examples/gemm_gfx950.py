"""bf16 NT GEMM on gfx950: 3-stage gmem::async_load -> LDS pipeline feeding a 2x2-wave tiled MFMA.

C[M, N] = A[M, K] @ B[N, K]^T, block tile 128x128x64, 4 waves (256 threads).
"""

from opus_ir import Access, Counter, DType, P, Y, cdiv, kernel, thread_x
from opus_ir.expr import block_x, block_y

BM, BN, BK, STAGES = 128, 128, 64, 3


@kernel(
    name="gemm_bf16_nt_gfx950",
    target="gfx950",
    block_size=256,
    metadata={"workload": "C[M,N] = A[M,K] @ B[N,K]^T", "tile": [BM, BN, BK]},
)
def gemm_bf16_nt_gfx950(lm):
    m = lm.symbol("M", minimum=1, maximum=1 << 20)
    n = lm.symbol("N", minimum=1, maximum=1 << 20)
    k = lm.symbol("K", minimum=BK, maximum=1 << 20, multiple_of=BK)
    lm.grid = (cdiv(n, BN), cdiv(m, BM), 1)

    a = lm.param("A", (m, k), DType.BF16, Access.READ)
    b = lm.param("B", (n, k), DType.BF16, Access.READ)
    c = lm.param("C", (m, n), DType.BF16, Access.WRITE)

    s_a = lm.lds("s_a", (BM, BK), DType.BF16, stages=STAGES)
    s_b = lm.lds("s_b", (BN, BK), DType.BF16, stages=STAGES)

    # make_tiled_mma<bf16, bf16, fp32>(seq<4,4,2>, seq<2,2,1>, seq<16,16,32>): 128x128x64 per block
    mma = lm.mma("mma", "mfma", DType.BF16, DType.BF16, DType.FP32, wave=(16, 16, 32), expand=(4, 4, 2), tile=(2, 2, 1))

    # 256 threads copy a 128x64 tile: 8 lanes x 8 elems per row, 32 rows per pass, 4 passes.
    copy_dims = ((Y, P), (P, Y))
    copy_shape = (BM // 32, 32, BK // 8, 8)
    copy_coord = (thread_x // 8, thread_x % 8)
    g_a = lm.layout("g_a", copy_shape, copy_dims, x_stride=(k, 1), p_coord=copy_coord)
    g_b = lm.layout("g_b", copy_shape, copy_dims, x_stride=(k, 1), p_coord=copy_coord)
    l_copy = lm.layout("l_copy", copy_shape, copy_dims, x_stride=(BK, 1), p_coord=copy_coord)

    l_a = lm.mma_layout("l_a", mma, "a", x_stride=(BK, 1))
    l_b = lm.mma_layout("l_b", mma, "b", x_stride=(BK, 1))
    g_c = lm.mma_layout("g_c", mma, "c", x_stride=(n, 1))

    v_a = lm.reg("v_a", DType.BF16, 64)
    v_b = lm.reg("v_b", DType.BF16, 64)
    v_c = lm.reg("v_c", DType.FP32, 64, zero_init=True)
    v_o = lm.reg("v_o", DType.BF16, 64)
    pipe = lm.pipeline("kloop", STAGES)

    row_a, row_b = block_y * BM * k, block_x * BN * k
    for s in range(STAGES - 1):  # prologue: stages 0..S-2 in flight
        lm.async_load(a, g_a, s_a, l_copy, vec=8, offset=row_a + s * BK, stage=s)
        lm.async_load(b, g_b, s_b, l_copy, vec=8, offset=row_b + s * BK, stage=s)

    per_stage = 2 * (BM // 32)  # 4 buffer_load...lds per operand per lane
    with lm.loop("k", cdiv(k, BK), pipeline=pipe) as it:
        lm.wait(Counter.VM, outstanding=per_stage * (STAGES - 2))  # stage `it` has landed
        lm.sync_threads()  # ...for every wave; also retires reads of the slot refilled next
        nxt = it + (STAGES - 1)
        lm.async_load(a, g_a, s_a, l_copy, vec=8, offset=row_a + nxt * BK, stage=nxt % STAGES)
        lm.async_load(b, g_b, s_b, l_copy, vec=8, offset=row_b + nxt * BK, stage=nxt % STAGES)
        lm.smem_load(s_a, l_a, v_a, vec=8, stage=it % STAGES)
        lm.smem_load(s_b, l_b, v_b, vec=8, stage=it % STAGES)
        lm.mma_op(mma, v_a, v_b, v_c)

    lm.cast(v_c, v_o)
    lm.gmem_store(v_o, c, g_c, vec=1, offset=block_y * BM * n + block_x * BN)
