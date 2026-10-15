"""bf16 NT GEMM on gfx1250: one TDM producer wave + four WMMA consumer waves, named-barrier handoff.

Block tile 128x128x64, 2-stage LDS ring, 5 waves of 32 lanes (160 threads).
"""

from opus_ir import Access, Counter, DType, RoleKind, cdiv, kernel
from opus_ir.expr import block_x, block_y

BM, BN, BK, STAGES = 128, 128, 64, 2


@kernel(name="gemm_bf16_nt_gfx1250_tdm", target="gfx1250", block_size=160)
def gemm_bf16_nt_gfx1250_tdm(lm):
    m = lm.symbol("M", minimum=1, maximum=1 << 20)
    n = lm.symbol("N", minimum=1, maximum=1 << 20)
    k = lm.symbol("K", minimum=BK, maximum=1 << 20, multiple_of=BK)
    lm.grid = (cdiv(n, BN), cdiv(m, BM), 1)

    a = lm.param("A", (m, k), DType.BF16, Access.READ)
    b = lm.param("B", (n, k), DType.BF16, Access.READ)
    c = lm.param("C", (m, n), DType.BF16, Access.WRITE)
    s_a = lm.lds("s_a", (BM, BK), DType.BF16, stages=STAGES)
    s_b = lm.lds("s_b", (BN, BK), DType.BF16, stages=STAGES)
    t_a = lm.tdm("t_a", a, s_a, tile=(BM, BK))
    t_b = lm.tdm("t_b", b, s_b, tile=(BN, BK))

    mma = lm.mma("mma", "wmma", DType.BF16, DType.BF16, DType.FP32, wave=(16, 16, 32), expand=(4, 4, 2), tile=(2, 2, 1))
    l_a = lm.mma_layout("l_a", mma, "a", x_stride=(BK, 1))
    l_b = lm.mma_layout("l_b", mma, "b", x_stride=(BK, 1))
    g_c = lm.mma_layout("g_c", mma, "c", x_stride=(n, 1))
    v_a = lm.reg("v_a", DType.BF16, 128)
    v_b = lm.reg("v_b", DType.BF16, 128)
    v_c = lm.reg("v_c", DType.FP32, 128, zero_init=True)
    v_o = lm.reg("v_o", DType.BF16, 128)

    load = lm.role("load", (4,), RoleKind.PRODUCER)
    math = lm.role("math", (0, 1, 2, 3), RoleKind.MMA)
    full = lm.named_barrier("full", (load, math))
    empty = lm.named_barrier("empty", (load, math))
    pipe = lm.pipeline("kloop", STAGES)

    m0, n0 = block_y * BM, block_x * BN
    with lm.on(load):
        lm.tdm_load(t_a, (m0, 0), stage=0)
        lm.tdm_load(t_b, (n0, 0), stage=0)

    with lm.loop("k", cdiv(k, BK), pipeline=pipe) as it:
        with lm.on(load):
            lm.wait(Counter.TENSOR, outstanding=0)  # stage `it` landed
            lm.barrier_arrive(full)
            lm.barrier_wait(empty)  # consumers are done with the slot refilled next
            lm.tdm_load(t_a, (m0, (it + 1) * BK), stage=(it + 1) % STAGES)
            lm.tdm_load(t_b, (n0, (it + 1) * BK), stage=(it + 1) % STAGES)
        with lm.on(math):
            lm.barrier_wait(full)
            lm.smem_load(s_a, l_a, v_a, vec=8, stage=it % STAGES)
            lm.smem_load(s_b, l_b, v_b, vec=8, stage=it % STAGES)
            lm.mma_op(mma, v_a, v_b, v_c)
            lm.barrier_arrive(empty)

    with lm.on(math):
        lm.cast(v_c, v_o)
        lm.gmem_store(v_o, c, g_c, vec=1, offset=m0 * n + n0)
