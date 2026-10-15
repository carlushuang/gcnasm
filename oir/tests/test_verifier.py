import json

import pytest

from opus_ir import Access, Counter, DType, Kernel, KernelBuilder, P, RoleKind, Y, cdiv, lane, thread_x, verify
from opus_ir.expr import block_x, block_y


def gemm(
    target="gfx950",
    stages=3,
    vec=8,
    pad=0,
    wait=True,
    wait_slack=0,
    sync=True,
    wave=(16, 16, 32),
    offset_lane=False,
) -> Kernel:
    """The examples/gemm_gfx950.py schedule with knobs that break it in specific ways."""
    lm = KernelBuilder("gemm", target, 256)
    m, n = lm.symbol("M", 1, 1 << 20), lm.symbol("N", 1, 1 << 20)
    k = lm.symbol("K", 64, 1 << 20, 64)
    lm.grid = (cdiv(n, 128), cdiv(m, 128))
    a = lm.param("A", (m, k), DType.BF16)
    b = lm.param("B", (n, k), DType.BF16)
    c = lm.param("C", (m, n), DType.BF16, Access.WRITE)
    s_a = lm.lds("s_a", (128, 64), DType.BF16, stages=stages, pad=pad)
    s_b = lm.lds("s_b", (128, 64), DType.BF16, stages=stages, pad=pad)
    expand = (4, 4, 64 // wave[2])
    mma = lm.mma("mma", "mfma", DType.BF16, DType.BF16, DType.FP32, wave=wave, expand=expand, tile=(2, 2, 1))
    dims, shape, coord = ((Y, P), (P, Y)), (4, 32, 8, 8), (thread_x // 8, thread_x % 8)
    g_a = lm.layout("g_a", shape, dims, (k, 1), coord)
    g_b = lm.layout("g_b", shape, dims, (k, 1), coord)
    l_copy = lm.layout("l_copy", shape, dims, (64 + pad, 1), coord)
    l_a = lm.mma_layout("l_a", mma, "a", (64 + pad, 1))
    l_b = lm.mma_layout("l_b", mma, "b", (64 + pad, 1))
    g_c = lm.mma_layout("g_c", mma, "c", (n, 1))
    v_a, v_b = lm.reg("v_a", DType.BF16, 64), lm.reg("v_b", DType.BF16, 64)
    v_c = lm.reg("v_c", DType.FP32, 64, zero_init=True)
    v_o = lm.reg("v_o", DType.BF16, 64)
    pipe = lm.pipeline("kloop", stages)
    off_a = block_y * 128 * k + (lane if offset_lane else 0)
    issues = 4 * 8 // vec
    for s in range(stages - 1):
        lm.async_load(a, g_a, s_a, l_copy, vec=vec, offset=off_a + s * 64, stage=s)
        lm.async_load(b, g_b, s_b, l_copy, vec=vec, offset=block_x * 128 * k + s * 64, stage=s)
    with lm.loop("k", cdiv(k, 64), pipeline=pipe) as it:
        if wait:
            lm.wait(Counter.VM, 2 * issues * (stages - 2) + wait_slack)
        if sync:
            lm.sync_threads()
        nxt = it + stages - 1
        lm.async_load(a, g_a, s_a, l_copy, vec=vec, offset=off_a + nxt * 64, stage=nxt % stages)
        lm.async_load(b, g_b, s_b, l_copy, vec=vec, offset=block_x * 128 * k + nxt * 64, stage=nxt % stages)
        lm.smem_load(s_a, l_a, v_a, vec=8, stage=it % stages)
        lm.smem_load(s_b, l_b, v_b, vec=8, stage=it % stages)
        lm.mma_op(mma, v_a, v_b, v_c)
    lm.cast(v_c, v_o)
    lm.gmem_store(v_o, c, g_c, offset=block_y * 128 * n + block_x * 128)
    return lm.build()


def codes(kernel: Kernel, severity="error"):
    return {d.code for d in verify(kernel).diagnostics if d.severity.value == severity}


def test_reference_schedule_verifies():
    assert verify(gemm()).ok


def test_round_trip_is_exact():
    kernel = gemm()
    again = Kernel.from_dict(json.loads(kernel.canonical_json()))
    assert again.canonical_json() == kernel.canonical_json()
    assert again.digest == kernel.digest == gemm().digest


def test_missing_wait_reads_in_flight_async_data():
    assert "SYNC002" in codes(gemm(wait=False))


def test_wait_one_instruction_too_loose():
    assert "SYNC002" in codes(gemm(wait_slack=1))


def test_missing_sync_threads_is_raw_and_war():
    assert {"SYNC003", "SYNC004"} <= codes(gemm(sync=False))


def test_two_stage_ring_needs_the_full_drain():
    assert verify(gemm(stages=2)).ok


def test_padded_lds_breaks_cdna_lane_contiguity():
    assert "ASYNC001" in codes(gemm(pad=8))


def test_gfx942_rejects_16_byte_async_and_overflows_lds():
    found = codes(gemm(target="gfx942"))
    assert {"TYPE003", "MEM004"} <= found


def test_gfx942_double_k_mfma_is_chained():
    infos = [d for d in verify(gemm(target="gfx942", stages=2, vec=2)).diagnostics if d.code == "TARGET010"]
    assert infos and "2 chained 16x16x16" in infos[0].message


def test_unknown_mma_shape_lists_alternatives():
    diags = [d for d in verify(gemm(wave=(32, 32, 32))).diagnostics if d.code == "TARGET001"]
    assert diags and "16x16x32" in diags[0].notes[0]


def test_divergent_scalar_offset_is_rejected():
    assert "SCHED009" in codes(gemm(offset_lane=True))


def _tiny(target="gfx950", **kw):
    lm = KernelBuilder("tiny", target, kw.get("block", 256))
    x = lm.param("X", (1024,), DType.FP32)
    y = lm.param("Y", (1024,), DType.FP32, Access.WRITE)
    u = lm.layout("u", (4,), x_stride=(1,))
    r = lm.reg("r", DType.FP32, 4)
    return lm, x, y, u, r


def test_register_read_before_write():
    lm, x, y, u, r = _tiny()
    lm.gmem_store(r, y, u, vec=4)
    assert "DATA001" in codes(lm.build())


def test_store_to_read_only_param():
    lm, x, y, u, r = _tiny()
    lm.gmem_load(x, u, r, vec=4)
    lm.gmem_store(r, x, u, vec=4)
    assert "ACCESS001" in codes(lm.build())


def test_gfx12_has_no_opus_wait_wrapper():
    lm, x, y, u, r = _tiny("gfx1201")
    lm.gmem_load(x, u, r, vec=4)
    lm.wait(Counter.VM, 0)
    lm.gmem_store(r, y, u, vec=4)
    assert "TARGET006" in codes(lm.build())


def test_sync_threads_inside_role_deadlocks():
    lm, x, y, u, r = _tiny()
    half = lm.role("half", (0, 1))
    lm.gmem_load(x, u, r, vec=4)
    with lm.on(half):
        lm.sync_threads()
    lm.gmem_store(r, y, u, vec=4)
    assert "SCHED003" in codes(lm.build())


def test_tdm_needs_gfx1250():
    lm, x, y, u, r = _tiny()
    s = lm.lds("s", (64,), DType.FP32)
    t = lm.tdm("t", x, s, tile=(64,))
    lm.tdm_load(t, (0,))
    assert "TARGET004" in codes(lm.build())


def test_block_size_must_be_whole_waves():
    lm, x, y, u, r = _tiny(block=96)
    lm.gmem_load(x, u, r, vec=4)
    assert "SCHED001" in codes(lm.build())


def tdm_gemm(empty_wait=True, tensor_wait=True) -> Kernel:
    lm = KernelBuilder("tdm_gemm", "gfx1250", 160)
    m, n = lm.symbol("M", 1, 1 << 20), lm.symbol("N", 1, 1 << 20)
    k = lm.symbol("K", 64, 1 << 20, 64)
    a, b = lm.param("A", (m, k), DType.BF16), lm.param("B", (n, k), DType.BF16)
    c = lm.param("C", (m, n), DType.BF16, Access.WRITE)
    s_a, s_b = lm.lds("s_a", (128, 64), DType.BF16, 2), lm.lds("s_b", (128, 64), DType.BF16, 2)
    t_a, t_b = lm.tdm("t_a", a, s_a, (128, 64)), lm.tdm("t_b", b, s_b, (128, 64))
    mma = lm.mma("mma", "wmma", DType.BF16, DType.BF16, DType.FP32, (16, 16, 32), (4, 4, 2), (2, 2, 1))
    l_a, l_b = lm.mma_layout("l_a", mma, "a", (64, 1)), lm.mma_layout("l_b", mma, "b", (64, 1))
    g_c = lm.mma_layout("g_c", mma, "c", (n, 1))
    v_a, v_b = lm.reg("v_a", DType.BF16, 128), lm.reg("v_b", DType.BF16, 128)
    v_c, v_o = lm.reg("v_c", DType.FP32, 128, True), lm.reg("v_o", DType.BF16, 128)
    load, math = lm.role("load", (4,), RoleKind.PRODUCER), lm.role("math", (0, 1, 2, 3), RoleKind.MMA)
    full, empty = lm.named_barrier("full", (load, math)), lm.named_barrier("empty", (load, math))
    with lm.on(load):
        lm.tdm_load(t_a, (0, 0), 0)
        lm.tdm_load(t_b, (0, 0), 0)
    with lm.loop("k", cdiv(k, 64), lm.pipeline("p", 2)) as it:
        with lm.on(load):
            if tensor_wait:
                lm.wait(Counter.TENSOR, 0)
            lm.barrier_arrive(full)
            if empty_wait:
                lm.barrier_wait(empty)
            lm.tdm_load(t_a, (0, (it + 1) * 64), (it + 1) % 2)
            lm.tdm_load(t_b, (0, (it + 1) * 64), (it + 1) % 2)
        with lm.on(math):
            lm.barrier_wait(full)
            lm.smem_load(s_a, l_a, v_a, 8, it % 2)
            lm.smem_load(s_b, l_b, v_b, 8, it % 2)
            lm.mma_op(mma, v_a, v_b, v_c)
            lm.barrier_arrive(empty)
    with lm.on(math):
        lm.cast(v_c, v_o)
        lm.gmem_store(v_o, c, g_c)
    return lm.build()


def test_tdm_warp_specialized_reference():
    assert verify(tdm_gemm()).ok


def test_tdm_missing_empty_handoff_is_war():
    assert "SYNC004" in codes(tdm_gemm(empty_wait=False))


def test_tdm_missing_tensor_wait_is_raw():
    assert "SYNC003" in codes(tdm_gemm(tensor_wait=False))


def test_builder_rejects_duplicate_names():
    lm, x, y, u, r = _tiny()
    with pytest.raises(ValueError):
        lm.reg("r", DType.FP32, 4)
