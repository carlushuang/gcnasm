"""
flash_mla — Flash MLA Decode for ROCm (MODEL1: head_dim=512, absorbed K=V)

Uses apache-tvm-ffi for the FFI layer, Ninja JIT compilation, and opus.hpp
(from aiter) for AMD GPU primitives.

MODEL1 FP8 layout (matching FlashMLA MODEL1_FP8Sparse):
  - d_qk=512: d_nope=448 (FP8 e4m3fnuz) + d_rope=64 (BF16)
  - Per-tile float32 scales: 7 tiles of 64 = 448 nope dims
  - Q[:,:,:448] = Q_nope, Q[:,:,448:512] = Q_rope

Sparse decode packed KV layout per token (584 bytes, AoS):
  [0:448)    FP8 e4m3fnuz nope
  [448:576)  BF16 rope (64 values)
  [576:583)  e8m0 scales (7 bytes, power-of-2 per-tile)
  [583:584)  padding
"""

import functools
import math

import tvm_ffi

from .jit import gen_flash_mla_module

HEAD_DIM = 512
D_NOPE = 448
D_ROPE = 64
NUM_TILES = 7
TILE_SIZE = 64
BYTES_PER_TOKEN = D_NOPE + 2 * D_ROPE + NUM_TILES + 1  # 448 + 128 + 7 + 1 = 584


@functools.cache
def _get_module():
    """JIT-compile (if needed) and load the flash_mla module."""
    return gen_flash_mla_module().build_and_load()


def _get_decode_func():
    return _get_module()["flash_mla_decode"]


def _get_sparse_decode_func():
    return _get_module()["flash_mla_sparse_decode"]


def _get_sparse_decode_splitk_func():
    return _get_module()["flash_mla_sparse_decode_splitk"]


def _compute_num_splits(topk, h_q, b, s_q, num_cus=304):
    """Choose num_splits balancing GPU occupancy vs combine overhead.

    Split-K is only needed when there aren't enough blocks to fill all CUs.
    """
    if topk <= 16:
        return 1
    hg2 = (h_q + 31) // 32
    blocks2 = hg2 * b * s_q
    if blocks2 >= num_cus and h_q >= 32:
        head_groups = hg2
    else:
        head_groups = (h_q + 15) // 16
    base_blocks = head_groups * b * s_q
    target = num_cus
    if base_blocks >= target:
        return 1
    desired = (target + base_blocks - 1) // base_blocks
    max_by_topk = max(1, topk // 16)
    return min(desired, max_by_topk, 64)


def decode(q, kv_nope_fp8, kv_rope, kv_scales, kv_indptr, num_heads, sm_scale=0.0):
    """
    Flash MLA decode — MODEL1 FP8 KV cache, MFMA 16x16x16 bf16.

    Uses gfx942 MFMA matrix cores for QK and PV GEMMs.
    Dequantizes FP8→BF16 with per-tile scales before MFMA.

    Args:
        q:            [batch_size, num_heads, 512] bf16
        kv_nope_fp8:  [total_tokens, 448] fp8 (float8_e4m3fnuz)
        kv_rope:      [total_tokens, 64] bf16
        kv_scales:    [total_tokens, 7] float32 (per-tile dequant scales)
        kv_indptr:    [batch_size + 1] int32
        num_heads:    number of Q attention heads
        sm_scale:     softmax scale (0 = auto = 1/sqrt(512))

    Returns:
        output: [batch_size, num_heads, 512] bf16 GPU tensor
    """
    import torch

    assert q.dtype == torch.bfloat16
    assert kv_nope_fp8.dtype == torch.float8_e4m3fnuz
    assert kv_rope.dtype == torch.bfloat16
    assert kv_scales.dtype == torch.float32
    assert kv_indptr.dtype == torch.int32
    assert q.dim() == 3 and q.shape[2] == HEAD_DIM
    assert kv_nope_fp8.dim() == 2 and kv_nope_fp8.shape[1] == D_NOPE
    assert kv_rope.dim() == 2 and kv_rope.shape[1] == D_ROPE
    assert kv_scales.dim() == 2 and kv_scales.shape[1] == NUM_TILES

    q = q.contiguous()
    kv_nope_fp8 = kv_nope_fp8.contiguous()
    kv_rope = kv_rope.contiguous()
    kv_scales = kv_scales.contiguous()

    output = torch.empty_like(q)

    q_tvm = tvm_ffi.from_dlpack(q)
    nope_tvm = tvm_ffi.from_dlpack(kv_nope_fp8)
    rope_tvm = tvm_ffi.from_dlpack(kv_rope)
    sc_tvm = tvm_ffi.from_dlpack(kv_scales)
    o_tvm = tvm_ffi.from_dlpack(output)
    indptr_tvm = tvm_ffi.from_dlpack(kv_indptr)

    _get_decode_func()(q_tvm, nope_tvm, rope_tvm, sc_tvm, o_tvm, indptr_tvm,
                       num_heads, sm_scale)

    return output


def sparse_attn_decode(
    q,
    kv,
    indices,
    topk_length=None,
    d_v=512,
    sm_scale=None,
):
    """
    Sparse attention decode with packed FP8 KV cache (MODEL1 layout).

    Matches FlashMLA's sparse_attn_decode_interface for d_qk=512.

    Packed KV layout per token (584 bytes, AoS):
      [0:448)    FP8 e4m3fnuz nope
      [448:576)  BF16 rope (64 bf16 values)
      [576:583)  e8m0 scales (7 bytes, power-of-2 per-tile)
      [583:584)  padding

    Args:
        q:            [b, s_q, h_q, d_qk] bf16, d_qk=512
        kv:           [num_blocks, page_block_size, 1, 584] uint8 (packed FP8 KV)
        indices:      [b, s_q, topk] int32 (flat indices into KV, <0 = invalid)
        topk_length:  [b] int32 (optional, attend to indices[:,:,:topk_length[b]])
        d_v:          must be 512
        sm_scale:     softmax scale (None = auto = 1/sqrt(d_qk))

    Returns:
        out: [b, s_q, h_q, d_v] bf16
        lse: [b, h_q, s_q] float32 (log-sum-exp of attention scores)
    """
    import torch

    assert q.dim() == 4, f"q must be 4D [b, s_q, h_q, d_qk], got {q.dim()}D"
    b, s_q, h_q, d_qk = q.shape
    assert d_qk == HEAD_DIM, f"d_qk must be {HEAD_DIM}, got {d_qk}"
    assert d_v == HEAD_DIM, f"d_v must be {HEAD_DIM}, got {d_v}"
    assert q.dtype == torch.bfloat16, f"q must be bf16, got {q.dtype}"
    assert kv.dtype == torch.uint8, f"kv must be uint8, got {kv.dtype}"
    assert indices.dtype == torch.int32, f"indices must be int32, got {indices.dtype}"
    assert indices.shape[:2] == (b, s_q)

    if kv.dim() == 4:
        assert kv.shape[2] == 1 and kv.shape[3] == BYTES_PER_TOKEN
    assert kv.is_contiguous(), "kv must be contiguous"

    topk = indices.shape[2]
    if sm_scale is None:
        sm_scale = d_qk ** (-0.5)

    q = q.contiguous()
    indices = indices.contiguous()

    # Sort indices to improve HBM access locality
    sort_keys = indices.clone()
    if topk_length is not None:
        topk_range = torch.arange(topk, device=q.device).view(1, 1, topk)
        tl_expanded = topk_length.view(b, 1, 1)
        sort_keys[topk_range >= tl_expanded] = -1
        topk_length = None
    neg_mask = sort_keys < 0
    sort_keys[neg_mask] = 0x7FFFFFFF
    sorted_idx = sort_keys.sort(dim=-1).values
    sorted_idx[sorted_idx == 0x7FFFFFFF] = -1
    indices = sorted_idx

    out = torch.empty(b, s_q, h_q, d_v, dtype=torch.bfloat16, device=q.device)
    lse = torch.empty(b, s_q, h_q, dtype=torch.float32, device=q.device)

    q_tvm = tvm_ffi.from_dlpack(q)
    kv_tvm = tvm_ffi.from_dlpack(kv.view(torch.uint8).flatten())
    idx_tvm = tvm_ffi.from_dlpack(indices)
    o_tvm = tvm_ffi.from_dlpack(out)
    lse_tvm = tvm_ffi.from_dlpack(lse)

    has_topk_length = 1 if topk_length is not None else 0

    if topk_length is not None:
        assert topk_length.dtype == torch.int32
        assert topk_length.shape == (b,)
        topk_length = topk_length.contiguous()
        tl_tvm = tvm_ffi.from_dlpack(topk_length)
    else:
        tl_tvm = tvm_ffi.from_dlpack(torch.empty(1, dtype=torch.int32, device=q.device))

    num_splits = _compute_num_splits(topk, h_q, b, s_q)

    if num_splits <= 1:
        _get_sparse_decode_func()(
            q_tvm, kv_tvm, idx_tvm, o_tvm, lse_tvm,
            h_q, topk, sm_scale, has_topk_length, tl_tvm
        )
    else:
        bsq = b * s_q
        o_partial = torch.empty(num_splits, bsq, h_q, d_v, dtype=torch.float32, device=q.device)
        lse_partial = torch.empty(num_splits, bsq, h_q, dtype=torch.float32, device=q.device)

        op_tvm = tvm_ffi.from_dlpack(o_partial)
        lp_tvm = tvm_ffi.from_dlpack(lse_partial)

        _get_sparse_decode_splitk_func()(
            q_tvm, kv_tvm, idx_tvm, op_tvm, lp_tvm, o_tvm, lse_tvm,
            h_q, topk, num_splits, sm_scale, has_topk_length, tl_tvm
        )

    lse_inf_mask = lse == float("-inf")
    lse[lse_inf_mask] = float("inf")

    return out, lse.transpose(1, 2)
