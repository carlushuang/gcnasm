"""
Tests for flash_mla decode kernels — MODEL1 FP8 layout
d_qk=512, d_nope=448 (FP8 e4m3fnuz), d_rope=64 (BF16)

Each test compares the kernel output against a PyTorch fp32 reference.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import math
import itertools

import flash_mla

HEAD_DIM = 512
D_NOPE = 448
D_ROPE = 64
TILE_SIZE = 64
NUM_TILES = D_NOPE // TILE_SIZE  # 7


# ═══════════════════════════════════════════════════════════════════════════════
# Reference implementation (fp32, PyTorch)
# ═══════════════════════════════════════════════════════════════════════════════

def quantize_kv_model1(kv_bf16):
    """
    Quantize bf16 KV [total_tokens, 512] into MODEL1 FP8 layout.

    Returns:
        kv_nope_fp8: [total_tokens, 448] float8_e4m3fnuz
        kv_rope:     [total_tokens, 64] bf16  (unchanged)
        kv_scales:   [total_tokens, 7] float32 (per-tile)
        kv_dequant:  [total_tokens, 512] float32 (dequant for reference)
    """
    T = kv_bf16.shape[0]
    device = kv_bf16.device

    kv_nope_f32 = kv_bf16[:, :D_NOPE].float()  # [T, 448]
    kv_rope = kv_bf16[:, D_NOPE:].contiguous()  # [T, 64] bf16 (must be contiguous)

    nope_tiles = kv_nope_f32.reshape(T, NUM_TILES, TILE_SIZE)  # [T, 7, 64]
    tile_absmax = nope_tiles.abs().amax(dim=-1)                 # [T, 7]

    fp8_max = torch.finfo(torch.float8_e4m3fnuz).max
    kv_scales = tile_absmax / fp8_max                           # [T, 7]
    kv_scales = kv_scales.clamp(min=1e-12)

    nope_scaled = nope_tiles / kv_scales.unsqueeze(-1)          # [T, 7, 64]
    nope_fp8 = nope_scaled.reshape(T, D_NOPE).to(torch.float8_e4m3fnuz)

    nope_dequant = nope_fp8.float().reshape(T, NUM_TILES, TILE_SIZE) * kv_scales.unsqueeze(-1)
    kv_dequant = torch.cat([nope_dequant.reshape(T, D_NOPE), kv_rope.float()], dim=1)

    return nope_fp8, kv_rope, kv_scales, kv_dequant


def ref_mla_decode_fp8(Q, kv_dequant, kv_indptr, sm_scale):
    """Reference fp32 decode using dequantized KV."""
    B, H, D = Q.shape
    O = torch.zeros_like(Q, dtype=torch.float32)
    Q_f = Q.float()
    for b in range(B):
        start, end = kv_indptr[b].item(), kv_indptr[b + 1].item()
        if start == end:
            continue
        kv_slice = kv_dequant[start:end]                    # [T, D]
        scores = (Q_f[b] @ kv_slice.T) * sm_scale
        scores_max = scores.max(dim=-1, keepdim=True).values
        exp_scores = torch.exp(scores - scores_max)
        attn = exp_scores / exp_scores.sum(dim=-1, keepdim=True)
        O[b] = attn @ kv_slice
    return O.to(Q.dtype)


# ═══════════════════════════════════════════════════════════════════════════════
# Test: FP8 MFMA decode (CSR indptr interface)
# ═══════════════════════════════════════════════════════════════════════════════

def test_flash_mla_decode():
    """Test MODEL1 FP8 KV MFMA decode against reference."""
    configs = [
        (1, 1, [10]),
        (1, 4, [32]),
        (2, 4, [16, 24]),
        (4, 8, [5, 10, 15, 20]),
        (1, 16, [100]),
        (2, 16, [50, 75]),
        (1, 1, [1]),
        (1, 3, [7]),
        (3, 5, [3, 11, 17]),
        # Partial MFMA tile (num_heads not multiple of 16)
        (1, 17, [20]),
        (2, 15, [30, 40]),
        # Partial KV tile (kv_length not multiple of 16)
        (1, 16, [3]),
        (1, 16, [17]),
        (2, 16, [1, 33]),
        # FlashMLA MODEL1 CONFIG style
        (1, 64, [128]),
        (2, 128, [256, 512]),
    ]

    device = "cuda"
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    passed = 0

    for batch_size, num_heads, kv_lengths in configs:
        assert len(kv_lengths) == batch_size
        total = sum(kv_lengths)
        indptr = [0]
        for l in kv_lengths:
            indptr.append(indptr[-1] + l)

        Q = torch.randn(batch_size, num_heads, HEAD_DIM, device=device, dtype=torch.bfloat16)
        KV = torch.randn(total, HEAD_DIM, device=device, dtype=torch.bfloat16)
        kv_indptr = torch.tensor(indptr, device=device, dtype=torch.int32)

        nope_fp8, rope, scales, kv_dequant = quantize_kv_model1(KV)

        ref = ref_mla_decode_fp8(Q, kv_dequant, kv_indptr, sm_scale)
        out = flash_mla.decode(Q, nope_fp8, rope, scales, kv_indptr, num_heads, sm_scale)

        max_err = (ref.float() - out.float()).abs().max().item()
        ok = max_err < 0.15
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] FP8 MFMA B={batch_size} H={num_heads} kvlen={kv_lengths} | max_err={max_err:.6f}")
        if ok:
            passed += 1

    total_tests = len(configs)
    print(f"\nFP8 MFMA decode: {passed}/{total_tests} passed\n")
    return passed == total_tests


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Sparse decode with packed FP8 KV (FlashMLA interface)
# ═══════════════════════════════════════════════════════════════════════════════

from flash_mla.quant import (
    quantize_kv_cache,
    dequantize_kv_cache,
    abs_indices_to_flat_indices,
    BYTES_PER_TOKEN,
)


def _gen_block_paged_kv(b, s_kv, block_size, h_q, d_qk, device):
    """Generate blocked KV cache, block table, and random top-k indices."""
    num_blocks_per_seq = (s_kv + block_size - 1) // block_size
    total_blocks = b * num_blocks_per_seq
    block_table = torch.randperm(total_blocks, device=device, dtype=torch.int32).view(b, num_blocks_per_seq)
    kv_bf16 = torch.randn(total_blocks, block_size, 1, d_qk, device=device, dtype=torch.bfloat16)
    kv_bf16.clamp_(-1.0, 1.0)
    return kv_bf16, block_table, num_blocks_per_seq


def ref_sparse_attn_decode(q, kv_dequant_flat, indices, topk_length, sm_scale):
    """
    Pure PyTorch reference for sparse attention decode.
    """
    b, s_q, h_q, d_qk = q.shape
    d_v = d_qk
    topk = indices.shape[2]

    idx = indices.clone()
    invalid_mask = idx < 0
    if topk_length is not None:
        topk_range = torch.arange(topk, device=q.device).view(1, 1, topk).expand(b, s_q, topk)
        invalid_mask |= topk_range >= topk_length.view(b, 1, 1)
    idx[invalid_mask] = 0

    flat_idx = idx.reshape(-1)
    gathered = kv_dequant_flat[flat_idx].view(b, s_q, topk, d_qk).float()

    q_f = q.float()
    attn = torch.einsum("bshd,bstd->bsht", q_f, gathered) * sm_scale
    attn[invalid_mask.unsqueeze(2).expand(b, s_q, h_q, topk)] = float("-inf")

    lse = attn.logsumexp(dim=-1)
    attn_weights = torch.exp(attn - lse.unsqueeze(-1))
    out = torch.einsum("bsht,bstd->bshd", attn_weights, gathered[..., :d_v])

    lonely = (lse == float("-inf"))
    out[lonely.unsqueeze(-1).expand_as(out)] = 0.0
    lse[lonely] = float("inf")

    return out.to(torch.bfloat16), lse.transpose(1, 2)


def test_sparse_attn_decode():
    """Test sparse_attn_decode with packed FP8 KV against PyTorch reference."""
    configs = [
        # (b, h_q, s_q, s_kv, topk, block_size, have_topk_length)
        (1, 16, 1, 128, 32, 64, False),
        (2, 64, 1, 256, 64, 64, False),
        (4, 64, 1, 512, 128, 64, False),
        (1, 16, 1, 64, 16, 16, False),
        (2, 128, 1, 1024, 256, 64, False),
        # With s_q > 1
        (2, 64, 2, 256, 64, 64, False),
        (1, 16, 3, 128, 32, 32, False),
        # With topk_length
        (2, 64, 1, 256, 64, 64, True),
        (4, 64, 1, 512, 128, 64, True),
        # Partial MFMA tile (h_q not multiple of 16)
        (1, 17, 1, 128, 32, 64, False),
        (2, 15, 1, 128, 32, 64, False),
        # Small topk
        (1, 16, 1, 64, 3, 32, False),
        (2, 64, 1, 128, 1, 64, False),
        # With invalid indices
        (2, 64, 1, 256, 64, 64, False),
    ]

    device = "cuda"
    d_qk = HEAD_DIM
    passed = 0

    for cfg in configs:
        b, h_q, s_q, s_kv, topk, block_size, have_topk_length = cfg
        sm_scale = d_qk ** (-0.55)

        kv_bf16, block_table, num_blocks_per_seq = _gen_block_paged_kv(
            b, s_kv, block_size, h_q, d_qk, device
        )

        kv_packed = quantize_kv_cache(kv_bf16)
        kv_dequant = dequantize_kv_cache(kv_packed)

        total_blocks = kv_bf16.shape[0]
        kv_dequant_flat = kv_dequant.view(total_blocks * block_size, d_qk)

        abs_indices = torch.stack([
            torch.randperm(s_kv, device=device)[:topk]
            for _ in range(b * s_q)
        ]).view(b, s_q, topk).to(torch.int32)

        if topk > 4:
            inv_mask = torch.rand(b, s_q, topk, device=device) < 0.1
            abs_indices[inv_mask] = -1

        flat_indices = abs_indices_to_flat_indices(abs_indices, block_table, block_size)

        topk_len = None
        if have_topk_length:
            topk_len = torch.randint(1, topk + 1, (b,), device=device, dtype=torch.int32)

        q = torch.randn(b, s_q, h_q, d_qk, device=device, dtype=torch.bfloat16)
        q.clamp_(-1.0, 1.0)

        ref_out, ref_lse = ref_sparse_attn_decode(q, kv_dequant_flat, flat_indices, topk_len, sm_scale)

        out, lse = flash_mla.sparse_attn_decode(
            q, kv_packed, flat_indices,
            topk_length=topk_len,
            d_v=d_qk,
            sm_scale=sm_scale,
        )

        out_err = (ref_out.float() - out.float()).abs().max().item()
        lse_mask = (ref_lse != float("inf")) & (ref_lse != float("-inf"))
        if lse_mask.any():
            lse_err = (ref_lse[lse_mask] - lse[lse_mask]).abs().max().item()
        else:
            lse_err = 0.0

        ok = out_err < 0.15 and lse_err < 0.5
        status = "PASS" if ok else "FAIL"
        tl_str = f" topk_len" if have_topk_length else ""
        print(f"  [{status}] sparse b={b} h={h_q} sq={s_q} sk={s_kv} topk={topk} bs={block_size}{tl_str} | out_err={out_err:.6f} lse_err={lse_err:.6f}")
        if ok:
            passed += 1

    total_tests = len(configs)
    print(f"\nSparse attn decode: {passed}/{total_tests} passed\n")
    return passed == total_tests


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("Flash MLA Decode Tests — MODEL1 FP8 Layout")
    print("  d_qk=512, d_nope=448 (FP8), d_rope=64 (BF16)")
    print("  tile_size=64, num_tiles=7, per-tile scales")
    print("=" * 70)

    all_pass = True

    print("\n--- FP8 MFMA decode (CSR indptr) ---")
    all_pass &= test_flash_mla_decode()

    print("\n--- Sparse attention decode (FlashMLA interface) ---")
    all_pass &= test_sparse_attn_decode()

    print("=" * 70)
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 70)
