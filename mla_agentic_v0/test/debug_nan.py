"""Debug the NaN in topk_length test case."""
import torch
import sys
sys.path.insert(0, "/mnt/raid0/carhuang/repo/gcnasm/mla")
import flash_mla
from flash_mla.quant import quantize_kv_cache, dequantize_kv_cache, abs_indices_to_flat_indices
from test import _gen_block_paged_kv, ref_sparse_attn_decode

device = "cuda"
d_qk = 512
b, h_q, s_q, s_kv, topk, block_size = 4, 64, 1, 512, 128, 64
sm_scale = d_qk ** (-0.55)

for trial in range(50):
    torch.manual_seed(trial * 7 + 13)

    kv_bf16, block_table, _ = _gen_block_paged_kv(b, s_kv, block_size, h_q, d_qk, device)
    kv_packed = quantize_kv_cache(kv_bf16)
    kv_dequant = dequantize_kv_cache(kv_packed)
    total_blocks = kv_bf16.shape[0]
    kv_dequant_flat = kv_dequant.view(total_blocks * block_size, d_qk)

    abs_indices = torch.stack([
        torch.randperm(s_kv, device=device)[:topk] for _ in range(b * s_q)
    ]).view(b, s_q, topk).to(torch.int32)

    if topk > 4:
        inv_mask = torch.rand(b, s_q, topk, device=device) < 0.1
        abs_indices[inv_mask] = -1

    flat_indices = abs_indices_to_flat_indices(abs_indices, block_table, block_size)
    topk_len = torch.randint(1, topk + 1, (b,), device=device, dtype=torch.int32)
    q = torch.randn(b, s_q, h_q, d_qk, device=device, dtype=torch.bfloat16)
    q.clamp_(-1.0, 1.0)

    out, lse = flash_mla.sparse_attn_decode(
        q, kv_packed, flat_indices, topk_length=topk_len, d_v=d_qk, sm_scale=sm_scale
    )

    if out.isnan().any():
        print(f"Trial {trial}: NaN! topk_length={topk_len.tolist()}")
        for bi in range(b):
            if out[bi].isnan().any():
                print(f"  Batch {bi}: {out[bi].isnan().sum().item()} NaN, topk_len={topk_len[bi].item()}")
                n_invalid = (flat_indices[bi, :, :topk_len[bi].item()] < 0).sum().item()
                print(f"    Invalid indices in range: {n_invalid}/{topk_len[bi].item()}")
        break
    else:
        print(f"Trial {trial}: OK")
