"""
MODEL1 FP8 KV cache quantization helpers for MI300X (gfx942).

Packed layout per token (AoS, 584 bytes total — matches FlashMLA MODEL1):
  [0:448)    FP8 e4m3fnuz nope  (448 bytes)
  [448:576)  BF16 rope          (128 bytes = 64 bf16)
  [576:583)  e8m0 scales        (7 bytes, one per 64-elem tile)
  [583:584)  padding            (1 byte)

e8m0 format: value = 2^(byte - 127), matching IEEE 754 biased exponent.
In the kernel, conversion is a single shift: float = uint_as_float(e8m0 << 23).

Block layout: [num_blocks, page_block_size, 1, BYTES_PER_TOKEN]
"""

import torch
from typing import Tuple

D_QK = 512
D_NOPE = 448
D_ROPE = 64
TILE_SIZE = 64
NUM_TILES = D_NOPE // TILE_SIZE  # 7
SCALE_BYTES = NUM_TILES + 1      # 7 e8m0 + 1 pad = 8
BYTES_PER_TOKEN = D_NOPE + 2 * D_ROPE + SCALE_BYTES  # 448 + 128 + 8 = 584


def _float32_to_e8m0(scales: torch.Tensor) -> torch.Tensor:
    """Convert positive float32 values to e8m0 uint8 (round up to power of 2).

    e8m0 stores the biased exponent of IEEE 754 float32 (sign=0, mantissa=0).
    value = 2^(byte - 127).
    """
    scales_pow2 = torch.pow(2.0, torch.ceil(torch.log2(scales.clamp(min=2**-126))))
    bits = scales_pow2.to(torch.float32).view(torch.int32)
    return ((bits >> 23) & 0xFF).to(torch.uint8)


def _e8m0_to_float32(e8m0: torch.Tensor) -> torch.Tensor:
    """Convert e8m0 uint8 to float32.  Inverse of _float32_to_e8m0."""
    bits = e8m0.to(torch.int32) << 23
    return bits.view(torch.float32)


def quantize_kv_cache(
    kv_bf16: torch.Tensor,  # [num_blocks, page_block_size, 1, D_QK] bf16
) -> torch.Tensor:
    """
    Quantize bf16 KV cache into MODEL1 packed FP8 layout for MI300.

    Per-tile scales are stored as e8m0 (1 byte each, power-of-2 values).

    Args:
        kv_bf16: [num_blocks, page_block_size, 1, 512] bf16

    Returns:
        packed: [num_blocks, page_block_size, 1, 584] uint8
    """
    num_blocks, block_size, h_kv, d = kv_bf16.shape
    assert h_kv == 1 and d == D_QK
    device = kv_bf16.device

    kv = kv_bf16.squeeze(2)  # [num_blocks, block_size, 512]

    packed = torch.zeros(num_blocks, block_size, BYTES_PER_TOKEN, dtype=torch.uint8, device=device)

    nope_f32 = kv[:, :, :D_NOPE].float()  # [NB, BS, 448]
    rope_bf16 = kv[:, :, D_NOPE:]          # [NB, BS, 64] bf16

    # Per-tile quantization with power-of-2 scales (e8m0)
    fp8_max = torch.finfo(torch.float8_e4m3fnuz).max
    nope_tiles = nope_f32.reshape(num_blocks, block_size, NUM_TILES, TILE_SIZE)
    tile_absmax = nope_tiles.abs().amax(dim=-1)  # [NB, BS, 7]
    scales_raw = (tile_absmax / fp8_max).clamp(min=2**-126)

    # Round up to power of 2 and convert to e8m0
    e8m0_bytes = _float32_to_e8m0(scales_raw)              # [NB, BS, 7] uint8
    scales_pow2 = _e8m0_to_float32(e8m0_bytes)             # [NB, BS, 7] float32

    nope_scaled = nope_tiles / scales_pow2.unsqueeze(-1)
    nope_fp8 = nope_scaled.reshape(num_blocks, block_size, D_NOPE).to(torch.float8_e4m3fnuz)

    # Pack into output buffer
    # Nope: bytes [0, 448)
    packed[:, :, :D_NOPE] = nope_fp8.view(torch.uint8)
    # Rope: bytes [448, 576)
    packed[:, :, D_NOPE:D_NOPE + 2 * D_ROPE] = rope_bf16.contiguous().view(torch.uint8)
    # Scales: bytes [576, 583) — 7 e8m0 bytes, byte [583] is pad (already 0)
    packed[:, :, D_NOPE + 2 * D_ROPE:D_NOPE + 2 * D_ROPE + NUM_TILES] = e8m0_bytes

    return packed.view(num_blocks, block_size, 1, BYTES_PER_TOKEN)


def dequantize_kv_cache(
    packed: torch.Tensor,  # [num_blocks, page_block_size, 1, BYTES_PER_TOKEN] uint8
) -> torch.Tensor:
    """
    Dequantize packed FP8 KV cache back to bf16.

    Returns:
        kv_bf16: [num_blocks, page_block_size, 1, 512] bf16
    """
    num_blocks, block_size, h_kv, bpt = packed.shape
    assert h_kv == 1 and bpt == BYTES_PER_TOKEN

    packed_flat = packed.view(num_blocks, block_size, BYTES_PER_TOKEN)

    # Unpack
    nope_fp8 = packed_flat[:, :, :D_NOPE].view(torch.float8_e4m3fnuz)
    rope_bf16 = packed_flat[:, :, D_NOPE:D_NOPE + 2 * D_ROPE].view(torch.bfloat16)
    e8m0_bytes = packed_flat[:, :, D_NOPE + 2 * D_ROPE:D_NOPE + 2 * D_ROPE + NUM_TILES]  # [NB, BS, 7]

    # Convert e8m0 → float32 scales
    scales = _e8m0_to_float32(e8m0_bytes)  # [NB, BS, 7]

    # Dequantize nope
    nope_f32 = nope_fp8.float().reshape(num_blocks, block_size, NUM_TILES, TILE_SIZE)
    nope_dequant = (nope_f32 * scales.unsqueeze(-1)).reshape(num_blocks, block_size, D_NOPE)

    result = torch.cat([nope_dequant.to(torch.bfloat16), rope_bf16], dim=-1)
    return result.view(num_blocks, block_size, 1, D_QK)


def abs_indices_to_flat_indices(
    abs_indices: torch.Tensor,  # [b, s_q, topk] — logical token indices (0..s_kv-1)
    block_table: torch.Tensor,  # [b, max_blocks_per_seq] int32
    block_size: int,
) -> torch.Tensor:
    """
    Convert absolute token indices to flat indices into the blocked KV cache.

    flat_index = block_table[batch, abs_idx // block_size] * block_size + abs_idx % block_size

    Invalid indices (< 0) are preserved as -1.
    """
    b, s_q, topk = abs_indices.shape
    _, max_blocks = block_table.shape

    idx = abs_indices.clone()
    invalid = idx < 0
    idx[invalid] = 0

    block_idx = idx // block_size
    offset = idx % block_size

    real_blocks = block_table.view(-1).index_select(
        0, (block_idx + torch.arange(b, device=idx.device).view(b, 1, 1) * max_blocks).reshape(-1)
    ).view(b, s_q, topk)

    flat = real_blocks * block_size + offset
    flat[invalid] = -1
    return flat
