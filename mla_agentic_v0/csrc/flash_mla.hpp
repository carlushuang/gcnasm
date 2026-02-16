#pragma once

/// Flash MLA Decode — MODEL1 FP8 KV cache, MFMA 16x16x16 bf16
///
/// Uses gfx942 MFMA matrix cores for QK and PV GEMMs.
/// Dequantizes FP8→BF16 with per-tile scales before MFMA.
///
/// @param Q          [batch_size, num_heads, 512] bf16
/// @param KV_nope    [total_tokens, 448] fp8 (e4m3fnuz, packed as int32)
/// @param KV_rope    [total_tokens, 64]  bf16
/// @param KV_scales  [total_tokens, 7]   float32 (per-tile dequant scales)
/// @param O          [batch_size, num_heads, 512] bf16
/// @param kv_indptr  [batch_size + 1] int32 (CSR-style ranges into KV)
void flash_mla_decode(
    const void* Q,
    const void* KV_nope,
    const void* KV_rope,
    const void* KV_scales,
    void* O,
    const void* kv_indptr,
    int batch_size,
    int num_heads,
    float sm_scale,
    void* stream = nullptr);

/// Sparse Decode — packed FP8 KV cache, MFMA 16x16x16 bf16
///
/// Matches FlashMLA's sparse_attn_decode_interface for MODEL1 (d_qk=512).
///
/// Packed KV layout per token (584 bytes, AoS):
///   [0:448)    FP8 e4m3fnuz nope
///   [448:576)  BF16 rope (64 values)
///   [576:583)  e8m0 scales (7 bytes, power-of-2 per-tile)
///   [583:584)  padding
///
/// @param Q           [b, s_q, h_q, 512] bf16
/// @param KV_packed   packed FP8 KV data (contiguous uint8)
/// @param indices     [b, s_q, topk] int32 (flat KV indices, <0 = invalid)
/// @param topk_length [b] int32, or nullptr (attend to all topk)
/// @param O           [b, s_q, h_q, 512] bf16 output
/// @param lse         [b, s_q, h_q] float32 log-sum-exp output
void flash_mla_sparse_decode(
    const void* Q,
    const void* KV_packed,
    const void* indices,
    const void* topk_length,
    void* O,
    void* lse,
    int b,
    int s_q,
    int num_heads,
    int topk,
    float sm_scale,
    void* stream = nullptr);

/// Split-K Sparse Decode — parallelizes across topk for higher GPU utilization
///
/// @param O_partial   [num_splits, b*s_q, h_q, 512] float32 (temp buffer)
/// @param lse_partial [num_splits, b*s_q, h_q]      float32 (temp buffer)
/// @param O_final     [b, s_q, h_q, 512] bf16 output
/// @param lse_final   [b, s_q, h_q] float32 output
void flash_mla_sparse_decode_splitk(
    const void* Q,
    const void* KV_packed,
    const void* indices,
    const void* topk_length,
    void* O_partial,
    void* lse_partial,
    void* O_final,
    void* lse_final,
    int b,
    int s_q,
    int num_heads,
    int topk,
    int num_splits,
    float sm_scale,
    void* stream = nullptr);
