# Sparse Paged Prefill Attention for DeepSeek-V4

Optimized sparse paged prefill attention using the [OPUS](https://github.com/ROCm/aiter) C++ template library for DeepSeek-V4 inference on AMD gfx950.

This directory targets the DeepSeek-V4 MQA prefill shape with configurable query-head count `H_Q` and `D = 512` head dimension. The kernel consumes two sparse K/V sources: a unified prefix cache with layout `[total_pages, D]` and a current extend K/V tensor with layout `[total_tokens, D]`.

## Features

- DeepSeek-V4 prefill attention shape: BF16 or FP16 Q/K/V/O, configurable `H_Q`, `D = 512`.
- MQA layout: Q/O carry the query-head dimension, while K/V rows are shared across query heads.
- Paged sparse attention through two CSR index ranges per query token:
  - `prefix`: indices into `unified_kv_ptr` / `[total_pages, D]`.
  - `extend`: indices into `kv_ptr` / `[total_tokens, D]`.
- Online softmax across both CSR ranges, plus a per-head attention sink in the denominator, with no materialized attention matrix.
- OPUS-based gfx950 kernel using BF16/FP16 MFMA, double-buffered K/V shared-memory tiles, and FP32 accumulation.
- Standalone host harness with random sparse/dense index generation and CPU reference validation.

## Files

```text
sparse_paged_attn/
|-- Makefile                         # Build rules for the standalone executable
|-- pa_defs.h                        # Kernel argument struct and compile-time traits
|-- pa_host.cc                       # Host launcher, test harness, and CPU reference
|-- pa_prefill_bf16_kernel.cc        # D=512 BF16 kernel instantiation
|-- pa_prefill_fp16_kernel.cc        # D=512 FP16 kernel instantiation
`-- pa_prefill_kernel_template.hpp   # OPUS/HIP kernel implementation
```

## Attention Model

For each query token `i`, the caller provides two CSR rows:

```text
prefix_rows = kv_indices_prefix[kv_indptr_prefix[i] : kv_indptr_prefix[i + 1]]
extend_rows = kv_indices_extend[kv_indptr_extend[i] : kv_indptr_extend[i + 1]]
```

The kernel computes scaled dot-product attention over `prefix_rows` followed by `extend_rows`. Prefix rows index `UnifiedKV`; extend rows index the current `KV` tensor. Both ranges share the same online-softmax state:

```text
scores = concat(
  Q[i, h, :] @ UnifiedKV[prefix_rows, :].T,
  Q[i, h, :] @ KV[extend_rows, :].T
) * softmax_scale
P = exp(scores) / (sum(exp(scores)) + exp(attn_sink[h]))
O[i, h, :] = P_prefix @ UnifiedKV[prefix_rows, :] + P_extend @ KV[extend_rows, :]
```

In the DeepSeek-V4 prefill path, these two logical ranges map naturally to:

- `prefix`: previously available state, such as the sliding-window tail and compressed cache pages.
- `extend`: K/V rows produced by the current prefill chunk.

## Tensor Layout

Q/K/V/O tensor data is selected with `-dtype bf16|fp16`. `AttnSink` and accumulation stay FP32.

| Tensor | Shape | Notes |
| --- | --- | --- |
| `Q` | `[N, H_Q, D]` | Query tokens. `H_Q` is configurable, such as `64` for DeepSeek-V4-Flash and `128` for DeepSeek-V4-Pro. |
| `UnifiedKV` | `[total_pages, D]` | Prefix K/V rows. |
| `KV` | `[total_tokens, D]` | Current extend K/V rows. |
| `AttnSink` | `[H_Q]` | Per-head sink score included in the softmax denominator only. |
| `O` | `[N, H_Q, D]` | Output tokens. |
| `kv_indptr_prefix` | `[N + 1]` | CSR row pointers for prefix rows. |
| `kv_indices_prefix` | `[nnz_prefix]` | Row indices into `UnifiedKV`. |
| `kv_indptr_extend` | `[N + 1]` | CSR row pointers for extend rows. |
| `kv_indices_extend` | `[nnz_extend]` | Row indices into `KV`. |

The kernel assumes row-major contiguous layout with `D` as the fastest-changing dimension.

## Kernel Configuration

The compiled instantiations are:

```cpp
pa_traits<16, 32, 512, 8, bf16_t>
pa_traits<16, 32, 512, 8, fp16_t>
```

| Parameter | Value | Meaning |
| --- | --- | --- |
| `Q_TILE_SIZE` | `16` | Query-head tile per wave. |
| `KV_TILE_SIZE` | `32` | K/V rows loaded per sparse tile. |
| `D_TILE_SIZE` | `512` | Head dimension. |
| `NUM_WARPS` | `8` | Waves per workgroup. |
| `BLOCK_SIZE` | `512` | AMD wavefront size `64` times `8` waves. |

One workgroup covers one query token and up to `Q_TILE_SIZE * NUM_WARPS = 128` query heads. Arbitrary `H_Q` values are supported, while multiples of 128 give the best occupancy because every workgroup handles a full head tile.

## Build

Prerequisites:

- ROCm 7+ with `hipcc`.
- gfx950 GPU target.
- OPUS headers from `aiter`, exposed through `OPUS_INCLUDE_DIR`.
- OpenMP support for the host reference path.

```bash
cd opus_attn/sparse_paged_attn
export OPUS_INCLUDE_DIR=/path/to/aiter/csrc/include
make -j
```

The executable is written to:

```text
build/pa_prefill.exe
```

## Run and Validate

Run with the DeepSeek-V4 MQA shape:

```bash
./build/pa_prefill.exe -dtype bf16 -h_q 128 -d 512 -n 256 -total_pages 1024 -total_tokens 2048 --verify
```

Useful options:

| Option | Default | Description |
| --- | --- | --- |
| `-h_q` | `128` | Number of query heads. Supports arbitrary positive values; multiples of `128` have the best performance. |
| `-d` | `512` | Head dimension. Only `512` is compiled in this directory. |
| `-n` | `1024` | Number of query tokens in the standalone harness. |
| `-total_pages` | `N` | Number of prefix rows in `UnifiedKV`. |
| `-total_tokens` | `N` | Number of extend rows in `KV`. |
| `-dtype` | `bf16` | Attention tensor dtype: `bf16` or `fp16`. |
| `--dense` | off | Generate dense CSR rows instead of random sparse rows. |
| `--verify` | off | Compare GPU output against the CPU reference implementation. |

The harness initializes random attention tensors in the selected dtype and random per-head sink scores, generates prefix and extend CSR index ranges, launches the kernel, optionally checks the result against `pa_attention_ref()` in `pa_host.cc`, and then reports benchmark timing.

## Integration Notes

- The caller owns CSR construction. Causal, sliding-window, compressed-cache, or top-k semantics should already be reflected in `kv_indices_*` and `kv_indptr_*`.
- Empty CSR rows are allowed; with only the sink in the denominator, the output becomes zero.
- Prefix indices must be in `[0, total_pages)`, and extend indices must be in `[0, total_tokens)`.
- The standalone harness uses `softmax_scale = 1 / sqrt(D)`; integrations pass the scale through `pa_kargs`.
- This directory intentionally contains only the D=512 prefill variant used by the DeepSeek-V4 MQA inference path.
