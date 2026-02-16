# Flash MLA Decode — MODEL1 FP8 (MI300X / gfx942)

> **Disclaimer:** This is `mla_agentic_v0` — the first round of an
> agentic-generated kernel. Performance is currently very slow (~3.9 TFlops
> geomean on MI308X, roughly 4–5% of H20 FlashMLA throughput). The kernel
> is functionally correct but requires significant optimization effort to
> close the gap. This version serves as a baseline for future iterations.

Flash-attention-style MLA decode for the MODEL1 absorbed MLA formulation
(`head_dim=512`, `K=V=compressed_latent`), targeting AMD MI300X (gfx942).

Uses gfx942 MFMA 16×16×16 bf16 matrix cores for both QK and PV GEMMs.

## MODEL1 FP8 Layout

Matches FlashMLA's `MODEL1_FP8Sparse` quantization:

| Component | Dims   | Type            | Notes                           |
|-----------|--------|-----------------|---------------------------------|
| Q_nope    | 0–447  | bf16            | Q[:,:,:448]                     |
| Q_rope    | 448–511| bf16            | Q[:,:,448:]                     |
| K/V nope  | 0–447  | fp8 (e4m3fnuz)  | Per-tile (64-elem) dequant scale|
| K/V rope  | 448–511| bf16            | Stored as-is                    |
| Scales    | 7      | float32 / e8m0  | One per tile (7 tiles × 64 = 448)|

Score = `Q_nope @ K_nope^T + Q_rope @ K_rope^T`
Output = `softmax(Score) @ V` where `V = [V_nope; V_rope]`

## Kernels

| Function              | Description                                        |
|-----------------------|----------------------------------------------------|
| `decode`              | FP8 KV decode with CSR indptr (MFMA, 16 heads/block) |
| `sparse_attn_decode`  | Sparse FP8 decode with packed KV + top-k indices   |

Both use MFMA 16×16×16 bf16. FP8 nope is dequantized to BF16 before GEMM
(matching FlashMLA's dequant-before-GEMM approach).

The sparse decode includes:
- **Adaptive multi-head dispatch**: processes 1 or 2 MFMA head groups per
  block depending on batch size to balance KV reuse vs. GPU occupancy
- **Split-K parallelism**: automatically splits the top-k dimension across
  multiple blocks when batch size is too small to fill all CUs
- **Index sorting**: sorts KV indices for better HBM access locality

## Usage

```python
import flash_mla

# FP8 KV decode (CSR indptr interface)
output = flash_mla.decode(
    q,              # [B, H, 512] bf16
    kv_nope_fp8,    # [T, 448] float8_e4m3fnuz
    kv_rope,        # [T, 64] bf16
    kv_scales,      # [T, 7] float32
    kv_indptr,      # [B+1] int32
    num_heads,
)

# Sparse attention decode (FlashMLA interface, packed KV)
out, lse = flash_mla.sparse_attn_decode(
    q,              # [b, s_q, h_q, 512] bf16
    kv_packed,      # [num_blocks, block_size, 1, 584] uint8
    indices,        # [b, s_q, topk] int32 (flat KV indices, <0 = invalid)
    topk_length=None,  # [b] int32, optional
    sm_scale=None,     # defaults to 1/sqrt(512)
)
```

**Important:** All input tensors must be contiguous. The Python API calls
`.contiguous()` automatically, but for best performance, ensure your tensors
are already contiguous to avoid copies.

## Build

JIT-compiled on first use via Ninja. Requires:
- ROCm with `hipcc` (default: `/opt/rocm`)
- `apache-tvm-ffi` (`pip install apache-tvm-ffi`)

Environment variables:
- `ROCM_PATH` — ROCm installation path (default: `/opt/rocm`)
- `GPU_ARCH` — GPU architecture (default: `native`)
- `FLASH_MLA_CACHE_DIR` — JIT build cache (default: `~/.cache/flash_mla`)
- `FLASH_MLA_JIT_VERBOSE=1` — show build output

## Test

Run correctness tests (compares against PyTorch fp32 reference):

```bash
cd mla
python3 test/test.py
```

## Benchmark

### Sparse decode benchmark (end-to-end, with H20 comparison)

Benchmarks `sparse_attn_decode` across FlashMLA MODEL1 configurations and
compares TFlops / GB/s / latency against H20 reference numbers:

```bash
cd mla
python3 test/bench_sparse_decode.py
```

This runs CONFIG1 (`h_q=64, topk=640`) and CONFIG2 (`h_q=128, topk=1152`)
at batch sizes [2, 64, 74, 128, 148, 256], plus peak-throughput configs
(`topk=16384`). Output includes a comparison table vs. H20 FlashMLA.

### Kernel-only benchmark (no Python overhead)

Profiles the raw HIP kernel dispatch time, bypassing Python tensor
allocation and index sorting overhead:

```bash
cd mla
python3 test/bench_kernel_only.py
```

Reports kernel execution time (us) and TFlops for a subset of configs.
Useful for isolating kernel performance from host-side overhead.
