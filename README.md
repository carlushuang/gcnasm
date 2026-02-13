# gcnasm -- GCN Assembly & HIP Programming Examples

A collection of AMD GPU programming examples targeting **CDNA / RDNA**
architectures (primarily **gfx942 / MI300**), covering hand-written GCN
assembly kernels, HIP C++ device code, and PyTorch/Triton extensions.

## Legend

| Tag | Meaning |
|-----|---------|
| `[A]` | Hand-written GCN assembly kernel (`.s`) |
| `[H]` | HIP / C++ / CUDA device code |
| `[A/H]` | Both hand-written assembly **and** HIP host code |
| `+` | Has Python / PyTorch / Triton interface (can run from Python directly) |

## Highlighted Examples

**[`bandwidth_memread/`](bandwidth_memread/)** `[H]` -- The go-to
**memory bandwidth microbenchmark**. Measures peak read-only and read+write
GPU memory bandwidth using float4 vectorized, non-temporal, persistent kernels.
Supports both ROCm and CUDA. Sweeps from ~78 KB to ~1.7 GB and reports GB/s
per size. Peak observed: **~4.56 TB/s read-only** on MI308X (gfx942).
See the [detailed README](bandwidth_memread/README.md).

**[`vector_add_asm/`](vector_add_asm/)** `[A/H]` -- A minimal but complete
hand-written GCN assembly kernel for `C[i] = A[i] + B[i]` on gfx942,
demonstrating persistent kernel launch, double LDS buffering with deep pipeline
fill, `buffer_load_dword ... offen lds` (async load to LDS), OOB-based control
flow (no exec mask), and `vmcnt(3)` pipelining.
See the [detailed README](vector_add_asm/README.md).

## Examples

### Vector / Element-wise

| Tag | Folder | Description |
|-----|--------|-------------|
| `[H]` | [`vector_add`](vector_add/) | Basic HIP vector add `C[i] = A[i] + B[i]` |
| `[A/H]` | [`vector_add_asm`](vector_add_asm/) | Hand-written asm vector add with `buffer_load...lds`, double LDS buffering, OOB control flow, `vmcnt(3)` pipelining (gfx942) |
| `[H]` | [`absdiff`](absdiff/) | SAD 8x8 absolute difference using `__builtin_amdgcn_sad_u16` |

### Matrix Multiply (GEMM)

| Tag | Folder | Description |
|-----|--------|-------------|
| `[A/H]` | [`hgemm`](hgemm/) | Half-precision GEMM 128x128 with hand-written MFMA asm |
| `[H+]` | [`hgemm_mfma`](hgemm_mfma/) | Half-precision GEMM with MFMA; Python codegen helpers |
| `[A/H]` | [`sgemm`](sgemm/) | SGEMM 128x128 with hand-written asm (VEGA64) |
| `[H]` | [`sgemm_reduction`](sgemm_reduction/) | SGEMM with warp reduction for large K, small M/N |

### Matrix Core / MFMA

| Tag | Folder | Description |
|-----|--------|-------------|
| `[H]` | [`matrix_core`](matrix_core/) | Matrix core MFMA demo (32x32x16 f16, gfx942) |
| `[H]` | [`matrix_core_a`](matrix_core_a/) | Alternate matrix core MFMA demo |
| `[A/H]` | [`matrix_core_asm`](matrix_core_asm/) | Hand-written GCN matrix core kernel |
| `[H]` | [`matrix_core_gfx950`](matrix_core_gfx950/) | Matrix core MFMA for gfx950 |
| `[H]` | [`matrix_core_opus`](matrix_core_opus/) | Matrix core via opus library |

### Memory Bandwidth / Memcpy

| Tag | Folder | Description |
|-----|--------|-------------|
| `[A/H]` | [`bandwidth`](bandwidth/) | Memory bandwidth memcpy with hand-written asm kernel |
| `[H]` | [`bandwidth_c`](bandwidth_c/) | Memory bandwidth benchmark (CUDA/HIP memcpy kernel) |
| `[H]` | [`bandwidth_hip_jit`](bandwidth_hip_jit/) | JIT HIP bandwidth -- kernel compiled and loaded at runtime |
| `[H]` | [`bandwidth_memread`](bandwidth_memread/) | **Memory bandwidth microbenchmark** -- read-only & read+write, float4, non-temporal, persistent kernels (ROCm / CUDA) |
| `[H]` | [`bandwidth_memread_2d`](bandwidth_memread_2d/) | 2D memory read bandwidth benchmark |
| `[H+]` | [`membench`](membench/) | Memory bandwidth test suite with JSON config |
| `[H]` | [`memcpy_async`](memcpy_async/) | Async memcpy via shared memory (global -> LDS -> global) |
| `[A/H]` | [`memcpy_example`](memcpy_example/) | Memcpy kernel with hand-written asm |
| `[A/H]` | [`memcpy_example_gfx1030`](memcpy_example_gfx1030/) | Memcpy kernels for gfx1030 (RDNA2) |
| `[+]` | [`triton_memread`](triton_memread/) | Triton memory read kernel |
| `[H]` | [`smid`](smid/) | Memcpy throughput benchmark (buffer load, swizzled) |

### Warp / Wave Primitives

| Tag | Folder | Description |
|-----|--------|-------------|
| `[H]` | [`wave_reduce_dpp`](wave_reduce_dpp/) | Wave-level reduction using DPP |
| `[H]` | [`warp_sort`](warp_sort/) | Warp-level sort using DPP |
| `[H+]` | [`warp_sort_bitonic`](warp_sort_bitonic/) | Bitonic merge sort (med3 + DPP/SHFL) with Python wrapper |
| `[H+]` | [`warp_histogram`](warp_histogram/) | PyTorch extension for warp-level histogram |
| `[H]` | [`ds_permute`](ds_permute/) | LDS `ds_permute` via `__builtin_amdgcn_ds_permute` |
| `[A/H]` | [`transpose-lds`](transpose-lds/) | Matrix transpose via LDS with hand-written asm |

### Reduction

| Tag | Folder | Description |
|-----|--------|-------------|
| `[H]` | [`nbuf_reduction`](nbuf_reduction/) | N-buffer reduction kernel |
| `[H]` | [`nbuf_reduction_async`](nbuf_reduction_async/) | Async N-buffer reduction |
| `[H]` | [`nbuf_reduction_w`](nbuf_reduction_w/) | N-buffer reduction variant |

### Atomics / Synchronization

| Tag | Folder | Description |
|-----|--------|-------------|
| `[H]` | [`cmpswap_atomic`](cmpswap_atomic/) | Atomic compare-swap with bf16x2 reduction |
| `[H]` | [`cmpswap_atomic_bench`](cmpswap_atomic_bench/) | Benchmark for atomic compare-swap reduction |
| `[H]` | [`cross-wg-sync`](cross-wg-sync/) | Cross-workgroup synchronization |

### Type Conversion

| Tag | Folder | Description |
|-----|--------|-------------|
| `[H]` | [`cvt_fp8`](cvt_fp8/) | FP8 (E4M3) conversion test using builtins |
| `[H]` | [`cvt_i4`](cvt_i4/) | Int4 conversion / IPS test |
| `[H]` | [`pk_cvt`](pk_cvt/) | FP8 packed conversion |
| `[H]` | [`opus_cast`](opus_cast/) | Fast tanh via opus (ROCm / CUDA) |
| `[H]` | [`mix_load`](mix_load/) | Mixed-load kernel (bf16x2, gfx942) |
| `[H]` | [`lqq`](lqq/) | LQQ quantization: i8 -> i4 with scale/zero |

### Integer Arithmetic

| Tag | Folder | Description |
|-----|--------|-------------|
| `[A/H]` | [`int_divide_mod`](int_divide_mod/) | Integer division / modulus via hand-written asm |
| `[A/H]` | [`int_divide_mod_2`](int_divide_mod_2/) | Integer division / modulus variant |
| `[A/H]` | [`magic_integer_division`](magic_integer_division/) | Magic-number integer division via GCN asm |

### Instruction Throughput / HW Probing

| Tag | Folder | Description |
|-----|--------|-------------|
| `[A/H]` | [`measure_ips`](measure_ips/) | Instruction throughput (IPS) measurement |
| `[A/H+]` | [`measure_ips_bench`](measure_ips_bench/) | IPS benchmark for many instructions; Python generates asm |
| `[H]` | [`hwreg`](hwreg/) | HW_ID register readout for wave layout |
| `[H]` | [`hwreg_mask`](hwreg_mask/) | HW_ID register access with masking |

### Buffer / LDS / Async

| Tag | Folder | Description |
|-----|--------|-------------|
| `[H]` | [`buffer_ld_oob`](buffer_ld_oob/) | Buffer load out-of-bounds behavior test |
| `[H]` | [`async_copy`](async_copy/) | Matrix transpose with HIP async copy |
| `[H]` | [`test_lds_inst`](test_lds_inst/) | LDS instruction test |

### Math Functions / Compiler Tests

| Tag | Folder | Description |
|-----|--------|-------------|
| `[H]` | [`test_exp2`](test_exp2/) | Device-side `exp2f()` test |
| `[H]` | [`test_tanh`](test_tanh/) | Device tanh test (ROCm / CUDA) |
| `[H]` | [`test_compiler`](test_compiler/) | Compiler builtin tests (`ds_permute`) |

### PyTorch Extensions / Python Tools

| Tag | Folder | Description |
|-----|--------|-------------|
| `[H+]` | [`opus_fmm`](opus_fmm/) | PyTorch extension for opus flat matrix multiply |
| `[+]` | [`hadmard_rotate`](hadmard_rotate/) | Hadamard / orthogonal matrix (PyTorch) |
| `[+]` | [`merge_w_kv_o`](merge_w_kv_o/) | Merge K/KV/O utility (Python) |
| `[+]` | [`co-exec`](co-exec/) | Python tool: compile + run asm/HIP kernels via HSA |
