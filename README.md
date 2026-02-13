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

## Highlighted Example

**[`vector_add_asm/`](vector_add_asm/)** `[A/H]` -- A minimal but complete
hand-written GCN assembly kernel for `C[i] = A[i] + B[i]` on gfx942,
demonstrating persistent kernel launch, double LDS buffering with deep pipeline
fill, `buffer_load_dword ... offen lds` (async load to LDS), OOB-based control
flow (no exec mask), and `vmcnt(3)` pipelining.
See the [detailed README](vector_add_asm/README.md) for a full write-up of
every technique and lesson learned.

## Examples

### Vector / Element-wise

| Folder | Tag | Description |
|--------|-----|-------------|
| `vector_add` | `[H]` | Basic HIP vector add `C[i] = A[i] + B[i]` |
| `vector_add_asm` | `[A/H]` | Hand-written asm vector add with `buffer_load...lds`, double LDS buffering, OOB control flow, `vmcnt(3)` pipelining (gfx942) |
| `absdiff` | `[H]` | SAD 8x8 absolute difference using `__builtin_amdgcn_sad_u16` |

### Matrix Multiply (GEMM)

| Folder | Tag | Description |
|--------|-----|-------------|
| `hgemm` | `[A/H]` | Half-precision GEMM 128x128 with hand-written MFMA asm |
| `hgemm_mfma` | `[H+]` | Half-precision GEMM with MFMA; Python codegen helpers |
| `sgemm` | `[A/H]` | SGEMM 128x128 with hand-written asm (VEGA64) |
| `sgemm_reduction` | `[H]` | SGEMM with warp reduction for large K, small M/N |

### Matrix Core / MFMA

| Folder | Tag | Description |
|--------|-----|-------------|
| `matrix_core` | `[H]` | Matrix core MFMA demo (32x32x16 f16, gfx942) |
| `matrix_core_a` | `[H]` | Alternate matrix core MFMA demo |
| `matrix_core_asm` | `[A/H]` | Hand-written GCN matrix core kernel |
| `matrix_core_gfx950` | `[H]` | Matrix core MFMA for gfx950 |
| `matrix_core_opus` | `[H]` | Matrix core via opus library |

### Memory Bandwidth / Memcpy

| Folder | Tag | Description |
|--------|-----|-------------|
| `bandwidth` | `[A/H]` | Memory bandwidth memcpy with hand-written asm kernel |
| `bandwidth_c` | `[H]` | Memory bandwidth benchmark (CUDA/HIP memcpy kernel) |
| `bandwidth_hip_jit` | `[H]` | JIT HIP bandwidth -- kernel compiled and loaded at runtime |
| `bandwidth_memread` | `[H]` | Global memory read bandwidth (persistent memcpy, stream) |
| `bandwidth_memread_2d` | `[H]` | 2D memory read bandwidth benchmark |
| `membench` | `[H+]` | Memory bandwidth test suite with JSON config |
| `memcpy_async` | `[H]` | Async memcpy via shared memory (global -> LDS -> global) |
| `memcpy_example` | `[A/H]` | Memcpy kernel with hand-written asm |
| `memcpy_example_gfx1030` | `[A/H]` | Memcpy kernels for gfx1030 (RDNA2) |
| `triton_memread` | `[+]` | Triton memory read kernel |
| `smid` | `[H]` | Memcpy throughput benchmark (buffer load, swizzled) |

### Warp / Wave Primitives

| Folder | Tag | Description |
|--------|-----|-------------|
| `wave_reduce_dpp` | `[H]` | Wave-level reduction using DPP |
| `warp_sort` | `[H]` | Warp-level sort using DPP |
| `warp_sort_bitonic` | `[H+]` | Bitonic merge sort (med3 + DPP/SHFL) with Python wrapper |
| `warp_histogram` | `[H+]` | PyTorch extension for warp-level histogram |
| `ds_permute` | `[H]` | LDS `ds_permute` via `__builtin_amdgcn_ds_permute` |
| `transpose-lds` | `[A/H]` | Matrix transpose via LDS with hand-written asm |

### Reduction

| Folder | Tag | Description |
|--------|-----|-------------|
| `nbuf_reduction` | `[H]` | N-buffer reduction kernel |
| `nbuf_reduction_async` | `[H]` | Async N-buffer reduction |
| `nbuf_reduction_w` | `[H]` | N-buffer reduction variant |

### Atomics / Synchronization

| Folder | Tag | Description |
|--------|-----|-------------|
| `cmpswap_atomic` | `[H]` | Atomic compare-swap with bf16x2 reduction |
| `cmpswap_atomic_bench` | `[H]` | Benchmark for atomic compare-swap reduction |
| `cross-wg-sync` | `[H]` | Cross-workgroup synchronization |

### Type Conversion

| Folder | Tag | Description |
|--------|-----|-------------|
| `cvt_fp8` | `[H]` | FP8 (E4M3) conversion test using builtins |
| `cvt_i4` | `[H]` | Int4 conversion / IPS test |
| `pk_cvt` | `[H]` | FP8 packed conversion |
| `opus_cast` | `[H]` | Fast tanh via opus (ROCm / CUDA) |
| `mix_load` | `[H]` | Mixed-load kernel (bf16x2, gfx942) |
| `lqq` | `[H]` | LQQ quantization: i8 -> i4 with scale/zero |

### Integer Arithmetic

| Folder | Tag | Description |
|--------|-----|-------------|
| `int_divide_mod` | `[A/H]` | Integer division / modulus via hand-written asm |
| `int_divide_mod_2` | `[A/H]` | Integer division / modulus variant |
| `magic_integer_division` | `[A/H]` | Magic-number integer division via GCN asm |

### Instruction Throughput / HW Probing

| Folder | Tag | Description |
|--------|-----|-------------|
| `measure_ips` | `[A/H]` | Instruction throughput (IPS) measurement |
| `measure_ips_bench` | `[A/H+]` | IPS benchmark for many instructions; Python generates asm |
| `hwreg` | `[H]` | HW_ID register readout for wave layout |
| `hwreg_mask` | `[H]` | HW_ID register access with masking |

### Buffer / LDS / Async

| Folder | Tag | Description |
|--------|-----|-------------|
| `buffer_ld_oob` | `[H]` | Buffer load out-of-bounds behavior test |
| `async_copy` | `[H]` | Matrix transpose with HIP async copy |
| `test_lds_inst` | `[H]` | LDS instruction test |

### Math Functions / Compiler Tests

| Folder | Tag | Description |
|--------|-----|-------------|
| `test_exp2` | `[H]` | Device-side `exp2f()` test |
| `test_tanh` | `[H]` | Device tanh test (ROCm / CUDA) |
| `test_compiler` | `[H]` | Compiler builtin tests (`ds_permute`) |

### PyTorch Extensions / Python Tools

| Folder | Tag | Description |
|--------|-----|-------------|
| `opus_fmm` | `[H+]` | PyTorch extension for opus flat matrix multiply |
| `hadmard_rotate` | `[+]` | Hadamard / orthogonal matrix (PyTorch) |
| `merge_w_kv_o` | `[+]` | Merge K/KV/O utility (Python) |
| `co-exec` | `[+]` | Python tool: compile + run asm/HIP kernels via HSA |
