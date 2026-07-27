# asm_f4gemm -- driving aiter's gfx950 MXFP4 GEMM code objects

A standalone HIP host driver for the hand-written **f4gemm** assembly kernels that ship as pre-built code objects in [ROCm/aiter](https://github.com/ROCm/aiter).

```
D[M, N] (bf16) = alpha * A[M, K] (mxfp4) * B[N, K] (mxfp4)^T + beta * C[M, N]
```

This directory contains **host launch logic only**. It re-implements the relevant parts of aiter's `csrc/py_itfs_cu/asm_gemm_a4w4.cu` with no torch and no aiter dependency, so the kernels can be poked at, verified against a CPU reference, and benchmarked from a plain HIP program.

## Dependency: the code objects live in aiter

The `.co` files are **not** copied into gcnasm. Point the driver at aiter's tree:

```
<aiter>/hsa/gfx950/f4gemm/
    f4gemm_bf16_per1x32Fp4.csv                          <- kernel manifest
    f4gemm_bf16_per1x32Fp4_BpreShuffle_<tileM>x<tileN>.co
    f4gemm_bf16_per1x32Fp4_noBpreShuffle_256x256.co
```

35 kernels total: 34 pre-shuffled-B tiles plus one non-pre-shuffled 256x256. The manifest CSV is parsed at runtime (columns `tile_M, tile_N, splitK, bpreshuffle, knl_name, co_name`), so nothing has to be regenerated when aiter adds a tile.

## Build and run

```bash
./build.sh                       # host-only; hipcc, no --offload-arch needed

export AITER_ASM_DIR=/path/to/aiter/hsa      # or pass --co-dir directly
./asm_f4gemm.exe -m 512 -n 1024 -k 2048
./asm_f4gemm.exe --list                      # dump the manifest
./run_tests.sh                               # full regression sweep
```

Options: `-m/-n/-k`, `--co-dir DIR`, `--kernel NAME` (mangled symbol or `.co` file name), `--bpreshuffle 0|1`, `--splitk L` (`-1` = let the heuristic try 2/4/8/16), `--iters N`, `--no-verify`, `--list`.

Shape constraints enforced by the driver: `K % 256 == 0` and `N % 16 == 0`. `M` is unconstrained (`M = 1` works).

## Kernel ABI

### Kernarg block

Every argument occupies a 16-byte slot; the kernels `s_load` them from fixed offsets. Offsets below are from the `.co` metadata (`llvm-readelf --notes`), and the "read" column is from the disassembly (`llvm-objdump -d --triple=amdgcn-amd-amdhsa --mcpu=gfx950`).

| offset | field | size | read by kernel |
|--------|-------|------|----------------|
| `0x00` | `ptr_D` | 8 | yes |
| `0x10` | `ptr_C` (bias) | 8 | yes |
| `0x20` | `ptr_A` | 8 | yes |
| `0x30` | `ptr_B` | 8 | yes |
| `0x40` | `alpha` | 4 | yes |
| `0x50` | `beta` | 4 | yes |
| `0x60` | `stride_D0` | 4 | **no** |
| `0x70` | `stride_D1` | 4 | no |
| `0x80` | `stride_C0` | 4 | yes -- used for **both** C and D |
| `0x90` | `stride_C1` | 4 | no |
| `0xa0` | `stride_A0` | 4 | yes |
| `0xb0` | `stride_A1` | 4 | no |
| `0xc0` | `stride_B0` | 4 | yes |
| `0xd0` | `stride_B1` | 4 | no |
| `0xe0` | `M` | 4 | yes |
| `0xf0` | `N` | 4 | yes |
| `0x100` | `K` | 4 | yes |
| `0x110` | `ptr_ScaleA` | 8 | yes |
| `0x120` | `ptr_ScaleB` | 8 | yes |
| `0x130` | `stride_ScaleA0` | 4 | yes |
| `0x140` | `stride_ScaleA1` | 4 | no |
| `0x150` | `stride_ScaleB0` | 4 | yes |
| `0x160` | `stride_ScaleB1` | 4 | no |
| `0x170` | `log2_k_split` | 4 | splitK kernels only |

`sizeof == 0x174`; `kernarg_segment_size` is 384 (368 for the noBpreShuffle kernel, which stops before `log2_k_split`).

Note that `stride_D0` is dead -- the kernel uses `stride_C0` for the D store too, which is why aiter assigns `out.stride(0)` to `stride_C0` and leaves `stride_D0` unset. A/B strides are counted in **fp4 elements**, not bytes: aiter passes `tensor.stride(0) * 2` because the tensors are stored as `fp4x2` bytes. The kernel does the `>> 1` back to bytes itself.

### Launch geometry

All 35 kernels: **256 threads** (4 x wave64), **160 KB LDS** (static `group_segment_fixed_size`, so `sharedMemBytes` stays 0), 512 VGPRs, 96 SGPRs.

```
gdx = ceil(N / tile_N)
gdy = ceil(M / tile_M)
gdz = 1, or the K-split count for splitK kernels
```

The kernel flattens `wg_y * gdx + wg_x` and re-swizzles it into groups of 32 N-tiles for L2 locality.

### Tile selection heuristic

`select_kernel()` ports aiter's `get_heuristic_kernel()`: for each manifest entry it computes `ceil(tiles / num_cu)` rounds and picks the fewest rounds, tie-broken by CU occupancy and by the `tile_M * tile_N / (tile_M + tile_N)` compute-to-memory ratio. One quirk carried over: the 128x512 tile is skipped unless `N % 512 == 0`.

aiter iterates an `unordered_map` here, so its tie-breaking is not reproducible run to run. This port iterates the CSV in order, which makes the choice stable.

## Data layouts

These mirror what the aiter python side produces, and getting any of them wrong is the usual reason a hand-rolled launch returns garbage.

### A -- activations, `[M, K/2]` packed MXFP4

Row-major, no shuffle. Byte `i` of a row holds element `2i` in the **low** nibble and `2i+1` in the high nibble. Values are OCP e2m1: `{0, .5, 1, 1.5, 2, 3, 4, 6}` with the sign in bit 3.

### B -- weights, `[N, K/2]` packed MXFP4, 16x16 tile-transposed

For the `BpreShuffle` kernels, B goes through aiter's `shuffle_weight(w, layout=(16, 16))` on the packed byte buffer:

```
src.view(N/16, 16, Kp/32, 2, 16).permute(0, 2, 3, 1, 4)      # Kp = K/2

src[n0*16 + n1][k0*32 + k1*16 + k2]
  -> dst[(((n0 * Kp/32 + k0) * 2 + k1) * 16 + n1) * 16 + k2]
```

The single `noBpreShuffle_256x256` kernel takes plain row-major B instead.

### A_scale / B_scale -- `[rows, K/32]` E8M0, padded and shuffled

E8M0 is a bare biased exponent: value = `2^(e - 127)`, with `0 -> 2^-126` and `0xFF -> NaN`. One scale per 32 K-elements.

The buffer is first **padded to `[round_up(rows, 256), round_up(K/32, 8)]`**, then shuffled (aiter's `shuffle_scale`, the non-`guinterleave` path):

```
padded.view(sm/32, 2, 16, sn/8, 2, 4).permute(0, 3, 5, 2, 4, 1)

padded[d0*32 + d1*16 + d2][d3*8 + d4*4 + d5]
  -> dst[((((d0 * sn/8 + d3) * 4 + d5) * 16 + d2) * 2 + d4) * 2 + d1]
```

`stride_ScaleA0` / `stride_ScaleB0` are the **padded** column count `sn`. This driver fills the pad with `0x7F` (2^0 == 1.0) so an edge tile can never pick up a NaN scale.

### D -- output, `[M, N]` bf16, rows padded to 32

aiter allocates `[(M + 31) / 32 * 32, N]` and slices `[:M]` afterwards; `stride_C0 = N`. A/B rows are over-allocated to a multiple of 256 here so an edge tile can never touch unmapped memory.

### C / bias

`beta = 0` and `ptr_C = nullptr` is the only path aiter's own op tests exercise, and it is what this driver uses. The bias path is wired through the kernarg block but not verified here -- aiter documents `bias` as `f32` while `stride_C0` comes from the bf16 output tensor, so the intended element type is ambiguous.

## Verification

Operands are random fp4 nibbles with exponents drawn from `2^-3 .. 2^3`, which keeps every partial product exactly representable in f32. The reference is a threaded f64-accumulate CPU GEMM over the dequantized operands.

Without splitK, the measured error is exactly the bf16 output rounding, `max_rel_err = 0.003891 ~= 2^-8`, across all 35 kernels.

**splitK accumulates in bf16.** The splitK epilogue writes each K-chunk's partial sum with `buffer_atomic_pk_add_bf16`, so the cross-chunk reduction itself runs at bf16 precision -- the error scales with the magnitude of the *partials*, not of the final element, and heavily-cancelling outputs can be off by a large relative amount. Measured on `M=256 N=1024 K=4096`:

| `log2_k_split` | max abs err | as a fraction of max\|ref\| |
|---|---|---|
| 0 | 103 | 0.28 % |
| 1 | 184 | 0.51 % |
| 2 | 224 | 0.62 % |
| 3 | 293 | 0.81 % |
| 4 | 416 | 1.15 % |

So the check switches to an absolute bound of 2 % of `max|ref|` whenever `gdz > 1`. This is a real property of the kernels, not a launch bug; aiter's own `op_tests/test_gemm_a4w4.py` leaves the splitK path commented out.

## Measured

MI355X (gfx950, 256 CU), `--iters 50`, heuristic kernel selection:

| M | N | K | kernel | us | TFLOP/s |
|---|---|---|--------|----|---------|
| 8192 | 8192 | 8192 | BpreShuffle_256x256 | 259.0 | 4245 |
| 2048 | 8192 | 8192 | BpreShuffle_256x256 | 64.4 | 4270 |
| 4096 | 4096 | 4096 | BpreShuffle_256x256 | 33.5 | 4103 |
| 128 | 16384 | 16384 | BpreShuffle_96x640 | 70.2 | 979 |

`run_tests.sh` covers 8 heuristic shapes, all 35 kernels at `M=300 N=2048 K=1024`, and 8 splitK configurations: 51 checks, all passing.

## Files

| File | Contents |
|------|----------|
| `f4gemm.hpp` | kernarg struct, manifest parser, tile heuristic, `hipModuleLoad` wrapper, grid setup |
| `f4gemm_ref.hpp` | fp4/e8m0 decode, B and scale shuffles, operand generation, CPU reference |
| `main.cpp` | CLI, buffer setup, verification, benchmark |
| `run_tests.sh` | regression sweep |
