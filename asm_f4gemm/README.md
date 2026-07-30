# asm_f4gemm -- driving aiter's gfx950 MXFP4 GEMM code objects

A standalone HIP host driver for the hand-written **f4gemm** assembly kernels that ship as pre-built code objects in [ROCm/aiter](https://github.com/ROCm/aiter).

```
D[M, N] (bf16) = alpha * A[M, K] (mxfp4) * B[N, K] (mxfp4)^T + beta * C[M, N]
```

This directory contains **host launch logic only**. It re-implements the relevant parts of aiter's `csrc/py_itfs_cu/asm_gemm_a4w4.cu` with no torch and no aiter dependency, so the kernels can be poked at, verified against a CPU reference, and benchmarked from a plain HIP program.

## Learning resources for AMDGPU assembly

If you want to read or hand-edit the kernels (see the round-trip section below), these are the references worth having open:

- **[CDNA4 Instruction Set Architecture](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf)** (AMD, PDF) — the gfx950 ISA: every instruction's encoding and semantics, the MFMA/MXFP4 tables, `s_waitcnt` rules, LDS/buffer details. The single most important document. (ISAs for other architectures are collected on the [ROCm GPU architecture page](https://rocm.docs.amd.com/en/latest/reference/gpu-arch/index.html).)
- **[LLVM AMDGPU usage](https://llvm.org/docs/AMDGPUUsage.html)** — how the LLVM assembler/disassembler talks about AMD GPUs: target IDs (`gfx950:xnack+`), code object versions, the `.amdhsa_kernel` / `.amdgpu_metadata` directives, and ELF note layouts. This is the ground truth for what `co2asm.py` reconstructs and what `clang -x assembler` accepts.
- **[AMD CDNA 4 architecture whitepaper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf)** (AMD, PDF) — the machine model: CUs, matrix pipes, LDS, cache hierarchy. Background for why kernels are pipelined the way they are.
- The disassembler/assembler themselves: `llvm-objdump -d --triple=amdgcn-amd-amdhsa --mcpu=gfx950` and `clang -x assembler` (both under `/opt/rocm/llvm/bin`) — when in doubt, assemble a snippet and disassemble it back.

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

## Round-tripping a code object: `.co` -> `.s` -> `.co`

aiter ships only the assembled objects, not the sources. The intended workflow is fully manual:

1. **Disassemble** a `.co` into reassemblable GCN assembly with `co2asm.py`.
2. **Edit the `.s` by hand** — that is the whole point.
3. **Reassemble** into a new `.co` with clang (`co2asm.py` prints the exact command, with the arch and code object version already detected).
4. **Run it** through the host driver: put the rebuilt `.co` in a directory with a one-line manifest CSV listing only that kernel, and point `--co-dir` at it — so whatever shape you ask for, the rebuilt object is provably the one that ran.

```bash
# 1. disassemble (writes <kernel>.s next to the .co by default; -o to redirect)
python3 co2asm.py /path/to/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co

# 2. edit the .s ... then

# 3. reassemble
/opt/rocm/llvm/bin/clang -x assembler -target amdgcn-amd-amdhsa \
    -mcpu=gfx950 -mcode-object-version=6 \
    f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.s \
    -o rebuilt/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co

# 4. manifest listing only the rebuilt kernel, then run it
SRC=/path/to/aiter/hsa/gfx950/f4gemm
head -1 $SRC/f4gemm_bf16_per1x32Fp4.csv > rebuilt/f4gemm_bf16_per1x32Fp4.csv
grep ",f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co" $SRC/f4gemm_bf16_per1x32Fp4.csv \
    >> rebuilt/f4gemm_bf16_per1x32Fp4.csv
./asm_f4gemm.exe --co-dir ./rebuilt -m 4096 -n 4096 -k 4096
```

The 256x256 preshuffled tile is a good starting point: it is the tile the heuristic picks for every large shape, and the fastest one measured above.

### Reconstructing the assembly

For a quick look at a kernel, raw `llvm-objdump` is all you need:

```bash
/opt/rocm/llvm/bin/llvm-objdump -d --triple=amdgcn-amd-amdhsa --mcpu=gfx950 <file.co>
```

That prints every section it can decode — `.text` as instructions (each line suffixed with a `// <addr>: <encoding>` comment) and the 64-byte kernel descriptor in `.rodata` rendered as `.amdhsa_*` directives. What it prints is **not** reassemblable as-is; `co2asm.py` is the delta between that raw dump and a `.s` that `clang -x assembler` accepts and reproduces the original with. Concretely, it:

1. **Keeps only `.text` from the instruction dump** and strips the `// <addr>: <encoding>` comment from every line. Kernel entry points (any symbol not named `label_XXXX`) become `.globl` + `.type @function`, with `.p2align 8` restored.
2. **Separately dumps the descriptor** with `llvm-objdump -d -j .rodata` and applies the two field fixups described below (`.amdhsa_next_free_sgpr` and `.amdhsa_reserve_xnack_mask`).
3. **Recovers the metadata note** — the msgpack kernarg/LDS/workgroup table — from `llvm-readelf --notes` and re-emits the YAML document between `.amdgpu_metadata` / `.end_amdgpu_metadata`.
4. **Maps the ELF header to assembler flags**: arch from `Flags:` (`--mcpu` for both objdump and clang), and `ABI Version` to the real code object version (`ABI 4` → `-mcode-object-version=6`, see below), then prints the exact `clang` reassembly command.

A code object holds three things the assembler needs, and each comes from a different tool:

| Piece | Recovered with | Becomes |
|---|---|---|
| instructions + local labels | `llvm-objdump -d` | `.text` |
| 64-byte kernel descriptor | `llvm-objdump -d -j .rodata` | `.amdhsa_kernel` ... `.end_amdhsa_kernel` |
| msgpack note (kernarg offsets, LDS, workgroup size) | `llvm-readelf --notes` | `.amdgpu_metadata` ... `.end_amdgpu_metadata` |

The branch targets survive because the objects keep their `label_XXXX` symbols in `.symtab`, so `llvm-objdump` emits real labels rather than raw addresses.

Three things about llvm-objdump's descriptor dump make it *not* directly reassemblable, all handled in `co2asm.py`:

- **`.amdhsa_next_free_sgpr` is not what it looks like.** The descriptor only stores the SGPR count granulated by 8, so objdump reports the top of the granule (104) *and* inverts the encoding assuming zero extra SGPRs. The assembler goes the other way and adds `getNumExtraSGPRs()` back before granulating — 6 on gfx8+ when flat_scratch is reserved, which is the default objdump never prints. Feeding 104 straight back is rejected outright (gfx9 addresses at most 102), and clamping to 102 silently lands one granule too high. Subtracting the 6 extras reproduces the encoded byte exactly.
- **`.amdhsa_reserve_xnack_mask` is only legal when the target id names xnack.** These objects are built for xnack `ANY`, so the directive is dropped and the target left as plain `gfx950` — which also keeps the rebuilt ELF feature flags bit-identical. Dropping it is safe precisely because the extra-SGPR count is driven by flat_scratch, not xnack.
- **`ABI Version` in the ELF header is not the code object version.** `ELFABIVERSION_AMDGPU_HSA_V2` is 0, V3 is 1, and so on, so the `ABI Version: 4` these objects report means `-mcode-object-version=6`. Building with `4` produces a silently different object.

### What "equivalent" means here

Before modifying anything, it is worth checking the plain round-trip against the original — four checks, all doable by hand:

```bash
ORIG=/path/to/aiter/hsa/gfx950/f4gemm/f4gemm_..._256x256.co
NEW=rebuilt/f4gemm_..._256x256.co
RL=/opt/rocm/llvm/bin

# 1. ELF header: arch + feature flags + ABI version must match
diff <($RL/llvm-readelf -h $ORIG | grep -E "ABI Version|Flags:") \
     <($RL/llvm-readelf -h $NEW  | grep -E "ABI Version|Flags:")

# 2. kernel descriptor: must be byte-identical, it drives SGPR/VGPR/LDS setup
$RL/llvm-objcopy --dump-section=.rodata=o.kd $ORIG /dev/null
$RL/llvm-objcopy --dump-section=.rodata=n.kd $NEW  /dev/null
cmp o.kd n.kd

# 3. .text: compare as disassembly, not bytes (see below)
diff <($RL/llvm-objdump -d --triple=amdgcn-amd-amdhsa --mcpu=gfx950 $ORIG | sed 's|//.*||') \
     <($RL/llvm-objdump -d --triple=amdgcn-amd-amdhsa --mcpu=gfx950 $NEW  | sed 's|//.*||')

# 4. metadata note: kernarg offsets, LDS, workgroup size
diff <($RL/llvm-readelf --notes $ORIG) <($RL/llvm-readelf --notes $NEW)
```

For an unmodified round-trip of the 256x256 tile, all four report identical — the descriptor and metadata are **byte-identical** and the rebuilt file is the same size as the original (35632 bytes).

`.text` is compared as *disassembly*, not as bytes, because 508 bytes genuinely differ. Every one of them is bit 13 and/or 14 of a `v_mfma_scale_f32_16x16x128_f8f6f4` first dword (`0xd3ac....`) — the src2 `op_sel` / `op_sel_hi` bits, which are don't-cares for the accumulator operand. Whatever assembled the original set them; LLVM's assembler emits 0 and its disassembler ignores them, so all 3413 instructions decode identically. Nothing else in `.text` moves.

### Verified on hardware

The rebuilt object was run through the same checks as the original on MI355X:

| | rebuilt | original |
|---|---|---|
| `M=4096 N=4096 K=4096` | PASSED, `max_abs_err=128`, `max_rel_err=0.003891` | PASSED, `max_abs_err=128`, `max_rel_err=0.003891` |
| same, TFLOP/s | 4048 | 3958 |
| `M=300 N=2048 K=1024` | PASSED | PASSED |
| `M=1 N=4096 K=2048` | PASSED | PASSED |
| `--splitk 2` / `--splitk 4` | PASSED | PASSED |

Identical error metrics, and the throughput difference is run-to-run noise.

### Pointing at the wrong directory

Because `rebuilt/` holds a single tile, a wrong `--co-dir` is not a silent fallback. If the manifest lists a kernel the directory does not contain, the driver names the missing file and exits non-zero rather than letting `hipModuleLoad` report a bare "file not found":

```
[f4gemm] no such code object: /tmp/f4probe/f4gemm_bf16_per1x32Fp4_BpreShuffle_96x640.co
[f4gemm]   the manifest lists this kernel but --co-dir does not contain it -- point --co-dir at a directory holding it
```

`co2asm.py` is not f4gemm-specific — it takes any AMDGPU code object and detects the arch and code object version from the ELF header.

## Files

| File | Contents |
|------|----------|
| `f4gemm.hpp` | kernarg struct, manifest parser, tile heuristic, `hipModuleLoad` wrapper, grid setup |
| `f4gemm_ref.hpp` | fp4/e8m0 decode, B and scale shuffles, operand generation, CPU reference |
| `main.cpp` | CLI, buffer setup, verification, benchmark |
| `run_tests.sh` | regression sweep |
| `co2asm.py` | code object -> reassemblable `.s` (any AMDGPU `.co`, not just f4gemm) |
