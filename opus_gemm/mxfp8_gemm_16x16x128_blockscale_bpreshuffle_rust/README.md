# mxfp8_gemm_16x16x128_blockscale_bpreshuffle_rust -- the opus MXFP8 GEMM in Rust

Rust port of [`../mxfp8_gemm_16x16x128_blockscale_bpreshuffle`](../mxfp8_gemm_16x16x128_blockscale_bpreshuffle): the gfx950 MXFP8 blockscale GEMM (A row-major FP8, B aiter `(16,16)` preshuffle, E8M0 1x128 / 128x128 scales, 256x256x128 tile, 8 waves, `v_mfma_scale_f32_16x16x128_f8f6f4`) with **both the GPU kernel and the host written in Rust**. The goal was to find out whether an opus-style, hand-scheduled, near-peak kernel can be written in Rust, and what it costs.

**Result: yes. The Rust kernel validates on every configuration and runs at parity with the C++/opus kernel built by the same LLVM (23.1).**

## Results (MI355X gfx950, 256 CU)

Both kernels are launched by the same Rust host (same inputs, launch, and hipEvent timing); `make compare`, w=200, i=100, median of 5 interleaved rounds:

| M x N x K | out | tiles/WG | C++/opus (clang 23.1.1) | Rust (rustc nightly, LLVM 23.1.1) | Rust vs C++ |
|---|---|---|---|---|---|
| 8192 x 8192 x 8192 | BF16 | 4 | 0.3801 ms, 2.893 PF | 0.3785 ms, 2.905 PF | +0.4% |
| 8192 x 8192 x 8192 | FP32 | 4 | 0.3885 ms, 2.830 PF | 0.3850 ms, 2.856 PF | +0.9% |
| 4096 x 4096 x 8192 | BF16 | 1 | 0.1002 ms, 2.745 PF | 0.0998 ms, 2.755 PF | +0.4% |
| 4096 x 4096 x 8192 | FP32 | 1 | 0.1022 ms, 2.689 PF | 0.1010 ms, 2.722 PF | +1.2% |

Run-to-run noise is about +-1%, so read this as parity. The opus numbers match the C++ README (0.3683 ms BF16 on its GPU2 with its clang 23 build).

Correctness: `make verify` passes the C++ README's matrix (6 shapes incl. batch 2 and both 1/4-tile specializations, BF16 and FP32, full double-precision CPU reference, NaN-poisoned output); 8192^3 is checked on 65536 sampled outputs (`-v 2`). ISA (`make check`): 320 scaled MFMA per specialization like the C++ kernel, 0 VGPR/SGPR spills; 256 VGPRs (4-tile) / 242 (1-tile), C++: 256 / 232.

## Layout

```
kernel/src/lib.rs     the kernel, statement by statement against gemm_a8w8_mxfp8_scale_kernel_template.hpp
kernel/src/layout.rs  every opus layout (ga/sa/ra/gb/sb/rb/gsf/ssf/rsfa/rsfb/gc) as closed-form per-lane offsets
kernel/src/amdgpu.rs  wrappers: buffer load/store, buffer->LDS, TR8, scaled MFMA, s_waitcnt, pins
tools/relink.py       finishes rustc's LTO/codegen: launch bounds + IR shims (see below)
host/src/main.rs      port of gemm_a8w8_mxfp8_scale_host.cc (same inputs, B preshuffle, reference, tolerance)
oracle/layout_dump.cc dumps the real opus layouts per thread; `make check-layouts` diffs layout.rs against it
```

## Build and run

```bash
rustup toolchain install nightly --component rust-src --component llvm-tools
make                      # kernel (rustc + tools/relink.py) and host -> build/
make check                # 1280 MFMA, 0 spills
make verify GPU=0         # README verification matrix vs CPU reference
make benchmark GPU=0      # 8192^3, sampled check + timing
# C++/opus reference side by side (needs an upstream clang 23 and the opus headers):
make compare GPU=0 CLANG23_ROOT=/path/to/llvm/build OPUS_INCLUDE_DIR=/path/to/aiter/csrc/include
make check-layouts CLANG23_ROOT=... OPUS_INCLUDE_DIR=...
```

`build/gemm_mxfp8_bpreshuffle` takes the C++ harness options (`-m -n -k -b -v -w -i --dtype --tiles --seed`), plus `--co rust:<co>|opus:<co>` (repeatable), `--rounds`, `-v 2 --samples N`.

## How the opus kernel maps to Rust

| C++ / opus | Rust |
|---|---|
| `__global__ __launch_bounds__(512, 1)` | `extern "gpu-kernel" fn` + `amdgpu-flat-work-group-size=1,512` / `amdgpu-waves-per-eu=1` added by relink.py |
| `opus_gemm_scale_kargs` by value | `#[repr(C)] struct Kargs` by value (identical 96-byte byref kernarg) |
| opus layouts (`make_layout`, `unfold_x_stride`, `partition_layout_c`, ...) | plain functions in layout.rs, checked lane-for-lane against opus |
| `make_gmem` / `load` / `store` | `v4i32` buffer resource + `llvm.amdgcn.raw.buffer.{load,store}` |
| `async_load` (`buffer_load_dwordx4 ... lds`) | `llvm.amdgcn.raw.buffer.load.lds` via IR shim |
| `__shared__` arrays (139264 B) | one launch-sized LDS region (`gpu_launch_sized_workgroup_mem`) |
| `__builtin_amdgcn_ds_read_tr8_b64_v2i32` | `llvm.amdgcn.ds.read.tr8.b64` via IR shim |
| `mfma_adaptor_swap_ab` scaled MFMA | `llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4`, op_sel as const generics |
| `sched_barrier`, `sched_group_barrier`, `s_barrier`, `readfirstlane` | `core::arch::amdgpu` |
| `s_waitcnt_vmcnt/lgkmcnt` | `llvm.amdgcn.s.waitcnt` with the same encodings |
| `asm volatile("" : "+v"(v_c_pin))`, `"+s"(local_role)`, `ds_read2st64_b32`, `ds_write_b8` asm | same inline asm, via IR shims |
| `#pragma unroll 8` on the K loop | hand-unrolled 8x + remainder (same 320-MFMA structure clang produces) |
| `cast<bf16_t>` (`v_cvt_pk_bf16_f32`) | `fptrunc` to `<2 x bfloat>` via IR shim |

## What Rust cannot express, and how it is bridged

rustc (nightly) has a working `amdgcn-amd-amdhsa` target, `extern "gpu-kernel"`, `core::arch::amdgpu`, and accepts `extern "llvm-intrinsic"` declarations. Four things are missing for a kernel like this one; `tools/relink.py` covers them by taking rustc's post-LTO-link bitcode (`-Csave-temps`), patching the IR, and finishing with rustc's own `opt`/`llc`/`ld.lld` (the `llvm-tools` component, so the whole pipeline is one LLVM):

1. **No launch bounds.** Every gpu-kernel gets `amdgpu-flat-work-group-size=1,1024`, which caps a 512-thread kernel at 128 VGPRs (this one needs 256). relink.py adds the attributes clang emits for `__launch_bounds__(512, 1)`.
2. **No address spaces.** Rust pointers are always flat; intrinsics that take `ptr addrspace(3)` (buffer->LDS loads, TR8 reads) cannot be declared, and rustc now rejects mismatched intrinsic signatures. The kernel calls `rk.*` placeholders; relink.py defines them in IR with an `addrspacecast`, and opt inlines them. (Overloaded intrinsics like `ds.read.tr8` did slip through with a generic pointer, but only because release rustc skips the IR verifier.)
3. **No bf16 type.** Output conversion goes through an IR shim doing `fptrunc` to `<2 x bfloat>`, which selects `v_cvt_pk_bf16_f32`.
4. **amdgpu inline asm has no register classes.** `asm!` only works without operands (`asm_experimental_arch`). The opus kernel's `"+v"`/`"+s"` pins and asm LDS accesses are reproduced as IR shims containing the exact same inline asm.

Other gotchas: `core::intrinsics::nontemporal_store` is silently a plain store on amdgpu (see `bandwidth_memread_rust`, [PR #52](https://github.com/carlushuang/gcnasm/pull/52)), so NT stores use the buffer intrinsic's aux bits. There is no `#pragma unroll`; the K loop is unrolled by hand. Tier-3 target: nightly, `-Zbuild-std=core`, `no_std`, one GPU arch per build.

## Getting to parity

First faithful port (all layouts verified, correct results): **72%** of the C++ kernel. Every step below was measured at 8192^3 BF16 against the C++ kernel in the same run:

| step | spills (4-tile) | Rust / C++ |
|---|---|---|
| straight port | 56 | 72% |
| operand-less `asm!("")` where C++ pins accumulators (scheduler fence only) | 42 | 82% |
| + `iterative-minreg` machine scheduler (experiment, not kept) | 6 | 98.5% |
| + global/C offsets split into voffset + soffset (experiment, not kept) | 21 | 96.4% |
| real `"=v,0"` accumulator pins + `"=s,0"` role pins + asm LDS ops via IR shims, C++-identical offsets | 0 | 100.4% |
| 1-tile kernel: pin `first_stage` so LDS addresses stay runtime values | 0 (1-tile: 8 -> 0) | 100.4% (1-tile, 4096x4096x8192) |

The whole gap was register pressure: the C++ kernel sits at exactly 256 VGPRs and relies on its `asm volatile("" : "+v"(acc))` pins to keep each 16-float accumulator group in one fixed register tuple. Without them the Rust build spilled; every spill reload is a `scratch_load` whose `s_waitcnt vmcnt(0)` also drains the in-flight global->LDS prefetches, which is what cost 28%. Once the same pins were expressible (IR shims), the Rust kernel compiled to the same schedule with zero spills.

## Feasibility summary

- **Expressiveness:** everything opus provides here (layouts, buffer/LDS/TR8 ops, scaled MFMA, scheduling barriers, waitcnts) maps onto Rust plus LLVM intrinsics. The opus layout algebra becomes explicit offset functions; they are more verbose but easy to verify (the oracle check).
- **Performance:** parity with C++ at the same LLVM version, but only after reproducing the C++ kernel's register-allocation hints. A Rust kernel at this level needs register-pinned inline asm, which today means a ~150-line IR post-processing step.
- **What would remove the relink step:** a launch-bounds attribute for `extern "gpu-kernel"`, address-space-qualified pointers (or LDS-typed intrinsics in `core::arch::amdgpu`), a bf16 type, and `vgpr`/`sgpr` register classes for amdgpu `asm!`.
