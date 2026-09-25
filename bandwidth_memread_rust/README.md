# bandwidth_memread_rust -- the bandwidth_memread kernels, written in Rust

Port of [`../bandwidth_memread`](../bandwidth_memread) where **both the GPU kernels and the host are Rust**. No HIP/C++ is involved; the host only calls the HIP runtime (`libamdhip64.so`) over FFI.

```
kernel/   #![no_std] cdylib for target amdgcn-amd-amdhsa  ->  bw_kernel.elf (HSA code object)
host/     std binary, include_bytes!(bw_kernel.elf) -> hipModuleLoadData -> hipModuleLaunchKernel
```

## Build / run

```bash
rustup toolchain install nightly --component rust-src   # target is tier 3: needs nightly + build-std
make ARCH=gfx950                                         # or gfx942, gfx1201, ...
./bandwidth_kernel.exe              # validate, then sweep [ro] / [ro-plain] / [rw]
./bandwidth_kernel.exe 268369920    # single size (dwords)
make asm                            # dump ISA to bw_kernel.<arch>.s
```

## How the Rust kernel maps to the HIP one

| HIP (`bandwidth_kernel.cu`) | Rust (`kernel/src/lib.rs`) |
|---|---|
| `__global__ void f(...)` | `#[unsafe(no_mangle)] pub unsafe extern "gpu-kernel" fn f(...)` (`abi_gpu_kernel`) |
| `threadIdx.x` / `blockIdx.x` | `core::arch::amdgpu::workitem_id_x()` / `workgroup_id_x()` (`stdarch_amdgpu`) |
| `float4` ext_vector_type | `core::simd::f32x4` (`portable_simd`) |
| `__builtin_nontemporal_load` | `llvm.amdgcn.raw.buffer.load.v4f32` with aux=NT via `extern "llvm-intrinsic"` |
| `__builtin_nontemporal_store` | `llvm.amdgcn.raw.buffer.store.v4f32` with aux=NT (see gotchas) |
| `hipcc --offload-arch=gfx950` | `cargo build --target amdgcn-amd-amdhsa -Ctarget-cpu=gfx950 -Zbuild-std=core` |

The output `.elf` is a normal HSA code object (v5 metadata, correct kernarg layout: 2x `global_buffer` + 2x `by_value`), so it loads via `hipModuleLoadData` like any `.co`.

## Results (MI355X gfx950, 256 CU, nightly 2026-09-23 / LLVM 23.1, vs hipcc ROCm 7.1.1)

All three kernels pass validation. Bandwidth, same session, back to back:

| size | HIP [ro] | Rust [ro] | | HIP [rw] | Rust [rw] | |
|---|---|---|---|---|---|---|
| 212 MB | 7189 | 7176 | -0.2% | 5786 | 5608 | -3.1% |
| 476 MB | 7168 | 7142 | -0.4% | 5765 | 5656 | -1.9% |
| 1.0 GB | 7002 | 6968 | -0.5% | 6002 | 5892 | -1.8% |
| 1.6 GB | 6450 | 6443 | -0.1% | 6228 | 6224 | -0.1% |
| 3.2 GB | 6653 | 6634 | -0.3% | 5782 | 5770 | -0.2% |
| 5.58 GB | 6934 | 6871 | -0.9% | 6441 | 6331 | -1.7% |

(GB/s.) At HBM-bound sizes Rust matches HIP to within ~1% read-only and ~0-3% read+write. `[ro-plain]` (ordinary temporal loads) is ~10% slower than both, so the NT hint really matters. Sizes <=64 MB are launch/cache dominated and differ by ±5-9% in both directions.

ISA: `memread_kernel` issues 4x `buffer_load_dwordx4 ... offen nt` per iteration + `v_pk_add_f32`, 24 VGPRs (HIP: 4x `global_load_dwordx4 ... nt`, 28 VGPRs).

## Gotchas found

1. **`core::intrinsics::nontemporal_store` is silently a plain store on amdgpu.** rustc only emits `!nontemporal` for archs in `WELL_BEHAVED_NONTEMPORAL_ARCHS` (aarch64/arm/riscv), so no `nt` bit appears in the ISA. Worked around with `raw.buffer.store`; the real fix is a one-line rustc change adding amdgpu to that list.
2. **No nontemporal *load* in Rust at all.** Only an LLVM intrinsic via `link_llvm_intrinsics` (+ `simd_ffi` to pass vector types) works. That forces the buffer path, which means a 4 GiB window per descriptor. Here each workgroup builds its own descriptor at its base, so 5.58 GB still works.
3. **Inline asm is operand-less only.** `asm!("s_nop 0")` compiles with `asm_experimental_arch`, but there's no `vgpr`/`sgpr` register class, so asm can't carry values. Anything like hand-written `global_load ... nt` or `v_mfma` inline asm is blocked. Intrinsics via `extern "llvm-intrinsic"` are the escape hatch.
4. **LLVM 23 runtime-unrolls more aggressively than hipcc.** By default the outer `iters` loop was unrolled 4x (16 loads in flight, 61 VGPRs) and was ~2% slower. There's no `#pragma nounroll` in Rust; `-Cllvm-args=-unroll-runtime=false` (crate-wide) restores hipcc's shape.
5. Use `u32` index math. The first `usize` version produced 64-bit `v_add_co/v_addc` pairs per address; HIP uses 32-bit `int` offsets.
6. Aggregate (`#[repr(C)]` struct) loads/stores get lowered to `i128` memcpys; use `core::simd` types for vector data.
7. Tier-3 target: nightly + `-Zbuild-std=core` only, no `std`/`alloc`, `panic=abort`, a `#[panic_handler]` is required (`core::arch::amdgpu::endpgm()`), and `-Ctarget-cpu` is mandatory (one arch per build, no fat binary).
