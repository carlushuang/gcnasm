# opus_attn_gfx1201 v201 — hand-tuned gfx1201 assembly (beyond #24's v88)

A hand-written **gfx1201 (RDNA4, wave32, WMMA) assembly** Flash-Attention-2 forward kernel,
derived from #24's `v88_kernel.s` and further tuned to reduce VMEM load-wait (`s_wait_loadcnt`)
stalls found via rocprofv3 ATT instruction tracing.

## Build & run
```bash
# assemble the .s into a HSACO
/opt/rocm/lib/llvm/bin/clang++ -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 \
    attn_gfx1201_kernel_v201.s -o v201.hsaco
# build the host loader
/opt/rocm/bin/hipcc -std=c++17 -O3 --offload-arch=gfx1201 attn_v88_host.cc -o attn_host.exe
# run (BLOCK_M=384 -> N must be a multiple of 384)
./attn_host.exe --hsaco=v201.hsaco --func=v88_kernel -b=1 -h=32 -n=1536 --iters=50
```

## Measured on RX 9070 XT (ROCm 7.2.3, bf16, fp32 acc, D=128), best-of-5 / alternating
| shape (H=32) | #24 v88 (ASM) | **v201 (this PR)** |
|---|---:|---:|
| N=1536 | ~113.5 | **~114.2 (+0.6%)** |
| N=3072 | ~106.7 | ~106.4 (≈) |
| N=6144 | ~99.7 | ~99.7 (≈) |

Correctness: `PASSED` / `n_bad=0` (bit-exact in bf16, identical fp32 reduction order to v88).

> **Status: actively being optimized.** This PR is updated as the autonomous profile-guided
> ASM loop finds further gains. Current edge over v88 is small but consistent (~0.5–0.7% at
> N=1536); the larger structural lever (cutting VGPR 192→170 for 3 waves/SIMD) is still being
> pursued and should widen the margin, especially at N=3072/6144.
