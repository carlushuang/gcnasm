#!/bin/sh
HSACO=kernel_asm.co

rm -rf $HSACO sgemm.exe
/opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn--amdhsa -mcpu=gfx906 sgemm128x128_cov3_v3.s -o $HSACO
/opt/rocm/hip/bin/hipcc sgemm.cc -mcpu=gfx906 -o sgemm.exe
