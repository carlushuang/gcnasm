#!/bin/sh

KSRC=memcpy_x4_kernel_gfx1030.s
KOUT=memcpy_x4_kernel_gfx1030.hsaco
SRC=main.cpp
TARGET=out.exe

rm -rf $KOUT
/opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn--amdhsa -mcpu=gfx1030 $KSRC -o $KOUT

rm -rf $TARGET
/opt/rocm/hip/bin/hipcc $SRC -mcpu=gfx1030 -o $TARGET
