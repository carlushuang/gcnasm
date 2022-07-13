#!/bin/sh

KSRC=memcpy_2d_x4_example_gfx1030.s
KOUT=memcpy_2d_x4_example_gfx1030.hsaco
SRC=main.cpp
TARGET=out.exe

# pre-delete the previous KOUT
rm -rf $KOUT
/opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn--amdhsa -mcpu=gfx1030 $KSRC -o $KOUT

rm -rf $TARGET
/opt/rocm/hip/bin/hipcc $SRC -mcpu=gfx1030 -o $TARGET
