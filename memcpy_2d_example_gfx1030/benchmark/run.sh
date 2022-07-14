#!/bin/sh

# g++ -std=c++11 benchmark.cpp parser.cpp -o out

KSRC=memcpy_2d_example_gfx1030.s
KOUT=memcpy_2d_example_gfx1030.hsaco
SRC1=benchmark.cpp 
SRC2=parser.cpp

TARGET=out.exe

# pre-delete the previous KOUT
rm -rf $KOUT
/opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn--amdhsa -mcpu=gfx1030 $KSRC -o $KOUT

rm -rf $TARGET
/opt/rocm/hip/bin/hipcc $SRC1 $SRC2 -mcpu=gfx1030 -o $TARGET