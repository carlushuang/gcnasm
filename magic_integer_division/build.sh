#!/bin/sh

KSRC=magic_div.s
KOUT=magic_div.hsaco
SRC=main.cc
TARGET=out.exe
CXXFLAGS="-D__HIP_PLATFORM_HCC__= -I/opt/rocm/hip/include -I/opt/rocm/hcc/include -I/opt/rocm/hsa/include -Wall -O2  -std=c++11  "
LDFLAGS=" -L/opt/rocm/lib -L/opt/rocm/lib64"\
" -Wl,-rpath=/opt/rocm/lib -ldl -lm -lpthread "\
" -Wl,--whole-archive -lamdhip64 -lhsa-runtime64 -lhsakmt -Wl,--no-whole-archive"

rm -rf $KOUT
/opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn--amdhsa -mcpu=gfx906 $KSRC -o $KOUT

rm -rf $TARGET
g++ $CXXFLAGS $SRC $LDFLAGS -o $TARGET
