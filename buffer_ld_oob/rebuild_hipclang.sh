#!/bin/sh

KSRC=buffer_ld_oob.s
KOUT=buffer_ld_oob.hsaco
SRC=main.cpp
TARGET=out.exe
CXXFLAGS="-D__HIP_PLATFORM_HCC__= -I/opt/rocm/hip/include -I/opt/rocm/hcc/include -I/opt/rocm/hsa/include -Wall -O2  -std=c++11  "
LDFLAGS=" -L/opt/rocm/lib -L/opt/rocm/lib64"\
" -Wl,-rpath=/opt/rocm/lib -ldl -lm -lpthread "\
" -Wl,--whole-archive -lhip_hcc -lhsa-runtime64 -lhsakmt -Wl,--no-whole-archive"

rm -rf $KOUT $KOUT.diss.s
/opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn--amdhsa -mcpu=gfx908 $KSRC -o $KOUT
/opt/rocm/llvm/bin/llvm-objdump  --disassemble --mcpu=gfx908 $KOUT >  $KOUT.diss.s


rm -rf $TARGET
g++ $CXXFLAGS $SRC $LDFLAGS -o $TARGET
