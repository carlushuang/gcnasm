#!/bin/sh

KSRC=kernel.s
KOBJ=kernel.o
KOUT=kernel.co
SRC=main.cpp
TARGET=out.exe
CXXFLAGS=`/opt/rocm/bin/hipconfig --cpp_config`" -Wall -O2   "
LDFLAGS=" -L/opt/rocm/hcc/lib -L/opt/rocm/lib -L/opt/rocm/lib64"\
" -Wl,--rpath=/opt/rocm/hcc/lib -ldl -lm -lpthread -lhc_am "\
" -Wl,--whole-archive -lmcwamp -lhip_hcc -lhsa-runtime64 -lhsakmt -Wl,--no-whole-archive"

rm -rf $KOBJ $KOUT
#/opt/rocm/hcc/bin/clang -x assembler -target amdgcn--amdhsa -mcpu=gfx906 -mno-code-object-v3 $KSRC -o $KOUT
/opt/rocm/hcc/bin/clang -x assembler -target amdgcn--amdhsa -mcpu=gfx906 kernel_cov3.s -o $KOUT

rm -rf $TARGET
g++ $CXXFLAGS $SRC $LDFLAGS -o $TARGET
