#!/bin/sh

KSRC=memcpy_kernel.s
KOUT=memcpy_kernel.hsaco
SRC=main.cpp
TARGET=out.exe
CXXFLAGS=`/opt/rocm/bin/hipconfig --cpp_config`" -Wall -O2  -std=c++11  "
LDFLAGS=" -L/opt/rocm/hcc/lib -L/opt/rocm/lib -L/opt/rocm/lib64"\
" -Wl,--rpath=/opt/rocm/hcc/lib -ldl -lm -lpthread -lhc_am "\
" -Wl,--whole-archive -lmcwamp -lhip_hcc -lhsa-runtime64 -lhsakmt -Wl,--no-whole-archive"

rm -rf $KOUT
/opt/rocm/hcc/bin/clang -x assembler -target amdgcn--amdhsa -mcpu=gfx906 $KSRC -o $KOUT

rm -rf $TARGET
g++ $CXXFLAGS $SRC $LDFLAGS -o $TARGET
