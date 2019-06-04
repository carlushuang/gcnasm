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
llvm-mc -arch=amdgcn -mcpu=gfx900 $KSRC -filetype=obj -o $KOBJ || exit 1
ld.lld -shared $KOBJ -o $KOUT || exit 1

rm -rf $TARGET
g++ $CXXFLAGS $SRC $LDFLAGS -o $TARGET
