#!/bin/sh
ARCH=gfx906
KSRC=kernel.s
KOBJ=kernel.o
KOUT=kernel.co
SRC=main.cpp
TARGET=out.exe
CXXFLAGS=`/opt/rocm/bin/hipconfig --cpp_config`" -Wall -O2  -std=c++11 "
LDFLAGS=" -L/opt/rocm/hcc/lib -L/opt/rocm/lib -L/opt/rocm/lib64"\
" -Wl,-rpath=/opt/rocm/hcc/lib:/opt/rocm/lib -ldl -lm -lpthread -lhc_am "\
" -Wl,--whole-archive -lmcwamp -lhip_hcc -lhsa-runtime64 -lhsakmt -Wl,--no-whole-archive"

rm -rf $KOBJ $KOUT $KOUT.dump.s
/opt/rocm/hcc/bin/clang -x assembler -target amdgcn--amdhsa -mcpu=$ARCH -mno-code-object-v3 $KSRC -o $KOUT || exit 1
/opt/rocm/hcc/bin/llvm-objdump -disassemble -mcpu=${ARCH} $KOUT > $KOUT.dump.s

rm -rf $TARGET
g++ $CXXFLAGS $SRC $LDFLAGS -o $TARGET || exit 1
