#!/bin/sh
ARCH=gfx906
KSRC_HIP=sgemm64x64_wr.hip.cc
#KSRC_ASM=sgemm128x128_v3.s
#KSRC_ASM=sgemm128x128.s
KOUT=kernel.co
SRC=sgemm.cc
USE_MKL=0
TARGET=sgemm.exe
CXXFLAGS=`/opt/rocm/bin/hipconfig --cpp_config`" -Wall -O2  -std=c++11 "
LDFLAGS=" -L/opt/rocm/hcc/lib -L/opt/rocm/lib -L/opt/rocm/lib64"\
" -Wl,-rpath=/opt/rocm/hcc/lib:/opt/rocm/lib -ldl -lm -lpthread -lhc_am "\
" -Wl,--whole-archive -lmcwamp -lhip_hcc -lhsa-runtime64 -lhsakmt -Wl,--no-whole-archive"
if [ "x$USE_MKL" = "x1" ]; then
    MKLROOT=/opt/intel/mkl
    CXXFLAGS="$CXXFLAGS -DUSE_MKL -DMKL_ILP64 -m64 -I${MKLROOT}/include"
    LDFLAGS="$LDFLAGS   -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl"
fi

rm -rf $KOUT dump*
export KMDUMPLLVM=1 && export KMDUMPISA=1
/opt/rocm/hip/bin/hipcc --genco  --targets $ARCH $KSRC_HIP -o $KOUT

#rm -rf kernel_asm.co
#/opt/rocm/hcc/bin/clang -x assembler -target amdgcn--amdhsa -mcpu=$ARCH -mno-code-object-v3 \
#  $KSRC_ASM -o kernel_asm.co
#/opt/rocm/hcc/bin/llvm-objdump -disassemble -mcpu=${ARCH} kernel_asm.co > dump.$KSRC_ASM


rm -rf $TARGET
g++ $CXXFLAGS $SRC $LDFLAGS -o $TARGET || exit 1
