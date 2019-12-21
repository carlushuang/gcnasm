#!/bin/sh
ARCH=gfx906
KSRC=sgemm128x128.hip.cc
KOUT=kernel.co
SRC=sgemm.cc
USE_MKL=1
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
/opt/rocm/hip/bin/hipcc --genco  --targets gfx900,gfx906 $KSRC -o $KOUT

rm -rf $TARGET
g++ $CXXFLAGS $SRC $LDFLAGS -o $TARGET || exit 1
