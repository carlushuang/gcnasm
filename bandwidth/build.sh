#!/bin/sh

DWORD_PER_UNIT=2
BLOCK_DIM_X=512
GRID_DIM_X=$((64*1))
GRID_DIM_Y=128
UNIT_PER_THRD=16
UNIT_STRIDE=64      # every XX unit
UNIT_STRIDE_SHIFT=$(printf "%.0f" $(echo "l(${UNIT_STRIDE})/l(2)" | bc -l))
P_LOOP=1
KSRC=kernel.s
KOBJ=kernel.o
KOUT=kernel.co
SRC=main.cpp
TARGET=out.exe
CXXFLAGS=`/opt/rocm/bin/hipconfig --cpp_config`" -Wall -O2 -Wno-deprecated-declarations  "
CXXFLAGS="${CXXFLAGS} -DDWORD_PER_UNIT=$DWORD_PER_UNIT -DBLOCK_DIM_X=$BLOCK_DIM_X -DGRID_DIM_X=$GRID_DIM_X"\
"  -DGRID_DIM_Y=$GRID_DIM_Y -DUNIT_PER_THRD=$UNIT_PER_THRD -DUNIT_STRIDE=$UNIT_STRIDE -DP_LOOP=$P_LOOP"
LDFLAGS=" -L/opt/rocm/hcc/lib -L/opt/rocm/lib -L/opt/rocm/lib64"\
" -Wl,--rpath=/opt/rocm/hcc/lib -ldl -lm -lpthread -lhc_am "\
" -Wl,--whole-archive -lmcwamp -lhip_hcc -lhsa-runtime64 -lhsakmt -Wl,--no-whole-archive"

AS_CLAGS="-defsym=DWORD_PER_UNIT=$DWORD_PER_UNIT -defsym=BLOCK_DIM_X=$BLOCK_DIM_X -defsym=GRID_DIM_X=$GRID_DIM_X"\
"  -defsym=GRID_DIM_Y=$GRID_DIM_Y -defsym=UNIT_PER_THRD=$UNIT_PER_THRD "\
"  -defsym=UNIT_STRIDE=$UNIT_STRIDE -defsym=UNIT_STRIDE_SHIFT=$UNIT_STRIDE_SHIFT -defsym=P_LOOP=$P_LOOP"
rm -rf $KOBJ $KOUT
llvm-mc -arch=amdgcn -mcpu=gfx900 $AS_CLAGS $KSRC -filetype=obj -o $KOBJ || exit 1
ld.lld -shared $KOBJ -o $KOUT || exit 1

rm -rf $TARGET
g++ $CXXFLAGS $SRC $LDFLAGS -o $TARGET
