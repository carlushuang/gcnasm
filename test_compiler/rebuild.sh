#!/bin/sh
SRC=test.cu
OUT=test.exe
TOP=`pwd`
BUILD="$TOP/build/"


rm -rf $BUILD ; mkdir $BUILD ; cd $BUILD

# comment out unwanted platform
# ROCM
/opt/rocm/bin/hipify-perl $TOP/$SRC > $BUILD/$SRC.hip.cc
/opt/rocm/bin/hipcc $BUILD/$SRC.hip.cc -std=c++17 -O3 --offload-arch=gfx90a -save-temps -o $BUILD/$OUT

# CUDA
nvcc -std=c++17 -O3 --ptxas-options=-v -arch=sm_80 $TOP/$SRC -o $BUILD/$OUT
cuobjdump -sass $BUILD/$OUT > $BUILD/$OUT.sass.s
