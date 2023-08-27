#!/bin/sh
SRC=test.cc
OUT=test.exe
TOP=`pwd`
BUILD="$TOP/build/"


rm -rf $BUILD ; mkdir $BUILD ; cd $BUILD

# ROCM
/opt/rocm/bin/hipify-perl $TOP/$SRC > $BUILD/$SRC.hip.cc
/opt/rocm/bin/hipcc $BUILD/$SRC.hip.cc -std=c++17 -O3 --offload-arch=gfx90a -save-temps -o $BUILD/$OUT

# CUDA
nvcc -std=c++17 -O3 memcpy_kernel.cu -o memcpy_kernel.exe