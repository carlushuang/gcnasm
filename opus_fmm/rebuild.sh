#!/bin/sh

ARCH=gfx950
SRC=flatmm_a8w8_${ARCH}.cc
OUT=flatmm.exe

TOP=`pwd`
BUILD="$TOP/build/"
OPUS_INCLUDE_DIR=/path/to/aiter/csrc/include

rm -rf $BUILD ; mkdir $BUILD ; cd $BUILD

/opt/rocm/bin/hipcc $TOP/$SRC -I$OPUS_INCLUDE_DIR -fPIC -std=c++17 -fopenmp -O3 -Wall --offload-arch=$ARCH -save-temps -Rpass-analysis=kernel-resource-usage -o $BUILD/$OUT