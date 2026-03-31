#!/bin/sh

SRC=gqa_gfx950.cc
OUT=gqa_attn.exe
TOP=`pwd`
BUILD="$TOP/build/"
OPUS_INCLUDE_DIR=/path/to/aiter/csrc/include

rm -rf $BUILD ; mkdir $BUILD ; cd $BUILD

/opt/rocm/bin/hipcc $TOP/$SRC -I$OPUS_INCLUDE_DIR -fPIC -std=c++17 -fopenmp -O3 -Wall --offload-arch=gfx950 -save-temps -Rpass-analysis=kernel-resource-usage -o $BUILD/$OUT
