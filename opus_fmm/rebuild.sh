#!/bin/sh

SRC=flatmm.cc
OUT=flatmm.exe
TOP=`pwd`
BUILD="$TOP/build/"
OPUS_INCLUDE_DIR=/path/to/aiter/csrc/include

rm -rf $BUILD ; mkdir $BUILD ; cd $BUILD

/opt/rocm/bin/hipcc $TOP/$SRC -I$OPUS_INCLUDE_DIR -fPIC -std=c++17 -O3 -Wall --offload-arch=gfx950 -save-temps -o $BUILD/$OUT

