#!/bin/sh

SRC=vector_add.cpp
OUT=vector_add.exe
TOP=`pwd`
BUILD="$TOP/build/"
OPUS_INCLUDE_DIR=/raid0/carhuang/repo/aiter/csrc/include

rm -rf $BUILD ; mkdir $BUILD ; cd $BUILD

/opt/rocm/bin/hipcc $TOP/$SRC -I$OPUS_INCLUDE_DIR -fPIC -std=c++17 -O3 -Wall --offload-arch=native -save-temps -o $BUILD/$OUT

