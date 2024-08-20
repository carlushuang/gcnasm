#!/bin/sh

SRC=test.cpp
OUT=test.exe
TOP=`pwd`
BUILD="$TOP/build/"

rm -rf $BUILD ; mkdir $BUILD ; cd $BUILD

/opt/rocm/bin/hipcc $TOP/$SRC -fPIC -std=c++17 -O3 -Wall --offload-arch=gfx90a -save-temps -o $BUILD/$OUT

