#!/bin/sh

SRC=test.cu
OUT=test.exe
TOP=`pwd`
BUILD="$TOP/build/"


if [ $# -ge 1 ] ; then
    TARGET=$1
else
    TARGET="rocm"
fi

rm -rf $BUILD ; mkdir $BUILD ; cd $BUILD


if [ "x$TARGET" = "xrocm" ] ; then
echo "===== build rocm"
/opt/rocm/bin/hipify-perl $TOP/$SRC > $BUILD/$SRC.hip.cc
/opt/rocm/bin/hipcc $BUILD/$SRC.hip.cc -fPIC -std=c++17 -O3 -Wall --offload-arch=gfx90a -save-temps -o $BUILD/$OUT
fi

if [ "x$TARGET" = "xcuda" ] ; then
echo "===== build cuda"
fi
