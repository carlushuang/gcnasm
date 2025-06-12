#!/bin/sh
BUILD=build
ARCH=gfx950
CXXCLAGS="-std=c++17 -O3 --offload-arch=$ARCH "

# build a test
rm -rf $BUILD ; mkdir $BUILD ; cd $BUILD
/opt/rocm/bin/hipcc -x hip $CXXCLAGS ../main.cpp -save-temps -o test.exe
