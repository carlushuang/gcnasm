#!/bin/sh
BUILD=build
ARCH=gfx942
CXXCLAGS="-std=c++17 -O0 --offload-arch=$ARCH "

# build a test
rm -rf $BUILD ; mkdir $BUILD ; cd $BUILD
/opt/rocm/bin/hipcc -x hip $CXXCLAGS ../main.hip.cc -save-temps -o test.exe
