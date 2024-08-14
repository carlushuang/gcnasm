#!/bin/sh
#ARCH=gfx942
ARCH=gfx90a

rm -rf build && mkdir build && cd build 

/opt/rocm/bin/hipcc -x hip ../memcpy_async.cc  --offload-arch=$ARCH  -O3 -Wall -save-temps -o memcpy_async.exe
