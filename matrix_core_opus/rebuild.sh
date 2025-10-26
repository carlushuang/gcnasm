#!/bin/sh
ARCH=gfx942
#ARCH=gfx90a

rm -rf build && mkdir build && cd build 

/opt/rocm/bin/hipcc -x hip ../matrix_core.cc -std=c++17 -I/raid0/carhuang/repo/rlogits/csrc/include --offload-arch=$ARCH  -O3 -Wall -save-temps -o matrix_core.exe
