#!/bin/sh
ARCH=gfx942
CK_DIR=/raid0/carhuang/repo/composable_kernel

rm -rf build && mkdir build && cd build 

/opt/rocm/bin/hipcc -x hip ../warp_sort.cc  -ICK_DIR/include --offload-arch=$ARCH  -O3 -Wall -save-temps -o warp_sort.exe
