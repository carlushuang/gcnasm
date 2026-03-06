#!/bin/sh
ARCH=native
OPUS_DIR=/raid0/carhuang/repo/aiter/csrc/include

rm -rf build && mkdir build && cd build 

/opt/rocm/bin/hipcc -x hip ../warp_sort_bitonic.cc  -I$OPUS_DIR --offload-arch=$ARCH  -O3 -Wall -save-temps -o warp_sort_bitonic.exe
