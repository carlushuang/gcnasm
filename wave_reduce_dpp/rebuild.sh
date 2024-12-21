#!/bin/sh
ARCH=gfx942
#ARCH=gfx90a

rm -rf build && mkdir build && cd build 

/opt/rocm/bin/hipcc -x hip ../wave_reduce_dpp.cc --offload-arch=$ARCH  -O3 -Wall -save-temps -o wave_reduce_dpp.exe
