#!/bin/sh
ARCH=gfx950
#ARCH=gfx90a

rm -rf build && mkdir build && cd build 

/opt/rocm/bin/hipcc -x hip ../matrix_core.cc  --offload-arch=$ARCH  -O3 -Wall -save-temps -o matrix_core.exe
