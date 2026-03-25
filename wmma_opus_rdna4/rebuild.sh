#!/bin/sh
ARCH=gfx1250
#ARCH=gfx90a

rm -rf build && mkdir build && cd build 

/opt/rocm/bin/hipcc -x hip ../matrix_core_tcopy_1p1c.cc -std=c++17 -fopenmp -Iwmma_opus_rdna4/opus --offload-arch=$ARCH  -O3 -Wall -save-temps -o matrix_core.exe
