#!/bin/sh
ARCH=gfx942
#ARCH=gfx90a
CK_TILE_PATH=/raid0/carhuang/repo/ck_layernorm_fusion

rm -rf build && mkdir build && cd build 

/opt/rocm/bin/hipcc -I$CK_TILE_PATH/include -x hip ../matrix_core.cc  --offload-arch=$ARCH  -O3 -Wall -save-temps -o matrix_core.exe
