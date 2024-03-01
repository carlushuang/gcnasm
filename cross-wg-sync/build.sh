#!/bin/sh
ARCH=gfx942
/opt/rocm/bin/hipcc -x hip cross-wg-sync.hip.cpp  --offload-arch=$ARCH  -O3 -Wall -save-temps -o cross-wg-sync.exe
