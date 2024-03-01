#!/bin/sh

/opt/rocm/bin/hipcc -x hip cross-wg-sync.hip.cpp  --offload-arch=gfx90a  -O3 -Wall -save-temps -o cross-wg-sync.exe
