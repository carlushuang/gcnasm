#!/bin/sh

/opt/rocm/hip/bin/hipcc -x hip cross-wg-sync.hip.cpp -mcpu=gfx90a  -O3 -Wall -save-temps -o cross-wg-sync.exe
