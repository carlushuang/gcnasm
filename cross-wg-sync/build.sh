#!/bin/sh

/opt/rocm/hip/bin/hipcc -x hip cross-wg-sync.hip.cpp -mcpu=gfx90a  -save-temps -o cross-wg-sync.exe
