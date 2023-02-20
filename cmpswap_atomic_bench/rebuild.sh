#!/bin/sh
SRC=main.hip.cc
OUT=test.exe

rm -rf $OUT
/opt/rocm/bin/hipcc $SRC -std=c++17 --amdgpu-target=gfx90a -save-temps -o $OUT