#!/bin/sh
SRC=main.hip.cc
OUT=test.exe
BF16ASU16=0

rm -rf $OUT
/opt/rocm/bin/hipcc $SRC -std=c++17 -DBF16ASU16=$BF16ASU16 --amdgpu-target=gfx90a -save-temps -o $OUT