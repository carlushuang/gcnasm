#!/bin/sh
# switch to whatever c++ compiler you want
CC=/opt/rocm/llvm/bin/clang++
rm -rf build && mkdir build && cd build

$CC -std=c++11 -O3 ../main.cpp -o test.exe
