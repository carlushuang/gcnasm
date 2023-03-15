#!/bin/sh
SRC=memcpy_driver.cpp
OUT=memcpy_driver.exe
PWD=`pwd`
TMP="$PWD/tmp/"

rm -rf $TMP ; mkdir $TMP
/opt/rocm/bin/hipcc $SRC -std=c++17 -DPWD=\"$PWD\" -DTMP=\"$TMP\" --amdgpu-target=gfx90a -ldl -o $TMP/$OUT

