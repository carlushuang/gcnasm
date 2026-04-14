#!/bin/sh

SRC=gqa_gfx950.cc
OUT=gqa_attn.exe
TOP=`pwd`
BUILD="$TOP/build/"
OPUS_INCLUDE_DIR=${OPUS_INCLUDE_DIR:-/home/carhuang/repo/aiter/csrc/include}
HIPCC=${HIPCXX:-/opt/rocm/bin/hipcc}

rm -rf $BUILD ; mkdir $BUILD ; cd $BUILD

echo "=== Monolithic build ==="
START=$(date +%s%N)
$HIPCC "$TOP/$SRC" \
  -I"$OPUS_INCLUDE_DIR" \
  -std=c++20 -fopenmp -O3 -Wall \
  --offload-arch=gfx950 -ffast-math \
  -save-temps -Rpass-analysis=kernel-resource-usage \
  -o "$BUILD/$OUT"
RC=$?
END=$(date +%s%N)
BUILD_MS=$(( (END - START) / 1000000 ))

echo "Build: ${BUILD_MS} ms (rc=$RC)"

if [ $RC -ne 0 ]; then
  exit $RC
fi

echo "Output: $BUILD/$OUT"