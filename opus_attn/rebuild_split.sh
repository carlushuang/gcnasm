#!/bin/sh

TOP=`pwd`
BUILD="$TOP/build/"
OPUS_INCLUDE_DIR=/home/carhuang/repo/aiter/csrc/include
HIPCC=${HIPCXX:-/opt/rocm/bin/hipcc}
COMMON_FLAGS="-I$TOP -I$OPUS_INCLUDE_DIR -std=c++20 -fopenmp -O3 -Wall --offload-arch=gfx950 -ffast-math"

rm -rf $BUILD ; mkdir $BUILD ; cd $BUILD

echo "=== Compiling device kernel (no hip_runtime.h) ==="
START_DEV=$(date +%s%N)
$HIPCC "$TOP/gqa_gfx950_kernel.cc" \
  $COMMON_FLAGS \
  -save-temps -Rpass-analysis=kernel-resource-usage \
  -c -o "$BUILD/gqa_kernel.o" 2>&1
DEV_RC=$?
END_DEV=$(date +%s%N)
DEV_MS=$(( (END_DEV - START_DEV) / 1000000 ))
echo "Device compile: ${DEV_MS} ms (rc=$DEV_RC)"

if [ $DEV_RC -ne 0 ]; then
  echo "Device compilation failed!"
  exit 1
fi

echo ""
echo "=== Compiling host code (with hip_runtime.h) ==="
START_HOST=$(date +%s%N)
$HIPCC "$TOP/gqa_gfx950_host.cc" \
  $COMMON_FLAGS \
  -c -o "$BUILD/gqa_host.o" 2>&1
HOST_RC=$?
END_HOST=$(date +%s%N)
HOST_MS=$(( (END_HOST - START_HOST) / 1000000 ))
echo "Host compile: ${HOST_MS} ms (rc=$HOST_RC)"

if [ $HOST_RC -ne 0 ]; then
  echo "Host compilation failed!"
  exit 1
fi

echo ""
echo "=== Linking ==="
START_LINK=$(date +%s%N)
$HIPCC "$BUILD/gqa_kernel.o" "$BUILD/gqa_host.o" \
  --offload-arch=gfx950 -fopenmp \
  -o "$BUILD/gqa_attn.exe" 2>&1
LINK_RC=$?
END_LINK=$(date +%s%N)
LINK_MS=$(( (END_LINK - START_LINK) / 1000000 ))
echo "Link: ${LINK_MS} ms (rc=$LINK_RC)"

echo ""
echo "=== Summary ==="
echo "Device compile: ${DEV_MS} ms"
echo "Host compile:   ${HOST_MS} ms"
echo "Link:           ${LINK_MS} ms"
TOTAL_MS=$(( DEV_MS + HOST_MS + LINK_MS ))
echo "Total:          ${TOTAL_MS} ms"
