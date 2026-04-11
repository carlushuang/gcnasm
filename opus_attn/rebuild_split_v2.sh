#!/bin/sh
# Split build: compile device and host as a single TU using preprocessor selection
# The kernel file does NOT include hip/hip_runtime.h — only hip_minimal.h
# But we compile it as one TU so the optimizer sees everything

TOP=`pwd`
BUILD="$TOP/build/"
OPUS_INCLUDE_DIR=/home/carhuang/repo/aiter/csrc/include
HIPCC=${HIPCXX:-/opt/rocm/bin/hipcc}

rm -rf $BUILD ; mkdir $BUILD ; cd $BUILD

echo "=== Device-only compile time (kernel.cc with hip_minimal.h) ==="
for i in 1 2 3; do
  START=$(date +%s%N)
  $HIPCC "$TOP/gqa_gfx950_kernel.cc" \
    -I"$TOP" -I"$OPUS_INCLUDE_DIR" \
    -std=c++20 -fopenmp -O3 -Wall --offload-arch=gfx950 -ffast-math \
    --cuda-device-only -c \
    -o /dev/null 2>/dev/null
  END=$(date +%s%N)
  MS=$(( (END - START) / 1000000 ))
  echo "  Device-only run $i: ${MS} ms"
done

echo ""
echo "=== Full single-TU build (original monolithic with hip_runtime.h) ==="
START=$(date +%s%N)
$HIPCC "$TOP/gqa_gfx950.cc" \
  -I"$OPUS_INCLUDE_DIR" \
  -std=c++20 -fopenmp -O3 -Wall --offload-arch=gfx950 -ffast-math \
  -Rpass-analysis=kernel-resource-usage \
  -o "$BUILD/gqa_attn.exe" 2>&1 | grep -E "VGPRs|Spill|Occupancy|TotalSGPR"
END=$(date +%s%N)
MONO_MS=$(( (END - START) / 1000000 ))
echo "Monolithic build: ${MONO_MS} ms"

echo ""
echo "=== Run ==="
./gqa_attn.exe
