#!/bin/sh
#
# Compare two compile paths for the current GQA kernel:
#   1) device-only compile of gqa_gfx950_kernel.cc (`--cuda-device-only`)
#   2) full monolithic build of gqa_gfx950.cc
#
# This is a comparison/experiment script, not the regular split build script.

set -eu

TOP=`pwd`
BUILD="$TOP/build/"
OUT=gqa_attn.exe
DEVICE_SRC="$TOP/gqa_gfx950_kernel.cc"
MONO_SRC="$TOP/gqa_gfx950.cc"
OPUS_INCLUDE_DIR=${OPUS_INCLUDE_DIR:-/home/carhuang/repo/aiter/csrc/include}
HIPCC=${HIPCXX:-/opt/rocm/bin/hipcc}
DEVICE_ONLY_RUNS=${DEVICE_ONLY_RUNS:-3}
RUN_BINARY=${RUN_BINARY:-1}

COMMON_FLAGS="-std=c++20 -fopenmp -O3 -Wall --offload-arch=gfx950 -ffast-math"
DEVICE_FLAGS="-I$TOP -I$OPUS_INCLUDE_DIR $COMMON_FLAGS"
MONO_FLAGS="-I$OPUS_INCLUDE_DIR $COMMON_FLAGS -Rpass-analysis=kernel-resource-usage"

now_ms() {
  date +%s%N
}

print_section() {
  echo ""
  echo "=== $1 ==="
}

if [ ! -d "$OPUS_INCLUDE_DIR/opus" ]; then
  echo "OPUS_INCLUDE_DIR is not set correctly: $OPUS_INCLUDE_DIR"
  echo "Please export OPUS_INCLUDE_DIR=/path/to/aiter/csrc/include"
  exit 1
fi

rm -rf "$BUILD"
mkdir -p "$BUILD"
cd "$BUILD"

print_section "Device-only compile timing"
echo "Source: $DEVICE_SRC"
echo "Runs:   $DEVICE_ONLY_RUNS"

DEVICE_TOTAL_MS=0
i=1
while [ "$i" -le "$DEVICE_ONLY_RUNS" ]; do
  START=$(now_ms)
  "$HIPCC" "$DEVICE_SRC" \
    $DEVICE_FLAGS \
    --cuda-device-only -c \
    -o /dev/null >/dev/null 2>&1
  END=$(now_ms)
  RUN_MS=$(( (END - START) / 1000000 ))
  DEVICE_TOTAL_MS=$(( DEVICE_TOTAL_MS + RUN_MS ))
  echo "  Device-only run $i: ${RUN_MS} ms"
  i=$(( i + 1 ))
done

DEVICE_AVG_MS=$(( DEVICE_TOTAL_MS / DEVICE_ONLY_RUNS ))

print_section "Monolithic build"
echo "Source: $MONO_SRC"
START=$(now_ms)
"$HIPCC" "$MONO_SRC" \
  $MONO_FLAGS \
  -o "$BUILD/$OUT" >"$BUILD/monolithic_build.log" 2>&1
MONO_RC=$?
END=$(now_ms)
MONO_MS=$(( (END - START) / 1000000 ))

if [ "$MONO_RC" -ne 0 ]; then
  echo "Monolithic build failed. Full log:"
  cat "$BUILD/monolithic_build.log"
  exit 1
fi

echo "Monolithic build: ${MONO_MS} ms"
echo "Compiler resource summary:"
grep -E "VGPRs|Spill|Occupancy|TotalSGPR" "$BUILD/monolithic_build.log" || true

if [ "$RUN_BINARY" = "1" ]; then
  print_section "Run monolithic executable"
  "./$OUT"
fi

print_section "Summary"
echo "Device-only build: ${DEVICE_AVG_MS} ms"
echo "Monolithic build:  ${MONO_MS} ms"
