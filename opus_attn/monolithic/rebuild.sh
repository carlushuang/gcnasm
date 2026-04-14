#!/bin/sh
TOP=$(cd "$(dirname "$0")" && pwd)
OPUS_INCLUDE_DIR=${OPUS_INCLUDE_DIR:-/home/carhuang/repo/aiter/csrc/include}
HIPCC=${HIPCXX:-/opt/rocm/bin/hipcc}
FLAGS="-I$TOP/.. -I$OPUS_INCLUDE_DIR -std=c++20 -fopenmp -O3 -Wall --offload-arch=gfx950 -ffast-math"

mkdir -p "$TOP/build"
echo "=== Monolithic build (host+device in one file) ==="
START=$(date +%s%N)
$HIPCC "$TOP/gqa_gfx950.cc" $FLAGS -Rpass-analysis=kernel-resource-usage -o "$TOP/build/gqa_attn.exe" 2>&1
END=$(date +%s%N)
echo "Build: $(( (END - START) / 1000000 )) ms"
echo "Output: $TOP/build/gqa_attn.exe"
