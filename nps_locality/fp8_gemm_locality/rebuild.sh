#!/bin/sh
# Build the IOD-locality-aware FP8 GEMM and the B-tile stride microbenchmark.
# OPUS_INCLUDE_DIR must contain opus/opus.hpp (aiter: csrc/include).
set -e
ARCH=${ARCH:-gfx950}
HIPCC=${HIPCC:-hipcc}
: "${OPUS_INCLUDE_DIR:?set OPUS_INCLUDE_DIR to aiter/csrc/include}"
cd "$(dirname "$0")"
mkdir -p build
"$HIPCC" fp8_gemm_locality.cc -I"$OPUS_INCLUDE_DIR" -std=c++17 -fopenmp -O3 --offload-arch="$ARCH" -o build/fp8_gemm_locality
"$HIPCC" tile_stride_bw.cpp -O3 --offload-arch="$ARCH" -o build/tile_stride_bw
echo "built into build/"
