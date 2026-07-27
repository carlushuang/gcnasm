#!/bin/bash
# Build the asm_f4gemm host driver.
#
# Only host code is compiled here -- the gfx950 kernels come pre-assembled from
# the aiter repo (hsa/gfx950/f4gemm), so no --offload-arch is needed.

set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

HIPCC=${HIPCC:-/opt/rocm/bin/hipcc}
SRC=main.cpp
TARGET=asm_f4gemm.exe

echo "=== Building asm_f4gemm (host only) ==="
rm -f $TARGET
$HIPCC -O3 -std=c++17 $SRC -o $TARGET -lpthread
echo "      OK -> $TARGET"
echo ""
echo "Run:  ./$TARGET --co-dir /path/to/aiter/hsa/gfx950/f4gemm"
