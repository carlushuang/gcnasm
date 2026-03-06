#!/bin/bash
# Build script for vector_add assembly kernel (gfx942)
# Run this inside the ROCm docker container.

set -e

KSRC=vector_add_kernel.s
KOUT=vector_add_kernel.hsaco
SRC=main.cpp
TARGET=vector_add_asm.exe

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "=== Building vector_add_asm (gfx942) ==="

# Step 1: Assemble the GCN assembly kernel into an HSACO code object
echo "[1/2] Assembling kernel: $KSRC -> $KOUT"
rm -f $KOUT
/opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn--amdhsa -mcpu=gfx942 $KSRC -o $KOUT
echo "      OK"

# Step 2: Build host code with hipcc
echo "[2/2] Building host code: $SRC -> $TARGET"
rm -f $TARGET
/opt/rocm/bin/hipcc $SRC -o $TARGET
echo "      OK"

echo ""
echo "Build complete!"
echo "  Kernel HSACO: $KOUT"
echo "  Executable:   $TARGET"
echo ""
echo "Run:  ./$TARGET"
