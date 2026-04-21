#!/bin/sh
TOP=$(cd "$(dirname "$0")" && pwd)
cd "$TOP"

make clean 2>/dev/null

echo "=== Parallel build (make -j) ==="
START=$(date +%s%N)
make -j 2>&1
RC=$?
END=$(date +%s%N)
TOTAL_MS=$(( (END - START) / 1000000 ))

if [ $RC -ne 0 ]; then
    echo "Build FAILED (rc=$RC)"
    exit 1
fi

echo ""
echo "=== Build time: ${TOTAL_MS} ms ==="
echo "Output: $TOP/build/gemm_a8w8_blockscale.exe"
echo ""

./build/gemm_a8w8_blockscale.exe
