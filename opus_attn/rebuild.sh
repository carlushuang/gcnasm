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
echo "Output: $TOP/build/gqa_attn.exe"
echo ""

# Run benchmark
echo "=== Causal ==="
./build/gqa_attn.exe --causal
echo ""
echo "=== Causal N=16384 ==="
./build/gqa_attn.exe --causal -n 16384
echo ""
echo "=== Non-causal ==="
./build/gqa_attn.exe --no-causal
echo ""
echo "=== Non-causal N=16384 ==="
./build/gqa_attn.exe --no-causal -n 16384
