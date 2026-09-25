#!/bin/sh
TOP=$(cd "$(dirname "$0")" && pwd)
cd "$TOP" || exit 1

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
echo "Output: $TOP/build/gemm_a8w8_mxfp8_scale.exe"
echo ""

# Both gates have regressed silently before, so never trust a build that has
# not passed them: <192 MFMA means the K loop did not unroll (~-13%), and a
# surviving s_setprio means the scheduler serialised the waves.
echo "=== Gate ==="
make check || exit 1
make regs

echo ""
./build/gemm_a8w8_mxfp8_scale.exe -m 8192 -n 8192 -k 8192 -b 1 -v 0 -w 200 -i 100
