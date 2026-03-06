#!/bin/bash
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "Building..."
bash build.sh

echo ""
echo "=========================================="
echo " [1/4] Overall Memory Latency"
echo "=========================================="
(cd mem_latency && ./mem_latency.exe)

echo ""
echo "=========================================="
echo " [2/4] LDS Detailed (ds_read/write b32/b64/b128)"
echo "=========================================="
(cd lds_detailed && ./lds_detailed.exe)

echo ""
echo "=========================================="
echo " [3/4] Cache Line Stride Analysis"
echo "=========================================="
(cd cacheline_stride && ./cacheline_stride.exe)

echo ""
echo "=========================================="
echo " [4/4] LDS Throughput (dwords/cycle)"
echo "=========================================="
(cd lds_throughput && ./lds_throughput.exe)
