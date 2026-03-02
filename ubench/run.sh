#!/bin/bash
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "Building..."
bash build.sh

echo ""
echo "=========================================="
echo " [1/3] Overall Memory Latency"
echo "=========================================="
(cd mem_latency && ./mem_latency.exe)

echo ""
echo "=========================================="
echo " [2/3] LDS Detailed (ds_read/write b32/b64/b128)"
echo "=========================================="
(cd lds_detailed && ./lds_detailed.exe)

echo ""
echo "=========================================="
echo " [3/3] Cache Line Stride Analysis"
echo "=========================================="
(cd cacheline_stride && ./cacheline_stride.exe)
