#!/bin/bash
# Build all memory latency micro-benchmarks (gfx942 / MI308)
# Run this inside the ROCm docker container.

set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

ASM="/opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn--amdhsa -mcpu=gfx942"
CXX="/opt/rocm/bin/hipcc"

echo "=== Building Memory Latency Micro-Benchmarks (gfx942) ==="

# --- Assemble shared kernels ---
echo ""
echo "[common]"
$ASM common/nop_loop.s -o common/nop_loop.hsaco
echo "  nop_loop.hsaco"
$ASM common/global_load_latency.s -o common/global_load_latency.hsaco
echo "  global_load_latency.hsaco"

# --- mem_latency ---
echo ""
echo "[mem_latency]"
$ASM mem_latency/lds_latency.s -o mem_latency/lds_latency.hsaco
echo "  lds_latency.hsaco"
cp common/global_load_latency.hsaco mem_latency/
cp common/nop_loop.hsaco mem_latency/
$CXX mem_latency/mem_latency.cpp -o mem_latency/mem_latency.exe
echo "  mem_latency.exe"

# --- lds_detailed ---
echo ""
echo "[lds_detailed]"
$ASM lds_detailed/lds_detailed.s -o lds_detailed/lds_detailed.hsaco
echo "  lds_detailed.hsaco"
cp common/nop_loop.hsaco lds_detailed/
cp common/global_load_latency.hsaco lds_detailed/
$CXX lds_detailed/lds_detailed.cpp -o lds_detailed/lds_detailed.exe
echo "  lds_detailed.exe"

# --- cacheline_stride ---
echo ""
echo "[cacheline_stride]"
cp common/global_load_latency.hsaco cacheline_stride/
cp common/nop_loop.hsaco cacheline_stride/
$CXX cacheline_stride/cacheline_stride.cpp -o cacheline_stride/cacheline_stride.exe
echo "  cacheline_stride.exe"

# --- lds_throughput ---
echo ""
echo "[lds_throughput]"
$ASM lds_throughput/lds_throughput.s -o lds_throughput/lds_throughput.hsaco
echo "  lds_throughput.hsaco"
$CXX lds_throughput/lds_throughput.cpp -o lds_throughput/lds_throughput.exe
echo "  lds_throughput.exe"

echo ""
echo "Build complete!"
