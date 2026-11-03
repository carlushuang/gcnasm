#!/bin/sh
# Build all probes. Override ARCH / HIPCC / BUILD from the environment if needed.
set -e
ARCH=${ARCH:-gfx950}
HIPCC=${HIPCC:-hipcc}
BUILD=${BUILD:-build}
cd "$(dirname "$0")"
mkdir -p "$BUILD"
for src in dpx_partition_probe dpx_concurrent_bw spx_interleave_probe spx_page_phase_bw spx_granule_map; do
    "$HIPCC" -O3 --offload-arch="$ARCH" -o "$BUILD/$src" "$src.cpp"
done
echo "built into $BUILD/"
