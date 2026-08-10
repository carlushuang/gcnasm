#!/usr/bin/env bash
# Build libfabric_symm.so.
#   ./build.sh                     # arch autodetected
#   GPU_ARCH=gfx950 ./build.sh     # explicit arch
#   HIPCC=/path/to/hipcc ./build.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCH="${GPU_ARCH:-native}"
OUT="$HERE/libfabric_symm.so"

# Build with the same ROCm that torch loads at runtime. A pip rocm-sdk install
# (_rocm_sdk_core) often shadows /opt/rocm -- sometimes it *is* /opt/rocm -- and the
# fabric export path needs ROCm >= 7.15. See the README for how to check.
if [ -z "${HIPCC:-}" ]; then
    ROCM="${ROCM_PATH:-/opt/rocm}"
    if [ -x "$ROCM/bin/hipcc" ]; then
        HIPCC="$ROCM/bin/hipcc"
    else
        HIPCC="$(command -v hipcc)"
    fi
fi

# Resolve the ROCm root hipcc actually links against.
HIPCONFIG="$(dirname "$HIPCC")/hipconfig"
if [ -x "$HIPCONFIG" ]; then
    ROCM_REAL="$("$HIPCONFIG" --rocmpath 2>/dev/null || echo "${ROCM_PATH:-/opt/rocm}")"
else
    ROCM_REAL="${ROCM_PATH:-/opt/rocm}"
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

"$HIPCC" -std=c++17 -O3 -fPIC -c --offload-arch="$ARCH" \
    "$HERE/fabric_symm.hip" -o "$TMP/fabric_symm.o"

# The rocm-sdk wheels ship libamdhip64.so.7 without the libamdhip64.so dev symlink.
# hipcc --hip-link puts that missing absolute path straight on the link line, so when
# it is absent we link the shared object ourselves and name the runtime by soname.
HIP_SO="$ROCM_REAL/lib/libamdhip64.so"
if [ -e "$HIP_SO" ]; then
    "$HIPCC" -shared "$TMP/fabric_symm.o" -o "$OUT"
else
    REAL_SO="$(ls "$ROCM_REAL"/lib/libamdhip64.so.* 2>/dev/null | head -1 || true)"
    if [ -z "$REAL_SO" ]; then
        echo "error: no libamdhip64 under $ROCM_REAL/lib" >&2
        exit 1
    fi
    echo "note: $HIP_SO missing; linking against $(basename "$REAL_SO") by soname"
    "${CXX:-g++}" -shared "$TMP/fabric_symm.o" -o "$OUT" \
        -L"$ROCM_REAL/lib" -l:"$(basename "$REAL_SO")" -Wl,-rpath,"$ROCM_REAL/lib"
fi

echo "built $OUT (arch=$ARCH, rocm=$ROCM_REAL)"

# The LD_PRELOAD shim (method "shim") is host-only C++ -- no device code, no hipcc.
SHIM="$HERE/libfabric_shim.so"
"${CXX:-g++}" -std=c++17 -O2 -fPIC -shared -D__HIP_PLATFORM_AMD__ \
    -I"$ROCM_REAL/include" "$HERE/fabric_shim.cpp" -o "$SHIM" -ldl

echo "built $SHIM"
