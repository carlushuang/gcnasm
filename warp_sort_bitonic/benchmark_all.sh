#!/bin/bash
# Compile-time benchmark: all 5 Python-HIP integration methods
# Measures hipcc compile time for each approach with full clean builds.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HIPCC=/opt/rocm/bin/hipcc
ARCH=native
ROCM_INC=/opt/rocm/include
OPUS_INC=/raid0/carhuang/repo/aiter/csrc/include
TORCH_LIB=/opt/venv/lib/python3.12/site-packages/torch/lib

NUM_RUNS=3

echo "========================================================================"
echo "  COMPILE-TIME BENCHMARK: 5 Python-HIP Integration Methods"
echo "  $NUM_RUNS runs each, clean builds"
echo "========================================================================"
echo ""

# ── Helper: measure time in ms (all command output goes to /dev/null) ──
measure() {
    local t0 t1
    t0=$(date +%s%N)
    eval "$@" > /dev/null 2>&1
    t1=$(date +%s%N)
    echo $(( (t1 - t0) / 1000000 ))
}

TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

# ── Warm-up hipcc ──
echo "Warming up hipcc..."
$HIPCC --offload-arch=$ARCH -O3 -fPIC -shared \
    -I$ROCM_INC -I$OPUS_INC \
    -I$SCRIPT_DIR/pyhip/csrc \
    -D__HIPCC_RTC__ \
    $SCRIPT_DIR/pyhip/csrc/warp_bitonic_sort.hip \
    -o $TMPDIR/warmup.so 2>/dev/null
echo ""

# ── 1) pyhip: hipcc with <<<>>> and C++ dispatch ──
echo "--- 1) pyhip (<<<>>> in C++, ctypes call to C++ dispatch) ---"
PYHIP_TIMES=()
for i in $(seq 1 $NUM_RUNS); do
    rm -f $TMPDIR/pyhip.so
    ms=$(measure "$HIPCC --offload-arch=$ARCH -O3 -fPIC -shared \
        -I$ROCM_INC -I$OPUS_INC \
        -I$SCRIPT_DIR/pyhip/csrc \
        -D__HIPCC_RTC__ \
        $SCRIPT_DIR/pyhip/csrc/warp_bitonic_sort.hip \
        -o $TMPDIR/pyhip.so 2>&1")
    PYHIP_TIMES+=($ms)
    printf "  run %d: %dms\n" $i $ms
done
echo ""

# ── 2) pyhip_v2: hipcc with NO <<<>>>, Python hipLaunchKernel ──
echo "--- 2) pyhip_v2 (no <<<>>> in C++, Python hipLaunchKernel) ---"
PYHIP2_TIMES=()
for i in $(seq 1 $NUM_RUNS); do
    rm -f $TMPDIR/pyhip_v2.so
    ms=$(measure "$HIPCC --offload-arch=$ARCH -O3 -fPIC -shared \
        -I$ROCM_INC -I$OPUS_INC \
        -I$SCRIPT_DIR/pyhip_v2/csrc \
        -D__HIPCC_RTC__ \
        $SCRIPT_DIR/pyhip_v2/csrc/warp_bitonic_sort.hip \
        -o $TMPDIR/pyhip_v2.so 2>&1")
    PYHIP2_TIMES+=($ms)
    printf "  run %d: %dms\n" $i $ms
done
echo ""

# ── 2b) pyhip_v3: hipcc --genco (device-only), Python hipModuleLaunchKernel ──
echo "--- 2b) pyhip_v3 (--genco device-only, Python hipModuleLaunchKernel) ---"
PYHIP3_TIMES=()
for i in $(seq 1 $NUM_RUNS); do
    rm -f $TMPDIR/pyhip_v3.hsaco
    ms=$(measure "$HIPCC --genco --offload-arch=$ARCH -O3 \
        -I$ROCM_INC -I$OPUS_INC \
        -I$SCRIPT_DIR/pyhip_v3/csrc \
        -D__HIPCC_RTC__ \
        $SCRIPT_DIR/pyhip_v3/csrc/warp_bitonic_sort.hip \
        -o $TMPDIR/pyhip_v3.hsaco")
    PYHIP3_TIMES+=($ms)
    printf "  run %d: %dms\n" $i $ms
done
echo ""

# ── 3) pybind: hipcc kernel + g++ pybind binding + ninja ──
echo "--- 3) pybind (hipcc kernel + g++ pybind binding, ninja JIT) ---"
PYBIND_TIMES=()
for i in $(seq 1 $NUM_RUNS); do
    rm -rf ~/.cache/warp_bitonic_sort_pybind
    ms=$(measure "cd $SCRIPT_DIR/pybind && python3 -c '
import sys, os
sys.path.insert(0, \".\")
os.environ[\"OPUS_DIR\"] = \"$OPUS_INC\"
from warp_bitonic_sort.jit import gen_sort_module
spec = gen_sort_module()
spec.build(verbose=False)
' 2>&1")
    PYBIND_TIMES+=($ms)
    printf "  run %d: %dms\n" $i $ms
done
echo ""

# ── 4) tvmffi: hipcc kernel + g++ tvm_ffi binding + ninja ──
echo "--- 4) tvmffi (hipcc kernel + g++ tvm_ffi binding, ninja JIT) ---"
TVMFFI_TIMES=()
for i in $(seq 1 $NUM_RUNS); do
    rm -rf ~/.cache/warp_bitonic_sort
    ms=$(measure "cd $SCRIPT_DIR/tvmffi && python3 -c '
import sys, os
sys.path.insert(0, \".\")
os.environ[\"OPUS_DIR\"] = \"$OPUS_INC\"
from warp_bitonic_sort.jit import gen_sort_module
spec = gen_sort_module()
spec.build(verbose=False)
' 2>&1")
    TVMFFI_TIMES+=($ms)
    printf "  run %d: %dms\n" $i $ms
done
echo ""

# ── 5) torch: setup.py build_ext ──
echo "--- 5) torch (setup.py build_ext, hipcc for kernel+binding, ninja) ---"
TORCH_TIMES=()
for i in $(seq 1 $NUM_RUNS); do
    cd $SCRIPT_DIR/torch
    rm -rf build dist *.egg-info *.so
    rm -f csrc/*_hip.*
    ms=$(measure "cd $SCRIPT_DIR/torch && \
        LD_LIBRARY_PATH=$TORCH_LIB:\$LD_LIBRARY_PATH \
        python3 setup.py build_ext --inplace")
    cd $SCRIPT_DIR/torch
    rm -rf build dist *.egg-info *.so
    rm -f csrc/*_hip.*
    TORCH_TIMES+=($ms)
    printf "  run %d: %dms\n" $i $ms
done
echo ""

# ── Breakdown: hipcc kernel-only compile times ──
echo "========================================================================"
echo "  BREAKDOWN: hipcc kernel .hip compile only (no binding, no ninja)"
echo "========================================================================"
echo ""

echo "--- pyhip kernel (with <<<>>> in C++) ---"
PYHIP_K_TIMES=()
for i in $(seq 1 $NUM_RUNS); do
    rm -f $TMPDIR/k.o
    ms=$(measure "$HIPCC --offload-arch=$ARCH -O3 -fPIC \
        -I$ROCM_INC -I$OPUS_INC \
        -I$SCRIPT_DIR/pyhip/csrc \
        -D__HIPCC_RTC__ \
        -c $SCRIPT_DIR/pyhip/csrc/warp_bitonic_sort.hip \
        -o $TMPDIR/k.o 2>&1")
    PYHIP_K_TIMES+=($ms)
    printf "  run %d: %dms\n" $i $ms
done
echo ""

echo "--- pyhip_v2 kernel (no <<<>>> in C++, -shared) ---"
PYHIP2_K_TIMES=()
for i in $(seq 1 $NUM_RUNS); do
    rm -f $TMPDIR/k.o
    ms=$(measure "$HIPCC --offload-arch=$ARCH -O3 -fPIC \
        -I$ROCM_INC -I$OPUS_INC \
        -I$SCRIPT_DIR/pyhip_v2/csrc \
        -D__HIPCC_RTC__ \
        -c $SCRIPT_DIR/pyhip_v2/csrc/warp_bitonic_sort.hip \
        -o $TMPDIR/k.o")
    PYHIP2_K_TIMES+=($ms)
    printf "  run %d: %dms\n" $i $ms
done
echo ""

echo "--- pyhip_v3 kernel (--genco, device-only) ---"
PYHIP3_K_TIMES=()
for i in $(seq 1 $NUM_RUNS); do
    rm -f $TMPDIR/k.hsaco
    ms=$(measure "$HIPCC --genco --offload-arch=$ARCH -O3 \
        -I$ROCM_INC -I$OPUS_INC \
        -I$SCRIPT_DIR/pyhip_v3/csrc \
        -D__HIPCC_RTC__ \
        $SCRIPT_DIR/pyhip_v3/csrc/warp_bitonic_sort.hip \
        -o $TMPDIR/k.hsaco")
    PYHIP3_K_TIMES+=($ms)
    printf "  run %d: %dms\n" $i $ms
done
echo ""

echo "--- torch/pybind/tvmffi kernel (shared .hip, with hip_runtime.h) ---"
SHARED_K_TIMES=()
for i in $(seq 1 $NUM_RUNS); do
    rm -f $TMPDIR/k.o
    ms=$(measure "$HIPCC --offload-arch=$ARCH -O3 -fPIC \
        -I$ROCM_INC -I$OPUS_INC \
        -I$SCRIPT_DIR/torch/csrc \
        -c $SCRIPT_DIR/torch/csrc/warp_bitonic_sort.hip \
        -o $TMPDIR/k.o 2>&1")
    SHARED_K_TIMES+=($ms)
    printf "  run %d: %dms\n" $i $ms
done
echo ""

# ── Preprocessed line counts ──
echo "--- Preprocessed line counts ---"
PP_PYHIP=$($HIPCC -E --offload-arch=$ARCH -D__HIPCC_RTC__ \
    -I$ROCM_INC -I$OPUS_INC -I$SCRIPT_DIR/pyhip/csrc \
    $SCRIPT_DIR/pyhip/csrc/warp_bitonic_sort.hip 2>/dev/null | wc -l)
PP_PYHIP2=$($HIPCC -E --offload-arch=$ARCH -D__HIPCC_RTC__ \
    -I$ROCM_INC -I$OPUS_INC -I$SCRIPT_DIR/pyhip_v2/csrc \
    $SCRIPT_DIR/pyhip_v2/csrc/warp_bitonic_sort.hip 2>/dev/null | wc -l)
PP_PYHIP3=$($HIPCC -E --offload-arch=$ARCH -D__HIPCC_RTC__ \
    -I$ROCM_INC -I$OPUS_INC -I$SCRIPT_DIR/pyhip_v3/csrc \
    $SCRIPT_DIR/pyhip_v3/csrc/warp_bitonic_sort.hip 2>/dev/null | wc -l)
PP_TORCH=$($HIPCC -E --offload-arch=$ARCH \
    -I$ROCM_INC -I$OPUS_INC -I$SCRIPT_DIR/torch/csrc \
    $SCRIPT_DIR/torch/csrc/warp_bitonic_sort.hip 2>/dev/null | wc -l)

printf "  torch/pybind/tvmffi kernel:  %6d lines\n" $PP_TORCH
printf "  pyhip kernel:                %6d lines\n" $PP_PYHIP
printf "  pyhip_v2 kernel:             %6d lines\n" $PP_PYHIP2
printf "  pyhip_v3 kernel:             %6d lines\n" $PP_PYHIP3
echo ""

# ── Summary ──
avg() {
    local sum=0
    for v in "$@"; do sum=$((sum + v)); done
    echo $((sum / $#))
}

pyhip_avg=$(avg "${PYHIP_TIMES[@]}")
pyhip2_avg=$(avg "${PYHIP2_TIMES[@]}")
pyhip3_avg=$(avg "${PYHIP3_TIMES[@]}")
pybind_avg=$(avg "${PYBIND_TIMES[@]}")
tvmffi_avg=$(avg "${TVMFFI_TIMES[@]}")
torch_avg=$(avg "${TORCH_TIMES[@]}")

pyhip_k_avg=$(avg "${PYHIP_K_TIMES[@]}")
pyhip2_k_avg=$(avg "${PYHIP2_K_TIMES[@]}")
pyhip3_k_avg=$(avg "${PYHIP3_K_TIMES[@]}")
shared_k_avg=$(avg "${SHARED_K_TIMES[@]}")

echo "========================================================================"
echo "  SUMMARY (averaged over $NUM_RUNS runs)"
echo "========================================================================"
echo ""
printf "  %-45s %8s  %s\n" "Method" "Time" "Speedup vs torch"
echo "  -------------------------------------------------------------------------"
ratio() { python3 -c "print(f'{$1/$2:.1f}')"; }
printf "  %-50s %6dms  (1.0x baseline)\n" "torch  (setup.py + hipcc + ninja)" $torch_avg
printf "  %-50s %6dms  (%sx)\n" "pybind (hipcc + g++ + ninja)" $pybind_avg $(ratio $torch_avg $pybind_avg)
printf "  %-50s %6dms  (%sx)\n" "tvmffi (hipcc + g++ + ninja)" $tvmffi_avg $(ratio $torch_avg $tvmffi_avg)
printf "  %-50s %6dms  (%sx)\n" "pyhip  (hipcc -shared, <<<>>> in C++)" $pyhip_avg $(ratio $torch_avg $pyhip_avg)
printf "  %-50s %6dms  (%sx)\n" "pyhip_v2 (hipcc -shared, Python hipLaunchKernel)" $pyhip2_avg $(ratio $torch_avg $pyhip2_avg)
printf "  %-50s %6dms  (%sx)\n" "pyhip_v3 (hipcc --genco, hipModuleLaunchKernel)" $pyhip3_avg $(ratio $torch_avg $pyhip3_avg)
echo ""
echo "  Kernel-only compile:"
printf "    %-48s %6dms\n" "torch/pybind/tvmffi (hipcc -c, hip_runtime.h)" $shared_k_avg
printf "    %-48s %6dms\n" "pyhip  (hipcc -c, hip_minimal.h + builtins)" $pyhip_k_avg
printf "    %-48s %6dms\n" "pyhip_v2 (hipcc -c, hip_minimal.h, no <<<>>>)" $pyhip2_k_avg
printf "    %-48s %6dms\n" "pyhip_v3 (hipcc --genco, device-only)" $pyhip3_k_avg
echo ""
