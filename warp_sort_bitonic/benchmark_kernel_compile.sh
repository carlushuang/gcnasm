#!/bin/bash
# A/B kernel compile benchmark: CK headers vs opus headers
# Measures: preprocessing, compilation, total, and file sizes

set -e

HIPCC=/opt/rocm/bin/hipcc
ARCH=native
CK_INC=/opt/rocm/include
OPUS_INC=/raid0/carhuang/repo/aiter/csrc/include

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CK_SRC=$SCRIPT_DIR/.bench_ck.cc
OPUS_SRC=$SCRIPT_DIR/.bench_opus.cc

NUM_RUNS=5
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

echo "========================================================================"
echo " KERNEL COMPILE BENCHMARK: CK vs OPUS"
echo " $NUM_RUNS runs each, architecture=$ARCH"
echo "========================================================================"
echo ""

# --- Step 1: header size comparison ---
echo "--- Header size comparison ---"
echo ""

# Preprocess to measure total included code
$HIPCC -x hip -E $CK_SRC -I$CK_INC --offload-arch=$ARCH -O3 2>/dev/null | wc -l > $TMPDIR/ck_pp_lines
$HIPCC -x hip -E $OPUS_SRC -I$OPUS_INC --offload-arch=$ARCH -O3 2>/dev/null | wc -l > $TMPDIR/opus_pp_lines

CK_PP_LINES=$(cat $TMPDIR/ck_pp_lines)
OPUS_PP_LINES=$(cat $TMPDIR/opus_pp_lines)

CK_SRC_LINES=$(wc -l < $CK_SRC)
OPUS_SRC_LINES=$(wc -l < $OPUS_SRC)

echo "  Source lines:       CK=$CK_SRC_LINES  opus=$OPUS_SRC_LINES"
echo "  Preprocessed lines: CK=$CK_PP_LINES  opus=$OPUS_PP_LINES"
echo ""

# --- Step 2: preprocessing time ---
echo "--- Preprocessing time (hipcc -E) ---"
CK_PP_TIMES=()
OPUS_PP_TIMES=()
for i in $(seq 1 $NUM_RUNS); do
    t0=$(date +%s%N)
    $HIPCC -x hip -E $CK_SRC -I$CK_INC --offload-arch=$ARCH -O3 > /dev/null 2>&1
    t1=$(date +%s%N)
    ms=$(( (t1 - t0) / 1000000 ))
    CK_PP_TIMES+=($ms)

    t0=$(date +%s%N)
    $HIPCC -x hip -E $OPUS_SRC -I$OPUS_INC --offload-arch=$ARCH -O3 > /dev/null 2>&1
    t1=$(date +%s%N)
    ms=$(( (t1 - t0) / 1000000 ))
    OPUS_PP_TIMES+=($ms)
done

ck_pp_sum=0; for v in "${CK_PP_TIMES[@]}"; do ck_pp_sum=$((ck_pp_sum + v)); done
opus_pp_sum=0; for v in "${OPUS_PP_TIMES[@]}"; do opus_pp_sum=$((opus_pp_sum + v)); done
ck_pp_avg=$((ck_pp_sum / NUM_RUNS))
opus_pp_avg=$((opus_pp_sum / NUM_RUNS))

printf "  CK:   %4dms avg  [%s]\n" $ck_pp_avg "$(printf '%dms ' "${CK_PP_TIMES[@]}")"
printf "  opus: %4dms avg  [%s]\n" $opus_pp_avg "$(printf '%dms ' "${OPUS_PP_TIMES[@]}")"
echo ""

# --- Step 3: compile only (hipcc -c) ---
echo "--- Compile time (hipcc -c, full compile) ---"
CK_CC_TIMES=()
OPUS_CC_TIMES=()
for i in $(seq 1 $NUM_RUNS); do
    t0=$(date +%s%N)
    $HIPCC -x hip -c $CK_SRC -I$CK_INC --offload-arch=$ARCH -O3 -o $TMPDIR/ck.o 2>/dev/null
    t1=$(date +%s%N)
    ms=$(( (t1 - t0) / 1000000 ))
    CK_CC_TIMES+=($ms)

    t0=$(date +%s%N)
    $HIPCC -x hip -c $OPUS_SRC -I$OPUS_INC --offload-arch=$ARCH -O3 -o $TMPDIR/opus.o 2>/dev/null
    t1=$(date +%s%N)
    ms=$(( (t1 - t0) / 1000000 ))
    OPUS_CC_TIMES+=($ms)
done

ck_cc_sum=0; for v in "${CK_CC_TIMES[@]}"; do ck_cc_sum=$((ck_cc_sum + v)); done
opus_cc_sum=0; for v in "${OPUS_CC_TIMES[@]}"; do opus_cc_sum=$((opus_cc_sum + v)); done
ck_cc_avg=$((ck_cc_sum / NUM_RUNS))
opus_cc_avg=$((opus_cc_sum / NUM_RUNS))

printf "  CK:   %4dms avg  [%s]\n" $ck_cc_avg "$(printf '%dms ' "${CK_CC_TIMES[@]}")"
printf "  opus: %4dms avg  [%s]\n" $opus_cc_avg "$(printf '%dms ' "${OPUS_CC_TIMES[@]}")"
echo ""

# --- Step 4: full build (compile + link) ---
echo "--- Full build time (hipcc compile + link to exe) ---"
CK_FULL_TIMES=()
OPUS_FULL_TIMES=()
for i in $(seq 1 $NUM_RUNS); do
    t0=$(date +%s%N)
    $HIPCC -x hip $CK_SRC -I$CK_INC --offload-arch=$ARCH -O3 -o $TMPDIR/ck.exe 2>/dev/null
    t1=$(date +%s%N)
    ms=$(( (t1 - t0) / 1000000 ))
    CK_FULL_TIMES+=($ms)

    t0=$(date +%s%N)
    $HIPCC -x hip $OPUS_SRC -I$OPUS_INC --offload-arch=$ARCH -O3 -o $TMPDIR/opus.exe 2>/dev/null
    t1=$(date +%s%N)
    ms=$(( (t1 - t0) / 1000000 ))
    OPUS_FULL_TIMES+=($ms)
done

ck_full_sum=0; for v in "${CK_FULL_TIMES[@]}"; do ck_full_sum=$((ck_full_sum + v)); done
opus_full_sum=0; for v in "${OPUS_FULL_TIMES[@]}"; do opus_full_sum=$((opus_full_sum + v)); done
ck_full_avg=$((ck_full_sum / NUM_RUNS))
opus_full_avg=$((opus_full_sum / NUM_RUNS))

printf "  CK:   %4dms avg  [%s]\n" $ck_full_avg "$(printf '%dms ' "${CK_FULL_TIMES[@]}")"
printf "  opus: %4dms avg  [%s]\n" $opus_full_avg "$(printf '%dms ' "${OPUS_FULL_TIMES[@]}")"
echo ""

# --- Step 5: object file sizes ---
CK_OBJ_SIZE=$(stat -c%s $TMPDIR/ck.o)
OPUS_OBJ_SIZE=$(stat -c%s $TMPDIR/opus.o)
CK_EXE_SIZE=$(stat -c%s $TMPDIR/ck.exe)
OPUS_EXE_SIZE=$(stat -c%s $TMPDIR/opus.exe)

echo "--- Output sizes ---"
printf "  Object:     CK=%d bytes  opus=%d bytes\n" $CK_OBJ_SIZE $OPUS_OBJ_SIZE
printf "  Executable: CK=%d bytes  opus=%d bytes\n" $CK_EXE_SIZE $OPUS_EXE_SIZE
echo ""

# --- Summary ---
echo "========================================================================"
echo " SUMMARY (averaged over $NUM_RUNS runs)"
echo "========================================================================"
printf "  %-25s %8s %8s %8s\n" "Stage" "CK" "opus" "diff"
echo "  ---------------------------------------------------------------"
pp_diff=$((opus_pp_avg - ck_pp_avg))
cc_diff=$((opus_cc_avg - ck_cc_avg))
full_diff=$((opus_full_avg - ck_full_avg))
printf "  %-25s %6dms %6dms %+5dms\n" "Preprocess (-E)" $ck_pp_avg $opus_pp_avg $pp_diff
printf "  %-25s %6dms %6dms %+5dms\n" "Compile (-c)" $ck_cc_avg $opus_cc_avg $cc_diff
printf "  %-25s %6dms %6dms %+5dms\n" "Full build (exe)" $ck_full_avg $opus_full_avg $full_diff
printf "  %-25s %8d %8d %+5d\n" "Preprocessed lines" $CK_PP_LINES $OPUS_PP_LINES $((OPUS_PP_LINES - CK_PP_LINES))
echo ""
