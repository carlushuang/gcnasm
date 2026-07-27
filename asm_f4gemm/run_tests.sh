#!/bin/bash
# Regression sweep: every kernel in the manifest x a few shapes, plus the
# heuristic path and the splitK path.
#
#   ./run_tests.sh [--co-dir DIR]
# Falls back to $AITER_F4GEMM_DIR / $AITER_ASM_DIR/gfx950/f4gemm.

set -u

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

EXE=./asm_f4gemm.exe
CO_ARGS=()
if [ "${1:-}" = "--co-dir" ]; then
    CO_ARGS=(--co-dir "$2")
    shift 2
fi

[ -x "$EXE" ] || ./build.sh

pass=0
fail=0
run() {
    local desc="$1"; shift
    local out
    out=$("$EXE" "${CO_ARGS[@]}" "$@" 2>&1)
    if echo "$out" | grep -q "PASSED"; then
        pass=$((pass + 1))
        printf "  ok   %s -- %s\n" "$desc" "$(echo "$out" | grep 'verify:' | sed 's/.*verify: //')"
    else
        fail=$((fail + 1))
        printf "  FAIL %s\n" "$desc"
        echo "$out" | sed 's/^/       /'
    fi
}

echo "== heuristic kernel selection over a shape sweep =="
for shape in "128 1024 1024" "256 2048 2048" "300 2048 1024" "1 4096 2048" "17 512 256" \
             "512 1024 2048" "2048 8192 1024" "4096 4096 512"; do
    set -- $shape
    run "heuristic M=$1 N=$2 K=$3" -m "$1" -n "$2" -k "$3" --iters 0
done

echo "== every kernel in the manifest (M=300 N=2048 K=1024) =="
mapfile -t kernels < <("$EXE" "${CO_ARGS[@]}" --list | awk '$5 ~ /\.co$/ {print $5}')
for co in "${kernels[@]}"; do
    bps=1
    [[ "$co" == *noBpreShuffle* ]] && bps=0
    run "$co" -m 300 -n 2048 -k 1024 --kernel "$co" --bpreshuffle $bps --iters 0
done

echo "== splitK (bf16 atomic accumulate) =="
for co in f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co \
          f4gemm_bf16_per1x32Fp4_BpreShuffle_128x512.co; do
    for s in 1 2 3 4; do
        run "$co splitk=$s" -m 256 -n 1024 -k 4096 --kernel "$co" --splitk "$s" --iters 0
    done
done

echo
echo "passed: $pass   failed: $fail"
[ "$fail" -eq 0 ]
