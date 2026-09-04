#!/bin/bash
# Low-noise benchmark: run the exe several times and report the distribution.
#
# Two separate sources of spread, do not confuse them:
#   * within one card, repeatability is under 0.2% -- a smaller difference than
#     that is noise, not a change;
#   * across cards, DVFS under the power wall spreads results by ~5%, so a
#     number is only meaningful next to the card it came from.
# Therefore always compare two versions on the SAME card, and always quote the
# card id when reporting a number.
#
# Usage: ./bench.sh [num_processes] [gpu_id]
#        SHAPE="-m 4096 -n 4096 -k 4096 -b 1" ./bench.sh 5 2
EXE=${EXE:-./build/gemm_a8w8_mxfp8_scale.exe}
SHAPE=${SHAPE:--m 8192 -n 8192 -k 8192 -b 1}
ITER=${ITER:--w 200 -i 100}
NP=${1:-5}
GPU=${2:-}

if [ ! -x "$EXE" ]; then
    echo "$EXE not found -- run 'make' first" >&2
    exit 1
fi

if [ -n "$GPU" ]; then
    export HIP_VISIBLE_DEVICES=$GPU
    CARD="GPU $GPU"
else
    CARD="default GPU"
fi

TMP=${RAW:-$(mktemp)}
for _ in $(seq 1 "$NP"); do
    $EXE $SHAPE -v 0 $ITER 2>&1 |
        grep -oE "[0-9]+\.[0-9]+ TFlops" | tail -1 | cut -d' ' -f1
done | sort -n > "$TMP"

awk -v card="$CARD" '{a[NR]=$1} END{
  n=NR;
  if(n==0){ print "no results parsed -- did the exe run?"; exit 1 }
  medi=int(n/2)+1; if(medi>n)medi=n;
  spread=(a[n]-a[1])/a[medi]*100;
  printf "%s  runs=%d  max=%.2f  median=%.2f  min=%.2f TFlops  spread=%.2f%%\n",
    card, n, a[n], a[medi], a[1], spread;
}' "$TMP"

if [ -z "$RAW" ]; then rm -f "$TMP"; fi
exit 0
