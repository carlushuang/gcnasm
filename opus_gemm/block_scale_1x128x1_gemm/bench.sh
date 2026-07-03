#!/bin/bash
# Low-noise benchmark: run the exe many times; each run does many internal rounds.
# DVFS makes per-process peak clock vary, so the cleanest cross-version signal is the
# MAX "best" across processes (both versions converge to the hardware clock ceiling).
# Usage: ./bench.sh [num_processes] [rounds_per_process]
EXE=${EXE:-./build_abl/gemm_a8w8_1x128x1_blockscale.exe}
NP=${1:-12}
ROUNDS=${2:-60}
TMP=${RAW:-$(mktemp)}
for i in $(seq 1 "$NP"); do
    BENCH_ROUNDS=$ROUNDS "$EXE" 2>&1 | grep -oE "best=[0-9]+\.[0-9]+" | head -1 | cut -d= -f2
done | sort -n > "$TMP"
awk '{a[NR]=$1} END{
  n=NR;
  p90i=int(n*0.9); if(p90i<1)p90i=1; if(p90i>n)p90i=n;
  medi=int(n/2)+1; if(medi>n)medi=n;
  printf "processes=%d  peak(max)=%.2f  p90=%.2f  median=%.2f  min=%.2f TFlops\n",
    n, a[n], a[p90i], a[medi], a[1];
}' "$TMP"
if [ -z "$RAW" ]; then rm -f "$TMP"; fi
exit 0
