#!/usr/bin/env bash
# Run all three methods back to back and collate their summary lines.
#   ./bench_methods.sh
#   NPROC=2 ./bench_methods.sh --buffer-mib 512 --sizes 64,512
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$(mktemp -d)"
trap 'rm -rf "$LOG"' EXIT

"$HERE/build.sh" >/dev/null

for m in own shim rebind; do
    echo "===== method=$m ====="
    if "$HERE/run.sh" --method "$m" "$@" >"$LOG/$m.txt" 2>&1; then
        grep -E "^\[probe\]|^\[summary\]|^SUCCESS" "$LOG/$m.txt" || true
    else
        echo "FAILED -- last lines:"
        tail -5 "$LOG/$m.txt"
    fi
    echo
done

echo "================ comparison ================"
printf "%-8s %10s %11s %12s %12s %12s %8s\n" \
    method setup_ms phys_MiB local_GB/s read_GB/s write_GB/s status
printf "%-8s %10s %11s %12s %12s %12s %8s\n" \
    -------- ---------- ----------- ------------ ------------ ------------ --------
for m in own shim rebind; do
    line="$(grep -h '^\[summary\]' "$LOG/$m.txt" 2>/dev/null || true)"
    if [ -z "$line" ]; then
        printf "%-8s %10s %11s %12s %12s %12s %8s\n" "$m" - - - - - FAIL
        continue
    fi
    get() { echo "$line" | tr ' ' '\n' | grep "^$1=" | cut -d= -f2; }
    printf "%-8s %10s %11s %12s %12s %12s %8s\n" \
        "$m" "$(get setup_ms)" "$(get phys_mib)" "$(get local_gbs)" \
        "$(get read_gbs)" "$(get write_gbs)" "$(get status)"
done
