#!/usr/bin/env bash
# Launch one rank per visible GPU. Extra args pass through to main.py.
#   ./run.sh --method rebind
#   ./run.sh --method shim            # LD_PRELOAD is set for you
#   NPROC=2 ./run.sh --method own --buffer-mib 512 --sizes 1,16,64,256,512
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "$HERE/libfabric_symm.so" ] || [ ! -f "$HERE/libfabric_shim.so" ]; then
    "$HERE/build.sh"
fi

# Only the shim method needs the interposer, and it must be in place before the
# torchrun workers start. The other two methods run with a clean environment.
for arg in "$@"; do
    if [ "$arg" = "shim" ]; then
        export LD_PRELOAD="$HERE/libfabric_shim.so${LD_PRELOAD:+:$LD_PRELOAD}"
        echo "LD_PRELOAD=$LD_PRELOAD"
        break
    fi
done

NPROC="${NPROC:-$(python3 -c 'import torch; print(torch.cuda.device_count())')}"
echo "launching $NPROC ranks"

exec torchrun --nnodes=1 --nproc_per_node="$NPROC" "$HERE/main.py" "$@"
