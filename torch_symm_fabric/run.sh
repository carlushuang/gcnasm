#!/usr/bin/env bash
# Launch one rank per visible GPU. Extra args pass through to main.py.
#   ./run.sh
#   NPROC=2 ./run.sh --window-mib 512 --sizes 1,16,64,256,512
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "$HERE/libfabric_symm.so" ]; then
    "$HERE/build.sh"
fi

NPROC="${NPROC:-$(python3 -c 'import torch; print(torch.cuda.device_count())')}"
echo "launching $NPROC ranks"

exec torchrun --nnodes=1 --nproc_per_node="$NPROC" "$HERE/main.py" "$@"
