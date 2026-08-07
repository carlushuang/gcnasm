#!/usr/bin/env bash
# Launch one rank per visible GPU, with the fabric shim preloaded so that torch's
# symm_mem allocations come back fabric-exportable. Extra args pass through to main.py.
#   ./run.sh
#   NPROC=2 ./run.sh --buffer-mib 512 --sizes 1,16,64,256,512
#   FABRIC_SHIM_VERBOSE=1 ./run.sh        # log every allocation the shim upgrades
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "$HERE/libfabric_symm.so" ] || [ ! -f "$HERE/libfabric_shim.so" ]; then
    "$HERE/build.sh"
fi

NPROC="${NPROC:-$(python3 -c 'import torch; print(torch.cuda.device_count())')}"
echo "launching $NPROC ranks"

# LD_PRELOAD is inherited by the torchrun workers, which is where it needs to take effect
export LD_PRELOAD="$HERE/libfabric_shim.so${LD_PRELOAD:+:$LD_PRELOAD}"

exec torchrun --nnodes=1 --nproc_per_node="$NPROC" "$HERE/main.py" "$@"
