#!/usr/bin/env bash
# Launch one rank per visible GPU. Nothing to build -- this side is pure python.
#   ./run.sh
#   NPROC=4 ./run.sh --buffer-mib 512 --sizes 64,512
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# H20 (and any GPU without NVLS multicast) makes symm_mem.rendezvous() raise
# "CUDA driver error: invalid argument" unless multicast is switched off explicitly.
export TORCH_SYMM_MEM_DISABLE_MULTICAST="${TORCH_SYMM_MEM_DISABLE_MULTICAST:-1}"

NPROC="${NPROC:-$(python3 -c 'import torch; print(torch.cuda.device_count())')}"
echo "launching $NPROC ranks"

exec torchrun --nnodes=1 --nproc_per_node="$NPROC" "$HERE/symm_mem_cuda.py" "$@"
