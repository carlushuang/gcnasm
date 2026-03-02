#!/bin/bash
# Launch ROCm docker and run all memory latency benchmarks.

DOCKER_IMAGE="rocm/atom:nightly_202601190317"
UBENCH_DIR="/raid0/carhuang/repo/gcnasm/ubench"

docker run -it --privileged --network=host \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    -v /home/carhuang:/dockerx \
    -v /mnt/raid0:/raid0 \
    "$DOCKER_IMAGE" \
    bash -c "cd $UBENCH_DIR && bash run.sh"
