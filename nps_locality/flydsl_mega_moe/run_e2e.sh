#!/bin/bash
# Host-side: optionally cap all 8 GPUs (CAP watts), run e2e_mega_moe.py in the container, always restore the caps.
CONTAINER=${CONTAINER:-carhuang_dsv41_n1029}; DIR=${DIR:-/home/carhuang/dsv41/fly_iod}; CAP=${CAP:-0}
restore() {
    if [ "$CAP" != 0 ]; then sudo rocm-smi --resetpoweroverdrive --autorespond y >/dev/null 2>&1; fi
    echo "power caps after run: $(for f in /sys/class/drm/card*/device/hwmon/hwmon*/power1_cap; do echo -n "$(( $(cat $f) / 1000000 )) "; done)W"
}
trap restore EXIT
if [ -n "$(ls /sys/class/kfd/kfd/proc/ 2>/dev/null)" ]; then echo "GPU busy (KFD processes present), abort"; exit 1; fi
if [ "$CAP" != 0 ]; then sudo rocm-smi --setpoweroverdrive "$CAP" --autorespond y 2>&1 | grep -E "GPU\[" | head -8; fi
echo "power caps: $(for f in /sys/class/drm/card*/device/hwmon/hwmon*/power1_cap; do echo -n "$(( $(cat $f) / 1000000 )) "; done)W"
docker exec "$CONTAINER" bash -c "cd $DIR && MORI_SHMEM_HEAP_SIZE=${HEAP:-48G} TOKENS=${TOKENS:-64,512,4096} \
    VARIANTS=${VARIANTS:-stock,s2q,s2q+s2w,s2q+s2x} ROUNDS=${ROUNDS:-5} ITERS=${ITERS:-20} SUSTAIN=${SUSTAIN:-1.0} \
    torchrun --nproc_per_node 8 --master_port 29931 e2e_mega_moe.py 2>&1 \
    | grep --line-buffered -v '^\[aiter\]\|flydsl_dispatch_combine_intranode_op\|Gloo\|^W1[01][0-9][0-9]\|^\s*$'"
