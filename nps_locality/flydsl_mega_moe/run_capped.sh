#!/bin/bash
# Host-side: lower the power cap of one GPU, run the GEMM1 benchmark inside the container, always restore the cap.
# BDF / SMI: PCI address and rocm-smi index of the container's HIP device DEV (see `rocm-smi --showbus`).
CONTAINER=${CONTAINER:-carhuang_dsv41_n1029}; DIR=${DIR:-/home/carhuang/dsv41/fly_iod}
DEV=${DEV:-0}; BDF=${BDF:-0000:75:00.0}; SMI=${SMI:-3}; CAP=${CAP:-1000}
CAPF=$(ls /sys/bus/pci/devices/$BDF/hwmon/hwmon*/power1_cap)
restore() { sudo rocm-smi -d "$SMI" --resetpoweroverdrive --autorespond y >/dev/null 2>&1; echo "restored cap: $(( $(cat $CAPF) / 1000000 )) W"; }
trap restore EXIT
if [ -n "$(ls /sys/class/kfd/kfd/proc/ 2>/dev/null)" ]; then echo "GPU busy (KFD processes present), abort"; exit 1; fi
sudo rocm-smi -d "$SMI" --setpoweroverdrive "$CAP" --autorespond y 2>&1 | grep -E "GPU\[$SMI\]"
docker exec "$CONTAINER" bash -c "cd $DIR && DEV=$DEV TPE=${TPE:-1,4,32} SECONDS=3 ROUNDS=3 python3 bench_gemm1_iod.py"
