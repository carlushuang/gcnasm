#!/bin/bash
# Rerun the locality GEMM under lowered power caps, then restore the default cap (trap on exit).
# DEV: HIP device; BDF: its PCI address; SMI: its rocm-smi index (see `rocm-smi --showbus`). Needs sudo and idle GPUs.
DEV=${DEV:-0}; BDF=${BDF:-0000:75:00.0}; SMI=${SMI:-3}; CAPS=${CAPS:-"1000 800"}; MS=${MS:-"4096 1024"}
CAPF=$(ls /sys/bus/pci/devices/$BDF/hwmon/hwmon*/power1_cap)
restore() { sudo rocm-smi -d "$SMI" --resetpoweroverdrive --autorespond y >/dev/null 2>&1; echo "restored cap: $(( $(cat $CAPF) / 1000000 )) W"; }
trap restore EXIT
cd "$(dirname "$0")"
if [ -n "$(ls /sys/class/kfd/kfd/proc/ 2>/dev/null)" ]; then echo "GPU busy (KFD processes present), abort"; exit 1; fi
for cap in $CAPS; do
    sudo rocm-smi -d "$SMI" --setpoweroverdrive "$cap" --autorespond y 2>&1 | grep -E "GPU\[$SMI\]" | head -2
    echo "cap now: $(( $(cat $CAPF) / 1000000 )) W"
    for m in $MS; do
        echo "######## M=$m cap=$cap"
        ./build/fp8_gemm_locality -d "$DEV" -m "$m" -n 65536 -s 3 -r 3
    done
done
