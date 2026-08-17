#!/bin/bash
# Regenerate device assembly for the kernel variants. HIP links the device side
# through LTO, so the compiler never writes a device .s during a normal build:
# run llc on the post-optimisation bitcode the linker wrapper hands to codegen
# (-save-temps=obj leaves it behind) with the same flags the build uses.
set -u
cd "$(dirname "$0")"
# Must be a pin-enabled build (see README); a stock llc rejects the second flag.
LLC=${LLC:-llc}
FLAGS=(-mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -O3
       -enable-post-misched=1 -amdgpu-expert-scheduling-mode)

mkdir -p asm
for spec in build_pin:pin build_nopin:nopin; do
    dir=${spec%%:*}; name=${spec##*:}
    bc=$(ls "$dir"/*.precodegen.bc 2>/dev/null | head -1)
    if [ -z "$bc" ]; then
        printf '  %-22s SKIPPED (no precodegen bitcode in %s)\n' "$name" "$dir"
        continue
    fi
    "$LLC" "${FLAGS[@]}" "$bc" -o "asm/$name.s" || continue
    printf '  %-22s %6d lines   wmma=%-4s ds_load=%-4s s_wait_alu=%-4s va_vdst0=%-4s vgpr=%s\n' \
        "asm/$name.s" "$(grep -c '' "asm/$name.s")" \
        "$(grep -c v_wmma "asm/$name.s")" \
        "$(grep -c ds_load "asm/$name.s")" \
        "$(grep -c s_wait_alu "asm/$name.s")" \
        "$(grep -c 'va_vdst(0)' "asm/$name.s")" \
        "$(grep -oE '\.vgpr_count:[ ]*[0-9]+' "asm/$name.s" | grep -oE '[0-9]+$' | head -1)"
done
