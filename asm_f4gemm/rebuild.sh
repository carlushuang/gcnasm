#!/bin/bash
# Round-trip one aiter code object: .co -> .s -> .co, verify, and drop the
# result into a self-contained --co-dir the host driver can be pointed at.
#
#   ./rebuild.sh [--co-dir SRC] [--kernel CO_NAME] [--out DIR]
#
# SRC defaults to $AITER_F4GEMM_DIR / $AITER_ASM_DIR/gfx950/f4gemm, the kernel
# to the 256x256 preshuffled tile, and DIR to ./rebuilt next to this script.
#
# The output dir gets the rebuilt .co plus a one-line manifest CSV, so pointing
# --co-dir at it exercises exactly the rebuilt kernel -- any other tile fails
# loudly with "cannot open", which is the point: it proves which object ran.

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

ROCM=${ROCM:-/opt/rocm}
CLANG="$ROCM/llvm/bin/clang"
OBJCOPY="$ROCM/llvm/bin/llvm-objcopy"
OBJDUMP="$ROCM/llvm/bin/llvm-objdump"

SRC_DIR="${AITER_F4GEMM_DIR:-${AITER_ASM_DIR:+$AITER_ASM_DIR/gfx950/f4gemm}}"
KERNEL="f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co"
OUT_DIR="$DIR/rebuilt"

while [ $# -gt 0 ]; do
    case "$1" in
        --co-dir) SRC_DIR="$2"; shift 2 ;;
        --kernel) KERNEL="$2"; shift 2 ;;
        --out)    OUT_DIR="$2"; shift 2 ;;
        *) echo "unknown option $1"; exit 1 ;;
    esac
done

if [ -z "${SRC_DIR:-}" ]; then
    echo "no source dir: pass --co-dir or set AITER_ASM_DIR" >&2
    exit 1
fi

ORIG="$SRC_DIR/$KERNEL"
MANIFEST="$SRC_DIR/f4gemm_bf16_per1x32Fp4.csv"
[ -f "$ORIG" ] || { echo "no such code object: $ORIG" >&2; exit 1; }

mkdir -p "$OUT_DIR"
ASM="$OUT_DIR/${KERNEL%.co}.s"
NEW="$OUT_DIR/$KERNEL"

echo "=== [1/4] disassemble: $KERNEL -> $(basename "$ASM")"
TARGET_ID=$(python3 co2asm.py "$ORIG" --print-target-id)
COV=$(python3 co2asm.py "$ORIG" -o /dev/null 2>/dev/null | sed -n 's/.*code-object-version=\([0-9]*\).*/\1/p' | head -1)
python3 co2asm.py "$ORIG" -o "$ASM" | sed 's/^/    /'

echo "=== [2/4] reassemble: $(basename "$ASM") -> $(basename "$NEW")"
"$CLANG" -x assembler -target amdgcn-amd-amdhsa \
    -mcpu="$TARGET_ID" -mcode-object-version="$COV" "$ASM" -o "$NEW"
echo "    OK ($(stat -c%s "$NEW") bytes, original $(stat -c%s "$ORIG"))"

echo "=== [3/4] verify against the original"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
rc=0

# ELF header: arch + feature flags + ABI version must match.
for f in "$ORIG" "$NEW"; do
    "$ROCM/llvm/bin/llvm-readelf" -h "$f" | grep -E "ABI Version|Flags:" > "$tmp/$(basename "$f").hdr"
done
if diff -q "$tmp/$(basename "$ORIG").hdr" "$tmp/$(basename "$NEW").hdr" >/dev/null; then
    echo "    ELF header    identical ($(sed -n 's/.*Flags: *//p' "$tmp/$(basename "$NEW").hdr"))"
else
    echo "    ELF header    DIFFERS"; diff "$tmp/$(basename "$ORIG").hdr" "$tmp/$(basename "$NEW").hdr" | sed 's/^/      /'; rc=1
fi

# Kernel descriptor: must be byte-identical, it drives SGPR/VGPR/LDS setup.
"$OBJCOPY" --dump-section=.rodata="$tmp/o.kd" "$ORIG" /dev/null 2>/dev/null
"$OBJCOPY" --dump-section=.rodata="$tmp/n.kd" "$NEW"  /dev/null 2>/dev/null
if cmp -s "$tmp/o.kd" "$tmp/n.kd"; then
    echo "    descriptor    identical (64 bytes)"
else
    echo "    descriptor    DIFFERS"; cmp -l "$tmp/o.kd" "$tmp/n.kd" | sed 's/^/      /'; rc=1
fi

# .text: compared as disassembly, not bytes. LLVM normalises a few don't-care
# operand bits (src2 op_sel on v_mfma_scale_*), so the encodings differ while
# every instruction decodes identically.
for f in "$ORIG:o" "$NEW:n"; do
    p=${f%:*}; t=${f#*:}
    "$OBJDUMP" -d --triple=amdgcn-amd-amdhsa --mcpu="${TARGET_ID%%:*}" "$p" \
        | sed 's|//.*||; s/[[:space:]]*$//' | tail -n +3 > "$tmp/$t.txt"
done
"$OBJCOPY" --dump-section=.text="$tmp/o.text" "$ORIG" /dev/null 2>/dev/null
"$OBJCOPY" --dump-section=.text="$tmp/n.text" "$NEW"  /dev/null 2>/dev/null
nbytes=$(cmp -l "$tmp/o.text" "$tmp/n.text" 2>/dev/null | wc -l || true)
if diff -q "$tmp/o.txt" "$tmp/n.txt" >/dev/null; then
    echo "    .text         $(wc -l < "$tmp/n.txt") lines, disassembly identical ($nbytes don't-care bits re-encoded)"
else
    echo "    .text         DISASSEMBLY DIFFERS"; diff "$tmp/o.txt" "$tmp/n.txt" | head -20 | sed 's/^/      /'; rc=1
fi

# Metadata note: kernarg offsets, LDS, workgroup size.
for f in "$ORIG:o" "$NEW:n"; do
    p=${f%:*}; t=${f#*:}
    "$ROCM/llvm/bin/llvm-readelf" --notes "$p" | sed -n '/---/,/^\s*\.\.\.$/p' > "$tmp/$t.meta"
done
if diff -q "$tmp/o.meta" "$tmp/n.meta" >/dev/null; then
    echo "    metadata      identical ($(grep -c '\.offset:' "$tmp/n.meta") kernarg slots)"
else
    echo "    metadata      DIFFERS"; diff "$tmp/o.meta" "$tmp/n.meta" | head -20 | sed 's/^/      /'; rc=1
fi

echo "=== [4/4] manifest for the rebuilt kernel"
head -1 "$MANIFEST" > "$OUT_DIR/f4gemm_bf16_per1x32Fp4.csv"
grep -F ",$KERNEL" "$MANIFEST" >> "$OUT_DIR/f4gemm_bf16_per1x32Fp4.csv"
echo "    $OUT_DIR/f4gemm_bf16_per1x32Fp4.csv"
sed 's/^/      /' "$OUT_DIR/f4gemm_bf16_per1x32Fp4.csv"

echo
if [ "$rc" -eq 0 ]; then
    echo "round-trip OK. Run the rebuilt kernel with:"
    echo "  ./asm_f4gemm.exe --co-dir $OUT_DIR -m 4096 -n 4096 -k 4096"
else
    echo "round-trip FAILED" >&2
fi
exit $rc
