#!/usr/bin/env python3
"""Turn a prebuilt AMDGPU code object back into reassemblable GCN assembly.

Reconstructs the three pieces the assembler needs from an existing .co:

  .text            <- llvm-objdump -d            (instructions + local labels)
  .amdhsa_kernel   <- llvm-objdump -d -j .rodata (the 64-byte kernel descriptor,
                                                  which llvm-objdump already
                                                  decodes into directives)
  .amdgpu_metadata <- llvm-readelf --notes       (the msgpack note, printed as YAML)

The result assembles back to a byte-identical .text with
    clang -x assembler -target amdgcn-amd-amdhsa -mcpu=<arch> -mcode-object-version=<v>

Written for aiter's hsa/gfx950/f4gemm kernels; nothing here is f4gemm-specific.
"""

import argparse
import os
import re
import subprocess
import sys

LABEL_RE = re.compile(r"^[0-9a-fA-F]+\s+<(.+)>:\s*$")
# llvm-objdump appends "// <addr>: <encoding>" to every instruction.
COMMENT_RE = re.compile(r"\s*//.*$")

# SGPRs the assembler silently adds on top of .amdhsa_next_free_sgpr on gfx8+
# when flat_scratch is reserved (llvm's AMDGPU::IsaInfo::getNumExtraSGPRs).
# .amdhsa_reserve_flat_scratch defaults to 1 and llvm-objdump never prints it,
# so this always applies here.
EXTRA_SGPRS = 6


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"[co2asm] command failed: {' '.join(cmd)}\n{p.stderr}")
    return p.stdout


# e_ident[EI_ABIVERSION] is NOT the code object version -- ELFABIVERSION_AMDGPU_HSA_V2
# is 0, V3 is 1, ... so an "ABI Version: 4" header means code object v6.
ABI_TO_COV = {0: 2, 1: 3, 2: 4, 3: 5, 4: 6}


def elf_flags(readelf, co):
    """(arch, code_object_version) straight out of the ELF header."""
    out = run([readelf, "-h", co])
    arch, cov = None, None
    for line in out.splitlines():
        if "Flags:" in line:
            m = re.search(r"(gfx\w+)", line)
            if m:
                arch = m.group(1)
        if "ABI Version:" in line:
            abi = int(line.split(":")[1].strip())
            cov = ABI_TO_COV.get(abi)
            if cov is None:
                sys.exit(f"[co2asm] unrecognised AMDGPU ELF ABI version {abi}")
    return arch, cov


def disasm_text(objdump, co, arch):
    """.text -> (kernel_names, body_lines)."""
    out = run([objdump, "-d", "--triple=amdgcn-amd-amdhsa", f"--mcpu={arch}", co])

    body, kernels, in_text = [], [], False
    for line in out.splitlines():
        if line.startswith("Disassembly of section"):
            in_text = ".text" in line
            continue
        if not in_text or not line.strip():
            continue

        m = LABEL_RE.match(line)
        if m:
            name = m.group(1)
            # Local branch targets are emitted as symbols too; only the ones the
            # kernel descriptor names are real entry points, but every label in
            # .text that is not "label_XXXX" is treated as a kernel here.
            if not name.startswith("label_"):
                kernels.append(name)
                body.append("")
                body.append(f"{name}:")
            else:
                body.append(f"{name}:")
            continue

        insn = COMMENT_RE.sub("", line).rstrip()
        if insn.strip():
            body.append(insn)

    if not kernels:
        sys.exit("[co2asm] no kernel symbol found in .text")
    return kernels, body


def disasm_kd(objdump, co, arch):
    """.rodata -> (.amdhsa_kernel blocks, target id to assemble with).

    llvm-objdump's descriptor dump is not directly reassemblable; two fields
    need fixing up. The README's round-trip section shows how to diff the
    reassembled .rodata against the original, which proves both fixups are
    lossless.

    ``.amdhsa_next_free_sgpr`` -- the descriptor only stores the SGPR count
    granulated by 8, so llvm-objdump reports the *top* of the granule, and it
    inverts the encoding assuming zero extra SGPRs. The assembler goes the other
    way and adds EXTRA_SGPRS back before granulating, so feeding objdump's
    number straight back lands one granule too high (and 104 is rejected
    outright, since gfx9 addresses at most 102). Subtracting the extras
    reproduces the original granule exactly.

    ``.amdhsa_reserve_xnack_mask`` -- only legal when the target id names xnack.
    These objects are built for xnack ANY, so the directive is dropped and the
    target id left unqualified, which also keeps the rebuilt ELF flags identical.
    A kernel that actually reserves the mask gets an ``:xnack+`` target instead.
    """
    out = run(
        [objdump, "-d", "-j", ".rodata", "--triple=amdgcn-amd-amdhsa", f"--mcpu={arch}", co]
    )
    blocks, cur, target_id = [], None, arch
    for line in out.splitlines():
        s = line.strip()
        if s.startswith(".amdhsa_kernel"):
            cur = [line.rstrip()]
        elif cur is not None:
            if s.startswith(".amdhsa_reserve_xnack_mask"):
                if s.split()[-1] == "0":
                    continue  # default for an xnack-unqualified target
                target_id = f"{arch}:xnack+"
            if s.startswith(".amdhsa_next_free_sgpr"):
                n = int(s.split()[-1])
                line = line.replace(str(n), str(max(1, n - EXTRA_SGPRS)))
            cur.append(line.rstrip())
            if s == ".end_amdhsa_kernel":
                blocks.append(cur)
                cur = None
    if not blocks:
        sys.exit("[co2asm] no kernel descriptor found in .rodata")
    return blocks, target_id


def note_metadata(readelf, co):
    """The NT_AMDGPU_METADATA note, as the YAML document the assembler wants."""
    out = run([readelf, "--notes", co])
    lines, collecting = [], False
    for line in out.splitlines():
        s = line.strip()
        if not collecting:
            if s == "---":
                collecting = True
                lines.append("---")
            continue
        lines.append(line)
        if s == "...":
            break
    if not lines or lines[-1].strip() != "...":
        sys.exit("[co2asm] could not extract the AMDGPU metadata note")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("co", help="input code object")
    ap.add_argument("-o", "--output", help="output .s (default: <co>.s next to it)")
    ap.add_argument("--rocm", default="/opt/rocm", help="ROCm install prefix")
    ap.add_argument("--mcpu", help="override the arch detected from the ELF header")
    ap.add_argument("--print-target-id", action="store_true",
                    help="print only the clang -mcpu target id and exit")
    args = ap.parse_args()

    objdump = os.path.join(args.rocm, "llvm/bin/llvm-objdump")
    readelf = os.path.join(args.rocm, "llvm/bin/llvm-readelf")

    arch, cov = elf_flags(readelf, args.co)
    if args.mcpu:
        arch = args.mcpu
    if not arch:
        sys.exit("[co2asm] could not detect the target arch; pass --mcpu")

    # llvm-objdump only accepts a bare arch; clang wants the full target id.
    kds, target_id = disasm_kd(objdump, args.co, arch)
    if args.print_target_id:
        print(target_id)
        return

    kernels, body = disasm_text(objdump, args.co, arch)
    meta = note_metadata(readelf, args.co)

    out_path = args.output or os.path.splitext(args.co)[0] + ".s"
    with open(out_path, "w") as f:
        w = lambda s="": f.write(s + "\n")
        w(f"// Regenerated from {os.path.basename(args.co)} by co2asm.py")
        w(f"// target {target_id}, code object v{cov}")
        w()
        w("\t.text")
        for k in kernels:
            w(f"\t.globl {k}")
            w(f"\t.type {k},@function")
        w("\t.p2align 8")
        for line in body:
            w(line)
        w()
        w('\t.section .rodata,"a",@progbits')
        w("\t.p2align 6")
        for kd in kds:
            for line in kd:
                w(line)
        w()
        w("\t.amdgpu_metadata")
        for line in meta:
            w(line)
        w("\t.end_amdgpu_metadata")

    print(f"[co2asm] {args.co} -> {out_path}")
    print(f"[co2asm]   target={target_id} code-object-version={cov} kernels={len(kernels)}")
    print(f"[co2asm] reassemble with:")
    print(f"  {args.rocm}/llvm/bin/clang -x assembler -target amdgcn-amd-amdhsa \\")
    print(f"      -mcpu={target_id} -mcode-object-version={cov} {out_path} -o <out>.co")


if __name__ == "__main__":
    main()
