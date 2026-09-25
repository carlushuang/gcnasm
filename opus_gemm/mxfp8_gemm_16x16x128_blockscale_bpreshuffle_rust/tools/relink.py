#!/usr/bin/env python3
"""Finish rustc's amdgpu LTO + codegen with the two things Rust cannot express.

Input is rustc's post-LTO-link bitcode (build with -Csave-temps). Then:

1. Kernel attributes. rustc has no __launch_bounds__, so every extern "gpu-kernel"
   fn defaults to amdgpu-flat-work-group-size=1,1024, which caps a 512-thread
   kernel at 128 VGPRs. --attr adds e.g. amdgpu-flat-work-group-size=1,512.
2. IR shims. Rust has only generic pointers and no bfloat type, so intrinsics that
   take `ptr addrspace(3)` or produce `bfloat` cannot be declared from Rust (rustc
   checks intrinsic signatures), and amdgpu inline asm cannot take register operands.
   The kernel calls `rk.*` extern "C" placeholders instead, and this script defines
   them in LLVM IR (SHIMS below); opt inlines them.

Then runs rustc's own LLVM (llvm-tools component): opt lto<O3>, llc, ld.lld.
"""
import argparse
import glob
import os
import re
import subprocess
import sys

TOOLS = os.path.expanduser(
    "~/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/bin")

SHIMS = {
    # buffer_load_dwordx4 ... lds (size/offset/aux are immargs, so fixed here)
    "rk.buffer.load.lds.b128": """
define internal void @rk.buffer.load.lds.b128(<4 x i32> %rsrc, ptr %lds, i32 %voffset, i32 %soffset) alwaysinline {
  %l = addrspacecast ptr %lds to ptr addrspace(3)
  call void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> %rsrc, ptr addrspace(3) %l, i32 16, i32 %voffset, i32 %soffset, i32 0, i32 0)
  ret void
}
declare void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32>, ptr addrspace(3), i32 immarg, i32, i32, i32 immarg, i32 immarg)
""",
    # ds_read_b64_tr_b8
    "rk.ds.read.tr8.b64": """
define internal <2 x i32> @rk.ds.read.tr8.b64(ptr %lds) alwaysinline {
  %l = addrspacecast ptr %lds to ptr addrspace(3)
  %r = call <2 x i32> @llvm.amdgcn.ds.read.tr8.b64.v2i32(ptr addrspace(3) %l)
  ret <2 x i32> %r
}
declare <2 x i32> @llvm.amdgcn.ds.read.tr8.b64.v2i32(ptr addrspace(3))
""",
    # the C++ kernel's `asm volatile("" : "+v"(v_c_pin[i]))`: keeps each 16-float accumulator
    # group in one fixed register tuple and orders the scheduler around it
    "rk.pin.v16f32": """
define internal <16 x float> @rk.pin.v16f32(<16 x float> %x) alwaysinline {
  %r = call <16 x float> asm sideeffect "", "=v,0"(<16 x float> %x)
  ret <16 x float> %r
}
""",
    # the C++ kernel's `asm volatile("" : "+s"(local_role))`: hides a wave-uniform value
    "rk.pin.s.i32": """
define internal i32 @rk.pin.s.i32(i32 %x) alwaysinline {
  %r = call i32 asm sideeffect "", "=s,0"(i32 %x)
  ret i32 %r
}
""",
    # the C++ kernel's inline-asm LDS accesses (volatile, memory clobber)
    "rk.asm.ds.read2st64.b32.o2": """
define internal <2 x i32> @rk.asm.ds.read2st64.b32.o2(ptr %lds) alwaysinline {
  %l = addrspacecast ptr %lds to ptr addrspace(3)
  %a = ptrtoint ptr addrspace(3) %l to i32
  %r = call <2 x i32> asm sideeffect "ds_read2st64_b32 $0, $1 offset0:0 offset1:2\0A", "=v,v,~{memory}"(i32 %a)
  ret <2 x i32> %r
}
""",
    "rk.asm.ds.write.b8": """
define internal void @rk.asm.ds.write.b8(ptr %lds, i32 %v) alwaysinline {
  %l = addrspacecast ptr %lds to ptr addrspace(3)
  %a = ptrtoint ptr addrspace(3) %l to i32
  call void asm sideeffect "ds_write_b8 $0, $1\0A", "v,v,~{memory}"(i32 %a, i32 %v)
  ret void
}
""",
    # two floats -> packed bf16 (round to nearest even), v_cvt_pk_bf16_f32 on gfx950
    "rk.cvt.pk.bf16.f32": """
define internal i32 @rk.cvt.pk.bf16.f32(float %a, float %b) alwaysinline {
  %v0 = insertelement <2 x float> poison, float %a, i32 0
  %v1 = insertelement <2 x float> %v0, float %b, i32 1
  %h = fptrunc <2 x float> %v1 to <2 x bfloat>
  %r = bitcast <2 x bfloat> %h to i32
  ret i32 %r
}
""",
}


def run(*cmd):
    subprocess.run(cmd, check=True)


def link_shims(ll):
    used = []
    for name, body in SHIMS.items():
        decl = re.compile(r"^declare [^\n]*@" + re.escape(name) + r"\([^\n]*$\n?", re.M)
        if not decl.search(ll):
            continue
        ll = decl.sub("", ll)
        # drop intrinsic declarations the shim re-declares
        for d in re.findall(r"^declare [^\n]*(@llvm\.[\w.]+)\(", body, re.M):
            ll = re.sub(r"^declare [^\n]*" + re.escape(d) + r"\([^\n]*$\n?", "", ll, flags=re.M)
        ll += body
        used.append(name)
    left = sorted(set(re.findall(r"@(rk\.[\w.]+)\(", ll)) - set(used))
    if left:
        sys.exit(f"no IR shim for {left}")
    return ll, used


def add_kernel_attrs(ll, attrs):
    kernels = re.findall(r"^define [^\n]*amdgpu_kernel [^\n]*@(\w+)\([^\n]*\) [^\n]*#(\d+) [^\n]*\{", ll, re.M)
    extra = " ".join(f'"{k}"="{v}"' if v else f'"{k}"' for k, v in attrs)
    for g in sorted({g for _, g in kernels}):
        ll, n = re.subn(rf"^(attributes #{g} = \{{)(.*)\}}$", rf"\1\2 {extra} }}", ll, flags=re.M)
        assert n == 1, g
    return ll, [k for k, _ in kernels]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target-dir", required=True)
    p.add_argument("--crate", required=True)
    p.add_argument("--mcpu", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--attr", action="append", default=[],
                   help="kernel attribute key=value or key, e.g. amdgpu-flat-work-group-size=1,512")
    p.add_argument("--llc-arg", action="append", default=[])
    a = p.parse_args()

    bcs = glob.glob(f"{a.target_dir}/**/{a.crate}.{a.crate}.*.lto.after-restriction.bc", recursive=True)
    if not bcs:
        sys.exit(f"no post-LTO-link bitcode under {a.target_dir}; build with -Csave-temps")
    bc = max(bcs, key=os.path.getmtime)  # different RUSTFLAGS land in different hash dirs
    stem = os.path.splitext(a.out)[0]
    run(f"{TOOLS}/llvm-dis", bc, "-o", f"{stem}.linked.ll")
    ll = open(f"{stem}.linked.ll").read()
    ll, shims = link_shims(ll)
    ll, kernels = add_kernel_attrs(ll, [(kv.split("=", 1) + [""])[:2] for kv in a.attr])
    open(f"{stem}.patched.ll", "w").write(ll)
    run(f"{TOOLS}/opt", "-passes=lto<O3>", f"{stem}.patched.ll", "-o", f"{stem}.opt.bc")
    run(f"{TOOLS}/llc", "-O3", f"-mcpu={a.mcpu}", "-filetype=obj", *a.llc_arg, f"{stem}.opt.bc", "-o", f"{stem}.o")
    run(f"{TOOLS}/gcc-ld/ld.lld", "-shared", f"{stem}.o", "-o", a.out)
    print(f"relink: {len(kernels)} kernels, shims {shims}, attrs {a.attr} -> {a.out}")


if __name__ == "__main__":
    main()
