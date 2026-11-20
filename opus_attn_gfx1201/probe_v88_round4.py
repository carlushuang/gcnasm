#!/usr/bin/env python3
"""Round 4: Fix dual-add bank conflicts, interleave bf16+sum properly,
and try the winning trunc_bf16 at all N sizes."""
import re, subprocess, os

BASE = "/mnt/sda1/carhuang/repo/gcnasm/opus_attn_gfx1201"
ASM = os.path.join(BASE, "v88_kernel.s")
HSACO = os.path.join(BASE, "v88.hsaco")
BIN = os.path.join(BASE, "attn_v88")

os.environ["PATH"] = "/opt/rocm/bin:/opt/rocm/llvm/bin:" + os.environ.get("PATH","")
os.environ["LD_LIBRARY_PATH"] = "/mnt/sda1/carhuang/lib"
os.environ["HIP_VISIBLE_DEVICES"] = "1"

with open(ASM) as f:
    baseline = f.readlines()

def find_line(lines, pattern, start=0):
    for i in range(start, len(lines)):
        if re.search(pattern, lines[i]):
            return i
    return -1

def replace_lines(lines, start, end, new_lines):
    out = lines[:start]
    for nl in new_lines:
        out.append(nl + "\n")
    out.extend(lines[end+1:])
    return out

def write_build_bench(tag, lines, n=7680, iters=500, runs=3, verify=False):
    with open(ASM, 'w') as f:
        f.writelines(lines)
    r = subprocess.run(
        ["clang++", "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
         "-mcpu=gfx1201", ASM, "-o", HSACO],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(f"{tag}: BUILD FAILED - {r.stderr[:400]}")
        return None
    if verify:
        for vn in [384, 7680]:
            r = subprocess.run(
                [BIN, "--verify", "1", "-b", "1" if vn==384 else "4",
                 "-h", "1" if vn==384 else "8",
                 "-n", str(vn), "-d", "128", "--iters", "10"],
                capture_output=True, text=True, cwd=BASE)
            m_pass = re.search(r'(PASSED|FAILED)', r.stdout)
            m_err = re.search(r'max_abs=([\d.]+)', r.stdout)
            status = m_pass.group(1) if m_pass else "?"
            err = m_err.group(1) if m_err else "?"
            print(f"  verify N={vn}: {status} (max_abs={err})")
            if status == "FAILED":
                return None
    results = []
    for i in range(runs):
        r = subprocess.run(
            [BIN, "--verify", "0", "-b", "4", "-h", "8",
             "-n", str(n), "-d", "128", "--iters", str(iters)],
            capture_output=True, text=True, cwd=BASE)
        m = re.search(r'([\d.]+) TFLOPS', r.stdout)
        if m:
            results.append(float(m.group(1)))
    if results:
        best = max(results[1:]) if len(results) > 1 else results[0]
        print(f"{tag}: {' / '.join(f'{x:.2f}' for x in results)}  best={best:.2f}")
        return best
    print(f"{tag}: NO RESULT")
    return None

# =====================================================================
# Interleave trunc bf16 with tree sum (no dual-add, just interleave for ILP)
# =====================================================================
print("=" * 70)
print("SECTION A: Interleave trunc bf16 with tree row sum")
print("=" * 70)

loop_s = find_line(baseline, r'\.L_n_loop:')
exp_s = find_line(baseline, r'v_exp_f32_e32 v97, v97', loop_s)
mul_l = find_line(baseline, r'v_mul_f32_e32 v136, v137, v136', exp_s)
sum_s = find_line(baseline, r'v_add_f32_e32 v186, v97, v98', mul_l)
sum_e = find_line(baseline, r'v_add_f32_e32 v136, v136, v186', sum_s)
bf16_s = find_line(baseline, r'v_bfe_u32 v186, v97, 16', sum_e)
bf16_e = find_line(baseline, r'v_perm_b32 v151', bf16_s)

# A1: trunc bf16 interleaved with tree sum (no dual constraints)
# Sum tree: 8 level-0 adds → 4 level-1 → 2 level-2 → 1 level-3 → bperm → final
# Trunc bf16: 8 independent v_perm instructions
# Interleave: do perms between sum adds when there's no dependency
mod = baseline[:]
new_block = [
    # Trunc bf16 pair 0,1 (independent of sum)
    "\tv_perm_b32 v144, v98, v97, 0x7060302",
    "\tv_perm_b32 v145, v100, v99, 0x7060302",
    # Sum level 0, first 4 pairs
    "\tv_add_f32_e32 v186, v97, v98",
    "\tv_add_f32_e32 v187, v99, v100",
    # Trunc bf16 pair 2,3
    "\tv_perm_b32 v146, v102, v101, 0x7060302",
    "\tv_perm_b32 v147, v104, v103, 0x7060302",
    # Sum level 0, next 4 pairs
    "\tv_add_f32_e32 v188, v101, v102",
    "\tv_add_f32_e32 v189, v103, v104",
    # Sum level 1 (first 2 merges, dep on level 0)
    "\ts_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)",
    "\tv_add_f32_e32 v186, v186, v187",
    "\tv_add_f32_e32 v188, v188, v189",
    # Trunc bf16 pair 4,5 (fills the dep gap before level 2)
    "\tv_perm_b32 v148, v106, v105, 0x7060302",
    "\tv_perm_b32 v149, v108, v107, 0x7060302",
    # Sum level 0, remaining pairs
    "\tv_add_f32_e32 v187, v105, v106",
    "\tv_add_f32_e32 v189, v107, v108",
    # Sum level 1, merge
    "\ts_delay_alu instid0(VALU_DEP_4)",
    "\tv_add_f32_e32 v186, v186, v188",
    "\tv_add_f32_e32 v187, v187, v189",
    # Trunc bf16 pair 6,7 (fills dep gap)
    "\tv_perm_b32 v150, v110, v109, 0x7060302",
    "\tv_perm_b32 v151, v112, v111, 0x7060302",
    # Sum level 0, last pairs
    "\tv_add_f32_e32 v188, v109, v110",
    "\tv_add_f32_e32 v189, v111, v112",
    # Sum level 1, merge
    "\ts_delay_alu instid0(VALU_DEP_4)",
    "\tv_add_f32_e32 v186, v186, v187",
    "\tv_add_f32_e32 v188, v188, v189",
    # Sum level 2
    "\ts_delay_alu instid0(VALU_DEP_2)",
    "\tv_add_f32_e32 v186, v186, v188",
    # Bpermute
    "\tds_bpermute_b32 v187, v138, v186",
    "\ts_wait_dscnt 0x0",
    "\tv_add_f32_e32 v186, v186, v187",
    "\ts_delay_alu instid0(VALU_DEP_1)",
    "\tv_add_f32_e32 v136, v136, v186",
]
mod = replace_lines(mod, sum_s, bf16_e, new_block)
write_build_bench("A1_interleave_bf16_sum", mod, verify=True)

for n in [1536, 2304, 3840]:
    write_build_bench(f"A1_N{n}", mod, n=n)

print()

# =====================================================================
print("=" * 70)
print("SECTION B: Trunc bf16 only (confirmed winner) at all sizes")
print("=" * 70)

mod = baseline[:]
new_bf16 = []
for dst, lo, hi in [
    (144, 97, 98), (145, 99, 100), (146, 101, 102), (147, 103, 104),
    (148, 105, 106), (149, 107, 108), (150, 109, 110), (151, 111, 112)
]:
    new_bf16.append(f"\tv_perm_b32 v{dst}, v{hi}, v{lo}, 0x7060302")
trunc_mod = replace_lines(mod, bf16_s, bf16_e, new_bf16)
for n in [1536, 2304, 3840, 7680]:
    write_build_bench(f"B1_trunc_N{n}", trunc_mod, n=n)

print()

# =====================================================================
print("=" * 70)
print("SECTION C: Dual score init + trunc bf16 combo (re-verify)")
print("=" * 70)
# The dual_init alone seemed to regress. Let's be careful - retest.
mod = trunc_mod[:]
init_s = find_line(mod, r'v_mov_b32_e32 v97, 0', loop_s)
new_init = [
    "\tv_dual_mov_b32 v97, 0 :: v_dual_mov_b32 v98, 0",
    "\tv_dual_mov_b32 v99, 0 :: v_dual_mov_b32 v100, 0",
    "\tv_dual_mov_b32 v101, 0 :: v_dual_mov_b32 v102, 0",
    "\tv_dual_mov_b32 v103, 0 :: v_dual_mov_b32 v104, 0",
    "\tv_dual_mov_b32 v105, 0 :: v_dual_mov_b32 v106, 0",
    "\tv_dual_mov_b32 v107, 0 :: v_dual_mov_b32 v108, 0",
    "\tv_dual_mov_b32 v109, 0 :: v_dual_mov_b32 v110, 0",
    "\tv_dual_mov_b32 v111, 0 :: v_dual_mov_b32 v112, 0",
]
combo = replace_lines(mod, init_s, init_s + 15, new_init)
write_build_bench("C1_dual_init+trunc", combo, verify=True)
for n in [1536, 2304, 3840]:
    write_build_bench(f"C1_N{n}", combo, n=n)

print()

# =====================================================================
print("=" * 70)
print("SECTION D: Best combo = interleave + dual_init")
print("=" * 70)

# Combine A1 (interleave bf16+sum) with dual score init
mod = baseline[:]
# Apply dual init first
init_s2 = find_line(mod, r'v_mov_b32_e32 v97, 0', loop_s)
mod = replace_lines(mod, init_s2, init_s2 + 15, new_init)
# Re-find landmarks
sum_s2 = find_line(mod, r'v_add_f32_e32 v186, v97, v98')
bf16_e2 = find_line(mod, r'v_perm_b32 v151', sum_s2)
mod = replace_lines(mod, sum_s2, bf16_e2, new_block)
write_build_bench("D1_interleave+dual_init", mod, verify=True)
for n in [1536, 2304, 3840]:
    write_build_bench(f"D1_N{n}", mod, n=n)

print()

# Restore
with open(ASM, 'w') as f:
    f.writelines(baseline)
subprocess.run(["clang++", "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
    "-mcpu=gfx1201", ASM, "-o", HSACO], capture_output=True)
print("Restored baseline.")
