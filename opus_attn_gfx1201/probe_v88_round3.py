#!/usr/bin/env python3
"""Round 3: Apply winning optimizations, fix dual-add, try more combos."""
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

def find_all(lines, pattern, start=0, end=None):
    end = end or len(lines)
    return [i for i in range(start, end) if re.search(pattern, lines[i])]

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
        r = subprocess.run(
            [BIN, "--verify", "1", "-b", "1", "-h", "1",
             "-n", "384", "-d", "128", "--iters", "10"],
            capture_output=True, text=True, cwd=BASE)
        m_pass = re.search(r'(PASSED|FAILED)', r.stdout)
        m_err = re.search(r'max_abs=([\d.]+)', r.stdout)
        status = m_pass.group(1) if m_pass else "?"
        err = m_err.group(1) if m_err else "?"
        if status == "FAILED":
            print(f"{tag}: VERIFY {status} (max_abs={err})")
            return None
        print(f"{tag}: verify OK (max_abs={err})")
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

def apply_trunc_bf16(lines):
    """Replace rounding bf16 with truncation (saves 32 instructions)."""
    mod = lines[:]
    loop_s = find_line(mod, r'\.L_n_loop:')
    bf16_s = find_line(mod, r'v_bfe_u32 v186, v97, 16', loop_s)
    bf16_e = find_line(mod, r'v_perm_b32 v151', bf16_s)
    if bf16_s < 0 or bf16_e < 0:
        return mod
    new_bf16 = []
    for dst, lo, hi in [
        (144, 97, 98), (145, 99, 100), (146, 101, 102), (147, 103, 104),
        (148, 105, 106), (149, 107, 108), (150, 109, 110), (151, 111, 112)
    ]:
        new_bf16.append(f"\tv_perm_b32 v{dst}, v{hi}, v{lo}, 0x7060302")
    return replace_lines(mod, bf16_s, bf16_e, new_bf16)

def apply_dual_score_init(lines):
    """Use v_dual_mov for score initialization (8 instructions instead of 16)."""
    mod = lines[:]
    loop_s = find_line(mod, r'\.L_n_loop:')
    init_s = find_line(mod, r'v_mov_b32_e32 v97, 0', loop_s)
    if init_s < 0:
        return mod
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
    return replace_lines(mod, init_s, init_s + 15, new_init)

def apply_dual_add_sum(lines):
    """Use v_dual_add for row sum tree (respecting even/odd constraint)."""
    mod = lines[:]
    loop_s = find_line(mod, r'\.L_n_loop:')
    exp_s = find_line(mod, r'v_exp_f32_e32 v97, v97', loop_s)
    sum_s = find_line(mod, r'v_add_f32_e32 v186, v97, v98', exp_s)
    sum_e = find_line(mod, r'v_add_f32_e32 v136, v136, v186', sum_s)
    if sum_s < 0 or sum_e < 0:
        return mod
    # v_dual constraint: dst regs must be one even, one odd
    # v186=even, v187=odd, v188=even, v189=odd — good pairs: (v186,v187), (v188,v189)
    # v97=odd, v98=even, v99=odd, v100=even — good pairs: (v98,v97), (v100,v99)
    new_sum = [
        # Level 0: 4 dual-adds
        "\tv_dual_add_f32 v186, v97, v98 :: v_dual_add_f32 v187, v99, v100",    # even+odd dst
        "\tv_dual_add_f32 v188, v101, v102 :: v_dual_add_f32 v189, v103, v104",  # even+odd dst
        "\tv_dual_add_f32 v98, v105, v106 :: v_dual_add_f32 v99, v107, v108",    # even+odd dst
        "\tv_dual_add_f32 v100, v109, v110 :: v_dual_add_f32 v97, v111, v112",   # even+odd dst
        # Level 1: 2 dual-adds
        "\ts_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)",
        "\tv_dual_add_f32 v186, v186, v187 :: v_dual_add_f32 v189, v188, v189",  # even+odd
        "\tv_dual_add_f32 v98, v98, v99 :: v_dual_add_f32 v97, v100, v97",       # even+odd
        # Level 2: 1 dual-add
        "\ts_delay_alu instid0(VALU_DEP_2)",
        "\tv_dual_add_f32 v186, v186, v189 :: v_dual_add_f32 v97, v98, v97",     # even+odd
        # Level 3: 1 single add
        "\ts_delay_alu instid0(VALU_DEP_1)",
        "\tv_add_f32_e32 v186, v186, v97",
        # Bpermute
        "\tds_bpermute_b32 v187, v138, v186",
        "\ts_wait_dscnt 0x0",
        "\tv_add_f32_e32 v186, v186, v187",
        "\ts_delay_alu instid0(VALU_DEP_1)",
        "\tv_add_f32_e32 v136, v136, v186",
    ]
    return replace_lines(mod, sum_s, sum_e, new_sum)

def apply_dual_max(lines):
    """Use v_dual for row max tree where possible."""
    # row max uses v_max3_num_f32 which can't be dualed.
    # But the merge levels use v_max_num_f32 which also can't be dualed.
    # No dual opportunity here - v_max3 and v_max aren't in the dual table.
    return lines

# =====================================================================
print("=" * 70)
print("SECTION A: Individual optimizations")
print("=" * 70)

# A1: trunc bf16 only
mod = apply_trunc_bf16(baseline)
write_build_bench("A1_trunc_bf16", mod)

# A2: dual score init only
mod = apply_dual_score_init(baseline)
write_build_bench("A2_dual_init", mod)

# A3: dual-add sum only
mod = apply_dual_add_sum(baseline)
write_build_bench("A3_dual_add_sum", mod, verify=True)

print()

# =====================================================================
print("=" * 70)
print("SECTION B: Combined optimizations")
print("=" * 70)

# B1: trunc_bf16 + dual_init
mod = apply_trunc_bf16(apply_dual_score_init(baseline))
write_build_bench("B1_trunc+dual_init", mod)

# B2: trunc_bf16 + dual_init + dual_add
mod = apply_dual_add_sum(apply_trunc_bf16(apply_dual_score_init(baseline)))
write_build_bench("B2_all_three", mod, verify=True)

# B3: at multiple N sizes
for n in [1536, 2304, 3840]:
    write_build_bench(f"B2_N{n}_all_three", mod, n=n)

print()

# =====================================================================
print("=" * 70)
print("SECTION C: More aggressive instruction reduction")
print("=" * 70)

# C1: Remove s_delay_alu from the trunc bf16 section
# (since trunc has no dependencies - just v_perm with no chain)
mod = apply_trunc_bf16(apply_dual_score_init(baseline))
# The trunc bf16 is just 8 v_perm_b32 with no dependencies between them
# No s_delay_alu should be present since we replaced the whole block
# But let's verify and remove any leftover
write_build_bench("C1_verify_no_stale_delay", mod, verify=True)

# C2: Use v_dual_subrev for the subtraction section
# Current: 8x v_dual_subrev_f32 vN, s32, vN :: v_dual_subrev_f32 vM, s32, vM
# This is already dual! Check if we can do better.
# Actually it's already optimal. Let's check something else.

# C3: Try interleaving the trunc bf16 perm with the row sum
# After exp2 completes, we need both row_sum and bf16 from the same data.
# row_sum reads v97..v112, bf16 trunc reads v97..v112.
# row_sum is a tree reduction, bf16 is 8 independent perms.
# Can we interleave?
# Level 0 of sum: v_dual_add v186, v97, v98 :: v_dual_add v187, v99, v100
# After this, v97 and v98 are consumed by sum. But bf16 needs v97/v98 too!
# So we need to do bf16 FIRST or in parallel.
#
# Actually: bf16 trunc perm doesn't destroy the source regs! It reads them.
# So we can do bf16 perm BEFORE or interleaved with sum.
# If we do bf16 first: v_perm v144, v98, v97 (reads v97,v98 into v144)
# Then sum: v_dual_add v186, v97, v98 (also reads v97,v98)
# Both are reads - no conflict!
# So interleave: bf16_perm(pair0), sum_add(pair0,pair1), bf16_perm(pair1), ...

mod = apply_dual_score_init(baseline)
loop_s = find_line(mod, r'\.L_n_loop:')
exp_s = find_line(mod, r'v_exp_f32_e32 v97, v97', loop_s)
mul_l = find_line(mod, r'v_mul_f32_e32 v136, v137, v136', exp_s)
sum_s = find_line(mod, r'v_add_f32_e32 v186, v97, v98', mul_l)
sum_e = find_line(mod, r'v_add_f32_e32 v136, v136, v186', sum_s)
bf16_s = find_line(mod, r'v_bfe_u32 v186, v97, 16', sum_e)
bf16_e = find_line(mod, r'v_perm_b32 v151', bf16_s)

if sum_s > 0 and bf16_e > 0:
    # Replace sum + bf16 with interleaved version
    new_interleaved = [
        # Trunc bf16 pair 0 + sum level 0 pair 0-1
        "\tv_perm_b32 v144, v98, v97, 0x7060302",    # bf16 pair 0
        "\tv_perm_b32 v145, v100, v99, 0x7060302",   # bf16 pair 1
        "\tv_dual_add_f32 v186, v97, v98 :: v_dual_add_f32 v187, v99, v100",
        "\tv_perm_b32 v146, v102, v101, 0x7060302",  # bf16 pair 2
        "\tv_perm_b32 v147, v104, v103, 0x7060302",  # bf16 pair 3
        "\tv_dual_add_f32 v188, v101, v102 :: v_dual_add_f32 v189, v103, v104",
        "\tv_perm_b32 v148, v106, v105, 0x7060302",  # bf16 pair 4
        "\tv_perm_b32 v149, v108, v107, 0x7060302",  # bf16 pair 5
        "\tv_dual_add_f32 v98, v105, v106 :: v_dual_add_f32 v99, v107, v108",
        "\tv_perm_b32 v150, v110, v109, 0x7060302",  # bf16 pair 6
        "\tv_perm_b32 v151, v112, v111, 0x7060302",  # bf16 pair 7
        "\tv_dual_add_f32 v100, v109, v110 :: v_dual_add_f32 v97, v111, v112",
        # Sum level 1
        "\ts_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)",
        "\tv_dual_add_f32 v186, v186, v187 :: v_dual_add_f32 v189, v188, v189",
        "\tv_dual_add_f32 v98, v98, v99 :: v_dual_add_f32 v97, v100, v97",
        # Sum level 2
        "\ts_delay_alu instid0(VALU_DEP_2)",
        "\tv_dual_add_f32 v186, v186, v189 :: v_dual_add_f32 v97, v98, v97",
        # Sum level 3
        "\ts_delay_alu instid0(VALU_DEP_1)",
        "\tv_add_f32_e32 v186, v186, v97",
        "\tds_bpermute_b32 v187, v138, v186",
        "\ts_wait_dscnt 0x0",
        "\tv_add_f32_e32 v186, v186, v187",
        "\ts_delay_alu instid0(VALU_DEP_1)",
        "\tv_add_f32_e32 v136, v136, v186",
    ]
    mod = replace_lines(mod, sum_s, bf16_e, new_interleaved)
    write_build_bench("C3_interleave_bf16_sum", mod, verify=True)

    for n in [1536, 2304, 3840]:
        write_build_bench(f"C3_N{n}_interleave", mod, n=n)

print()

# =====================================================================
print("=" * 70)
print("SECTION D: v_dual_subrev reduction")
print("=" * 70)
# Already using v_dual_subrev. But what about the exp2 inputs?
# Can we use v_dual for anything else?

# D1: Dual the rescale multiplies are already dual. Check store section.
# The output store section (after loop) has bf16 conversion + stores.
# Not in the hot loop, so skip.

# D2: Try removing the s_wait_alu after v_readfirstlane in subtraction
# Currently: v_readfirstlane_b32 s32, v188 / s_wait_alu 0xfffe / v_dual_subrev
# The s_wait_alu 0xfffe waits for SALU (s32 write). But v_readfirstlane writes
# to SGPR which needs an SALU wait. What if we put more exp2 or other work
# between the readfirstlane and the subtraction?
mod = baseline[:]
sub_rfirst = find_line(mod, r'v_readfirstlane_b32 s32, v188', loop_s)
sub_wait = find_line(mod, r's_wait_alu 0xfffe', sub_rfirst)
# Insert SALU precompute between readfirstlane and wait
# Move the SALU precompute (s34-s39) here
salu_s = find_line(mod, r's_mul_i32 s34, 2, s29', loop_s)
salu_lines = mod[salu_s:salu_s+6]
# Remove from current position
for i in range(6):
    mod[salu_s + i] = ""
mod = [l for l in mod if l != ""]
# Re-find positions
sub_rfirst = find_line(mod, r'v_readfirstlane_b32 s32, v188', loop_s)
sub_wait = find_line(mod, r's_wait_alu 0xfffe', sub_rfirst)
# Insert SALU precomp between readfirstlane and wait
for i, sl in enumerate(salu_lines):
    mod.insert(sub_rfirst + 1 + i, sl)
write_build_bench("D2_salu_between_rfirst", mod)

print()

# =====================================================================
print("=" * 70)
print("SECTION E: Wave occupancy experiments")
print("=" * 70)
# With 192 VGPRs, we get 256/192 = 1 wave per SIMD (1 wave occupancy).
# Actually RDNA4 has 512 VGPRs per SIMD (gfx12), so 512/192 = 2 waves.
# Let's check: reducing to 160 VGPRs would give 512/160 = 3 waves.
# But we actually use v186-v189 as temps, so minimum is 190.

# E1: Check if declaring fewer VGPRs helps (might enable higher occupancy)
# Current: .amdhsa_next_free_vgpr 192
# We use up to v189, so 190 is minimum
mod = [l.replace('.amdhsa_next_free_vgpr 192',
                  '.amdhsa_next_free_vgpr 190') for l in baseline]
write_build_bench("E1_vgpr_190", mod)

# E2: Pad to 256 VGPRs (exactly 2 waves/SIMD)
mod = [l.replace('.amdhsa_next_free_vgpr 192',
                  '.amdhsa_next_free_vgpr 256') for l in baseline]
write_build_bench("E2_vgpr_256", mod)

# E3: Try 196 (just over 192, might round differently)
mod = [l.replace('.amdhsa_next_free_vgpr 192',
                  '.amdhsa_next_free_vgpr 196') for l in baseline]
write_build_bench("E3_vgpr_196", mod)

print()

# Restore
with open(ASM, 'w') as f:
    f.writelines(baseline)
subprocess.run(["clang++", "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
    "-mcpu=gfx1201", ASM, "-o", HSACO], capture_output=True)
print("Restored baseline.")
