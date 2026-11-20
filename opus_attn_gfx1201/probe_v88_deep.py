#!/usr/bin/env python3
"""Deep optimization probes for v88 kernel.
Explores: load patterns, clause sizes, scheduling modes, nop insertion,
WMMA grouping, V address strategies, bf16 conversion alternatives.
"""
import re, subprocess, os, copy, sys

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

def replace_line(lines, idx, new):
    out = lines[:]
    out[idx] = new + "\n"
    return out

def insert_after(lines, idx, new_lines):
    out = lines[:]
    for i, nl in enumerate(new_lines):
        out.insert(idx + 1 + i, nl + "\n")
    return out

def insert_before(lines, idx, new_lines):
    out = lines[:]
    for i, nl in enumerate(new_lines):
        out.insert(idx + i, nl + "\n")
    return out

def replace_lines(lines, start, end, new_lines):
    out = lines[:start]
    for nl in new_lines:
        out.append(nl + "\n")
    out.extend(lines[end+1:])
    return out

def comment_lines(lines, indices):
    out = lines[:]
    for i in indices:
        if not out[i].startswith(";"):
            out[i] = "; " + out[i].lstrip()
    return out

def comment_pattern_in_range(lines, start, end, patterns):
    out = lines[:]
    for i in range(start, min(end+1, len(out))):
        for p in patterns:
            if re.search(p, out[i]):
                out[i] = "; " + out[i].lstrip()
                break
    return out

def write_build_bench(tag, lines, n=7680, iters=500, runs=3):
    with open(ASM, 'w') as f:
        f.writelines(lines)
    r = subprocess.run(
        ["clang++", "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
         "-mcpu=gfx1201", ASM, "-o", HSACO],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(f"{tag}: BUILD FAILED - {r.stderr[:300]}")
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
    else:
        print(f"{tag}: NO RESULT - {r.stdout[:200]}")
        return None

# Find key landmarks
loop_start = find_line(baseline, r'\.L_n_loop:')
softmax_start = find_line(baseline, r'v_max3_num_f32 v186, v97, v98', loop_start)
v_prefetch = find_line(baseline, r's_clause 0x3', loop_start + 50)  # V prefetch after QKT
salu_precomp = find_line(baseline, r's_mul_i32 s34, 2, s29', v_prefetch)
sub_start = find_line(baseline, r'v_readfirstlane_b32 s32, v188', softmax_start)
exp_start = find_line(baseline, r'v_exp_f32_e32 v97, v97', sub_start)
sum_start = find_line(baseline, r'v_add_f32_e32 v186, v97, v98', exp_start)
sum_end = find_line(baseline, r'v_add_f32_e32 v136, v136, v186', sum_start)
bf16_start = find_line(baseline, r'v_bfe_u32 v186, v97, 16', sum_end)
bf16_end = find_line(baseline, r'v_perm_b32 v151', bf16_start)
rescale_start = find_line(baseline, r'v_cmp_neq_f32', bf16_end)
skip_rescale = find_line(baseline, r'\.L_skip_rescale:', rescale_start)
pv_start = find_line(baseline, r's_or_b32 exec_lo, exec_lo, s31', skip_rescale)
ptr_adv = find_line(baseline, r'v_add_co_u32 v129.*0x2000', pv_start)
loop_branch = find_line(baseline, r's_branch .L_n_loop', ptr_adv)
loop_end = find_line(baseline, r'\.L_n_done:', loop_branch)

print(f"Landmarks:")
print(f"  loop_start={loop_start+1} softmax={softmax_start+1} v_prefetch={v_prefetch+1}")
print(f"  salu_precomp={salu_precomp+1} exp={exp_start+1} sum={sum_start+1}")
print(f"  bf16={bf16_start+1}-{bf16_end+1} rescale={rescale_start+1}-{skip_rescale+1}")
print(f"  pv_start={pv_start+1} ptr_adv={ptr_adv+1} loop_end={loop_end+1}")
print()

# =====================================================================
# SECTION A: Baseline at multiple N sizes
# =====================================================================
print("=" * 70)
print("SECTION A: Baseline reference")
print("=" * 70)
for n in [1536, 3840, 7680]:
    write_build_bench(f"A0_baseline_N{n}", baseline, n=n)
print()

# =====================================================================
# SECTION B: QKT load clause size experiments
# =====================================================================
print("=" * 70)
print("SECTION B: QKT load patterns")
print("=" * 70)

# B1: Break s_clause 0x1 in QKT into individual loads (no clause)
mod = baseline[:]
qkt_clauses = find_all(mod, r's_clause 0x1', loop_start, softmax_start)
for idx in reversed(qkt_clauses):
    mod[idx] = "; " + mod[idx].lstrip()  # comment out s_clause
write_build_bench("B1_no_QKT_clause", mod)

# B2: Use s_clause 0x3 (4 loads) instead of s_clause 0x1 (2 loads) in QKT
# Group K loads into quads instead of pairs
# This is tricky - need to restructure the QKT loop significantly
# For now, just measure breaking clauses

# B3: Add s_nop before each QKT WMMA pair (test if WMMA launch needs spacing)
mod = baseline[:]
qkt_wmmas = find_all(mod, r'v_wmma_f32_16x16x16_bf16', loop_start, softmax_start)
for idx in reversed(qkt_wmmas):
    mod.insert(idx, "\ts_nop 1\n")
write_build_bench("B3_nop_before_QKT_wmma", mod)

print()

# =====================================================================
# SECTION C: PV load patterns
# =====================================================================
print("=" * 70)
print("SECTION C: PV load patterns")
print("=" * 70)

# C1: Break s_clause 0x1 in PV into individual loads
mod = baseline[:]
pv_clauses = find_all(mod, r's_clause 0x1', pv_start, ptr_adv)
for idx in reversed(pv_clauses):
    mod[idx] = "; " + mod[idx].lstrip()
write_build_bench("C1_no_PV_clause", mod)

# C2: Add s_nop before PV WMMAs
mod = baseline[:]
pv_wmmas = find_all(mod, r'v_wmma_f32_16x16x16_bf16', pv_start, ptr_adv)
for idx in reversed(pv_wmmas):
    mod.insert(idx, "\ts_nop 1\n")
write_build_bench("C2_nop_before_PV_wmma", mod)

# C3: Remove V dt=1 prefetch (keep only dt=0), load dt=1 during PV
# The 4 V prefetch loads are the s_clause 0x3 block after last QKT WMMA
# Current: loads dt=0 col0, dt=0 col1+offset, dt=1 col0, dt=1 col1+offset
vpref_clause = find_line(baseline, r's_clause 0x3', softmax_start - 10)
# Comment out dt=1 loads (3rd and 4th in the clause)
mod = baseline[:]
# Change s_clause 0x3 to s_clause 0x1 (only 2 loads)
mod[vpref_clause] = "\ts_clause 0x1\n"
# Comment out the 3rd and 4th loads
vpref_load3 = vpref_clause + 3
vpref_load4 = vpref_clause + 4
mod[vpref_load3] = "; " + mod[vpref_load3].lstrip()
mod[vpref_load4] = "; " + mod[vpref_load4].lstrip()
write_build_bench("C3_V_dt0_only_prefetch", mod)

print()

# =====================================================================
# SECTION D: Scheduling mode experiments
# =====================================================================
print("=" * 70)
print("SECTION D: Scheduling modes")
print("=" * 70)

# D1: round_robin=0 (reference - already know this is worse)
mod = [l.replace('.amdhsa_round_robin_scheduling 1',
                  '.amdhsa_round_robin_scheduling 0') for l in baseline]
write_build_bench("D1_round_robin_0", mod)

# D2: forward_progress=0
mod = [l.replace('.amdhsa_forward_progress 1',
                  '.amdhsa_forward_progress 0') for l in baseline]
write_build_bench("D2_no_fwd_progress", mod)

# D3: round_robin=0 + forward_progress=0
mod = [l.replace('.amdhsa_round_robin_scheduling 1',
                  '.amdhsa_round_robin_scheduling 0')
        .replace('.amdhsa_forward_progress 1',
                  '.amdhsa_forward_progress 0') for l in baseline]
write_build_bench("D3_rr0_fp0", mod)

# D4: inst_pref_size variations
for pref in [1, 4, 12, 16]:
    mod = [l.replace('.amdhsa_inst_pref_size 8',
                      f'.amdhsa_inst_pref_size {pref}') for l in baseline]
    write_build_bench(f"D4_inst_pref_{pref}", mod)

print()

# =====================================================================
# SECTION E: s_delay_alu experiments
# =====================================================================
print("=" * 70)
print("SECTION E: s_delay_alu experiments")
print("=" * 70)

# E1: Remove ALL s_delay_alu in the inner loop
mod = baseline[:]
delay_alus = find_all(mod, r's_delay_alu', loop_start, loop_end)
for idx in reversed(delay_alus):
    mod[idx] = "; " + mod[idx].lstrip()
write_build_bench("E1_no_delay_alu", mod)

# E2: Remove s_delay_alu only in softmax section
mod = baseline[:]
delay_alus = find_all(mod, r's_delay_alu', softmax_start, sum_end)
for idx in reversed(delay_alus):
    mod[idx] = "; " + mod[idx].lstrip()
write_build_bench("E2_no_delay_alu_softmax", mod)

# E3: Remove s_delay_alu only in bf16 section
mod = baseline[:]
delay_alus = find_all(mod, r's_delay_alu', bf16_start, bf16_end)
for idx in reversed(delay_alus):
    mod[idx] = "; " + mod[idx].lstrip()
write_build_bench("E3_no_delay_alu_bf16", mod)

print()

# =====================================================================
# SECTION F: BF16 conversion alternatives
# =====================================================================
print("=" * 70)
print("SECTION F: BF16 conversion alternatives")
print("=" * 70)

# F1: Use v_cvt_pk_bf16_f32 if available on gfx1201
# This would replace 5 instructions (2x v_bfe + 2x v_add3 + v_perm) with 1
# Let's try it for the first pair
mod = baseline[:]
# Replace the first bf16 conversion pair
first_bfe = bf16_start
# Replace: v_bfe_u32 v186, v97, 16, 1 / v_add3_u32 v97, v97, v186, 0x7fff
#           v_bfe_u32 v186, v98, 16, 1 / v_add3_u32 v98, v98, v186, 0x7fff
#           v_perm_b32 v144, v98, v97, 0x7060302
# With: v_cvt_pk_bf16_f32 v144, v97, v98
# Find the lines for the first conversion
perm_line = find_line(mod, r'v_perm_b32 v144', first_bfe)
if perm_line > 0:
    # Replace the 5-instruction block with 1 instruction
    new_lines = ["\tv_cvt_pk_bf16_f32 v144, v97, v98"]
    # Also need to handle delay_alus
    # Find all lines from first_bfe to perm_line
    old_start = first_bfe
    old_end = perm_line
    # Count how many s_delay_alu are between
    delays_in_range = find_all(mod, r's_delay_alu', old_start, old_end)
    # Replace entire block including delays
    mod2 = mod[:old_start]
    for nl in new_lines:
        mod2.append(nl + "\n")
    # Now do the same for all 8 pairs
    # Actually let's just try the full replacement
    mod = baseline[:]
    new_bf16 = []
    for i, (dst, lo, hi) in enumerate([
        (144, 97, 98), (145, 99, 100), (146, 101, 102), (147, 103, 104),
        (148, 105, 106), (149, 107, 108), (150, 109, 110), (151, 111, 112)
    ]):
        new_bf16.append(f"\tv_cvt_pk_bf16_f32 v{dst}, v{lo}, v{hi}")
    mod = replace_lines(mod, bf16_start, bf16_end, new_bf16)
    write_build_bench("F1_cvt_pk_bf16", mod)

# F2: Remove rounding (just shift+pack, skip v_bfe+v_add3)
# Use v_perm_b32 directly to extract upper 16 bits of each float
mod = baseline[:]
new_bf16 = []
for i, (dst, lo, hi) in enumerate([
    (144, 97, 98), (145, 99, 100), (146, 101, 102), (147, 103, 104),
    (148, 105, 106), (149, 107, 108), (150, 109, 110), (151, 111, 112)
]):
    new_bf16.append(f"\tv_perm_b32 v{dst}, v{hi}, v{lo}, 0x7060302")
mod = replace_lines(mod, bf16_start, bf16_end, new_bf16)
write_build_bench("F2_trunc_bf16_no_round", mod)

print()

# =====================================================================
# SECTION G: Interleave softmax with V prefetch
# =====================================================================
print("=" * 70)
print("SECTION G: Load/compute overlap experiments")
print("=" * 70)

# G1: Move K prefetch (at loop bottom) up into the PV section
# Currently K prefetch is: s_clause 0x3 + 4 loads after the loop branch test
# Try moving it earlier, right after ptr_adv
k_pref_clause = find_line(baseline, r's_clause 0x3', ptr_adv)
k_pref_end = find_line(baseline, r's_branch .L_n_loop', k_pref_clause)
if k_pref_clause > 0 and k_pref_end > 0:
    # Extract the K prefetch block
    k_pref_block = baseline[k_pref_clause:k_pref_end]
    # Move it right after the loop counter check (before s_cbranch_scc0)
    branch_line = find_line(baseline, r's_cbranch_scc0 .L_n_done', ptr_adv)
    mod = baseline[:]
    # Remove from old location
    for idx in range(k_pref_clause, k_pref_end):
        mod[idx] = ""
    # Insert before branch
    for i, kl in enumerate(k_pref_block):
        mod.insert(branch_line + i, kl)
    # Clean up empty lines
    mod = [l for l in mod if l != ""]
    write_build_bench("G1_K_prefetch_early", mod)

# G2: Try issuing ALL 4 V dt=0 loads (not just 2) right after QKT
# Currently we issue loads for dt=0 and dt=1 (4 total)
# Try 4 dt=0 loads (covering D tiles 0-3, each with 2 loads = 8 loads)
# This needs more VGPRs or reuse, so skip if too complex

# G3: Move SALU precompute into the softmax section (overlap with exp2)
# Already done - SALU precompute is right after V prefetch, which is during softmax
# Verify it's optimal by moving it earlier

print()

# =====================================================================
# SECTION H: Loop unrolling / N-tile fusion
# =====================================================================
print("=" * 70)
print("SECTION H: Miscellaneous experiments")
print("=" * 70)

# H1: Add s_nop at loop top (alignment)
mod = baseline[:]
mod.insert(loop_start + 1, "\ts_nop 3\n")
write_build_bench("H1_nop_loop_top", mod)

# H2: Add .p2align 6 before loop (64-byte align)
mod = baseline[:]
mod.insert(loop_start, "\t.p2align 6\n")
write_build_bench("H2_align64_loop", mod)

# H3: Add .p2align 7 before loop (128-byte align)
mod = baseline[:]
mod.insert(loop_start, "\t.p2align 7\n")
write_build_bench("H3_align128_loop", mod)

# H4: Reorder score init: interleave s0/s1 movs
mod = baseline[:]
score_init_start = loop_start + 1
# Current: v97=0, v105=0, v98=0, v106=0, ... (already interleaved)
# Try: all s0 first, then all s1
new_init = [
    "\tv_mov_b32_e32 v97, 0", "\tv_mov_b32_e32 v98, 0",
    "\tv_mov_b32_e32 v99, 0", "\tv_mov_b32_e32 v100, 0",
    "\tv_mov_b32_e32 v101, 0", "\tv_mov_b32_e32 v102, 0",
    "\tv_mov_b32_e32 v103, 0", "\tv_mov_b32_e32 v104, 0",
    "\tv_mov_b32_e32 v105, 0", "\tv_mov_b32_e32 v106, 0",
    "\tv_mov_b32_e32 v107, 0", "\tv_mov_b32_e32 v108, 0",
    "\tv_mov_b32_e32 v109, 0", "\tv_mov_b32_e32 v110, 0",
    "\tv_mov_b32_e32 v111, 0", "\tv_mov_b32_e32 v112, 0",
]
mod = replace_lines(mod, score_init_start, score_init_start + 15, new_init)
write_build_bench("H4_score_init_sequential", mod)

# H5: Use v_dual_mov for score init (8 dual ops instead of 16 singles)
new_init_dual = [
    "\tv_dual_mov_b32 v97, 0 :: v_dual_mov_b32 v98, 0",
    "\tv_dual_mov_b32 v99, 0 :: v_dual_mov_b32 v100, 0",
    "\tv_dual_mov_b32 v101, 0 :: v_dual_mov_b32 v102, 0",
    "\tv_dual_mov_b32 v103, 0 :: v_dual_mov_b32 v104, 0",
    "\tv_dual_mov_b32 v105, 0 :: v_dual_mov_b32 v106, 0",
    "\tv_dual_mov_b32 v107, 0 :: v_dual_mov_b32 v108, 0",
    "\tv_dual_mov_b32 v109, 0 :: v_dual_mov_b32 v110, 0",
    "\tv_dual_mov_b32 v111, 0 :: v_dual_mov_b32 v112, 0",
]
mod = replace_lines(baseline, score_init_start, score_init_start + 15, new_init_dual)
write_build_bench("H5_score_init_dual", mod)

# H6: Try different s_wait_loadcnt values for first QKT WMMA pair
# Current: s_wait_loadcnt 0x2 (wait for oldest 2 of 4 prefetch loads)
first_wait = find_line(baseline, r's_wait_loadcnt 0x2', loop_start)
if first_wait > 0 and first_wait < softmax_start:
    # Try s_wait_loadcnt 0x0 (wait for all 4)
    mod = replace_line(baseline, first_wait, "\ts_wait_loadcnt 0x0")
    write_build_bench("H6a_first_wait_0", mod)

    # Try s_wait_loadcnt 0x3 (wait for only oldest 1)
    mod = replace_line(baseline, first_wait, "\ts_wait_loadcnt 0x3")
    write_build_bench("H6b_first_wait_3", mod)

# H7: Reduce VGPR count (try .amdhsa_next_free_vgpr lower)
# Current is 192, try 190 (tighter fit, might improve occupancy)
mod = [l.replace('.amdhsa_next_free_vgpr 192',
                  '.amdhsa_next_free_vgpr 190') for l in baseline]
write_build_bench("H7_vgpr_190", mod)

print()

# =====================================================================
# SECTION I: Combined best-of experiments
# =====================================================================
print("=" * 70)
print("SECTION I: Combined experiments")
print("=" * 70)

# I1: v_cvt_pk_bf16 + v_dual score init (if both work)
mod = baseline[:]
# dual score init
score_init_start2 = find_line(mod, r'v_mov_b32_e32 v97, 0', loop_start)
if score_init_start2 > 0:
    new_init_dual2 = [
        "\tv_dual_mov_b32 v97, 0 :: v_dual_mov_b32 v98, 0",
        "\tv_dual_mov_b32 v99, 0 :: v_dual_mov_b32 v100, 0",
        "\tv_dual_mov_b32 v101, 0 :: v_dual_mov_b32 v102, 0",
        "\tv_dual_mov_b32 v103, 0 :: v_dual_mov_b32 v104, 0",
        "\tv_dual_mov_b32 v105, 0 :: v_dual_mov_b32 v106, 0",
        "\tv_dual_mov_b32 v107, 0 :: v_dual_mov_b32 v108, 0",
        "\tv_dual_mov_b32 v109, 0 :: v_dual_mov_b32 v110, 0",
        "\tv_dual_mov_b32 v111, 0 :: v_dual_mov_b32 v112, 0",
    ]
    mod = replace_lines(mod, score_init_start2, score_init_start2 + 15, new_init_dual2)
# cvt bf16
bf16_s2 = find_line(mod, r'v_bfe_u32 v186, v97, 16', loop_start)
bf16_e2 = find_line(mod, r'v_perm_b32 v151', bf16_s2)
if bf16_s2 > 0 and bf16_e2 > 0:
    new_bf16_2 = []
    for dst, lo, hi in [
        (144, 97, 98), (145, 99, 100), (146, 101, 102), (147, 103, 104),
        (148, 105, 106), (149, 107, 108), (150, 109, 110), (151, 111, 112)
    ]:
        new_bf16_2.append(f"\tv_cvt_pk_bf16_f32 v{dst}, v{lo}, v{hi}")
    mod = replace_lines(mod, bf16_s2, bf16_e2, new_bf16_2)
write_build_bench("I1_dual_init+cvt_bf16", mod)

# I2: trunc bf16 + dual score init
mod = baseline[:]
score_init_start3 = find_line(mod, r'v_mov_b32_e32 v97, 0', loop_start)
if score_init_start3 > 0:
    mod = replace_lines(mod, score_init_start3, score_init_start3 + 15, new_init_dual)
bf16_s3 = find_line(mod, r'v_bfe_u32 v186, v97, 16', loop_start)
bf16_e3 = find_line(mod, r'v_perm_b32 v151', bf16_s3)
if bf16_s3 > 0 and bf16_e3 > 0:
    new_bf16_3 = []
    for dst, lo, hi in [
        (144, 97, 98), (145, 99, 100), (146, 101, 102), (147, 103, 104),
        (148, 105, 106), (149, 107, 108), (150, 109, 110), (151, 111, 112)
    ]:
        new_bf16_3.append(f"\tv_perm_b32 v{dst}, v{hi}, v{lo}, 0x7060302")
    mod = replace_lines(mod, bf16_s3, bf16_e3, new_bf16_3)
write_build_bench("I2_dual_init+trunc_bf16", mod)

# I3: At multiple N sizes for best combo
for n in [1536, 3840]:
    write_build_bench(f"I1_N{n}_dual+cvt_bf16", mod, n=n)

print()

# Restore baseline
with open(ASM, 'w') as f:
    f.writelines(baseline)
subprocess.run(["clang++", "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
    "-mcpu=gfx1201", ASM, "-o", HSACO], capture_output=True)
print("Restored baseline.")
