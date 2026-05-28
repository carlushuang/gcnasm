#!/usr/bin/env python3
"""Probe v88 kernel by commenting out sections to find critical path."""
import re, subprocess, sys, os

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

def comment_range(lines, start, end, pattern=None):
    out = lines[:]
    for i in range(start, end+1):
        if pattern is None or re.search(pattern, out[i]):
            out[i] = "; " + out[i].lstrip()
    return out

def comment_pattern_in_range(lines, start, end, patterns):
    out = lines[:]
    for i in range(start, end+1):
        for p in patterns:
            if re.search(p, out[i]):
                out[i] = "; " + out[i].lstrip()
                break
    return out

def write_build_bench(tag, lines, n=7680):
    with open(ASM, 'w') as f:
        f.writelines(lines)
    r = subprocess.run(
        ["clang++", "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
         "-mcpu=gfx1201", ASM, "-o", HSACO],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(f"{tag}: BUILD FAILED")
        print(r.stderr[:500])
        return
    results = []
    for i in range(3):
        r = subprocess.run(
            [BIN, "--verify", "0", "-b", "4", "-h", "8",
             "-n", str(n), "-d", "128", "--iters", "500"],
            capture_output=True, text=True, cwd=BASE)
        m = re.search(r'([\d.]+) TFLOPS', r.stdout)
        if m:
            results.append(float(m.group(1)))
    if results:
        best = max(results[1:]) if len(results) > 1 else results[0]
        print(f"{tag}: {' / '.join(f'{x:.2f}' for x in results)} TFLOPS  (best={best:.2f})")
    else:
        print(f"{tag}: NO RESULT")
        print(r.stdout[:300])

# Find key section boundaries
loop_start = find_line(baseline, r'\.L_n_loop:')
loop_end = find_line(baseline, r'\.L_n_done:')
softmax_start = find_line(baseline, r'v_max3_num_f32 v186, v97, v98', loop_start)
sub_start = find_line(baseline, r'v_readfirstlane_b32 s32, v188', softmax_start)
exp_start = find_line(baseline, r'v_exp_f32_e32 v97, v97', sub_start)
sum_start = find_line(baseline, r'v_add_f32_e32 v186, v97, v98', exp_start)
bf16_start = find_line(baseline, r'v_bfe_u32 v186, v97, 16', sum_start)
bf16_end = find_line(baseline, r'v_perm_b32 v151', bf16_start)
rescale_start = find_line(baseline, r'v_cmp_neq_f32', bf16_end)
skip_rescale = find_line(baseline, r'\.L_skip_rescale:', rescale_start)
pv_start = find_line(baseline, r's_or_b32 exec_lo, exec_lo, s31', skip_rescale)
ptr_adv = find_line(baseline, r'v_add_co_u32 v129.*0x2000', pv_start)
# V prefetch loads (after QKT, before softmax)
vprefetch_start = find_line(baseline, r'global_load_b128 v\[113:116\], v\[140:141\]', loop_start)

print(f"Section map:")
print(f"  loop: {loop_start+1}-{loop_end+1}")
print(f"  softmax_max: {softmax_start+1}")
print(f"  subtraction: {sub_start+1}")
print(f"  exp2: {exp_start+1}")
print(f"  row_sum: {sum_start+1}")
print(f"  bf16_pack: {bf16_start+1}-{bf16_end+1}")
print(f"  rescale: {rescale_start+1}-{skip_rescale+1}")
print(f"  PV: {pv_start+1}-{ptr_adv+1}")
print(f"  V_prefetch: {vprefetch_start+1}")
print()

# P0: Baseline
print("=" * 60)
write_build_bench("P0_baseline", baseline)

# P1: No QKT global_loads (comment out K loads between loop_start and softmax)
mod = comment_pattern_in_range(baseline, loop_start, softmax_start-1,
    [r'global_load_b128 v\[1[12][0-9]', r's_clause'])
write_build_bench("P1_no_QKT_loads", mod)

# P2: No PV global_loads
mod = comment_pattern_in_range(baseline, pv_start, ptr_adv-1,
    [r'global_load_b128', r's_clause'])
write_build_bench("P2_no_PV_loads", mod)

# P3: No QKT WMMAs
mod = comment_pattern_in_range(baseline, loop_start, softmax_start-1,
    [r'v_wmma_f32_16x16x16_bf16 v\[(?:97|105)'])
write_build_bench("P3_no_QKT_wmma", mod)

# P4: No PV WMMAs
mod = comment_pattern_in_range(baseline, pv_start, ptr_adv-1,
    [r'v_wmma_f32_16x16x16_bf16'])
write_build_bench("P4_no_PV_wmma", mod)

# P5: No softmax (max+sub+exp+sum, keep bf16 pack)
sum_end = find_line(baseline, r'v_add_f32_e32 v136, v136, v186', sum_start)
mod = comment_range(baseline, softmax_start, sum_end)
write_build_bench("P5_no_softmax", mod)

# P6: No BF16 pack only
mod = comment_range(baseline, bf16_start, bf16_end)
write_build_bench("P6_no_bf16pack", mod)

# P7: No conditional rescale
mod = comment_range(baseline, rescale_start, skip_rescale-1)
write_build_bench("P7_no_rescale", mod)

# P8: No V dt=0 prefetch
mod = comment_pattern_in_range(baseline, vprefetch_start, vprefetch_start+2,
    [r'global_load_b128'])
write_build_bench("P8_no_V_prefetch", mod)

# P9: No next-tile K prefetch
k_pref = find_line(baseline, r's_clause 0x3', ptr_adv)
k_pref_end = find_line(baseline, r's_branch .L_n_loop', k_pref)
mod = comment_pattern_in_range(baseline, k_pref, k_pref_end-1,
    [r'global_load_b128', r's_clause'])
write_build_bench("P9_no_K_prefetch", mod)

# P10: No QKT loads AND no QKT WMMAs (pure softmax+PV)
mod = comment_pattern_in_range(baseline, loop_start, softmax_start-1,
    [r'global_load_b128', r's_clause', r'v_wmma_f32_16x16x16_bf16', r's_wait_loadcnt'])
write_build_bench("P10_no_QKT_all", mod)

# P11: No PV loads AND no PV WMMAs (pure QKT+softmax)
mod = comment_pattern_in_range(baseline, pv_start, ptr_adv-1,
    [r'global_load_b128', r's_clause', r'v_wmma_f32_16x16x16_bf16', r's_wait_loadcnt'])
write_build_bench("P11_no_PV_all", mod)

# Scheduling mode probes
# P12: round_robin_scheduling = 1
mod = [l.replace('.amdhsa_round_robin_scheduling 0',
                  '.amdhsa_round_robin_scheduling 1') for l in baseline]
write_build_bench("P12_round_robin", mod)

# P13: inst_pref_size = 1
mod = [l.replace('.amdhsa_inst_pref_size 8',
                  '.amdhsa_inst_pref_size 1') for l in baseline]
write_build_bench("P13_inst_pref_1", mod)

# P14: inst_pref_size = 16
mod = [l.replace('.amdhsa_inst_pref_size 8',
                  '.amdhsa_inst_pref_size 16') for l in baseline]
write_build_bench("P14_inst_pref_16", mod)

# P15: forward_progress = 0
mod = [l.replace('.amdhsa_forward_progress 1',
                  '.amdhsa_forward_progress 0') for l in baseline]
write_build_bench("P15_no_fwd_progress", mod)

# Restore baseline
with open(ASM, 'w') as f:
    f.writelines(baseline)
subprocess.run(["clang++", "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
    "-mcpu=gfx1201", ASM, "-o", HSACO], capture_output=True)
print("\nRestored baseline.")
