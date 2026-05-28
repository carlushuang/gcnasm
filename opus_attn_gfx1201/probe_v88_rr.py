#!/usr/bin/env python3
"""Quick probes on round-robin baseline."""
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

def comment_pattern_in_range(lines, start, end, patterns):
    out = lines[:]
    for i in range(start, min(end+1, len(out))):
        for p in patterns:
            if re.search(p, out[i]):
                out[i] = "; " + out[i].lstrip()
                break
    return out

def comment_range(lines, start, end):
    out = lines[:]
    for i in range(start, min(end+1, len(out))):
        out[i] = "; " + out[i].lstrip()
    return out

def write_build_bench(tag, lines, n=7680):
    with open(ASM, 'w') as f:
        f.writelines(lines)
    r = subprocess.run(
        ["clang++", "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
         "-mcpu=gfx1201", ASM, "-o", HSACO],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(f"{tag}: BUILD FAILED - {r.stderr[:200]}")
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
        print(f"{tag}: {' / '.join(f'{x:.2f}' for x in results)}  best={best:.2f}")

loop_start = find_line(baseline, r'\.L_n_loop:')
softmax_start = find_line(baseline, r'v_max3_num_f32 v186, v97, v98', loop_start)
sum_end = find_line(baseline, r'v_add_f32_e32 v136, v136, v186', softmax_start)
bf16_start = find_line(baseline, r'v_bfe_u32 v186, v97, 16', sum_end)
bf16_end = find_line(baseline, r'v_perm_b32 v151', bf16_start)
rescale_start = find_line(baseline, r'v_cmp_neq_f32', bf16_end)
skip_rescale = find_line(baseline, r'\.L_skip_rescale:', rescale_start)
pv_start = find_line(baseline, r's_or_b32 exec_lo, exec_lo, s31', skip_rescale)
ptr_adv = find_line(baseline, r'v_add_co_u32 v129.*0x2000', pv_start)

write_build_bench("P0_baseline_rr", baseline)

# No QKT loads
mod = comment_pattern_in_range(baseline, loop_start, softmax_start-1,
    [r'global_load_b128 v\[1[12][0-9]', r's_clause'])
write_build_bench("P1_no_QKT_loads", mod)

# No PV loads
mod = comment_pattern_in_range(baseline, pv_start, ptr_adv-1,
    [r'global_load_b128', r's_clause'])
write_build_bench("P2_no_PV_loads", mod)

# No QKT WMMAs
mod = comment_pattern_in_range(baseline, loop_start, softmax_start-1,
    [r'v_wmma_f32_16x16x16_bf16 v\[(?:97|105)'])
write_build_bench("P3_no_QKT_wmma", mod)

# No PV WMMAs
mod = comment_pattern_in_range(baseline, pv_start, ptr_adv-1,
    [r'v_wmma_f32_16x16x16_bf16'])
write_build_bench("P4_no_PV_wmma", mod)

# No softmax
mod = comment_range(baseline, softmax_start, sum_end)
write_build_bench("P5_no_softmax", mod)

# No bf16 pack
mod = comment_range(baseline, bf16_start, bf16_end)
write_build_bench("P6_no_bf16pack", mod)

# Also try N=1536 where the improvement was biggest
write_build_bench("P0_N1536_rr", baseline, n=1536)

mod = comment_pattern_in_range(baseline, pv_start, ptr_adv-1,
    [r'global_load_b128', r's_clause'])
write_build_bench("P2_N1536_no_PV_loads", mod, n=1536)

mod = comment_range(baseline, softmax_start, sum_end)
write_build_bench("P5_N1536_no_softmax", mod, n=1536)

mod = comment_range(baseline, bf16_start, bf16_end)
write_build_bench("P6_N1536_no_bf16pack", mod, n=1536)

# Restore
with open(ASM, 'w') as f:
    f.writelines(baseline)
subprocess.run(["clang++", "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
    "-mcpu=gfx1201", ASM, "-o", HSACO], capture_output=True)
print("Restored.")
