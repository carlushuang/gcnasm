#!/usr/bin/env python3
"""Round 2 probes: build on findings from deep probe.
Focus: trunc bf16 correctness, instruction reordering, PV scheduling,
exp2/bf16 overlap, dual_add for row sum, and more aggressive V prefetch.
"""
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
        print(f"{tag}: BUILD FAILED - {r.stderr[:300]}")
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
        print(f"{tag}: VERIFY {status} (max_abs={err})")
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
        print(f"{tag}: NO RESULT")
        return None

loop_start = find_line(baseline, r'\.L_n_loop:')
softmax_start = find_line(baseline, r'v_max3_num_f32 v186, v97, v98', loop_start)
exp_start = find_line(baseline, r'v_exp_f32_e32 v97, v97')
sum_start = find_line(baseline, r'v_add_f32_e32 v186, v97, v98', exp_start)
sum_end = find_line(baseline, r'v_add_f32_e32 v136, v136, v186', sum_start)
bf16_start = find_line(baseline, r'v_bfe_u32 v186, v97, 16', sum_end)
bf16_end = find_line(baseline, r'v_perm_b32 v151', bf16_start)
rescale_start = find_line(baseline, r'v_cmp_neq_f32', bf16_end)
skip_rescale = find_line(baseline, r'\.L_skip_rescale:', rescale_start)
pv_start = find_line(baseline, r's_or_b32 exec_lo, exec_lo, s31', skip_rescale)
ptr_adv = find_line(baseline, r'v_add_co_u32 v129.*0x2000', pv_start)

print("=" * 70)
print("SECTION A: Verify trunc bf16 correctness")
print("=" * 70)

# A1: trunc bf16 with verification
mod = baseline[:]
new_bf16 = []
for dst, lo, hi in [
    (144, 97, 98), (145, 99, 100), (146, 101, 102), (147, 103, 104),
    (148, 105, 106), (149, 107, 108), (150, 109, 110), (151, 111, 112)
]:
    new_bf16.append(f"\tv_perm_b32 v{dst}, v{hi}, v{lo}, 0x7060302")
mod = replace_lines(mod, bf16_start, bf16_end, new_bf16)
write_build_bench("A1_trunc_bf16_verify", mod, n=384, verify=True)

# A1b: Also verify at larger N
write_build_bench("A1b_trunc_bf16_N7680", mod, n=7680, verify=True)

print()

# =====================================================================
print("=" * 70)
print("SECTION B: Interleave exp2 with bf16 conversion")
print("=" * 70)
# exp2 runs on HWTransVALU (parallel with VALU). The bf16 conversion uses
# VALU (v_bfe, v_add3, v_perm). If we interleave them, exp2 and bf16 can
# run concurrently, saving ~32 cycles of bf16 compute.
#
# Current order: exp2(v97..v112) → row_sum → bf16_pack(v97..v112)
# The problem: row_sum needs exp2 results, bf16 needs exp2 results
# So we can't overlap exp2 with bf16 directly.
#
# BUT: we can interleave exp2 with the row_sum tree reduction.
# After the first pair of exp2 results arrive, we can start summing.
# Currently exp2 takes ~4 cycles each on TransVALU, and we issue 16.
# The row sum starts only after all 16 exp2 complete.
#
# Alternative: start bf16 conversion for EARLY exp2 results while LATE
# exp2 are still executing. exp2 latency is ~4 cy per instruction.
# If we issue v_exp_f32 v97..v104, then start bf16 on v97/v98 while
# v105..v112 exp2 are still running.

# B1: Interleave exp2(col1) with bf16_pack(col0)
# Current: 16x exp2, then row_sum, then 8x bf16
# New: 8x exp2(col0), row_sum_partial(col0), bf16(col0),
#      8x exp2(col1), row_sum_partial(col1), bf16(col1)
# This is complex - let's first try the simpler: overlap last exp2 with first bf16

# B1: Move bf16 of first pair right after exp2 of first pair + row sum of first pair
# Actually, the simplest overlap: issue exp2 and bf16 interleaved
# exp2 results for v97 are ready 4 cycles after v_exp_f32 v97
# So if we do: exp2(v97), exp2(v98), exp2(v99), exp2(v100), bf16(v97,v98)
# the exp2 results for v97/v98 should be ready by then
mod = baseline[:]
exp_end = find_line(mod, r'v_exp_f32_e32 v112, v112', exp_start)
lsum_mul = find_line(mod, r'v_mul_f32_e32 v136, v137, v136', exp_end)

# New interleaved exp2 + bf16 pack, after row sum
new_exp_bf16 = [
    # First 8 exp2 (col0)
    "\tv_exp_f32_e32 v97, v97",
    "\tv_exp_f32_e32 v98, v98",
    "\tv_exp_f32_e32 v99, v99",
    "\tv_exp_f32_e32 v100, v100",
    "\tv_exp_f32_e32 v101, v101",
    "\tv_exp_f32_e32 v102, v102",
    "\tv_exp_f32_e32 v103, v103",
    "\tv_exp_f32_e32 v104, v104",
    # Second 8 exp2 (col1) - interleave with bf16 pack of col0
    "\tv_exp_f32_e32 v105, v105",
    "\tv_exp_f32_e32 v106, v106",
    # Start bf16 on v97/v98 (exp2 finished ~8 TransVALU cycles ago)
    "\tv_bfe_u32 v186, v97, 16, 1",
    "\ts_delay_alu instid0(VALU_DEP_1)",
    "\tv_add3_u32 v97, v97, v186, 0x7fff",
    "\tv_exp_f32_e32 v107, v107",
    "\tv_exp_f32_e32 v108, v108",
    "\tv_bfe_u32 v186, v98, 16, 1",
    "\ts_delay_alu instid0(VALU_DEP_1)",
    "\tv_add3_u32 v98, v98, v186, 0x7fff",
    "\ts_delay_alu instid0(VALU_DEP_1)",
    "\tv_perm_b32 v144, v98, v97, 0x7060302",
    "\tv_exp_f32_e32 v109, v109",
    "\tv_exp_f32_e32 v110, v110",
    "\tv_bfe_u32 v186, v99, 16, 1",
    "\ts_delay_alu instid0(VALU_DEP_1)",
    "\tv_add3_u32 v99, v99, v186, 0x7fff",
    "\tv_exp_f32_e32 v111, v111",
    "\tv_exp_f32_e32 v112, v112",
    "\tv_bfe_u32 v186, v100, 16, 1",
    "\ts_delay_alu instid0(VALU_DEP_1)",
    "\tv_add3_u32 v100, v100, v186, 0x7fff",
    "\ts_delay_alu instid0(VALU_DEP_1)",
    "\tv_perm_b32 v145, v100, v99, 0x7060302",
]
# This is getting complex. Let me try a simpler approach first.

# B2: Simple test - move 4 bf16 pairs into the gap between last exp2 and row_sum
# Actually row_sum needs exp2 results, so we can't skip it.
# Let's just try: after row_sum, interleave remaining bf16 with nothing
# (already the current pattern)

# The real opportunity is: row_sum uses v97..v112. After row_sum completes,
# bf16 also uses v97..v112. They're sequential and can't overlap.
# The only overlap opportunity is exp2 (TransVALU) with VALU work.

# B2: Issue exp2 for col1 while doing row_max tree for col0+col1
# Currently: QKT → row_max(all 16) → sub → exp2(all 16) → row_sum → bf16
# New: QKT → row_max(col0 8) → sub(col0 8) → exp2(col0 8) →
#      row_max_merge(col1 into result) → sub(col1 8) → exp2(col1 8) →
#      during col1 exp2: start row_sum(col0) → finish row_sum → bf16

# Too complex for a probe. Let's focus on what we can actually test.

print("=" * 70)
print("SECTION C: Deeper V prefetch (triple buffer)")
print("=" * 70)
# Currently: 4 V loads after QKT (dt=0 col0/col1, dt=1 col0/col1)
# PV starts with wait_loadcnt 0x2, uses dt=0 immediately, then loads dt=2..dt=7
# Try: prefetch dt=0,dt=1,dt=2 (6 loads) — needs 2 more VGPRs for load targets
# Actually we can't easily add more loads without more registers.
# But we CAN try: prefetch dt=0,dt=1 AND also issue dt=2 load at the start of PV
# before waiting for dt=0.

# C1: In PV section, issue dt=2 load before first s_wait_loadcnt
# Current PV dt=0:
#   s_wait_loadcnt 0x2
#   wmma v[1:8], v[113:116], v[144:147], v[1:8]
#   wmma v[1:8], v[117:120], v[148:151], v[1:8]
# After dt=0 WMMAs, v[113:120] are free, issue dt=2 load immediately
# Try: issue dt=2 load BEFORE wait, then wait for dt=0 only
mod = baseline[:]
# Find the first PV wait+wmma
pv_wait = find_line(mod, r's_wait_loadcnt 0x2', pv_start)
if pv_wait > 0:
    # Insert dt=2 V load before the wait (using the same buffer that dt=0 will free)
    # dt=2 offset = s34 (2*s29)
    # But we need to compute the address first...
    # v142 already holds v140+s29 from earlier. dt=2 = v140 + s34
    # Actually s34 = 2*s29 is already computed!
    # Insert: compute dt=2 address and issue load before wait
    new_lines = [
        "\tv_add_co_u32 v142, vcc_lo, v140, s34",
        "\ts_wait_alu 0xfffd",
        "\tv_add_co_ci_u32_e64 v143, null, v141, 0, vcc_lo",
    ]
    for i, nl in enumerate(new_lines):
        mod.insert(pv_wait + i, nl + "\n")
    # Now update the wait count - we have 4 prefetch + 0 new = still 4 in flight
    # Wait needs to change from 0x2 to 0x2 (same, since we haven't issued new loads yet)
    # Actually the address compute doesn't issue loads. We need to issue the load too.
    # But if we issue dt=2 load before wait, then wait count needs to account for 5 loads
    # s_wait_loadcnt 0x3 would mean "wait until at most 3 remain" = 2 done
    # We want dt=0 loads done = oldest 2. With 5 total, wait 0x3 means 2 done.
    # Find the wait line (it moved due to inserts)
    pv_wait2 = find_line(mod, r's_wait_loadcnt 0x2', pv_start)
    # Insert 2 loads for dt=2 before the wait
    load_insert = [
        "\ts_clause 0x1",
        "\tglobal_load_b128 v[113:116], v[142:143], off",  # will be overwritten by dt=0 wmma using same regs... NO
    ]
    # Problem: v[113:116] and v[117:120] are the dt=0 data we're about to use!
    # We can't overwrite them. We need different registers for dt=2.
    # Skip this approach - register pressure is the issue.
    pass

# C2: Instead, try: issue dt=2 load RIGHT AFTER dt=0 WMMAs consume the data
# After the first 2 PV WMMAs, v[113:116] and v[117:120] are free.
# Issue dt=2 load into them immediately, before the next wait.
# This is already somewhat what happens - after each pair of WMMAs, we load next pair.
# The current structure is:
#   wait 2, wmma dt=0 x2, addr+load dt=2 x2, wait 2, wmma dt=1 x2, addr+load dt=3 x2, ...
# So it's already doing progressive loads. The V prefetch for dt=0+dt=1 means
# they start 100+ cycles early during softmax.

# C3: Try issuing dt=2 and dt=3 loads during the bf16/rescale section
# After bf16 pack completes, we have v[113:128] free (they held the prefetch data
# but we haven't used them for PV yet... actually they're used at pv_start!)
# So v[113:128] hold V dt=0 and dt=1 data during bf16 section.
# After PV dt=0 WMMAs consume v[113:116], v[117:120], those are free for dt=2.
# After PV dt=1 WMMAs consume v[121:124], v[125:128], those are free for dt=3.
# The current code already does this! Let me verify...

# Actually let me trace the PV section more carefully.
# The issue is: can we start dt=2/dt=3 loads EARLIER?
# They're issued right after dt=0/dt=1 WMMAs respectively.
# Could we issue them during rescale (before PV starts)?
# Problem: v[113:128] still hold dt=0/dt=1 V data needed by PV WMMAs.
# So NO, we can't issue more loads until we consume the prefetch data.

# Let's try something different: use different VGPRs for dt=2 loads.

print("(C: skipped - register pressure prevents deeper V prefetch)")
print()

print("=" * 70)
print("SECTION D: exp2/VALU overlap via instruction reordering")
print("=" * 70)

# exp2 runs on TransVALU, parallel with regular VALU.
# Can we overlap exp2 with the SALU precompute that's already there?
# SALU runs on a separate pipe too. Let's see if we can stuff more
# VALU work between exp2 issues.

# D1: Interleave v_mul_f32 v136 (l_sum rescale) between exp2 instructions
# Currently: exp2(v97..v112), then v_mul_f32 v136
# Try: exp2(v97..v108), v_mul_f32 v136, exp2(v109..v112)
mod = baseline[:]
mul_l = find_line(mod, r'v_mul_f32_e32 v136, v137, v136', exp_start)
exp_last = find_line(mod, r'v_exp_f32_e32 v112', exp_start)
if mul_l > 0 and exp_last > 0:
    # Move mul_l line to between exp2(v108) and exp2(v109)
    # First find exp2 v108 and v109
    exp108 = find_line(mod, r'v_exp_f32_e32 v108', exp_start)
    exp109 = find_line(mod, r'v_exp_f32_e32 v109', exp_start)
    if exp108 > 0 and exp109 > 0:
        # Get the mul+waits
        # Need to find the s_wait_alu before mul
        wait_before_mul = mul_l - 1
        wait2_before = mul_l - 2  # might be another s_wait_alu
        mul_line_text = mod[mul_l]
        wait_line_text = mod[wait_before_mul] if re.search(r's_wait_alu', mod[wait_before_mul]) else None

        # Remove mul + its wait from current position
        lines_to_move = []
        if wait_line_text:
            lines_to_move = [mod[wait_before_mul], mod[mul_l]]
            mod[wait_before_mul] = ""
            mod[mul_l] = ""
        else:
            lines_to_move = [mod[mul_l]]
            mod[mul_l] = ""

        # Insert between exp108 and exp109
        # Re-find positions after removal
        mod = [l for l in mod if l != ""]
        exp108_new = find_line(mod, r'v_exp_f32_e32 v108', exp_start)
        for i, l in enumerate(lines_to_move):
            mod.insert(exp108_new + 1 + i, l)

write_build_bench("D1_mul_between_exp2", mod)

# D2: Move SALU precompute into exp2 section (if not already there)
# SALU precompute is already right after V prefetch which is during exp2 section
# Let's verify by checking positions
print(f"  exp2 range: {exp_start+1}-{exp_start+16}")
salu = find_line(baseline, r's_mul_i32 s34', loop_start)
print(f"  SALU precomp at: {salu+1}")

# D3: Add v_nop between exp2 and row_sum to let exp2 drain
mod = baseline[:]
exp_last2 = find_line(mod, r'v_exp_f32_e32 v112', exp_start)
waits_after_exp = find_all(mod, r's_wait_alu', exp_last2, exp_last2 + 5)
# Insert nops after last exp2 + wait
insert_pos = waits_after_exp[-1] + 1 if waits_after_exp else exp_last2 + 1
mod.insert(insert_pos, "\ts_nop 3\n")
mod.insert(insert_pos, "\ts_nop 3\n")
write_build_bench("D3_nop_after_exp2", mod)

print()

print("=" * 70)
print("SECTION E: PV n_pipe=2 via D-tile interleave (different approach)")
print("=" * 70)
# Previous attempt with s_clause 0x3 failed. Try: no clause at all,
# issue individual loads with maximal distance from WMMA consumption.
#
# Current PV pattern (dt=0):
#   s_wait_loadcnt 0x2  ; wait for V dt=0 (2 loads)
#   wmma v[1:8], v[113:116], P0, v[1:8]    ; dt=0 col0
#   wmma v[1:8], v[117:120], P1, v[1:8]    ; dt=0 col1
#   addr_dt=2 ; load_dt=2 x2
#   s_wait_loadcnt 0x2  ; wait for V dt=1 (2 loads, was prefetched)
#   wmma v[9:16], v[121:124], P0, v[9:16]  ; dt=1 col0
#   wmma v[9:16], v[125:128], P1, v[9:16]  ; dt=1 col1
#
# n_pipe issue: both WMMAs in each pair write same accumulator (v[1:8] or v[9:16])
# To get n_pipe=2, alternate accumulators:
#   wmma v[1:8], ..., v[1:8]     ; dt=0 col0
#   wmma v[9:16], ..., v[9:16]   ; dt=1 col0 ← different accum!
#   wmma v[1:8], ..., v[1:8]     ; dt=0 col1
#   wmma v[9:16], ..., v[9:16]   ; dt=1 col1
#
# But this needs BOTH dt=0 and dt=1 V data available before first WMMA.
# With current 4-load prefetch, both are available! The wait is s_wait_loadcnt 0x2
# which waits for oldest 2 of 4. But we need all 4.
#
# Try: s_wait_loadcnt 0x0 (all 4 ready), then interleave dt=0/dt=1 WMMAs

# E1: Full PV rewrite with interleaved D-tiles for dt=0..dt=7
# This is the big one. Let me construct it carefully.
# For each pair of D-tiles (dt, dt+1):
#   - V data for dt and dt+1 must be available
#   - Issue: wmma(dt,col0), wmma(dt+1,col0), wmma(dt,col1), wmma(dt+1,col1)
#   - After last WMMA, load V for dt+2 and dt+3

# The PV section currently has 8 D-tile pairs. Let me rewrite just the first
# pair to test the concept, keeping the rest the same.
mod = baseline[:]

# Find current PV dt=0 block: s_wait_loadcnt 0x2, 2 WMMAs
pv_wait = find_line(mod, r's_wait_loadcnt 0x2', pv_start)
# Find current PV dt=1 block
pv_dt1_wait = find_line(mod, r's_wait_loadcnt 0x2', pv_wait + 1)

if pv_wait > 0 and pv_dt1_wait > 0:
    # Current dt=0:
    #   s_wait_loadcnt 0x2
    #   wmma v[1:8], v[113:116], v[144:147], v[1:8]
    #   wmma v[1:8], v[117:120], v[148:151], v[1:8]
    #   addr_compute + load dt=2 x2
    # Current dt=1:
    #   s_wait_loadcnt 0x2
    #   wmma v[9:16], v[121:124], v[144:147], v[9:16]
    #   wmma v[9:16], v[125:128], v[148:151], v[9:16]
    #   addr_compute + load dt=3 x2

    # Find end of dt=1 block (next s_wait or WMMA with different accum)
    dt1_wmma1 = pv_dt1_wait + 1  # first wmma of dt=1
    dt1_wmma2 = pv_dt1_wait + 2  # second wmma of dt=1
    # After dt=1 WMMAs: addr compute for dt=3 + loads
    dt1_addr = find_line(mod, r'v_add_co_u32 v142', pv_dt1_wait)
    dt1_clause = find_line(mod, r's_clause 0x1', dt1_addr)
    dt1_load2 = dt1_clause + 2  # end of dt=1 block (2 loads after clause)

    # New interleaved dt=0+dt=1:
    new_pv_01 = [
        "\ts_wait_loadcnt 0x0",  # wait for all 4 prefetch loads
        # Interleaved WMMAs: dt=0 col0, dt=1 col0, dt=0 col1, dt=1 col1
        "\tv_wmma_f32_16x16x16_bf16 v[1:8], v[113:116], v[144:147], v[1:8]",    # dt0 P0
        "\tv_wmma_f32_16x16x16_bf16 v[9:16], v[121:124], v[144:147], v[9:16]",  # dt1 P0
        "\tv_wmma_f32_16x16x16_bf16 v[1:8], v[117:120], v[148:151], v[1:8]",    # dt0 P1
        "\tv_wmma_f32_16x16x16_bf16 v[9:16], v[125:128], v[148:151], v[9:16]",  # dt1 P1
        # Now v[113:128] are free, load dt=2 and dt=3
        "\tv_add_co_u32 v142, vcc_lo, v140, s34",
        "\ts_wait_alu 0xfffd",
        "\tv_add_co_ci_u32_e64 v143, null, v141, 0, vcc_lo",
        "\ts_clause 0x3",
        "\tglobal_load_b128 v[113:116], v[142:143], off",
        "\tglobal_load_b128 v[117:120], v[142:143], off offset:32",
    ]
    # Need dt=3 address too
    new_pv_01 += [
        "\tv_add_co_u32 v142, vcc_lo, v140, s35",
        "\ts_wait_alu 0xfffd",
        "\tv_add_co_ci_u32_e64 v143, null, v141, 0, vcc_lo",
        "\tglobal_load_b128 v[121:124], v[142:143], off",
        "\tglobal_load_b128 v[125:128], v[142:143], off offset:32",
    ]
    # Wait for dt=2 ready for dt=2+dt=3 WMMAs
    # Hmm, but s_clause 0x3 requires 4 consecutive loads, and we have
    # addr compute in between. Let me fix: issue loads separately.

    new_pv_01 = [
        "\ts_wait_loadcnt 0x0",
        "\tv_wmma_f32_16x16x16_bf16 v[1:8], v[113:116], v[144:147], v[1:8]",
        "\tv_wmma_f32_16x16x16_bf16 v[9:16], v[121:124], v[144:147], v[9:16]",
        "\tv_wmma_f32_16x16x16_bf16 v[1:8], v[117:120], v[148:151], v[1:8]",
        "\tv_wmma_f32_16x16x16_bf16 v[9:16], v[125:128], v[148:151], v[9:16]",
        # Load dt=2 into v[113:120]
        "\tv_add_co_u32 v142, vcc_lo, v140, s34",
        "\ts_wait_alu 0xfffd",
        "\tv_add_co_ci_u32_e64 v143, null, v141, 0, vcc_lo",
        "\ts_clause 0x1",
        "\tglobal_load_b128 v[113:116], v[142:143], off",
        "\tglobal_load_b128 v[117:120], v[142:143], off offset:32",
        # Load dt=3 into v[121:128]
        "\tv_add_co_u32 v142, vcc_lo, v140, s35",
        "\ts_wait_alu 0xfffd",
        "\tv_add_co_ci_u32_e64 v143, null, v141, 0, vcc_lo",
        "\ts_clause 0x1",
        "\tglobal_load_b128 v[121:124], v[142:143], off",
        "\tglobal_load_b128 v[125:128], v[142:143], off offset:32",
    ]

    # Replace from pv_wait to dt1_load2 with new interleaved block
    mod = replace_lines(mod, pv_wait, dt1_load2, new_pv_01)

    # Now need to continue with dt=2+dt=3 interleaved, then dt=4+dt=5, dt=6+dt=7
    # For now just update dt=2+dt=3 to be interleaved too
    # Find current dt=2 section in mod (after our insert)
    next_wait = find_line(mod, r's_wait_loadcnt', pv_wait + len(new_pv_01))
    if next_wait > 0:
        # dt=2 currently:
        #   s_wait_loadcnt 0x2
        #   wmma v[17:24], v[113:116], P0, v[17:24]
        #   wmma v[17:24], v[117:120], P1, v[17:24]
        #   addr + load dt=4 x2
        # dt=3:
        #   s_wait_loadcnt 0x2
        #   wmma v[25:32], v[121:124], P0, v[25:32]
        #   wmma v[25:32], v[125:128], P1, v[25:32]
        #   addr + load dt=5 x2

        # Find dt=3 wait
        dt3_wait = find_line(mod, r's_wait_loadcnt', next_wait + 1)
        if dt3_wait > 0:
            dt3_addr = find_line(mod, r'v_add_co_u32 v142', dt3_wait)
            if dt3_addr > 0:
                dt3_clause = find_line(mod, r's_clause 0x1', dt3_addr)
                dt3_load_end = dt3_clause + 2 if dt3_clause > 0 else dt3_addr + 5

                new_pv_23 = [
                    "\ts_wait_loadcnt 0x0",
                    "\tv_wmma_f32_16x16x16_bf16 v[17:24], v[113:116], v[144:147], v[17:24]",
                    "\tv_wmma_f32_16x16x16_bf16 v[25:32], v[121:124], v[144:147], v[25:32]",
                    "\tv_wmma_f32_16x16x16_bf16 v[17:24], v[117:120], v[148:151], v[17:24]",
                    "\tv_wmma_f32_16x16x16_bf16 v[25:32], v[125:128], v[148:151], v[25:32]",
                    # Load dt=4
                    "\tv_add_co_u32 v142, vcc_lo, v140, s36",
                    "\ts_wait_alu 0xfffd",
                    "\tv_add_co_ci_u32_e64 v143, null, v141, 0, vcc_lo",
                    "\ts_clause 0x1",
                    "\tglobal_load_b128 v[113:116], v[142:143], off",
                    "\tglobal_load_b128 v[117:120], v[142:143], off offset:32",
                    # Load dt=5
                    "\tv_add_co_u32 v142, vcc_lo, v140, s37",
                    "\ts_wait_alu 0xfffd",
                    "\tv_add_co_ci_u32_e64 v143, null, v141, 0, vcc_lo",
                    "\ts_clause 0x1",
                    "\tglobal_load_b128 v[121:124], v[142:143], off",
                    "\tglobal_load_b128 v[125:128], v[142:143], off offset:32",
                ]
                mod = replace_lines(mod, next_wait, dt3_load_end, new_pv_23)

                # Similarly do dt=4+dt=5 and dt=6+dt=7
                next2_wait = find_line(mod, r's_wait_loadcnt', next_wait + len(new_pv_23))
                if next2_wait > 0:
                    dt5_wait = find_line(mod, r's_wait_loadcnt', next2_wait + 1)
                    if dt5_wait > 0:
                        dt5_addr = find_line(mod, r'v_add_co_u32 v142', dt5_wait)
                        if dt5_addr > 0:
                            dt5_clause = find_line(mod, r's_clause 0x1', dt5_addr)
                            dt5_end = dt5_clause + 2 if dt5_clause > 0 else dt5_addr + 5

                            new_pv_45 = [
                                "\ts_wait_loadcnt 0x0",
                                "\tv_wmma_f32_16x16x16_bf16 v[33:40], v[113:116], v[144:147], v[33:40]",
                                "\tv_wmma_f32_16x16x16_bf16 v[41:48], v[121:124], v[144:147], v[41:48]",
                                "\tv_wmma_f32_16x16x16_bf16 v[33:40], v[117:120], v[148:151], v[33:40]",
                                "\tv_wmma_f32_16x16x16_bf16 v[41:48], v[125:128], v[148:151], v[41:48]",
                                # Load dt=6
                                "\tv_add_co_u32 v142, vcc_lo, v140, s38",
                                "\ts_wait_alu 0xfffd",
                                "\tv_add_co_ci_u32_e64 v143, null, v141, 0, vcc_lo",
                                "\ts_clause 0x1",
                                "\tglobal_load_b128 v[113:116], v[142:143], off",
                                "\tglobal_load_b128 v[117:120], v[142:143], off offset:32",
                                # Load dt=7
                                "\tv_add_co_u32 v142, vcc_lo, v140, s39",
                                "\ts_wait_alu 0xfffd",
                                "\tv_add_co_ci_u32_e64 v143, null, v141, 0, vcc_lo",
                                "\ts_clause 0x1",
                                "\tglobal_load_b128 v[121:124], v[142:143], off",
                                "\tglobal_load_b128 v[125:128], v[142:143], off offset:32",
                            ]
                            mod = replace_lines(mod, next2_wait, dt5_end, new_pv_45)

                            # Finally dt=6+dt=7 (last pair, no more loads needed)
                            next3_wait = find_line(mod, r's_wait_loadcnt', next2_wait + len(new_pv_45))
                            if next3_wait > 0:
                                dt7_wait = find_line(mod, r's_wait_loadcnt', next3_wait + 1)
                                if dt7_wait > 0:
                                    # Find end of dt=7 block
                                    dt7_wmma2 = dt7_wait + 2
                                    new_pv_67 = [
                                        "\ts_wait_loadcnt 0x0",
                                        "\tv_wmma_f32_16x16x16_bf16 v[49:56], v[113:116], v[144:147], v[49:56]",
                                        "\tv_wmma_f32_16x16x16_bf16 v[57:64], v[121:124], v[144:147], v[57:64]",
                                        "\tv_wmma_f32_16x16x16_bf16 v[49:56], v[117:120], v[148:151], v[49:56]",
                                        "\tv_wmma_f32_16x16x16_bf16 v[57:64], v[125:128], v[148:151], v[57:64]",
                                    ]
                                    mod = replace_lines(mod, next3_wait, dt7_wmma2, new_pv_67)

    write_build_bench("E1_PV_interleave_dtiles", mod, verify=True)
    for n in [1536, 3840]:
        write_build_bench(f"E1_N{n}_interleave", mod, n=n)

print()

print("=" * 70)
print("SECTION F: Dual-add for row sum")
print("=" * 70)
# Currently row sum uses tree reduction with v_add_f32_e32 (1 per cycle)
# Can we use v_dual_add for 2 adds per cycle?
mod = baseline[:]
sum_s = find_line(mod, r'v_add_f32_e32 v186, v97, v98', exp_start)
sum_e = find_line(mod, r'v_add_f32_e32 v136, v136, v186', sum_s)
if sum_s > 0 and sum_e > 0:
    # Current tree sum (15 instructions):
    # Level 0: 8 pairwise adds (v186..v189 as temps)
    # Level 1: 4 merge adds
    # Level 2: 2 merge adds
    # Level 3: 1 final add + bpermute + cross-half add + final accumulate
    #
    # With v_dual_add_f32:
    # Level 0: v_dual_add v186, v97, v98 :: v_dual_add v187, v99, v100
    #          v_dual_add v188, v101, v102 :: v_dual_add v189, v103, v104
    #          v_dual_add v97, v105, v106 :: v_dual_add v98, v107, v108
    #          v_dual_add v99, v109, v110 :: v_dual_add v100, v111, v112
    # Level 1: v_dual_add v186, v186, v187 :: v_dual_add v188, v188, v189
    #          v_dual_add v97, v97, v98 :: v_dual_add v99, v99, v100
    # Level 2: v_dual_add v186, v186, v188 :: v_dual_add v97, v97, v99
    # Level 3: v_add_f32 v186, v186, v97
    #          bpermute, cross-half add, accumulate
    #
    # 4 + 2 + 1 + 1 = 8 dual + 1 single = 9 instructions vs 15 (but same dep depth)
    # Actually v_dual_add is 1 cycle same as single, so savings are pure instruction count.
    # With RDNA4 issue rate of 1 VALU/cycle, 4 dual ops save 4 cycles.

    new_sum = [
        # l_sum rescale (v_mul was before sum, keep it)
        # Level 0: 4 dual-adds (8 pairwise sums in 4 cycles)
        "\tv_dual_add_f32 v186, v97, v98 :: v_dual_add_f32 v187, v99, v100",
        "\tv_dual_add_f32 v188, v101, v102 :: v_dual_add_f32 v189, v103, v104",
        "\tv_dual_add_f32 v97, v105, v106 :: v_dual_add_f32 v98, v107, v108",
        "\tv_dual_add_f32 v99, v109, v110 :: v_dual_add_f32 v100, v111, v112",
        # Level 1: 2 dual-adds
        "\ts_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)",
        "\tv_dual_add_f32 v186, v186, v187 :: v_dual_add_f32 v188, v188, v189",
        "\tv_dual_add_f32 v97, v97, v98 :: v_dual_add_f32 v99, v99, v100",
        # Level 2: 1 dual-add
        "\ts_delay_alu instid0(VALU_DEP_2)",
        "\tv_dual_add_f32 v186, v186, v188 :: v_dual_add_f32 v97, v97, v99",
        # Level 3: 1 add
        "\ts_delay_alu instid0(VALU_DEP_1)",
        "\tv_add_f32_e32 v186, v186, v97",
        # Bpermute
        "\tds_bpermute_b32 v187, v138, v186",
        "\ts_wait_dscnt 0x0",
        "\tv_add_f32_e32 v186, v186, v187",
        "\ts_delay_alu instid0(VALU_DEP_1)",
        "\tv_add_f32_e32 v136, v136, v186",
    ]
    mod = replace_lines(mod, sum_s, sum_e, new_sum)
    write_build_bench("F1_dual_add_sum", mod)
    # Verify
    write_build_bench("F1_dual_add_verify", mod, n=384, verify=True)

print()

# Restore
with open(ASM, 'w') as f:
    f.writelines(baseline)
subprocess.run(["clang++", "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
    "-mcpu=gfx1201", ASM, "-o", HSACO], capture_output=True)
print("Restored baseline.")
