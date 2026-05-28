#!/usr/bin/env python3
"""Generate v80 ASM kernel: BLOCK_M=384, BLOCK_N=64, D=128 for gfx1201."""

import sys

BLOCK_M = 384
BLOCK_N = 64
D = 128
W_K = 16
DK = D // W_K  # 8
NUM_WAVES = BLOCK_M // 16  # 24
BLOCK_SIZE = NUM_WAVES * 32  # 768

FUNC_NAME = "opus_attn_gfx1201_kernel_v80"
# Debug: skip QKT/softmax, hardcode P=1.0 bf16, only 1 tile
DEBUG_UNIFORM_P = False
DEBUG_WRITE_DIAG = False  # Write diagnostics instead of attention output

# --- SGPR allocation (must respect 4-alignment for s_load_b128) ---
# s[0:1]   = kernarg_segment_ptr (ABI, user SGPR)
# s[4:7]   = ptr_q(s[4:5]), ptr_k(s[6:7])       -- loaded at 0x00
# s[8:11]  = ptr_v(s[8:9]), ptr_o(s[10:11])      -- loaded at 0x10
# s[12:15] = B(s12), H(s13), N(s14), D(s15)      -- loaded at 0x20
# s16      = scale                                -- loaded at 0x30
# s17      = h (blockIdx.y)
# s18      = b (blockIdx.z)
# s19      = stride_h = N*D
# s20      = stride_b = H*N*D
# s[22:23] = Qp (4-aligned for potential future use)
# s[24:25] = Kp
# s[26:27] = Op
# s[28:29] = (scratch)
# s[30:31] = VTp
# s32      = qscale / temp
# s33      = k_col_stride_bytes = 16*D*2
# s34      = vt_stride_d_bytes = N*2
# s35      = num_kv_tiles
# s36      = n_tile counter
# s37      = n_base
# s38-40   = temps
# s41      = bx (block index X, preserved through prologue)
# s42-44   = temps

S_KARG   = 's[0:1]'
S_PTR_Q  = 's[4:5]';  S_PTR_K = 's[6:7]'
S_PTR_V  = 's[8:9]';  S_PTR_O = 's[10:11]'
S_B = 's12'; S_H = 's13'; S_N = 's14'; S_D = 's15'
S_SCALE  = 's16'
S_h      = 's17'   # blockIdx.y
S_b      = 's18'   # blockIdx.z
S_STRIDE_H = 's19'
S_STRIDE_B = 's20'
# s21 = temp
S_QP     = 's[22:23]'
S_KP     = 's[24:25]'
S_OP     = 's[26:27]'
S_VTP    = 's[30:31]'
S_QSCALE = 's32'
S_KCOL_STRIDE = 's33'
S_VT_STRIDE_D = 's34'
S_NUM_TILES = 's35'
S_NTILE  = 's36'
S_NBASE  = 's37'


def o_dt(dt):
    return f"v[{1+dt*8}:{8+dt*8}]"

def q_dt(dt):
    return f"v[{65+dt*4}:{68+dt*4}]"

def s_col(c):
    return f"v[{97+c*8}:{104+c*8}]"


def emit_prologue():
    L = []
    a = L.append

    a('.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"')
    a('\t.amdhsa_code_object_version 6')
    a('\t.text')
    a(f'\t.globl\t{FUNC_NAME}')
    a(f'\t.p2align\t8')
    a(f'\t.type\t{FUNC_NAME},@function')
    a(f'{FUNC_NAME}:')
    a('')

    # Load kargs (4-aligned SGPRs for b128)
    a(f'\ts_load_b128 s[4:7], {S_KARG}, 0x0    ; ptr_q(s[4:5]), ptr_k(s[6:7])')
    a(f'\ts_load_b128 s[8:11], {S_KARG}, 0x10 ; ptr_v(s[8:9]), ptr_o(s[10:11])')
    a(f'\ts_load_b128 s[12:15], {S_KARG}, 0x20')
    a(f'\ts_load_b32 {S_SCALE}, {S_KARG}, 0x30')
    a('')

    # Thread indexing
    a('\tv_and_b32_e32 v159, 15, v0')
    a('\tv_lshrrev_b32_e32 v158, 4, v0')
    a('\tv_and_b32_e32 v158, 1, v158')
    a('\tv_lshlrev_b32_e32 v158, 3, v158   ; row8_elem')
    a('\tv_xor_b32_e32 v157, 16, v0')
    a('\tv_lshlrev_b32_e32 v157, 2, v157   ; bpermute_idx')
    a('')

    # Block index: 1D grid, decompose linear_id = ttmp9
    # linear_id = bx + nb * (h + H * b)
    # nb = N / BLOCK_M  (BLOCK_M=384 is NOT power of 2, must use reciprocal)
    a('\ts_wait_kmcnt 0x0')
    a('')
    # nb = N / BLOCK_M  (BLOCK_M={BLOCK_M}, magic = ceil(2^32/{BLOCK_M}))
    a(f'\ts_mov_b32 s28, 0x{((1 << 32) // BLOCK_M + 1):08x}  ; magic for div by {BLOCK_M}')
    a(f'\ts_wait_alu 0xfffe')
    a(f'\ts_mul_hi_u32 s21, {S_N}, s28    ; nb = N / {BLOCK_M}')
    a(f'\ts_wait_alu 0xfffe')
    # q = ttmp9 / nb, bx = ttmp9 % nb  (via float div + correction)
    a(f'\tv_cvt_f32_u32_e32 v180, ttmp9')
    a(f'\tv_cvt_f32_u32_e32 v181, s21')
    a(f'\ts_delay_alu instid0(VALU_DEP_1)')
    a(f'\tv_rcp_iflag_f32_e32 v181, v181')
    a(f'\ts_delay_alu instid0(TRANS32_DEP_1)')
    a(f'\tv_mul_f32_e32 v180, v180, v181  ; float(ttmp9) / float(nb)')
    a(f'\ts_delay_alu instid0(VALU_DEP_1)')
    a(f'\tv_cvt_u32_f32_e32 v180, v180    ; q_approx')
    a(f'\tv_readfirstlane_b32 s29, v180   ; q')
    a(f'\ts_wait_alu 0xfffe')
    # Correction: if q*nb > ttmp9, q--
    a(f'\ts_mul_i32 s28, s29, s21')
    a(f'\ts_wait_alu 0xfffe')
    a(f'\ts_cmp_gt_u32 s28, ttmp9')
    a(f'\ts_cbranch_scc0 .L_div_ok1')
    a(f'\ts_sub_i32 s29, s29, 1')
    a(f'\ts_wait_alu 0xfffe')
    a(f'\ts_mul_i32 s28, s29, s21')
    a(f'\ts_wait_alu 0xfffe')
    a(f'.L_div_ok1:')
    a(f'\ts_sub_i32 s41, ttmp9, s28       ; bx = ttmp9 - q*nb  (saved in s41)')
    a(f'\ts_wait_alu 0xfffe')
    # h = q % H, b = q / H  (via float div + correction)
    a(f'\tv_cvt_f32_u32_e32 v180, s29')
    a(f'\tv_cvt_f32_u32_e32 v181, {S_H}')
    a(f'\ts_delay_alu instid0(VALU_DEP_1)')
    a(f'\tv_rcp_iflag_f32_e32 v181, v181')
    a(f'\ts_delay_alu instid0(TRANS32_DEP_1)')
    a(f'\tv_mul_f32_e32 v180, v180, v181  ; float(q) / float(H)')
    a(f'\ts_delay_alu instid0(VALU_DEP_1)')
    a(f'\tv_cvt_u32_f32_e32 v180, v180    ; b_approx')
    a(f'\tv_readfirstlane_b32 {S_b}, v180')
    a(f'\ts_wait_alu 0xfffe')
    # Correction: if b*H > q, b--
    a(f'\ts_mul_i32 s38, {S_b}, {S_H}')
    a(f'\ts_wait_alu 0xfffe')
    a(f'\ts_cmp_gt_u32 s38, s29')
    a(f'\ts_cbranch_scc0 .L_div_ok2')
    a(f'\ts_sub_i32 {S_b}, {S_b}, 1')
    a(f'\ts_wait_alu 0xfffe')
    a(f'\ts_mul_i32 s38, {S_b}, {S_H}')
    a(f'\ts_wait_alu 0xfffe')
    a(f'.L_div_ok2:')
    a(f'\ts_sub_i32 {S_h}, s29, s38       ; h = q - b*H')
    a(f'\ts_wait_alu 0xfffe')
    # s41 = bx (preserved through bh offset computation)
    a(f'\t; bx=s41, h={S_h}, b={S_b}')
    a('')

    # Strides
    a(f'\ts_mul_i32 {S_STRIDE_H}, {S_N}, {S_D}')
    a(f'\ts_mul_i32 {S_STRIDE_B}, {S_H}, {S_STRIDE_H}')
    a('')

    # bh byte offset
    a(f'\ts_mul_i32 s21, {S_b}, {S_STRIDE_B}')
    a(f'\ts_mul_i32 s28, {S_h}, {S_STRIDE_H}')
    a(f'\ts_add_i32 s21, s21, s28')
    a(f'\ts_lshl_b32 s28, s21, 1')
    a(f'\ts_ashr_i32 s29, s28, 31')
    a('')

    # Q/K/O base pointers
    a(f'\ts_add_u32 s22, s4, s28')
    a(f'\ts_addc_u32 s23, s5, s29    ; Qp')
    a(f'\ts_add_u32 s24, s6, s28')
    a(f'\ts_addc_u32 s25, s7, s29    ; Kp')
    a(f'\ts_add_u32 s26, s10, s28')
    a(f'\ts_addc_u32 s27, s11, s29   ; Op')
    a('')

    # VT base
    a(f'\ts_mul_i32 s28, {S_D}, {S_N}        ; vt_stride_h = D*N')
    a(f'\ts_mul_i32 s29, {S_H}, s28           ; vt_stride_b = H*D*N')
    a(f'\ts_mul_i32 s21, {S_b}, s29')
    a(f'\ts_mul_i32 s38, {S_h}, s28')
    a(f'\ts_add_i32 s21, s21, s38')
    a(f'\ts_lshl_b32 s38, s21, 1')
    a(f'\ts_ashr_i32 s39, s38, 31')
    a(f'\ts_add_u32 s30, s8, s38')
    a(f'\ts_addc_u32 s31, s9, s39    ; VTp')
    a('')

    # q_m_base = bx * BLOCK_M + wave_id * 16
    a(f'\tv_lshrrev_b32_e32 v180, 5, v0')
    a(f'\tv_lshlrev_b32_e32 v180, 4, v180')
    a(f'\ts_mul_i32 {S_QSCALE}, s41, {BLOCK_M}  ; bx * BLOCK_M')
    a(f'\ts_wait_alu 0xfffe')
    a(f'\tv_add_nc_u32_e32 v180, {S_QSCALE}, v180')
    a('')

    # Q row byte-offset = ((q_m_base + col16) * D + row8_elem) * 2
    a('\tv_add_nc_u32_e32 v181, v180, v159')
    a(f'\tv_mul_lo_u32 v181, {S_D}, v181')
    a('\tv_add_nc_u32_e32 v181, v181, v158')
    a('\tv_lshlrev_b32_e32 v176, 1, v181')
    a('\tv_ashrrev_i32_e32 v177, 31, v176')
    a('')

    # Q load
    a('\tv_add_co_u32 v178, vcc_lo, s22, v176')
    a('\ts_wait_alu 0xfffd')
    a('\tv_add_co_ci_u32_e64 v179, null, s23, v177, vcc_lo')
    a('')
    for dt in range(DK):
        off = dt * W_K * 2
        a(f'\tglobal_load_b128 {q_dt(dt)}, v[178:179], off offset:{off}')
    a('\ts_wait_loadcnt 0x0')
    a('')

    # Scale Q
    a(f'\ts_mul_f32 {S_QSCALE}, {S_SCALE}, 0x3fb8aa3b ; qscale = scale * LOG2_E')
    a('\ts_wait_alu 0xfffe')
    a('')
    for dt in range(DK):
        for r in range(4):
            v = 65 + dt * 4 + r
            a(f'\tv_lshlrev_b32_e32 v180, 16, v{v}')
            a(f'\tv_and_b32_e32 v181, 0xffff0000, v{v}')
            a(f'\tv_mul_f32_e32 v180, {S_QSCALE}, v180')
            a(f'\tv_mul_f32_e32 v181, {S_QSCALE}, v181')
            a(f'\tv_bfe_u32 v182, v180, 16, 1')
            a(f'\tv_add3_u32 v180, v180, v182, 0x7fff')
            a(f'\tv_bfe_u32 v182, v181, 16, 1')
            a(f'\tv_add3_u32 v181, v181, v182, 0x7fff')
            a(f'\tv_perm_b32 v{v}, v181, v180, 0x7060302')
    a('')

    # Init output accumulators = 0
    for i in range(1, 65, 2):
        a(f'\tv_dual_mov_b32 v{i}, 0 :: v_dual_mov_b32 v{i+1}, 0')
    a('')

    # Init m_row, l_row
    a('\tv_mov_b32_e32 v155, 0xff7fc99e  ; m_row = -3.4e38')
    a('\tv_mov_b32_e32 v156, 0           ; l_row = 0')
    a('')

    # Constants
    a(f'\ts_lshl_b32 {S_KCOL_STRIDE}, {S_D}, 5   ; 16*D*2 bytes')
    a(f'\ts_lshl_b32 {S_VT_STRIDE_D}, {S_N}, 1   ; N*2 bytes')
    a(f'\ts_lshr_b32 {S_NUM_TILES}, {S_N}, 6      ; N/64')
    # DEBUG: force 1 N-tile
    if DEBUG_UNIFORM_P:
        a(f'\ts_mov_b32 {S_NUM_TILES}, 1              ; DEBUG: single tile')
    else:
        a(f'\t;s_mov_b32 {S_NUM_TILES}, 1              ; DEBUG: single tile')
    a(f'\ts_cmp_eq_u32 {S_NUM_TILES}, 0')
    a(f'\ts_cbranch_scc1 .L_epilogue')
    a(f'\ts_mov_b32 {S_NTILE}, 0')
    a('')

    if DEBUG_WRITE_DIAG:
        pass

    return '\n'.join(L)


def emit_n_tile_loop():
    L = []
    a = L.append

    a('.L_n_tile_loop:')
    a('')

    # n_base
    a(f'\ts_lshl_b32 {S_NBASE}, {S_NTILE}, 6')
    a('\ts_wait_alu 0xfffe')
    a('')

    if DEBUG_UNIFORM_P:
        # Skip QKT+softmax entirely, hardcode P = bf16(1.0) for all cols
        # bf16 1.0 = 0x3f80, packed pair = 0x3f803f80
        a('\t; DEBUG: P = bf16(1.0) uniform')
        for c in range(4):
            for r in range(4):
                a(f'\tv_mov_b32_e32 v{160+c*4+r}, 0x3f803f80')
        a('\tv_mov_b32_e32 v189, 1.0  ; no rescale')
        a('\tv_mov_b32_e32 v156, 64.0 ; l_row = 64')
        a(f'\ts_branch .L_pv_start')
        a('')

    # K col0: Kp + ((n_base + col16) * D + row8_elem) * 2 (bytes)
    # v_add_nc_u32 only takes 2 src for VOP2. Use s37 via v_add:
    a(f'\tv_mov_b32_e32 v180, {S_NBASE}')
    a('\tv_add_nc_u32_e32 v180, v159, v180  ; n_base + col16')
    a(f'\tv_mul_lo_u32 v180, {S_D}, v180')
    a('\tv_add_nc_u32_e32 v180, v158, v180  ; + row8_elem')
    a('\tv_lshlrev_b32_e32 v180, 1, v180    ; bytes')
    a('\tv_ashrrev_i32_e32 v181, 31, v180')
    a('\tv_add_co_u32 v145, vcc_lo, s24, v180')
    a('\ts_wait_alu 0xfffd')
    a('\tv_add_co_ci_u32_e64 v146, null, s25, v181, vcc_lo')
    a('')

    # K col1 = col0 + s33
    a(f'\ts_ashr_i32 s38, {S_KCOL_STRIDE}, 31')
    a('\ts_wait_alu 0xfffe')
    a(f'\tv_add_co_u32 v147, vcc_lo, v145, {S_KCOL_STRIDE}')
    a('\ts_wait_alu 0xfffd')
    a('\tv_add_co_ci_u32_e64 v148, null, v146, s38, vcc_lo')
    a('')

    # K col2 = col0 + 2*s33
    a(f'\ts_lshl_b32 s39, {S_KCOL_STRIDE}, 1')
    a('\ts_ashr_i32 s40, s39, 31')
    a('\ts_wait_alu 0xfffe')
    a('\tv_add_co_u32 v149, vcc_lo, v145, s39')
    a('\ts_wait_alu 0xfffd')
    a('\tv_add_co_ci_u32_e64 v150, null, v146, s40, vcc_lo')
    a('')

    # K col3 = col0 + 3*s33
    a(f'\ts_mul_i32 s39, {S_KCOL_STRIDE}, 3')
    a('\ts_ashr_i32 s40, s39, 31')
    a('\ts_wait_alu 0xfffe')
    a('\tv_add_co_u32 v151, vcc_lo, v145, s39')
    a('\ts_wait_alu 0xfffd')
    a('\tv_add_co_ci_u32_e64 v152, null, v146, s40, vcc_lo')
    a('')

    # Zero score accumulators
    for c in range(4):
        base = 97 + c * 8
        for r in range(0, 8, 2):
            a(f'\tv_dual_mov_b32 v{base+r}, 0 :: v_dual_mov_b32 v{base+r+1}, 0')
    a('')

    # QKT: 8 D-tiles × 4 cols = 32 WMMAs
    a('\t; ===== QKT: 32 WMMAs =====')
    for dt in range(DK):
        off = dt * W_K * 2
        a(f'\ts_clause 0x1')
        a(f'\tglobal_load_b128 v[129:132], v[145:146], off offset:{off}')
        a(f'\tglobal_load_b128 v[133:136], v[147:148], off offset:{off}')
        a(f'\ts_wait_loadcnt 0x1')
        a(f'\tv_wmma_f32_16x16x16_bf16 {s_col(0)}, v[129:132], {q_dt(dt)}, {s_col(0)}')
        a(f'\ts_wait_loadcnt 0x0')
        a(f'\tv_wmma_f32_16x16x16_bf16 {s_col(1)}, v[133:136], {q_dt(dt)}, {s_col(1)}')
        a(f'\ts_clause 0x1')
        a(f'\tglobal_load_b128 v[129:132], v[149:150], off offset:{off}')
        a(f'\tglobal_load_b128 v[133:136], v[151:152], off offset:{off}')
        a(f'\ts_wait_loadcnt 0x1')
        a(f'\tv_wmma_f32_16x16x16_bf16 {s_col(2)}, v[129:132], {q_dt(dt)}, {s_col(2)}')
        a(f'\ts_wait_loadcnt 0x0')
        a(f'\tv_wmma_f32_16x16x16_bf16 {s_col(3)}, v[133:136], {q_dt(dt)}, {s_col(3)}')
        a('')

    # SOFTMAX
    a('\t; ===== SOFTMAX =====')
    a('')

    # Row max: reduce 32 values -> v129
    a('\tv_max_num_f32_e32 v129, v97, v98')
    a('\ts_delay_alu instid0(VALU_DEP_1)')
    a('\tv_max3_num_f32 v129, v129, v99, v100')
    a('\ts_delay_alu instid0(VALU_DEP_1)')
    a('\tv_max3_num_f32 v129, v129, v101, v102')
    a('\ts_delay_alu instid0(VALU_DEP_1)')
    a('\tv_max3_num_f32 v129, v129, v103, v104')
    for v in range(105, 129, 2):
        a('\ts_delay_alu instid0(VALU_DEP_1)')
        a(f'\tv_max3_num_f32 v129, v129, v{v}, v{v+1}')
    a('')

    # Cross-half max
    a('\tds_bpermute_b32 v130, v157, v129')
    a('\ts_wait_dscnt 0x0')
    a('\tv_max3_num_f32 v130, v155, v129, v130')
    a('')

    # Rescale
    a('\tv_sub_f32_e32 v131, v155, v130')
    a('\tv_cmp_neq_f32_e32 vcc_lo, v130, v155')
    a('\tv_mov_b32_e32 v155, v130       ; m_row = new_m')
    a('\ts_delay_alu instid0(VALU_DEP_3)')
    a('\tv_exp_f32_e32 v131, v131       ; rescale')
    a('\ts_wait_alu 0xfffe')
    a('\ts_delay_alu instid0(TRANS32_DEP_1)')
    a('\tv_cndmask_b32_e32 v189, 1.0, v131, vcc_lo')
    a('\tv_mul_f32_e32 v156, v189, v156 ; l_row *= rescale')
    a('')

    # exp2(score - new_m) for 32 values
    for c in range(4):
        for j in range(8):
            v = 97 + c * 8 + j
            a(f'\tv_sub_f32_e32 v{v}, v{v}, v155')
    a('')
    for c in range(4):
        for j in range(8):
            v = 97 + c * 8 + j
            a(f'\tv_exp_f32_e32 v{v}, v{v}')
    a('')

    # Sum 32 exp2 values
    a('\ts_delay_alu instid0(TRANS32_DEP_1)')
    a('\tv_add_f32_e32 v129, v97, v98')
    for i in range(2, 32):
        v = 97 + i
        a('\ts_delay_alu instid0(VALU_DEP_1)')
        a(f'\tv_add_f32_e32 v129, v129, v{v}')
    a('')

    # Cross-half sum
    a('\tds_bpermute_b32 v130, v157, v129')
    a('\ts_wait_dscnt 0x0')
    a('\tv_add_f32_e32 v129, v129, v130')
    a('\tv_add_f32_e32 v156, v156, v129 ; l_row += row_sum')
    a('')

    # bf16 pack -> P0..P3
    for c in range(4):
        p_base = 160 + c * 4
        s_base = 97 + c * 8
        for pair in range(4):
            lo = s_base + pair * 2
            hi = lo + 1
            p_reg = p_base + pair
            a(f'\tv_bfe_u32 v129, v{lo}, 16, 1')
            a(f'\tv_add3_u32 v{lo}, v{lo}, v129, 0x7fff')
            a(f'\tv_bfe_u32 v129, v{hi}, 16, 1')
            a(f'\tv_add3_u32 v{hi}, v{hi}, v129, 0x7fff')
            a(f'\tv_perm_b32 v{p_reg}, v{hi}, v{lo}, 0x7060302')
    a('')

    # PV: 8 D-tiles × 4 V-cols = 32 WMMAs
    a('.L_pv_start:')
    a('\t; ===== PV: 32 WMMAs =====')
    a('')

    # DEBUG: Copy V load for dt=0 into output accumulators instead of doing PV
    # Uncomment next line to enable:
    #a('\ts_branch .L_debug_dump_v')

    for dt in range(DK):
        dtw = dt * W_K
        a(f'\t; PV D-tile {dt}')
        # VT addr = VTp + (dt*16 + col16) * N + n_base + row8_elem  (elements)
        if dtw == 0:
            a(f'\tv_mul_lo_u32 v180, {S_N}, v159  ; col16 * N')
        else:
            a(f'\tv_add_nc_u32_e32 v180, {dtw}, v159')
            a(f'\tv_mul_lo_u32 v180, {S_N}, v180')
        a(f'\tv_mov_b32_e32 v181, {S_NBASE}')
        a(f'\tv_add_nc_u32_e32 v180, v181, v180  ; + n_base')
        a(f'\tv_add_nc_u32_e32 v180, v158, v180  ; + row8_elem')
        a(f'\tv_lshlrev_b32_e32 v180, 1, v180    ; bytes')
        a(f'\tv_ashrrev_i32_e32 v181, 31, v180')
        a(f'\tv_add_co_u32 v153, vcc_lo, s30, v180')
        a(f'\ts_wait_alu 0xfffd')
        a(f'\tv_add_co_ci_u32_e64 v154, null, s31, v181, vcc_lo')
        a('')

        # 4 V loads (col offsets 0,16,32,48 elements = 0,32,64,96 bytes)
        a(f'\ts_clause 0x3')
        a(f'\tglobal_load_b128 v[129:132], v[153:154], off offset:0')
        a(f'\tglobal_load_b128 v[133:136], v[153:154], off offset:32')
        a(f'\tglobal_load_b128 v[137:140], v[153:154], off offset:64')
        a(f'\tglobal_load_b128 v[141:144], v[153:154], off offset:96')
        a('')

        # Rescale output
        for j in range(8):
            a(f'\tv_mul_f32_e32 v{1+dt*8+j}, v189, v{1+dt*8+j}')
        a('')

        # 4 WMMAs
        a(f'\ts_wait_loadcnt 0x3')
        a(f'\tv_wmma_f32_16x16x16_bf16 {o_dt(dt)}, v[129:132], v[160:163], {o_dt(dt)}')
        a(f'\ts_wait_loadcnt 0x2')
        a(f'\tv_wmma_f32_16x16x16_bf16 {o_dt(dt)}, v[133:136], v[164:167], {o_dt(dt)}')
        a(f'\ts_wait_loadcnt 0x1')
        a(f'\tv_wmma_f32_16x16x16_bf16 {o_dt(dt)}, v[137:140], v[168:171], {o_dt(dt)}')
        a(f'\ts_wait_loadcnt 0x0')
        a(f'\tv_wmma_f32_16x16x16_bf16 {o_dt(dt)}, v[141:144], v[172:175], {o_dt(dt)}')
        a('')

    # Loop
    a(f'\ts_add_i32 {S_NTILE}, {S_NTILE}, 1')
    a('\ts_wait_alu 0xfffe')
    a(f'\ts_cmp_lt_u32 {S_NTILE}, {S_NUM_TILES}')
    a('\ts_cbranch_scc1 .L_n_tile_loop')
    a('')

    return '\n'.join(L)


def emit_epilogue():
    L = []
    a = L.append

    a('.L_epilogue:')
    a('')

    if DEBUG_WRITE_DIAG:
        # Write diagnostic info: Op_lo, Op_hi, h, b, bh_byte_offset, ttmp7, ttmp9, stride_h
        # Only thread 0 and blockIdx.x == 0 writes to Op + 0
        a('\t; DEBUG: write diagnostic info')
        a('\tv_cmp_eq_u32_e32 vcc_lo, 0, v0  ; thread 0 only')
        a('\ts_wait_alu 0xfffd')
        a('\ts_and_saveexec_b32 s38, vcc_lo')
        a(f'\ts_cmp_eq_u32 ttmp9, 0  ; blockIdx.x == 0')
        a('\ts_cbranch_scc0 .L_diag_skip')
        a('')
        # Write to Op directly (s26:s27)
        a(f'\tv_mov_b32_e32 v1, s26   ; Op lo')
        a(f'\tv_mov_b32_e32 v2, s27   ; Op hi')
        a(f'\tv_mov_b32_e32 v3, {S_h} ; h')
        a(f'\tv_mov_b32_e32 v4, {S_b} ; b')
        a(f'\tv_mov_b32_e32 v5, s26   ; Op lo')
        a(f'\tv_mov_b32_e32 v6, s27   ; Op hi')
        # diag[0:3] = h, b, Op_lo, Op_hi
        # diag[4:7] = ttmp7, stride_h, v176, ttmp9
        a(f'\tv_mov_b32_e32 v129, s26')
        a(f'\tv_mov_b32_e32 v130, s27')
        a('\tglobal_store_b128 v[129:130], v[3:6], off offset:0')
        a(f'\tv_mov_b32_e32 v3, ttmp7')
        a(f'\tv_mov_b32_e32 v4, {S_STRIDE_H}')
        a(f'\tv_mov_b32_e32 v5, v176')
        a(f'\tv_mov_b32_e32 v6, ttmp9')
        a('\tglobal_store_b128 v[129:130], v[3:6], off offset:16')
        a('')
        a('.L_diag_skip:')
        a('\ts_or_b32 exec_lo, exec_lo, s38')
        a('\ts_endpgm')
        a('')
    else:
        # inv = 1/l_row
        a('\tv_cmp_lt_f32_e32 vcc_lo, 0, v156  ; l_row > 0?')
        a('\tv_rcp_f32_e32 v129, v156')
        a('\ts_wait_alu 0xfffe')
        a('\ts_delay_alu instid0(TRANS32_DEP_1)')
        a('\tv_fma_f32 v130, -v156, v129, 1.0')
        a('\ts_delay_alu instid0(VALU_DEP_1)')
        a('\tv_fma_f32 v129, v130, v129, v129')
        a('\tv_cndmask_b32_e32 v129, 0, v129, vcc_lo')
        a('')

        # O row addr = Op + v[176:177]
        a('\tv_add_co_u32 v178, vcc_lo, s26, v176')
        a('\ts_wait_alu 0xfffd')
        a('\tv_add_co_ci_u32_e64 v179, null, s27, v177, vcc_lo')
        a('')

        for dt in range(DK):
            off = dt * W_K * 2
            for j in range(8):
                v = 1 + dt * 8 + j
                a(f'\tv_mul_f32_e32 v{v}, v129, v{v}')

            for pair in range(4):
                lo = 1 + dt * 8 + pair * 2
                hi = lo + 1
                a(f'\tv_bfe_u32 v130, v{lo}, 16, 1')
                a(f'\tv_add3_u32 v{lo}, v{lo}, v130, 0x7fff')
                a(f'\tv_bfe_u32 v130, v{hi}, 16, 1')
                a(f'\tv_add3_u32 v{hi}, v{hi}, v130, 0x7fff')
                a(f'\tv_perm_b32 v{1+dt*8+pair}, v{hi}, v{lo}, 0x7060302')

            base = 1 + dt * 8
            a(f'\tglobal_store_b128 v[178:179], v[{base}:{base+3}], off offset:{off}')
            a('')

        a('\ts_endpgm')
    a('')

    return '\n'.join(L)


def emit_metadata():
    L = []
    a = L.append

    VGPR_COUNT = 190
    SGPR_COUNT = 45

    a(f'\t.section\t.rodata,#alloc')
    a(f'\t.p2align\t6')
    a(f'\t.amdhsa_kernel {FUNC_NAME}')
    a('\t\t.amdhsa_group_segment_fixed_size 0')
    a('\t\t.amdhsa_private_segment_fixed_size 0')
    a('\t\t.amdhsa_kernarg_size 56')
    a('\t\t.amdhsa_user_sgpr_count 2')
    a('\t\t.amdhsa_user_sgpr_dispatch_ptr 0')
    a('\t\t.amdhsa_user_sgpr_queue_ptr 0')
    a('\t\t.amdhsa_user_sgpr_kernarg_segment_ptr 1')
    a('\t\t.amdhsa_user_sgpr_dispatch_id 0')
    a('\t\t.amdhsa_user_sgpr_private_segment_size 0')
    a('\t\t.amdhsa_wavefront_size32 1')
    a('\t\t.amdhsa_uses_dynamic_stack 0')
    a('\t\t.amdhsa_enable_private_segment 0')
    a('\t\t.amdhsa_system_sgpr_workgroup_id_x 1')
    a('\t\t.amdhsa_system_sgpr_workgroup_id_y 1')
    a('\t\t.amdhsa_system_sgpr_workgroup_id_z 1')
    a('\t\t.amdhsa_system_sgpr_workgroup_info 0')
    a('\t\t.amdhsa_system_vgpr_workitem_id 0')
    a(f'\t\t.amdhsa_next_free_vgpr {VGPR_COUNT}')
    a(f'\t\t.amdhsa_next_free_sgpr {SGPR_COUNT}')
    a('\t\t.amdhsa_reserve_vcc 1')
    a('\t\t.amdhsa_float_round_mode_32 0')
    a('\t\t.amdhsa_float_round_mode_16_64 0')
    a('\t\t.amdhsa_float_denorm_mode_32 3')
    a('\t\t.amdhsa_float_denorm_mode_16_64 3')
    a('\t\t.amdhsa_fp16_overflow 0')
    a('\t\t.amdhsa_workgroup_processor_mode 1')
    a('\t\t.amdhsa_memory_ordered 1')
    a('\t\t.amdhsa_forward_progress 1')
    a(f'\t.end_amdhsa_kernel')
    a('')

    a('\t.amdgpu_metadata')
    a('---')
    a('amdhsa.kernels:')
    a('  - .args:')
    a('      - .offset:         0')
    a('        .size:           56')
    a('        .value_kind:     by_value')
    a(f'    .group_segment_fixed_size: 0')
    a(f'    .kernarg_segment_align: 8')
    a(f'    .kernarg_segment_size: 56')
    a(f'    .max_flat_workgroup_size: {BLOCK_SIZE}')
    a(f'    .name:           {FUNC_NAME}')
    a(f'    .private_segment_fixed_size: 0')
    a(f'    .sgpr_count:     {SGPR_COUNT}')
    a(f'    .sgpr_spill_count: 0')
    a(f'    .symbol:         {FUNC_NAME}.kd')
    a(f'    .uniform_work_group_size: 1')
    a(f'    .uses_dynamic_stack: false')
    a(f'    .vgpr_count:     {VGPR_COUNT}')
    a(f'    .vgpr_spill_count: 0')
    a(f'    .wavefront_size: 32')
    a(f'    .workgroup_processor_mode: 1')
    a('amdhsa.target:   amdgcn-amd-amdhsa--gfx1201')
    a('amdhsa.version:')
    a('  - 1')
    a('  - 2')
    a('...')
    a('')
    a('\t.end_amdgpu_metadata')
    a('')

    return '\n'.join(L)


def main():
    parts = [emit_prologue(), emit_n_tile_loop(), emit_epilogue(), emit_metadata()]
    output = '\n'.join(parts)

    outfile = '/tmp/v80_kernel.s'
    if len(sys.argv) > 1:
        outfile = sys.argv[1]

    with open(outfile, 'w') as f:
        f.write(output)

    print(f"Generated {outfile}")
    print(f"  BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, D={D}")
    print(f"  {DK} D-tiles, {BLOCK_SIZE} threads/block")
    print(f"  QKT: {DK*4} WMMAs, PV: {DK*4} WMMAs, Total: {DK*8} WMMAs/tile")


if __name__ == '__main__':
    main()
