.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.text
	.globl	opus_attn_gfx1201_kernel_v80
	.p2align	8
	.type	opus_attn_gfx1201_kernel_v80,@function
opus_attn_gfx1201_kernel_v80:

	s_load_b128 s[4:7], s[0:1], 0x0    ; ptr_q(s[4:5]), ptr_k(s[6:7])
	s_load_b128 s[8:11], s[0:1], 0x10 ; ptr_v(s[8:9]), ptr_o(s[10:11])
	s_load_b128 s[12:15], s[0:1], 0x20
	s_load_b32 s16, s[0:1], 0x30

	v_and_b32_e32 v159, 15, v0
	v_lshrrev_b32_e32 v158, 4, v0
	v_and_b32_e32 v158, 1, v158
	v_lshlrev_b32_e32 v158, 3, v158   ; row8_elem
	v_xor_b32_e32 v157, 16, v0
	v_lshlrev_b32_e32 v157, 2, v157   ; bpermute_idx

	s_wait_kmcnt 0x0

	s_mov_b32 s28, 0x00aaaaab  ; magic for div by 384
	s_wait_alu 0xfffe
	s_mul_hi_u32 s21, s14, s28    ; nb = N / 384
	s_wait_alu 0xfffe
	v_cvt_f32_u32_e32 v180, ttmp9
	v_cvt_f32_u32_e32 v181, s21
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_iflag_f32_e32 v181, v181
	s_delay_alu instid0(TRANS32_DEP_1)
	v_mul_f32_e32 v180, v180, v181  ; float(ttmp9) / float(nb)
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_u32_f32_e32 v180, v180    ; q_approx
	v_readfirstlane_b32 s29, v180   ; q
	s_wait_alu 0xfffe
	s_mul_i32 s28, s29, s21
	s_wait_alu 0xfffe
	s_cmp_gt_u32 s28, ttmp9
	s_cbranch_scc0 .L_div_ok1
	s_sub_i32 s29, s29, 1
	s_wait_alu 0xfffe
	s_mul_i32 s28, s29, s21
	s_wait_alu 0xfffe
.L_div_ok1:
	s_sub_i32 s41, ttmp9, s28       ; bx = ttmp9 - q*nb  (saved in s41)
	s_wait_alu 0xfffe
	v_cvt_f32_u32_e32 v180, s29
	v_cvt_f32_u32_e32 v181, s13
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_iflag_f32_e32 v181, v181
	s_delay_alu instid0(TRANS32_DEP_1)
	v_mul_f32_e32 v180, v180, v181  ; float(q) / float(H)
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_u32_f32_e32 v180, v180    ; b_approx
	v_readfirstlane_b32 s18, v180
	s_wait_alu 0xfffe
	s_mul_i32 s38, s18, s13
	s_wait_alu 0xfffe
	s_cmp_gt_u32 s38, s29
	s_cbranch_scc0 .L_div_ok2
	s_sub_i32 s18, s18, 1
	s_wait_alu 0xfffe
	s_mul_i32 s38, s18, s13
	s_wait_alu 0xfffe
.L_div_ok2:
	s_sub_i32 s17, s29, s38       ; h = q - b*H
	s_wait_alu 0xfffe
	; bx=s41, h=s17, b=s18

	s_mul_i32 s19, s14, s15
	s_mul_i32 s20, s13, s19

	s_mul_i32 s21, s18, s20
	s_mul_i32 s28, s17, s19
	s_add_i32 s21, s21, s28
	s_lshl_b32 s28, s21, 1
	s_ashr_i32 s29, s28, 31

	s_add_u32 s22, s4, s28
	s_addc_u32 s23, s5, s29    ; Qp
	s_add_u32 s24, s6, s28
	s_addc_u32 s25, s7, s29    ; Kp
	s_add_u32 s26, s10, s28
	s_addc_u32 s27, s11, s29   ; Op

	s_mul_i32 s28, s15, s14        ; vt_stride_h = D*N
	s_mul_i32 s29, s13, s28           ; vt_stride_b = H*D*N
	s_mul_i32 s21, s18, s29
	s_mul_i32 s38, s17, s28
	s_add_i32 s21, s21, s38
	s_lshl_b32 s38, s21, 1
	s_ashr_i32 s39, s38, 31
	s_add_u32 s30, s8, s38
	s_addc_u32 s31, s9, s39    ; VTp

	v_lshrrev_b32_e32 v180, 5, v0
	v_lshlrev_b32_e32 v180, 4, v180
	s_mul_i32 s32, s41, 384  ; bx * BLOCK_M
	s_wait_alu 0xfffe
	v_add_nc_u32_e32 v180, s32, v180

	v_add_nc_u32_e32 v181, v180, v159
	v_mul_lo_u32 v181, s15, v181
	v_add_nc_u32_e32 v181, v181, v158
	v_lshlrev_b32_e32 v176, 1, v181
	v_ashrrev_i32_e32 v177, 31, v176

	v_add_co_u32 v178, vcc_lo, s22, v176
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v179, null, s23, v177, vcc_lo

	global_load_b128 v[65:68], v[178:179], off offset:0
	global_load_b128 v[69:72], v[178:179], off offset:32
	global_load_b128 v[73:76], v[178:179], off offset:64
	global_load_b128 v[77:80], v[178:179], off offset:96
	global_load_b128 v[81:84], v[178:179], off offset:128
	global_load_b128 v[85:88], v[178:179], off offset:160
	global_load_b128 v[89:92], v[178:179], off offset:192
	global_load_b128 v[93:96], v[178:179], off offset:224
	s_wait_loadcnt 0x0

	s_mul_f32 s32, s16, 0x3fb8aa3b ; qscale = scale * LOG2_E
	s_wait_alu 0xfffe

	v_lshlrev_b32_e32 v180, 16, v65
	v_and_b32_e32 v181, 0xffff0000, v65
	v_mul_f32_e32 v180, s32, v180
	v_mul_f32_e32 v181, s32, v181
	v_bfe_u32 v182, v180, 16, 1
	v_add3_u32 v180, v180, v182, 0x7fff
	v_bfe_u32 v182, v181, 16, 1
	v_add3_u32 v181, v181, v182, 0x7fff
	v_perm_b32 v65, v181, v180, 0x7060302
	v_lshlrev_b32_e32 v180, 16, v66
	v_and_b32_e32 v181, 0xffff0000, v66
	v_mul_f32_e32 v180, s32, v180
	v_mul_f32_e32 v181, s32, v181
	v_bfe_u32 v182, v180, 16, 1
	v_add3_u32 v180, v180, v182, 0x7fff
	v_bfe_u32 v182, v181, 16, 1
	v_add3_u32 v181, v181, v182, 0x7fff
	v_perm_b32 v66, v181, v180, 0x7060302
	v_lshlrev_b32_e32 v180, 16, v67
	v_and_b32_e32 v181, 0xffff0000, v67
	v_mul_f32_e32 v180, s32, v180
	v_mul_f32_e32 v181, s32, v181
	v_bfe_u32 v182, v180, 16, 1
	v_add3_u32 v180, v180, v182, 0x7fff
	v_bfe_u32 v182, v181, 16, 1
	v_add3_u32 v181, v181, v182, 0x7fff
	v_perm_b32 v67, v181, v180, 0x7060302
	v_lshlrev_b32_e32 v180, 16, v68
	v_and_b32_e32 v181, 0xffff0000, v68
	v_mul_f32_e32 v180, s32, v180
	v_mul_f32_e32 v181, s32, v181
	v_bfe_u32 v182, v180, 16, 1
	v_add3_u32 v180, v180, v182, 0x7fff
	v_bfe_u32 v182, v181, 16, 1
	v_add3_u32 v181, v181, v182, 0x7fff
	v_perm_b32 v68, v181, v180, 0x7060302
	v_lshlrev_b32_e32 v180, 16, v69
	v_and_b32_e32 v181, 0xffff0000, v69
	v_mul_f32_e32 v180, s32, v180
	v_mul_f32_e32 v181, s32, v181
	v_bfe_u32 v182, v180, 16, 1
	v_add3_u32 v180, v180, v182, 0x7fff
	v_bfe_u32 v182, v181, 16, 1
	v_add3_u32 v181, v181, v182, 0x7fff
	v_perm_b32 v69, v181, v180, 0x7060302
	v_lshlrev_b32_e32 v180, 16, v70
	v_and_b32_e32 v181, 0xffff0000, v70
	v_mul_f32_e32 v180, s32, v180
	v_mul_f32_e32 v181, s32, v181
	v_bfe_u32 v182, v180, 16, 1
	v_add3_u32 v180, v180, v182, 0x7fff
	v_bfe_u32 v182, v181, 16, 1
	v_add3_u32 v181, v181, v182, 0x7fff
	v_perm_b32 v70, v181, v180, 0x7060302
	v_lshlrev_b32_e32 v180, 16, v71
	v_and_b32_e32 v181, 0xffff0000, v71
	v_mul_f32_e32 v180, s32, v180
	v_mul_f32_e32 v181, s32, v181
	v_bfe_u32 v182, v180, 16, 1
	v_add3_u32 v180, v180, v182, 0x7fff
	v_bfe_u32 v182, v181, 16, 1
	v_add3_u32 v181, v181, v182, 0x7fff
	v_perm_b32 v71, v181, v180, 0x7060302
	v_lshlrev_b32_e32 v180, 16, v72
	v_and_b32_e32 v181, 0xffff0000, v72
	v_mul_f32_e32 v180, s32, v180
	v_mul_f32_e32 v181, s32, v181
	v_bfe_u32 v182, v180, 16, 1
	v_add3_u32 v180, v180, v182, 0x7fff
	v_bfe_u32 v182, v181, 16, 1
	v_add3_u32 v181, v181, v182, 0x7fff
	v_perm_b32 v72, v181, v180, 0x7060302
	v_lshlrev_b32_e32 v180, 16, v73
	v_and_b32_e32 v181, 0xffff0000, v73
	v_mul_f32_e32 v180, s32, v180
	v_mul_f32_e32 v181, s32, v181
	v_bfe_u32 v182, v180, 16, 1
	v_add3_u32 v180, v180, v182, 0x7fff
	v_bfe_u32 v182, v181, 16, 1
	v_add3_u32 v181, v181, v182, 0x7fff
	v_perm_b32 v73, v181, v180, 0x7060302
	v_lshlrev_b32_e32 v180, 16, v74
	v_and_b32_e32 v181, 0xffff0000, v74
	v_mul_f32_e32 v180, s32, v180
	v_mul_f32_e32 v181, s32, v181
	v_bfe_u32 v182, v180, 16, 1
	v_add3_u32 v180, v180, v182, 0x7fff
	v_bfe_u32 v182, v181, 16, 1
	v_add3_u32 v181, v181, v182, 0x7fff
	v_perm_b32 v74, v181, v180, 0x7060302
	v_lshlrev_b32_e32 v180, 16, v75
	v_and_b32_e32 v181, 0xffff0000, v75
	v_mul_f32_e32 v180, s32, v180
	v_mul_f32_e32 v181, s32, v181
	v_bfe_u32 v182, v180, 16, 1
	v_add3_u32 v180, v180, v182, 0x7fff
	v_bfe_u32 v182, v181, 16, 1
	v_add3_u32 v181, v181, v182, 0x7fff
	v_perm_b32 v75, v181, v180, 0x7060302
	v_lshlrev_b32_e32 v180, 16, v76
	v_and_b32_e32 v181, 0xffff0000, v76
	v_mul_f32_e32 v180, s32, v180
	v_mul_f32_e32 v181, s32, v181
	v_bfe_u32 v182, v180, 16, 1
	v_add3_u32 v180, v180, v182, 0x7fff
	v_bfe_u32 v182, v181, 16, 1
	v_add3_u32 v181, v181, v182, 0x7fff
	v_perm_b32 v76, v181, v180, 0x7060302
	v_lshlrev_b32_e32 v180, 16, v77
	v_and_b32_e32 v181, 0xffff0000, v77
	v_mul_f32_e32 v180, s32, v180
	v_mul_f32_e32 v181, s32, v181
	v_bfe_u32 v182, v180, 16, 1
	v_add3_u32 v180, v180, v182, 0x7fff
	v_bfe_u32 v182, v181, 16, 1
	v_add3_u32 v181, v181, v182, 0x7fff
	v_perm_b32 v77, v181, v180, 0x7060302
	v_lshlrev_b32_e32 v180, 16, v78
	v_and_b32_e32 v181, 0xffff0000, v78
	v_mul_f32_e32 v180, s32, v180
	v_mul_f32_e32 v181, s32, v181
	v_bfe_u32 v182, v180, 16, 1
	v_add3_u32 v180, v180, v182, 0x7fff
	v_bfe_u32 v182, v181, 16, 1
	v_add3_u32 v181, v181, v182, 0x7fff
	v_perm_b32 v78, v181, v180, 0x7060302
	v_lshlrev_b32_e32 v180, 16, v79
	v_and_b32_e32 v181, 0xffff0000, v79
	v_mul_f32_e32 v180, s32, v180
	v_mul_f32_e32 v181, s32, v181
	v_bfe_u32 v182, v180, 16, 1
	v_add3_u32 v180, v180, v182, 0x7fff
	v_bfe_u32 v182, v181, 16, 1
	v_add3_u32 v181, v181, v182, 0x7fff
	v_perm_b32 v79, v181, v180, 0x7060302
	v_lshlrev_b32_e32 v180, 16, v80
	v_and_b32_e32 v181, 0xffff0000, v80
	v_mul_f32_e32 v180, s32, v180
	v_mul_f32_e32 v181, s32, v181
	v_bfe_u32 v182, v180, 16, 1
	v_add3_u32 v180, v180, v182, 0x7fff
	v_bfe_u32 v182, v181, 16, 1
	v_add3_u32 v181, v181, v182, 0x7fff
	v_perm_b32 v80, v181, v180, 0x7060302
	v_lshlrev_b32_e32 v180, 16, v81
	v_and_b32_e32 v181, 0xffff0000, v81
	v_mul_f32_e32 v180, s32, v180
	v_mul_f32_e32 v181, s32, v181
	v_bfe_u32 v182, v180, 16, 1
	v_add3_u32 v180, v180, v182, 0x7fff
	v_bfe_u32 v182, v181, 16, 1
	v_add3_u32 v181, v181, v182, 0x7fff
	v_perm_b32 v81, v181, v180, 0x7060302
	v_lshlrev_b32_e32 v180, 16, v82
	v_and_b32_e32 v181, 0xffff0000, v82
	v_mul_f32_e32 v180, s32, v180
	v_mul_f32_e32 v181, s32, v181
	v_bfe_u32 v182, v180, 16, 1
	v_add3_u32 v180, v180, v182, 0x7fff
	v_bfe_u32 v182, v181, 16, 1
	v_add3_u32 v181, v181, v182, 0x7fff
	v_perm_b32 v82, v181, v180, 0x7060302
	v_lshlrev_b32_e32 v180, 16, v83
	v_and_b32_e32 v181, 0xffff0000, v83
	v_mul_f32_e32 v180, s32, v180
	v_mul_f32_e32 v181, s32, v181
	v_bfe_u32 v182, v180, 16, 1
	v_add3_u32 v180, v180, v182, 0x7fff
	v_bfe_u32 v182, v181, 16, 1
	v_add3_u32 v181, v181, v182, 0x7fff
	v_perm_b32 v83, v181, v180, 0x7060302
	v_lshlrev_b32_e32 v180, 16, v84
	v_and_b32_e32 v181, 0xffff0000, v84
	v_mul_f32_e32 v180, s32, v180
	v_mul_f32_e32 v181, s32, v181
	v_bfe_u32 v182, v180, 16, 1
	v_add3_u32 v180, v180, v182, 0x7fff
	v_bfe_u32 v182, v181, 16, 1
	v_add3_u32 v181, v181, v182, 0x7fff
	v_perm_b32 v84, v181, v180, 0x7060302
	v_lshlrev_b32_e32 v180, 16, v85
	v_and_b32_e32 v181, 0xffff0000, v85
	v_mul_f32_e32 v180, s32, v180
	v_mul_f32_e32 v181, s32, v181
	v_bfe_u32 v182, v180, 16, 1
	v_add3_u32 v180, v180, v182, 0x7fff
	v_bfe_u32 v182, v181, 16, 1
	v_add3_u32 v181, v181, v182, 0x7fff
	v_perm_b32 v85, v181, v180, 0x7060302
	v_lshlrev_b32_e32 v180, 16, v86
	v_and_b32_e32 v181, 0xffff0000, v86
	v_mul_f32_e32 v180, s32, v180
	v_mul_f32_e32 v181, s32, v181
	v_bfe_u32 v182, v180, 16, 1
	v_add3_u32 v180, v180, v182, 0x7fff
	v_bfe_u32 v182, v181, 16, 1
	v_add3_u32 v181, v181, v182, 0x7fff
	v_perm_b32 v86, v181, v180, 0x7060302
	v_lshlrev_b32_e32 v180, 16, v87
	v_and_b32_e32 v181, 0xffff0000, v87
	v_mul_f32_e32 v180, s32, v180
	v_mul_f32_e32 v181, s32, v181
	v_bfe_u32 v182, v180, 16, 1
	v_add3_u32 v180, v180, v182, 0x7fff
	v_bfe_u32 v182, v181, 16, 1
	v_add3_u32 v181, v181, v182, 0x7fff
	v_perm_b32 v87, v181, v180, 0x7060302
	v_lshlrev_b32_e32 v180, 16, v88
	v_and_b32_e32 v181, 0xffff0000, v88
	v_mul_f32_e32 v180, s32, v180
	v_mul_f32_e32 v181, s32, v181
	v_bfe_u32 v182, v180, 16, 1
	v_add3_u32 v180, v180, v182, 0x7fff
	v_bfe_u32 v182, v181, 16, 1
	v_add3_u32 v181, v181, v182, 0x7fff
	v_perm_b32 v88, v181, v180, 0x7060302
	v_lshlrev_b32_e32 v180, 16, v89
	v_and_b32_e32 v181, 0xffff0000, v89
	v_mul_f32_e32 v180, s32, v180
	v_mul_f32_e32 v181, s32, v181
	v_bfe_u32 v182, v180, 16, 1
	v_add3_u32 v180, v180, v182, 0x7fff
	v_bfe_u32 v182, v181, 16, 1
	v_add3_u32 v181, v181, v182, 0x7fff
	v_perm_b32 v89, v181, v180, 0x7060302
	v_lshlrev_b32_e32 v180, 16, v90
	v_and_b32_e32 v181, 0xffff0000, v90
	v_mul_f32_e32 v180, s32, v180
	v_mul_f32_e32 v181, s32, v181
	v_bfe_u32 v182, v180, 16, 1
	v_add3_u32 v180, v180, v182, 0x7fff
	v_bfe_u32 v182, v181, 16, 1
	v_add3_u32 v181, v181, v182, 0x7fff
	v_perm_b32 v90, v181, v180, 0x7060302
	v_lshlrev_b32_e32 v180, 16, v91
	v_and_b32_e32 v181, 0xffff0000, v91
	v_mul_f32_e32 v180, s32, v180
	v_mul_f32_e32 v181, s32, v181
	v_bfe_u32 v182, v180, 16, 1
	v_add3_u32 v180, v180, v182, 0x7fff
	v_bfe_u32 v182, v181, 16, 1
	v_add3_u32 v181, v181, v182, 0x7fff
	v_perm_b32 v91, v181, v180, 0x7060302
	v_lshlrev_b32_e32 v180, 16, v92
	v_and_b32_e32 v181, 0xffff0000, v92
	v_mul_f32_e32 v180, s32, v180
	v_mul_f32_e32 v181, s32, v181
	v_bfe_u32 v182, v180, 16, 1
	v_add3_u32 v180, v180, v182, 0x7fff
	v_bfe_u32 v182, v181, 16, 1
	v_add3_u32 v181, v181, v182, 0x7fff
	v_perm_b32 v92, v181, v180, 0x7060302
	v_lshlrev_b32_e32 v180, 16, v93
	v_and_b32_e32 v181, 0xffff0000, v93
	v_mul_f32_e32 v180, s32, v180
	v_mul_f32_e32 v181, s32, v181
	v_bfe_u32 v182, v180, 16, 1
	v_add3_u32 v180, v180, v182, 0x7fff
	v_bfe_u32 v182, v181, 16, 1
	v_add3_u32 v181, v181, v182, 0x7fff
	v_perm_b32 v93, v181, v180, 0x7060302
	v_lshlrev_b32_e32 v180, 16, v94
	v_and_b32_e32 v181, 0xffff0000, v94
	v_mul_f32_e32 v180, s32, v180
	v_mul_f32_e32 v181, s32, v181
	v_bfe_u32 v182, v180, 16, 1
	v_add3_u32 v180, v180, v182, 0x7fff
	v_bfe_u32 v182, v181, 16, 1
	v_add3_u32 v181, v181, v182, 0x7fff
	v_perm_b32 v94, v181, v180, 0x7060302
	v_lshlrev_b32_e32 v180, 16, v95
	v_and_b32_e32 v181, 0xffff0000, v95
	v_mul_f32_e32 v180, s32, v180
	v_mul_f32_e32 v181, s32, v181
	v_bfe_u32 v182, v180, 16, 1
	v_add3_u32 v180, v180, v182, 0x7fff
	v_bfe_u32 v182, v181, 16, 1
	v_add3_u32 v181, v181, v182, 0x7fff
	v_perm_b32 v95, v181, v180, 0x7060302
	v_lshlrev_b32_e32 v180, 16, v96
	v_and_b32_e32 v181, 0xffff0000, v96
	v_mul_f32_e32 v180, s32, v180
	v_mul_f32_e32 v181, s32, v181
	v_bfe_u32 v182, v180, 16, 1
	v_add3_u32 v180, v180, v182, 0x7fff
	v_bfe_u32 v182, v181, 16, 1
	v_add3_u32 v181, v181, v182, 0x7fff
	v_perm_b32 v96, v181, v180, 0x7060302

	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v4, 0
	v_dual_mov_b32 v5, 0 :: v_dual_mov_b32 v6, 0
	v_dual_mov_b32 v7, 0 :: v_dual_mov_b32 v8, 0
	v_dual_mov_b32 v9, 0 :: v_dual_mov_b32 v10, 0
	v_dual_mov_b32 v11, 0 :: v_dual_mov_b32 v12, 0
	v_dual_mov_b32 v13, 0 :: v_dual_mov_b32 v14, 0
	v_dual_mov_b32 v15, 0 :: v_dual_mov_b32 v16, 0
	v_dual_mov_b32 v17, 0 :: v_dual_mov_b32 v18, 0
	v_dual_mov_b32 v19, 0 :: v_dual_mov_b32 v20, 0
	v_dual_mov_b32 v21, 0 :: v_dual_mov_b32 v22, 0
	v_dual_mov_b32 v23, 0 :: v_dual_mov_b32 v24, 0
	v_dual_mov_b32 v25, 0 :: v_dual_mov_b32 v26, 0
	v_dual_mov_b32 v27, 0 :: v_dual_mov_b32 v28, 0
	v_dual_mov_b32 v29, 0 :: v_dual_mov_b32 v30, 0
	v_dual_mov_b32 v31, 0 :: v_dual_mov_b32 v32, 0
	v_dual_mov_b32 v33, 0 :: v_dual_mov_b32 v34, 0
	v_dual_mov_b32 v35, 0 :: v_dual_mov_b32 v36, 0
	v_dual_mov_b32 v37, 0 :: v_dual_mov_b32 v38, 0
	v_dual_mov_b32 v39, 0 :: v_dual_mov_b32 v40, 0
	v_dual_mov_b32 v41, 0 :: v_dual_mov_b32 v42, 0
	v_dual_mov_b32 v43, 0 :: v_dual_mov_b32 v44, 0
	v_dual_mov_b32 v45, 0 :: v_dual_mov_b32 v46, 0
	v_dual_mov_b32 v47, 0 :: v_dual_mov_b32 v48, 0
	v_dual_mov_b32 v49, 0 :: v_dual_mov_b32 v50, 0
	v_dual_mov_b32 v51, 0 :: v_dual_mov_b32 v52, 0
	v_dual_mov_b32 v53, 0 :: v_dual_mov_b32 v54, 0
	v_dual_mov_b32 v55, 0 :: v_dual_mov_b32 v56, 0
	v_dual_mov_b32 v57, 0 :: v_dual_mov_b32 v58, 0
	v_dual_mov_b32 v59, 0 :: v_dual_mov_b32 v60, 0
	v_dual_mov_b32 v61, 0 :: v_dual_mov_b32 v62, 0
	v_dual_mov_b32 v63, 0 :: v_dual_mov_b32 v64, 0

	v_mov_b32_e32 v155, 0xff7fc99e  ; m_row = -3.4e38
	v_mov_b32_e32 v156, 0           ; l_row = 0

	s_lshl_b32 s33, s15, 5   ; 16*D*2 bytes
	s_lshl_b32 s34, s14, 1   ; N*2 bytes
	s_lshr_b32 s35, s14, 6      ; N/64
	;s_mov_b32 s35, 1              ; DEBUG: single tile
	s_cmp_eq_u32 s35, 0
	s_cbranch_scc1 .L_epilogue
	s_mov_b32 s36, 0

.L_n_tile_loop:

	s_lshl_b32 s37, s36, 6
	s_wait_alu 0xfffe

	v_mov_b32_e32 v180, s37
	v_add_nc_u32_e32 v180, v159, v180  ; n_base + col16
	v_mul_lo_u32 v180, s15, v180
	v_add_nc_u32_e32 v180, v158, v180  ; + row8_elem
	v_lshlrev_b32_e32 v180, 1, v180    ; bytes
	v_ashrrev_i32_e32 v181, 31, v180
	v_add_co_u32 v145, vcc_lo, s24, v180
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v146, null, s25, v181, vcc_lo

	s_ashr_i32 s38, s33, 31
	s_wait_alu 0xfffe
	v_add_co_u32 v147, vcc_lo, v145, s33
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v148, null, v146, s38, vcc_lo

	s_lshl_b32 s39, s33, 1
	s_ashr_i32 s40, s39, 31
	s_wait_alu 0xfffe
	v_add_co_u32 v149, vcc_lo, v145, s39
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v150, null, v146, s40, vcc_lo

	s_mul_i32 s39, s33, 3
	s_ashr_i32 s40, s39, 31
	s_wait_alu 0xfffe
	v_add_co_u32 v151, vcc_lo, v145, s39
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v152, null, v146, s40, vcc_lo

	v_dual_mov_b32 v97, 0 :: v_dual_mov_b32 v98, 0
	v_dual_mov_b32 v99, 0 :: v_dual_mov_b32 v100, 0
	v_dual_mov_b32 v101, 0 :: v_dual_mov_b32 v102, 0
	v_dual_mov_b32 v103, 0 :: v_dual_mov_b32 v104, 0
	v_dual_mov_b32 v105, 0 :: v_dual_mov_b32 v106, 0
	v_dual_mov_b32 v107, 0 :: v_dual_mov_b32 v108, 0
	v_dual_mov_b32 v109, 0 :: v_dual_mov_b32 v110, 0
	v_dual_mov_b32 v111, 0 :: v_dual_mov_b32 v112, 0
	v_dual_mov_b32 v113, 0 :: v_dual_mov_b32 v114, 0
	v_dual_mov_b32 v115, 0 :: v_dual_mov_b32 v116, 0
	v_dual_mov_b32 v117, 0 :: v_dual_mov_b32 v118, 0
	v_dual_mov_b32 v119, 0 :: v_dual_mov_b32 v120, 0
	v_dual_mov_b32 v121, 0 :: v_dual_mov_b32 v122, 0
	v_dual_mov_b32 v123, 0 :: v_dual_mov_b32 v124, 0
	v_dual_mov_b32 v125, 0 :: v_dual_mov_b32 v126, 0
	v_dual_mov_b32 v127, 0 :: v_dual_mov_b32 v128, 0

	; ===== QKT: 32 WMMAs =====
	s_clause 0x1
	global_load_b128 v[129:132], v[145:146], off offset:0
	global_load_b128 v[133:136], v[147:148], off offset:0
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_bf16 v[97:104], v[129:132], v[65:68], v[97:104]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_bf16 v[105:112], v[133:136], v[65:68], v[105:112]
	s_clause 0x1
	global_load_b128 v[129:132], v[149:150], off offset:0
	global_load_b128 v[133:136], v[151:152], off offset:0
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_bf16 v[113:120], v[129:132], v[65:68], v[113:120]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_bf16 v[121:128], v[133:136], v[65:68], v[121:128]

	s_clause 0x1
	global_load_b128 v[129:132], v[145:146], off offset:32
	global_load_b128 v[133:136], v[147:148], off offset:32
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_bf16 v[97:104], v[129:132], v[69:72], v[97:104]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_bf16 v[105:112], v[133:136], v[69:72], v[105:112]
	s_clause 0x1
	global_load_b128 v[129:132], v[149:150], off offset:32
	global_load_b128 v[133:136], v[151:152], off offset:32
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_bf16 v[113:120], v[129:132], v[69:72], v[113:120]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_bf16 v[121:128], v[133:136], v[69:72], v[121:128]

	s_clause 0x1
	global_load_b128 v[129:132], v[145:146], off offset:64
	global_load_b128 v[133:136], v[147:148], off offset:64
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_bf16 v[97:104], v[129:132], v[73:76], v[97:104]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_bf16 v[105:112], v[133:136], v[73:76], v[105:112]
	s_clause 0x1
	global_load_b128 v[129:132], v[149:150], off offset:64
	global_load_b128 v[133:136], v[151:152], off offset:64
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_bf16 v[113:120], v[129:132], v[73:76], v[113:120]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_bf16 v[121:128], v[133:136], v[73:76], v[121:128]

	s_clause 0x1
	global_load_b128 v[129:132], v[145:146], off offset:96
	global_load_b128 v[133:136], v[147:148], off offset:96
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_bf16 v[97:104], v[129:132], v[77:80], v[97:104]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_bf16 v[105:112], v[133:136], v[77:80], v[105:112]
	s_clause 0x1
	global_load_b128 v[129:132], v[149:150], off offset:96
	global_load_b128 v[133:136], v[151:152], off offset:96
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_bf16 v[113:120], v[129:132], v[77:80], v[113:120]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_bf16 v[121:128], v[133:136], v[77:80], v[121:128]

	s_clause 0x1
	global_load_b128 v[129:132], v[145:146], off offset:128
	global_load_b128 v[133:136], v[147:148], off offset:128
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_bf16 v[97:104], v[129:132], v[81:84], v[97:104]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_bf16 v[105:112], v[133:136], v[81:84], v[105:112]
	s_clause 0x1
	global_load_b128 v[129:132], v[149:150], off offset:128
	global_load_b128 v[133:136], v[151:152], off offset:128
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_bf16 v[113:120], v[129:132], v[81:84], v[113:120]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_bf16 v[121:128], v[133:136], v[81:84], v[121:128]

	s_clause 0x1
	global_load_b128 v[129:132], v[145:146], off offset:160
	global_load_b128 v[133:136], v[147:148], off offset:160
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_bf16 v[97:104], v[129:132], v[85:88], v[97:104]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_bf16 v[105:112], v[133:136], v[85:88], v[105:112]
	s_clause 0x1
	global_load_b128 v[129:132], v[149:150], off offset:160
	global_load_b128 v[133:136], v[151:152], off offset:160
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_bf16 v[113:120], v[129:132], v[85:88], v[113:120]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_bf16 v[121:128], v[133:136], v[85:88], v[121:128]

	s_clause 0x1
	global_load_b128 v[129:132], v[145:146], off offset:192
	global_load_b128 v[133:136], v[147:148], off offset:192
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_bf16 v[97:104], v[129:132], v[89:92], v[97:104]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_bf16 v[105:112], v[133:136], v[89:92], v[105:112]
	s_clause 0x1
	global_load_b128 v[129:132], v[149:150], off offset:192
	global_load_b128 v[133:136], v[151:152], off offset:192
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_bf16 v[113:120], v[129:132], v[89:92], v[113:120]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_bf16 v[121:128], v[133:136], v[89:92], v[121:128]

	s_clause 0x1
	global_load_b128 v[129:132], v[145:146], off offset:224
	global_load_b128 v[133:136], v[147:148], off offset:224
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_bf16 v[97:104], v[129:132], v[93:96], v[97:104]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_bf16 v[105:112], v[133:136], v[93:96], v[105:112]
	s_clause 0x1
	global_load_b128 v[129:132], v[149:150], off offset:224
	global_load_b128 v[133:136], v[151:152], off offset:224
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_bf16 v[113:120], v[129:132], v[93:96], v[113:120]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_bf16 v[121:128], v[133:136], v[93:96], v[121:128]

	; ===== SOFTMAX =====

	v_max_num_f32_e32 v129, v97, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_max3_num_f32 v129, v129, v99, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_max3_num_f32 v129, v129, v101, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_max3_num_f32 v129, v129, v103, v104
	s_delay_alu instid0(VALU_DEP_1)
	v_max3_num_f32 v129, v129, v105, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_max3_num_f32 v129, v129, v107, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_max3_num_f32 v129, v129, v109, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_max3_num_f32 v129, v129, v111, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_max3_num_f32 v129, v129, v113, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_max3_num_f32 v129, v129, v115, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_max3_num_f32 v129, v129, v117, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_max3_num_f32 v129, v129, v119, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_max3_num_f32 v129, v129, v121, v122
	s_delay_alu instid0(VALU_DEP_1)
	v_max3_num_f32 v129, v129, v123, v124
	s_delay_alu instid0(VALU_DEP_1)
	v_max3_num_f32 v129, v129, v125, v126
	s_delay_alu instid0(VALU_DEP_1)
	v_max3_num_f32 v129, v129, v127, v128

	ds_bpermute_b32 v130, v157, v129
	s_wait_dscnt 0x0
	v_max3_num_f32 v130, v155, v129, v130

	v_sub_f32_e32 v131, v155, v130
	v_cmp_neq_f32_e32 vcc_lo, v130, v155
	v_mov_b32_e32 v155, v130       ; m_row = new_m
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v131, v131       ; rescale
	s_wait_alu 0xfffe
	s_delay_alu instid0(TRANS32_DEP_1)
	v_cndmask_b32_e32 v189, 1.0, v131, vcc_lo
	v_mul_f32_e32 v156, v189, v156 ; l_row *= rescale

	v_sub_f32_e32 v97, v97, v155
	v_sub_f32_e32 v98, v98, v155
	v_sub_f32_e32 v99, v99, v155
	v_sub_f32_e32 v100, v100, v155
	v_sub_f32_e32 v101, v101, v155
	v_sub_f32_e32 v102, v102, v155
	v_sub_f32_e32 v103, v103, v155
	v_sub_f32_e32 v104, v104, v155
	v_sub_f32_e32 v105, v105, v155
	v_sub_f32_e32 v106, v106, v155
	v_sub_f32_e32 v107, v107, v155
	v_sub_f32_e32 v108, v108, v155
	v_sub_f32_e32 v109, v109, v155
	v_sub_f32_e32 v110, v110, v155
	v_sub_f32_e32 v111, v111, v155
	v_sub_f32_e32 v112, v112, v155
	v_sub_f32_e32 v113, v113, v155
	v_sub_f32_e32 v114, v114, v155
	v_sub_f32_e32 v115, v115, v155
	v_sub_f32_e32 v116, v116, v155
	v_sub_f32_e32 v117, v117, v155
	v_sub_f32_e32 v118, v118, v155
	v_sub_f32_e32 v119, v119, v155
	v_sub_f32_e32 v120, v120, v155
	v_sub_f32_e32 v121, v121, v155
	v_sub_f32_e32 v122, v122, v155
	v_sub_f32_e32 v123, v123, v155
	v_sub_f32_e32 v124, v124, v155
	v_sub_f32_e32 v125, v125, v155
	v_sub_f32_e32 v126, v126, v155
	v_sub_f32_e32 v127, v127, v155
	v_sub_f32_e32 v128, v128, v155

	v_exp_f32_e32 v97, v97
	v_exp_f32_e32 v98, v98
	v_exp_f32_e32 v99, v99
	v_exp_f32_e32 v100, v100
	v_exp_f32_e32 v101, v101
	v_exp_f32_e32 v102, v102
	v_exp_f32_e32 v103, v103
	v_exp_f32_e32 v104, v104
	v_exp_f32_e32 v105, v105
	v_exp_f32_e32 v106, v106
	v_exp_f32_e32 v107, v107
	v_exp_f32_e32 v108, v108
	v_exp_f32_e32 v109, v109
	v_exp_f32_e32 v110, v110
	v_exp_f32_e32 v111, v111
	v_exp_f32_e32 v112, v112
	v_exp_f32_e32 v113, v113
	v_exp_f32_e32 v114, v114
	v_exp_f32_e32 v115, v115
	v_exp_f32_e32 v116, v116
	v_exp_f32_e32 v117, v117
	v_exp_f32_e32 v118, v118
	v_exp_f32_e32 v119, v119
	v_exp_f32_e32 v120, v120
	v_exp_f32_e32 v121, v121
	v_exp_f32_e32 v122, v122
	v_exp_f32_e32 v123, v123
	v_exp_f32_e32 v124, v124
	v_exp_f32_e32 v125, v125
	v_exp_f32_e32 v126, v126
	v_exp_f32_e32 v127, v127
	v_exp_f32_e32 v128, v128

	s_delay_alu instid0(TRANS32_DEP_1)
	v_add_f32_e32 v129, v97, v98
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v99
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v100
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v102
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v104
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v121
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v122
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v123
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v124
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v125
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v126
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v127
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v129, v129, v128

	ds_bpermute_b32 v130, v157, v129
	s_wait_dscnt 0x0
	v_add_f32_e32 v129, v129, v130
	v_add_f32_e32 v156, v156, v129 ; l_row += row_sum

	v_bfe_u32 v129, v97, 16, 1
	v_add3_u32 v97, v97, v129, 0x7fff
	v_bfe_u32 v129, v98, 16, 1
	v_add3_u32 v98, v98, v129, 0x7fff
	v_perm_b32 v160, v98, v97, 0x7060302
	v_bfe_u32 v129, v99, 16, 1
	v_add3_u32 v99, v99, v129, 0x7fff
	v_bfe_u32 v129, v100, 16, 1
	v_add3_u32 v100, v100, v129, 0x7fff
	v_perm_b32 v161, v100, v99, 0x7060302
	v_bfe_u32 v129, v101, 16, 1
	v_add3_u32 v101, v101, v129, 0x7fff
	v_bfe_u32 v129, v102, 16, 1
	v_add3_u32 v102, v102, v129, 0x7fff
	v_perm_b32 v162, v102, v101, 0x7060302
	v_bfe_u32 v129, v103, 16, 1
	v_add3_u32 v103, v103, v129, 0x7fff
	v_bfe_u32 v129, v104, 16, 1
	v_add3_u32 v104, v104, v129, 0x7fff
	v_perm_b32 v163, v104, v103, 0x7060302
	v_bfe_u32 v129, v105, 16, 1
	v_add3_u32 v105, v105, v129, 0x7fff
	v_bfe_u32 v129, v106, 16, 1
	v_add3_u32 v106, v106, v129, 0x7fff
	v_perm_b32 v164, v106, v105, 0x7060302
	v_bfe_u32 v129, v107, 16, 1
	v_add3_u32 v107, v107, v129, 0x7fff
	v_bfe_u32 v129, v108, 16, 1
	v_add3_u32 v108, v108, v129, 0x7fff
	v_perm_b32 v165, v108, v107, 0x7060302
	v_bfe_u32 v129, v109, 16, 1
	v_add3_u32 v109, v109, v129, 0x7fff
	v_bfe_u32 v129, v110, 16, 1
	v_add3_u32 v110, v110, v129, 0x7fff
	v_perm_b32 v166, v110, v109, 0x7060302
	v_bfe_u32 v129, v111, 16, 1
	v_add3_u32 v111, v111, v129, 0x7fff
	v_bfe_u32 v129, v112, 16, 1
	v_add3_u32 v112, v112, v129, 0x7fff
	v_perm_b32 v167, v112, v111, 0x7060302
	v_bfe_u32 v129, v113, 16, 1
	v_add3_u32 v113, v113, v129, 0x7fff
	v_bfe_u32 v129, v114, 16, 1
	v_add3_u32 v114, v114, v129, 0x7fff
	v_perm_b32 v168, v114, v113, 0x7060302
	v_bfe_u32 v129, v115, 16, 1
	v_add3_u32 v115, v115, v129, 0x7fff
	v_bfe_u32 v129, v116, 16, 1
	v_add3_u32 v116, v116, v129, 0x7fff
	v_perm_b32 v169, v116, v115, 0x7060302
	v_bfe_u32 v129, v117, 16, 1
	v_add3_u32 v117, v117, v129, 0x7fff
	v_bfe_u32 v129, v118, 16, 1
	v_add3_u32 v118, v118, v129, 0x7fff
	v_perm_b32 v170, v118, v117, 0x7060302
	v_bfe_u32 v129, v119, 16, 1
	v_add3_u32 v119, v119, v129, 0x7fff
	v_bfe_u32 v129, v120, 16, 1
	v_add3_u32 v120, v120, v129, 0x7fff
	v_perm_b32 v171, v120, v119, 0x7060302
	v_bfe_u32 v129, v121, 16, 1
	v_add3_u32 v121, v121, v129, 0x7fff
	v_bfe_u32 v129, v122, 16, 1
	v_add3_u32 v122, v122, v129, 0x7fff
	v_perm_b32 v172, v122, v121, 0x7060302
	v_bfe_u32 v129, v123, 16, 1
	v_add3_u32 v123, v123, v129, 0x7fff
	v_bfe_u32 v129, v124, 16, 1
	v_add3_u32 v124, v124, v129, 0x7fff
	v_perm_b32 v173, v124, v123, 0x7060302
	v_bfe_u32 v129, v125, 16, 1
	v_add3_u32 v125, v125, v129, 0x7fff
	v_bfe_u32 v129, v126, 16, 1
	v_add3_u32 v126, v126, v129, 0x7fff
	v_perm_b32 v174, v126, v125, 0x7060302
	v_bfe_u32 v129, v127, 16, 1
	v_add3_u32 v127, v127, v129, 0x7fff
	v_bfe_u32 v129, v128, 16, 1
	v_add3_u32 v128, v128, v129, 0x7fff
	v_perm_b32 v175, v128, v127, 0x7060302

.L_pv_start:
	; ===== PV: 32 WMMAs =====

	; PV D-tile 0
	v_mul_lo_u32 v180, s14, v159  ; col16 * N
	v_mov_b32_e32 v181, s37
	v_add_nc_u32_e32 v180, v181, v180  ; + n_base
	v_add_nc_u32_e32 v180, v158, v180  ; + row8_elem
	v_lshlrev_b32_e32 v180, 1, v180    ; bytes
	v_ashrrev_i32_e32 v181, 31, v180
	v_add_co_u32 v153, vcc_lo, s30, v180
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v154, null, s31, v181, vcc_lo

	s_clause 0x3
	global_load_b128 v[129:132], v[153:154], off offset:0
	global_load_b128 v[133:136], v[153:154], off offset:32
	global_load_b128 v[137:140], v[153:154], off offset:64
	global_load_b128 v[141:144], v[153:154], off offset:96

	v_mul_f32_e32 v1, v189, v1
	v_mul_f32_e32 v2, v189, v2
	v_mul_f32_e32 v3, v189, v3
	v_mul_f32_e32 v4, v189, v4
	v_mul_f32_e32 v5, v189, v5
	v_mul_f32_e32 v6, v189, v6
	v_mul_f32_e32 v7, v189, v7
	v_mul_f32_e32 v8, v189, v8

	s_wait_loadcnt 0x3
	v_wmma_f32_16x16x16_bf16 v[1:8], v[129:132], v[160:163], v[1:8]
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x16_bf16 v[1:8], v[133:136], v[164:167], v[1:8]
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_bf16 v[1:8], v[137:140], v[168:171], v[1:8]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_bf16 v[1:8], v[141:144], v[172:175], v[1:8]

	; PV D-tile 1
	v_add_nc_u32_e32 v180, 16, v159
	v_mul_lo_u32 v180, s14, v180
	v_mov_b32_e32 v181, s37
	v_add_nc_u32_e32 v180, v181, v180  ; + n_base
	v_add_nc_u32_e32 v180, v158, v180  ; + row8_elem
	v_lshlrev_b32_e32 v180, 1, v180    ; bytes
	v_ashrrev_i32_e32 v181, 31, v180
	v_add_co_u32 v153, vcc_lo, s30, v180
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v154, null, s31, v181, vcc_lo

	s_clause 0x3
	global_load_b128 v[129:132], v[153:154], off offset:0
	global_load_b128 v[133:136], v[153:154], off offset:32
	global_load_b128 v[137:140], v[153:154], off offset:64
	global_load_b128 v[141:144], v[153:154], off offset:96

	v_mul_f32_e32 v9, v189, v9
	v_mul_f32_e32 v10, v189, v10
	v_mul_f32_e32 v11, v189, v11
	v_mul_f32_e32 v12, v189, v12
	v_mul_f32_e32 v13, v189, v13
	v_mul_f32_e32 v14, v189, v14
	v_mul_f32_e32 v15, v189, v15
	v_mul_f32_e32 v16, v189, v16

	s_wait_loadcnt 0x3
	v_wmma_f32_16x16x16_bf16 v[9:16], v[129:132], v[160:163], v[9:16]
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x16_bf16 v[9:16], v[133:136], v[164:167], v[9:16]
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_bf16 v[9:16], v[137:140], v[168:171], v[9:16]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_bf16 v[9:16], v[141:144], v[172:175], v[9:16]

	; PV D-tile 2
	v_add_nc_u32_e32 v180, 32, v159
	v_mul_lo_u32 v180, s14, v180
	v_mov_b32_e32 v181, s37
	v_add_nc_u32_e32 v180, v181, v180  ; + n_base
	v_add_nc_u32_e32 v180, v158, v180  ; + row8_elem
	v_lshlrev_b32_e32 v180, 1, v180    ; bytes
	v_ashrrev_i32_e32 v181, 31, v180
	v_add_co_u32 v153, vcc_lo, s30, v180
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v154, null, s31, v181, vcc_lo

	s_clause 0x3
	global_load_b128 v[129:132], v[153:154], off offset:0
	global_load_b128 v[133:136], v[153:154], off offset:32
	global_load_b128 v[137:140], v[153:154], off offset:64
	global_load_b128 v[141:144], v[153:154], off offset:96

	v_mul_f32_e32 v17, v189, v17
	v_mul_f32_e32 v18, v189, v18
	v_mul_f32_e32 v19, v189, v19
	v_mul_f32_e32 v20, v189, v20
	v_mul_f32_e32 v21, v189, v21
	v_mul_f32_e32 v22, v189, v22
	v_mul_f32_e32 v23, v189, v23
	v_mul_f32_e32 v24, v189, v24

	s_wait_loadcnt 0x3
	v_wmma_f32_16x16x16_bf16 v[17:24], v[129:132], v[160:163], v[17:24]
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x16_bf16 v[17:24], v[133:136], v[164:167], v[17:24]
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_bf16 v[17:24], v[137:140], v[168:171], v[17:24]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_bf16 v[17:24], v[141:144], v[172:175], v[17:24]

	; PV D-tile 3
	v_add_nc_u32_e32 v180, 48, v159
	v_mul_lo_u32 v180, s14, v180
	v_mov_b32_e32 v181, s37
	v_add_nc_u32_e32 v180, v181, v180  ; + n_base
	v_add_nc_u32_e32 v180, v158, v180  ; + row8_elem
	v_lshlrev_b32_e32 v180, 1, v180    ; bytes
	v_ashrrev_i32_e32 v181, 31, v180
	v_add_co_u32 v153, vcc_lo, s30, v180
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v154, null, s31, v181, vcc_lo

	s_clause 0x3
	global_load_b128 v[129:132], v[153:154], off offset:0
	global_load_b128 v[133:136], v[153:154], off offset:32
	global_load_b128 v[137:140], v[153:154], off offset:64
	global_load_b128 v[141:144], v[153:154], off offset:96

	v_mul_f32_e32 v25, v189, v25
	v_mul_f32_e32 v26, v189, v26
	v_mul_f32_e32 v27, v189, v27
	v_mul_f32_e32 v28, v189, v28
	v_mul_f32_e32 v29, v189, v29
	v_mul_f32_e32 v30, v189, v30
	v_mul_f32_e32 v31, v189, v31
	v_mul_f32_e32 v32, v189, v32

	s_wait_loadcnt 0x3
	v_wmma_f32_16x16x16_bf16 v[25:32], v[129:132], v[160:163], v[25:32]
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x16_bf16 v[25:32], v[133:136], v[164:167], v[25:32]
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_bf16 v[25:32], v[137:140], v[168:171], v[25:32]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_bf16 v[25:32], v[141:144], v[172:175], v[25:32]

	; PV D-tile 4
	v_add_nc_u32_e32 v180, 64, v159
	v_mul_lo_u32 v180, s14, v180
	v_mov_b32_e32 v181, s37
	v_add_nc_u32_e32 v180, v181, v180  ; + n_base
	v_add_nc_u32_e32 v180, v158, v180  ; + row8_elem
	v_lshlrev_b32_e32 v180, 1, v180    ; bytes
	v_ashrrev_i32_e32 v181, 31, v180
	v_add_co_u32 v153, vcc_lo, s30, v180
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v154, null, s31, v181, vcc_lo

	s_clause 0x3
	global_load_b128 v[129:132], v[153:154], off offset:0
	global_load_b128 v[133:136], v[153:154], off offset:32
	global_load_b128 v[137:140], v[153:154], off offset:64
	global_load_b128 v[141:144], v[153:154], off offset:96

	v_mul_f32_e32 v33, v189, v33
	v_mul_f32_e32 v34, v189, v34
	v_mul_f32_e32 v35, v189, v35
	v_mul_f32_e32 v36, v189, v36
	v_mul_f32_e32 v37, v189, v37
	v_mul_f32_e32 v38, v189, v38
	v_mul_f32_e32 v39, v189, v39
	v_mul_f32_e32 v40, v189, v40

	s_wait_loadcnt 0x3
	v_wmma_f32_16x16x16_bf16 v[33:40], v[129:132], v[160:163], v[33:40]
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x16_bf16 v[33:40], v[133:136], v[164:167], v[33:40]
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_bf16 v[33:40], v[137:140], v[168:171], v[33:40]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_bf16 v[33:40], v[141:144], v[172:175], v[33:40]

	; PV D-tile 5
	v_add_nc_u32_e32 v180, 80, v159
	v_mul_lo_u32 v180, s14, v180
	v_mov_b32_e32 v181, s37
	v_add_nc_u32_e32 v180, v181, v180  ; + n_base
	v_add_nc_u32_e32 v180, v158, v180  ; + row8_elem
	v_lshlrev_b32_e32 v180, 1, v180    ; bytes
	v_ashrrev_i32_e32 v181, 31, v180
	v_add_co_u32 v153, vcc_lo, s30, v180
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v154, null, s31, v181, vcc_lo

	s_clause 0x3
	global_load_b128 v[129:132], v[153:154], off offset:0
	global_load_b128 v[133:136], v[153:154], off offset:32
	global_load_b128 v[137:140], v[153:154], off offset:64
	global_load_b128 v[141:144], v[153:154], off offset:96

	v_mul_f32_e32 v41, v189, v41
	v_mul_f32_e32 v42, v189, v42
	v_mul_f32_e32 v43, v189, v43
	v_mul_f32_e32 v44, v189, v44
	v_mul_f32_e32 v45, v189, v45
	v_mul_f32_e32 v46, v189, v46
	v_mul_f32_e32 v47, v189, v47
	v_mul_f32_e32 v48, v189, v48

	s_wait_loadcnt 0x3
	v_wmma_f32_16x16x16_bf16 v[41:48], v[129:132], v[160:163], v[41:48]
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x16_bf16 v[41:48], v[133:136], v[164:167], v[41:48]
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_bf16 v[41:48], v[137:140], v[168:171], v[41:48]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_bf16 v[41:48], v[141:144], v[172:175], v[41:48]

	; PV D-tile 6
	v_add_nc_u32_e32 v180, 96, v159
	v_mul_lo_u32 v180, s14, v180
	v_mov_b32_e32 v181, s37
	v_add_nc_u32_e32 v180, v181, v180  ; + n_base
	v_add_nc_u32_e32 v180, v158, v180  ; + row8_elem
	v_lshlrev_b32_e32 v180, 1, v180    ; bytes
	v_ashrrev_i32_e32 v181, 31, v180
	v_add_co_u32 v153, vcc_lo, s30, v180
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v154, null, s31, v181, vcc_lo

	s_clause 0x3
	global_load_b128 v[129:132], v[153:154], off offset:0
	global_load_b128 v[133:136], v[153:154], off offset:32
	global_load_b128 v[137:140], v[153:154], off offset:64
	global_load_b128 v[141:144], v[153:154], off offset:96

	v_mul_f32_e32 v49, v189, v49
	v_mul_f32_e32 v50, v189, v50
	v_mul_f32_e32 v51, v189, v51
	v_mul_f32_e32 v52, v189, v52
	v_mul_f32_e32 v53, v189, v53
	v_mul_f32_e32 v54, v189, v54
	v_mul_f32_e32 v55, v189, v55
	v_mul_f32_e32 v56, v189, v56

	s_wait_loadcnt 0x3
	v_wmma_f32_16x16x16_bf16 v[49:56], v[129:132], v[160:163], v[49:56]
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x16_bf16 v[49:56], v[133:136], v[164:167], v[49:56]
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_bf16 v[49:56], v[137:140], v[168:171], v[49:56]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_bf16 v[49:56], v[141:144], v[172:175], v[49:56]

	; PV D-tile 7
	v_add_nc_u32_e32 v180, 112, v159
	v_mul_lo_u32 v180, s14, v180
	v_mov_b32_e32 v181, s37
	v_add_nc_u32_e32 v180, v181, v180  ; + n_base
	v_add_nc_u32_e32 v180, v158, v180  ; + row8_elem
	v_lshlrev_b32_e32 v180, 1, v180    ; bytes
	v_ashrrev_i32_e32 v181, 31, v180
	v_add_co_u32 v153, vcc_lo, s30, v180
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v154, null, s31, v181, vcc_lo

	s_clause 0x3
	global_load_b128 v[129:132], v[153:154], off offset:0
	global_load_b128 v[133:136], v[153:154], off offset:32
	global_load_b128 v[137:140], v[153:154], off offset:64
	global_load_b128 v[141:144], v[153:154], off offset:96

	v_mul_f32_e32 v57, v189, v57
	v_mul_f32_e32 v58, v189, v58
	v_mul_f32_e32 v59, v189, v59
	v_mul_f32_e32 v60, v189, v60
	v_mul_f32_e32 v61, v189, v61
	v_mul_f32_e32 v62, v189, v62
	v_mul_f32_e32 v63, v189, v63
	v_mul_f32_e32 v64, v189, v64

	s_wait_loadcnt 0x3
	v_wmma_f32_16x16x16_bf16 v[57:64], v[129:132], v[160:163], v[57:64]
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x16_bf16 v[57:64], v[133:136], v[164:167], v[57:64]
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x16_bf16 v[57:64], v[137:140], v[168:171], v[57:64]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_bf16 v[57:64], v[141:144], v[172:175], v[57:64]

	s_add_i32 s36, s36, 1
	s_wait_alu 0xfffe
	s_cmp_lt_u32 s36, s35
	s_cbranch_scc1 .L_n_tile_loop

.L_epilogue:

	v_cmp_lt_f32_e32 vcc_lo, 0, v156  ; l_row > 0?
	v_rcp_f32_e32 v129, v156
	s_wait_alu 0xfffe
	s_delay_alu instid0(TRANS32_DEP_1)
	v_fma_f32 v130, -v156, v129, 1.0
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v129, v130, v129, v129
	v_cndmask_b32_e32 v129, 0, v129, vcc_lo

	v_add_co_u32 v178, vcc_lo, s26, v176
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v179, null, s27, v177, vcc_lo

	v_mul_f32_e32 v1, v129, v1
	v_mul_f32_e32 v2, v129, v2
	v_mul_f32_e32 v3, v129, v3
	v_mul_f32_e32 v4, v129, v4
	v_mul_f32_e32 v5, v129, v5
	v_mul_f32_e32 v6, v129, v6
	v_mul_f32_e32 v7, v129, v7
	v_mul_f32_e32 v8, v129, v8
	v_bfe_u32 v130, v1, 16, 1
	v_add3_u32 v1, v1, v130, 0x7fff
	v_bfe_u32 v130, v2, 16, 1
	v_add3_u32 v2, v2, v130, 0x7fff
	v_perm_b32 v1, v2, v1, 0x7060302
	v_bfe_u32 v130, v3, 16, 1
	v_add3_u32 v3, v3, v130, 0x7fff
	v_bfe_u32 v130, v4, 16, 1
	v_add3_u32 v4, v4, v130, 0x7fff
	v_perm_b32 v2, v4, v3, 0x7060302
	v_bfe_u32 v130, v5, 16, 1
	v_add3_u32 v5, v5, v130, 0x7fff
	v_bfe_u32 v130, v6, 16, 1
	v_add3_u32 v6, v6, v130, 0x7fff
	v_perm_b32 v3, v6, v5, 0x7060302
	v_bfe_u32 v130, v7, 16, 1
	v_add3_u32 v7, v7, v130, 0x7fff
	v_bfe_u32 v130, v8, 16, 1
	v_add3_u32 v8, v8, v130, 0x7fff
	v_perm_b32 v4, v8, v7, 0x7060302
	global_store_b128 v[178:179], v[1:4], off offset:0

	v_mul_f32_e32 v9, v129, v9
	v_mul_f32_e32 v10, v129, v10
	v_mul_f32_e32 v11, v129, v11
	v_mul_f32_e32 v12, v129, v12
	v_mul_f32_e32 v13, v129, v13
	v_mul_f32_e32 v14, v129, v14
	v_mul_f32_e32 v15, v129, v15
	v_mul_f32_e32 v16, v129, v16
	v_bfe_u32 v130, v9, 16, 1
	v_add3_u32 v9, v9, v130, 0x7fff
	v_bfe_u32 v130, v10, 16, 1
	v_add3_u32 v10, v10, v130, 0x7fff
	v_perm_b32 v9, v10, v9, 0x7060302
	v_bfe_u32 v130, v11, 16, 1
	v_add3_u32 v11, v11, v130, 0x7fff
	v_bfe_u32 v130, v12, 16, 1
	v_add3_u32 v12, v12, v130, 0x7fff
	v_perm_b32 v10, v12, v11, 0x7060302
	v_bfe_u32 v130, v13, 16, 1
	v_add3_u32 v13, v13, v130, 0x7fff
	v_bfe_u32 v130, v14, 16, 1
	v_add3_u32 v14, v14, v130, 0x7fff
	v_perm_b32 v11, v14, v13, 0x7060302
	v_bfe_u32 v130, v15, 16, 1
	v_add3_u32 v15, v15, v130, 0x7fff
	v_bfe_u32 v130, v16, 16, 1
	v_add3_u32 v16, v16, v130, 0x7fff
	v_perm_b32 v12, v16, v15, 0x7060302
	global_store_b128 v[178:179], v[9:12], off offset:32

	v_mul_f32_e32 v17, v129, v17
	v_mul_f32_e32 v18, v129, v18
	v_mul_f32_e32 v19, v129, v19
	v_mul_f32_e32 v20, v129, v20
	v_mul_f32_e32 v21, v129, v21
	v_mul_f32_e32 v22, v129, v22
	v_mul_f32_e32 v23, v129, v23
	v_mul_f32_e32 v24, v129, v24
	v_bfe_u32 v130, v17, 16, 1
	v_add3_u32 v17, v17, v130, 0x7fff
	v_bfe_u32 v130, v18, 16, 1
	v_add3_u32 v18, v18, v130, 0x7fff
	v_perm_b32 v17, v18, v17, 0x7060302
	v_bfe_u32 v130, v19, 16, 1
	v_add3_u32 v19, v19, v130, 0x7fff
	v_bfe_u32 v130, v20, 16, 1
	v_add3_u32 v20, v20, v130, 0x7fff
	v_perm_b32 v18, v20, v19, 0x7060302
	v_bfe_u32 v130, v21, 16, 1
	v_add3_u32 v21, v21, v130, 0x7fff
	v_bfe_u32 v130, v22, 16, 1
	v_add3_u32 v22, v22, v130, 0x7fff
	v_perm_b32 v19, v22, v21, 0x7060302
	v_bfe_u32 v130, v23, 16, 1
	v_add3_u32 v23, v23, v130, 0x7fff
	v_bfe_u32 v130, v24, 16, 1
	v_add3_u32 v24, v24, v130, 0x7fff
	v_perm_b32 v20, v24, v23, 0x7060302
	global_store_b128 v[178:179], v[17:20], off offset:64

	v_mul_f32_e32 v25, v129, v25
	v_mul_f32_e32 v26, v129, v26
	v_mul_f32_e32 v27, v129, v27
	v_mul_f32_e32 v28, v129, v28
	v_mul_f32_e32 v29, v129, v29
	v_mul_f32_e32 v30, v129, v30
	v_mul_f32_e32 v31, v129, v31
	v_mul_f32_e32 v32, v129, v32
	v_bfe_u32 v130, v25, 16, 1
	v_add3_u32 v25, v25, v130, 0x7fff
	v_bfe_u32 v130, v26, 16, 1
	v_add3_u32 v26, v26, v130, 0x7fff
	v_perm_b32 v25, v26, v25, 0x7060302
	v_bfe_u32 v130, v27, 16, 1
	v_add3_u32 v27, v27, v130, 0x7fff
	v_bfe_u32 v130, v28, 16, 1
	v_add3_u32 v28, v28, v130, 0x7fff
	v_perm_b32 v26, v28, v27, 0x7060302
	v_bfe_u32 v130, v29, 16, 1
	v_add3_u32 v29, v29, v130, 0x7fff
	v_bfe_u32 v130, v30, 16, 1
	v_add3_u32 v30, v30, v130, 0x7fff
	v_perm_b32 v27, v30, v29, 0x7060302
	v_bfe_u32 v130, v31, 16, 1
	v_add3_u32 v31, v31, v130, 0x7fff
	v_bfe_u32 v130, v32, 16, 1
	v_add3_u32 v32, v32, v130, 0x7fff
	v_perm_b32 v28, v32, v31, 0x7060302
	global_store_b128 v[178:179], v[25:28], off offset:96

	v_mul_f32_e32 v33, v129, v33
	v_mul_f32_e32 v34, v129, v34
	v_mul_f32_e32 v35, v129, v35
	v_mul_f32_e32 v36, v129, v36
	v_mul_f32_e32 v37, v129, v37
	v_mul_f32_e32 v38, v129, v38
	v_mul_f32_e32 v39, v129, v39
	v_mul_f32_e32 v40, v129, v40
	v_bfe_u32 v130, v33, 16, 1
	v_add3_u32 v33, v33, v130, 0x7fff
	v_bfe_u32 v130, v34, 16, 1
	v_add3_u32 v34, v34, v130, 0x7fff
	v_perm_b32 v33, v34, v33, 0x7060302
	v_bfe_u32 v130, v35, 16, 1
	v_add3_u32 v35, v35, v130, 0x7fff
	v_bfe_u32 v130, v36, 16, 1
	v_add3_u32 v36, v36, v130, 0x7fff
	v_perm_b32 v34, v36, v35, 0x7060302
	v_bfe_u32 v130, v37, 16, 1
	v_add3_u32 v37, v37, v130, 0x7fff
	v_bfe_u32 v130, v38, 16, 1
	v_add3_u32 v38, v38, v130, 0x7fff
	v_perm_b32 v35, v38, v37, 0x7060302
	v_bfe_u32 v130, v39, 16, 1
	v_add3_u32 v39, v39, v130, 0x7fff
	v_bfe_u32 v130, v40, 16, 1
	v_add3_u32 v40, v40, v130, 0x7fff
	v_perm_b32 v36, v40, v39, 0x7060302
	global_store_b128 v[178:179], v[33:36], off offset:128

	v_mul_f32_e32 v41, v129, v41
	v_mul_f32_e32 v42, v129, v42
	v_mul_f32_e32 v43, v129, v43
	v_mul_f32_e32 v44, v129, v44
	v_mul_f32_e32 v45, v129, v45
	v_mul_f32_e32 v46, v129, v46
	v_mul_f32_e32 v47, v129, v47
	v_mul_f32_e32 v48, v129, v48
	v_bfe_u32 v130, v41, 16, 1
	v_add3_u32 v41, v41, v130, 0x7fff
	v_bfe_u32 v130, v42, 16, 1
	v_add3_u32 v42, v42, v130, 0x7fff
	v_perm_b32 v41, v42, v41, 0x7060302
	v_bfe_u32 v130, v43, 16, 1
	v_add3_u32 v43, v43, v130, 0x7fff
	v_bfe_u32 v130, v44, 16, 1
	v_add3_u32 v44, v44, v130, 0x7fff
	v_perm_b32 v42, v44, v43, 0x7060302
	v_bfe_u32 v130, v45, 16, 1
	v_add3_u32 v45, v45, v130, 0x7fff
	v_bfe_u32 v130, v46, 16, 1
	v_add3_u32 v46, v46, v130, 0x7fff
	v_perm_b32 v43, v46, v45, 0x7060302
	v_bfe_u32 v130, v47, 16, 1
	v_add3_u32 v47, v47, v130, 0x7fff
	v_bfe_u32 v130, v48, 16, 1
	v_add3_u32 v48, v48, v130, 0x7fff
	v_perm_b32 v44, v48, v47, 0x7060302
	global_store_b128 v[178:179], v[41:44], off offset:160

	v_mul_f32_e32 v49, v129, v49
	v_mul_f32_e32 v50, v129, v50
	v_mul_f32_e32 v51, v129, v51
	v_mul_f32_e32 v52, v129, v52
	v_mul_f32_e32 v53, v129, v53
	v_mul_f32_e32 v54, v129, v54
	v_mul_f32_e32 v55, v129, v55
	v_mul_f32_e32 v56, v129, v56
	v_bfe_u32 v130, v49, 16, 1
	v_add3_u32 v49, v49, v130, 0x7fff
	v_bfe_u32 v130, v50, 16, 1
	v_add3_u32 v50, v50, v130, 0x7fff
	v_perm_b32 v49, v50, v49, 0x7060302
	v_bfe_u32 v130, v51, 16, 1
	v_add3_u32 v51, v51, v130, 0x7fff
	v_bfe_u32 v130, v52, 16, 1
	v_add3_u32 v52, v52, v130, 0x7fff
	v_perm_b32 v50, v52, v51, 0x7060302
	v_bfe_u32 v130, v53, 16, 1
	v_add3_u32 v53, v53, v130, 0x7fff
	v_bfe_u32 v130, v54, 16, 1
	v_add3_u32 v54, v54, v130, 0x7fff
	v_perm_b32 v51, v54, v53, 0x7060302
	v_bfe_u32 v130, v55, 16, 1
	v_add3_u32 v55, v55, v130, 0x7fff
	v_bfe_u32 v130, v56, 16, 1
	v_add3_u32 v56, v56, v130, 0x7fff
	v_perm_b32 v52, v56, v55, 0x7060302
	global_store_b128 v[178:179], v[49:52], off offset:192

	v_mul_f32_e32 v57, v129, v57
	v_mul_f32_e32 v58, v129, v58
	v_mul_f32_e32 v59, v129, v59
	v_mul_f32_e32 v60, v129, v60
	v_mul_f32_e32 v61, v129, v61
	v_mul_f32_e32 v62, v129, v62
	v_mul_f32_e32 v63, v129, v63
	v_mul_f32_e32 v64, v129, v64
	v_bfe_u32 v130, v57, 16, 1
	v_add3_u32 v57, v57, v130, 0x7fff
	v_bfe_u32 v130, v58, 16, 1
	v_add3_u32 v58, v58, v130, 0x7fff
	v_perm_b32 v57, v58, v57, 0x7060302
	v_bfe_u32 v130, v59, 16, 1
	v_add3_u32 v59, v59, v130, 0x7fff
	v_bfe_u32 v130, v60, 16, 1
	v_add3_u32 v60, v60, v130, 0x7fff
	v_perm_b32 v58, v60, v59, 0x7060302
	v_bfe_u32 v130, v61, 16, 1
	v_add3_u32 v61, v61, v130, 0x7fff
	v_bfe_u32 v130, v62, 16, 1
	v_add3_u32 v62, v62, v130, 0x7fff
	v_perm_b32 v59, v62, v61, 0x7060302
	v_bfe_u32 v130, v63, 16, 1
	v_add3_u32 v63, v63, v130, 0x7fff
	v_bfe_u32 v130, v64, 16, 1
	v_add3_u32 v64, v64, v130, 0x7fff
	v_perm_b32 v60, v64, v63, 0x7060302
	global_store_b128 v[178:179], v[57:60], off offset:224

	s_endpgm

	.section	.rodata,#alloc
	.p2align	6
	.amdhsa_kernel opus_attn_gfx1201_kernel_v80
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 56
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 190
		.amdhsa_next_free_sgpr 45
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
	.end_amdhsa_kernel

	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .offset:         0
        .size:           56
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 56
    .max_flat_workgroup_size: 768
    .name:           opus_attn_gfx1201_kernel_v80
    .private_segment_fixed_size: 0
    .sgpr_count:     45
    .sgpr_spill_count: 0
    .symbol:         opus_attn_gfx1201_kernel_v80.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     190
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
