	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.text
	.globl	v88_kernel
	.p2align	8
	.type	v88_kernel,@function
v88_kernel:
	s_clause 0x2
	s_load_b256 s[4:11], s[0:1], 0x0
	s_load_b128 s[12:15], s[0:1], 0x20
	s_load_b32 s16, s[0:1], 0x30
	s_and_b32 s17, ttmp7, 0xffff
	s_lshr_b32 s18, ttmp7, 16
	v_and_b32_e32 v186, 15, v0
	v_lshrrev_b32_e32 v187, 4, v0
	v_and_b32_e32 v187, 1, v187
	v_lshlrev_b32_e32 v188, 3, v187
	v_xor_b32_e32 v138, 16, v0
	v_lshlrev_b32_e32 v138, 2, v138
	s_wait_kmcnt 0x0
	s_mov_b32 s19, 128
	s_mul_i32 s20, s14, s19
	s_mul_i32 s21, s13, s20
	s_mul_i32 s22, s19, s14
	s_mul_i32 s23, s13, s22
	s_lshr_b32 s24, s14, 5
	v_lshrrev_b32_e32 v189, 5, v0
	v_readfirstlane_b32 s25, v189
	s_mul_i32 s26, ttmp9, 384
	s_lshl_b32 s27, s25, 4
	s_add_i32 s26, s26, s27
	s_mul_i32 s27, s18, s21
	s_mul_i32 s28, s17, s20
	s_add_i32 s27, s27, s28
	v_mul_u32_u24_e32 v189, 128, v186
	v_add_nc_u32_e32 v189, v189, v188
	s_mul_i32 s28, s26, s19
	s_add_i32 s28, s27, s28
	v_add_nc_u32_e32 v186, s28, v189
	v_lshlrev_b32_e32 v186, 1, v186
	v_add_co_u32 v186, vcc_lo, s4, v186
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v187, null, s5, 0, vcc_lo
	global_load_b128 v[65:68], v[186:187], off offset:0
	global_load_b128 v[69:72], v[186:187], off offset:32
	global_load_b128 v[73:76], v[186:187], off offset:64
	global_load_b128 v[77:80], v[186:187], off offset:96
	global_load_b128 v[81:84], v[186:187], off offset:128
	global_load_b128 v[85:88], v[186:187], off offset:160
	global_load_b128 v[89:92], v[186:187], off offset:192
	global_load_b128 v[93:96], v[186:187], off offset:224
	v_mov_b32_e32 v139, s16
	v_mul_f32_e32 v139, 0x3fb8aa3b, v139
	v_add_nc_u32_e32 v186, s27, v189
	v_lshlrev_b32_e32 v186, 1, v186
	v_add_co_u32 v129, vcc_lo, s6, v186
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v130, null, s7, 0, vcc_lo
	v_add_co_u32 v131, vcc_lo, v129, 0x1000
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v132, null, v130, 0, vcc_lo
	s_mul_i32 s28, s18, s23
	s_mul_i32 s29, s17, s22
	s_add_i32 s28, s28, s29
	v_and_b32_e32 v186, 15, v0
	v_mul_lo_u32 v186, s14, v186
	v_lshrrev_b32_e32 v187, 4, v0
	v_and_b32_e32 v187, 1, v187
	v_lshlrev_b32_e32 v187, 3, v187
	v_add_nc_u32_e32 v186, v186, v187
	v_add_nc_u32_e32 v186, s28, v186
	v_lshlrev_b32_e32 v186, 1, v186
	v_add_co_u32 v140, vcc_lo, s8, v186
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v141, null, s9, 0, vcc_lo
	s_lshl_b32 s29, s14, 5
	s_mul_i32 s28, s26, s19
	s_add_i32 s28, s27, s28
	v_and_b32_e32 v186, 15, v0
	v_mul_u32_u24_e32 v186, 128, v186
	v_lshrrev_b32_e32 v187, 4, v0
	v_and_b32_e32 v187, 1, v187
	v_lshlrev_b32_e32 v187, 3, v187
	v_add_nc_u32_e32 v186, v186, v187
	v_add_nc_u32_e32 v186, s28, v186
	v_lshlrev_b32_e32 v186, 1, v186
	v_add_co_u32 v133, vcc_lo, s10, v186
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v134, null, s11, 0, vcc_lo
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v186, 16, v65
	v_mul_f32_e32 v186, v139, v186
	v_and_b32_e32 v187, 0xffff0000, v65
	v_mul_f32_e32 v187, v139, v187
	v_bfe_u32 v188, v186, 16, 1
	v_add3_u32 v186, v186, v188, 0x7fff
	v_bfe_u32 v188, v187, 16, 1
	v_add3_u32 v187, v187, v188, 0x7fff
	v_perm_b32 v65, v187, v186, 0x7060302
	v_lshlrev_b32_e32 v186, 16, v66
	v_mul_f32_e32 v186, v139, v186
	v_and_b32_e32 v187, 0xffff0000, v66
	v_mul_f32_e32 v187, v139, v187
	v_bfe_u32 v188, v186, 16, 1
	v_add3_u32 v186, v186, v188, 0x7fff
	v_bfe_u32 v188, v187, 16, 1
	v_add3_u32 v187, v187, v188, 0x7fff
	v_perm_b32 v66, v187, v186, 0x7060302
	v_lshlrev_b32_e32 v186, 16, v67
	v_mul_f32_e32 v186, v139, v186
	v_and_b32_e32 v187, 0xffff0000, v67
	v_mul_f32_e32 v187, v139, v187
	v_bfe_u32 v188, v186, 16, 1
	v_add3_u32 v186, v186, v188, 0x7fff
	v_bfe_u32 v188, v187, 16, 1
	v_add3_u32 v187, v187, v188, 0x7fff
	v_perm_b32 v67, v187, v186, 0x7060302
	v_lshlrev_b32_e32 v186, 16, v68
	v_mul_f32_e32 v186, v139, v186
	v_and_b32_e32 v187, 0xffff0000, v68
	v_mul_f32_e32 v187, v139, v187
	v_bfe_u32 v188, v186, 16, 1
	v_add3_u32 v186, v186, v188, 0x7fff
	v_bfe_u32 v188, v187, 16, 1
	v_add3_u32 v187, v187, v188, 0x7fff
	v_perm_b32 v68, v187, v186, 0x7060302
	v_lshlrev_b32_e32 v186, 16, v69
	v_mul_f32_e32 v186, v139, v186
	v_and_b32_e32 v187, 0xffff0000, v69
	v_mul_f32_e32 v187, v139, v187
	v_bfe_u32 v188, v186, 16, 1
	v_add3_u32 v186, v186, v188, 0x7fff
	v_bfe_u32 v188, v187, 16, 1
	v_add3_u32 v187, v187, v188, 0x7fff
	v_perm_b32 v69, v187, v186, 0x7060302
	v_lshlrev_b32_e32 v186, 16, v70
	v_mul_f32_e32 v186, v139, v186
	v_and_b32_e32 v187, 0xffff0000, v70
	v_mul_f32_e32 v187, v139, v187
	v_bfe_u32 v188, v186, 16, 1
	v_add3_u32 v186, v186, v188, 0x7fff
	v_bfe_u32 v188, v187, 16, 1
	v_add3_u32 v187, v187, v188, 0x7fff
	v_perm_b32 v70, v187, v186, 0x7060302
	v_lshlrev_b32_e32 v186, 16, v71
	v_mul_f32_e32 v186, v139, v186
	v_and_b32_e32 v187, 0xffff0000, v71
	v_mul_f32_e32 v187, v139, v187
	v_bfe_u32 v188, v186, 16, 1
	v_add3_u32 v186, v186, v188, 0x7fff
	v_bfe_u32 v188, v187, 16, 1
	v_add3_u32 v187, v187, v188, 0x7fff
	v_perm_b32 v71, v187, v186, 0x7060302
	v_lshlrev_b32_e32 v186, 16, v72
	v_mul_f32_e32 v186, v139, v186
	v_and_b32_e32 v187, 0xffff0000, v72
	v_mul_f32_e32 v187, v139, v187
	v_bfe_u32 v188, v186, 16, 1
	v_add3_u32 v186, v186, v188, 0x7fff
	v_bfe_u32 v188, v187, 16, 1
	v_add3_u32 v187, v187, v188, 0x7fff
	v_perm_b32 v72, v187, v186, 0x7060302
	v_lshlrev_b32_e32 v186, 16, v73
	v_mul_f32_e32 v186, v139, v186
	v_and_b32_e32 v187, 0xffff0000, v73
	v_mul_f32_e32 v187, v139, v187
	v_bfe_u32 v188, v186, 16, 1
	v_add3_u32 v186, v186, v188, 0x7fff
	v_bfe_u32 v188, v187, 16, 1
	v_add3_u32 v187, v187, v188, 0x7fff
	v_perm_b32 v73, v187, v186, 0x7060302
	v_lshlrev_b32_e32 v186, 16, v74
	v_mul_f32_e32 v186, v139, v186
	v_and_b32_e32 v187, 0xffff0000, v74
	v_mul_f32_e32 v187, v139, v187
	v_bfe_u32 v188, v186, 16, 1
	v_add3_u32 v186, v186, v188, 0x7fff
	v_bfe_u32 v188, v187, 16, 1
	v_add3_u32 v187, v187, v188, 0x7fff
	v_perm_b32 v74, v187, v186, 0x7060302
	v_lshlrev_b32_e32 v186, 16, v75
	v_mul_f32_e32 v186, v139, v186
	v_and_b32_e32 v187, 0xffff0000, v75
	v_mul_f32_e32 v187, v139, v187
	v_bfe_u32 v188, v186, 16, 1
	v_add3_u32 v186, v186, v188, 0x7fff
	v_bfe_u32 v188, v187, 16, 1
	v_add3_u32 v187, v187, v188, 0x7fff
	v_perm_b32 v75, v187, v186, 0x7060302
	v_lshlrev_b32_e32 v186, 16, v76
	v_mul_f32_e32 v186, v139, v186
	v_and_b32_e32 v187, 0xffff0000, v76
	v_mul_f32_e32 v187, v139, v187
	v_bfe_u32 v188, v186, 16, 1
	v_add3_u32 v186, v186, v188, 0x7fff
	v_bfe_u32 v188, v187, 16, 1
	v_add3_u32 v187, v187, v188, 0x7fff
	v_perm_b32 v76, v187, v186, 0x7060302
	v_lshlrev_b32_e32 v186, 16, v77
	v_mul_f32_e32 v186, v139, v186
	v_and_b32_e32 v187, 0xffff0000, v77
	v_mul_f32_e32 v187, v139, v187
	v_bfe_u32 v188, v186, 16, 1
	v_add3_u32 v186, v186, v188, 0x7fff
	v_bfe_u32 v188, v187, 16, 1
	v_add3_u32 v187, v187, v188, 0x7fff
	v_perm_b32 v77, v187, v186, 0x7060302
	v_lshlrev_b32_e32 v186, 16, v78
	v_mul_f32_e32 v186, v139, v186
	v_and_b32_e32 v187, 0xffff0000, v78
	v_mul_f32_e32 v187, v139, v187
	v_bfe_u32 v188, v186, 16, 1
	v_add3_u32 v186, v186, v188, 0x7fff
	v_bfe_u32 v188, v187, 16, 1
	v_add3_u32 v187, v187, v188, 0x7fff
	v_perm_b32 v78, v187, v186, 0x7060302
	v_lshlrev_b32_e32 v186, 16, v79
	v_mul_f32_e32 v186, v139, v186
	v_and_b32_e32 v187, 0xffff0000, v79
	v_mul_f32_e32 v187, v139, v187
	v_bfe_u32 v188, v186, 16, 1
	v_add3_u32 v186, v186, v188, 0x7fff
	v_bfe_u32 v188, v187, 16, 1
	v_add3_u32 v187, v187, v188, 0x7fff
	v_perm_b32 v79, v187, v186, 0x7060302
	v_lshlrev_b32_e32 v186, 16, v80
	v_mul_f32_e32 v186, v139, v186
	v_and_b32_e32 v187, 0xffff0000, v80
	v_mul_f32_e32 v187, v139, v187
	v_bfe_u32 v188, v186, 16, 1
	v_add3_u32 v186, v186, v188, 0x7fff
	v_bfe_u32 v188, v187, 16, 1
	v_add3_u32 v187, v187, v188, 0x7fff
	v_perm_b32 v80, v187, v186, 0x7060302
	v_lshlrev_b32_e32 v186, 16, v81
	v_mul_f32_e32 v186, v139, v186
	v_and_b32_e32 v187, 0xffff0000, v81
	v_mul_f32_e32 v187, v139, v187
	v_bfe_u32 v188, v186, 16, 1
	v_add3_u32 v186, v186, v188, 0x7fff
	v_bfe_u32 v188, v187, 16, 1
	v_add3_u32 v187, v187, v188, 0x7fff
	v_perm_b32 v81, v187, v186, 0x7060302
	v_lshlrev_b32_e32 v186, 16, v82
	v_mul_f32_e32 v186, v139, v186
	v_and_b32_e32 v187, 0xffff0000, v82
	v_mul_f32_e32 v187, v139, v187
	v_bfe_u32 v188, v186, 16, 1
	v_add3_u32 v186, v186, v188, 0x7fff
	v_bfe_u32 v188, v187, 16, 1
	v_add3_u32 v187, v187, v188, 0x7fff
	v_perm_b32 v82, v187, v186, 0x7060302
	v_lshlrev_b32_e32 v186, 16, v83
	v_mul_f32_e32 v186, v139, v186
	v_and_b32_e32 v187, 0xffff0000, v83
	v_mul_f32_e32 v187, v139, v187
	v_bfe_u32 v188, v186, 16, 1
	v_add3_u32 v186, v186, v188, 0x7fff
	v_bfe_u32 v188, v187, 16, 1
	v_add3_u32 v187, v187, v188, 0x7fff
	v_perm_b32 v83, v187, v186, 0x7060302
	v_lshlrev_b32_e32 v186, 16, v84
	v_mul_f32_e32 v186, v139, v186
	v_and_b32_e32 v187, 0xffff0000, v84
	v_mul_f32_e32 v187, v139, v187
	v_bfe_u32 v188, v186, 16, 1
	v_add3_u32 v186, v186, v188, 0x7fff
	v_bfe_u32 v188, v187, 16, 1
	v_add3_u32 v187, v187, v188, 0x7fff
	v_perm_b32 v84, v187, v186, 0x7060302
	v_lshlrev_b32_e32 v186, 16, v85
	v_mul_f32_e32 v186, v139, v186
	v_and_b32_e32 v187, 0xffff0000, v85
	v_mul_f32_e32 v187, v139, v187
	v_bfe_u32 v188, v186, 16, 1
	v_add3_u32 v186, v186, v188, 0x7fff
	v_bfe_u32 v188, v187, 16, 1
	v_add3_u32 v187, v187, v188, 0x7fff
	v_perm_b32 v85, v187, v186, 0x7060302
	v_lshlrev_b32_e32 v186, 16, v86
	v_mul_f32_e32 v186, v139, v186
	v_and_b32_e32 v187, 0xffff0000, v86
	v_mul_f32_e32 v187, v139, v187
	v_bfe_u32 v188, v186, 16, 1
	v_add3_u32 v186, v186, v188, 0x7fff
	v_bfe_u32 v188, v187, 16, 1
	v_add3_u32 v187, v187, v188, 0x7fff
	v_perm_b32 v86, v187, v186, 0x7060302
	v_lshlrev_b32_e32 v186, 16, v87
	v_mul_f32_e32 v186, v139, v186
	v_and_b32_e32 v187, 0xffff0000, v87
	v_mul_f32_e32 v187, v139, v187
	v_bfe_u32 v188, v186, 16, 1
	v_add3_u32 v186, v186, v188, 0x7fff
	v_bfe_u32 v188, v187, 16, 1
	v_add3_u32 v187, v187, v188, 0x7fff
	v_perm_b32 v87, v187, v186, 0x7060302
	v_lshlrev_b32_e32 v186, 16, v88
	v_mul_f32_e32 v186, v139, v186
	v_and_b32_e32 v187, 0xffff0000, v88
	v_mul_f32_e32 v187, v139, v187
	v_bfe_u32 v188, v186, 16, 1
	v_add3_u32 v186, v186, v188, 0x7fff
	v_bfe_u32 v188, v187, 16, 1
	v_add3_u32 v187, v187, v188, 0x7fff
	v_perm_b32 v88, v187, v186, 0x7060302
	v_lshlrev_b32_e32 v186, 16, v89
	v_mul_f32_e32 v186, v139, v186
	v_and_b32_e32 v187, 0xffff0000, v89
	v_mul_f32_e32 v187, v139, v187
	v_bfe_u32 v188, v186, 16, 1
	v_add3_u32 v186, v186, v188, 0x7fff
	v_bfe_u32 v188, v187, 16, 1
	v_add3_u32 v187, v187, v188, 0x7fff
	v_perm_b32 v89, v187, v186, 0x7060302
	v_lshlrev_b32_e32 v186, 16, v90
	v_mul_f32_e32 v186, v139, v186
	v_and_b32_e32 v187, 0xffff0000, v90
	v_mul_f32_e32 v187, v139, v187
	v_bfe_u32 v188, v186, 16, 1
	v_add3_u32 v186, v186, v188, 0x7fff
	v_bfe_u32 v188, v187, 16, 1
	v_add3_u32 v187, v187, v188, 0x7fff
	v_perm_b32 v90, v187, v186, 0x7060302
	v_lshlrev_b32_e32 v186, 16, v91
	v_mul_f32_e32 v186, v139, v186
	v_and_b32_e32 v187, 0xffff0000, v91
	v_mul_f32_e32 v187, v139, v187
	v_bfe_u32 v188, v186, 16, 1
	v_add3_u32 v186, v186, v188, 0x7fff
	v_bfe_u32 v188, v187, 16, 1
	v_add3_u32 v187, v187, v188, 0x7fff
	v_perm_b32 v91, v187, v186, 0x7060302
	v_lshlrev_b32_e32 v186, 16, v92
	v_mul_f32_e32 v186, v139, v186
	v_and_b32_e32 v187, 0xffff0000, v92
	v_mul_f32_e32 v187, v139, v187
	v_bfe_u32 v188, v186, 16, 1
	v_add3_u32 v186, v186, v188, 0x7fff
	v_bfe_u32 v188, v187, 16, 1
	v_add3_u32 v187, v187, v188, 0x7fff
	v_perm_b32 v92, v187, v186, 0x7060302
	v_lshlrev_b32_e32 v186, 16, v93
	v_mul_f32_e32 v186, v139, v186
	v_and_b32_e32 v187, 0xffff0000, v93
	v_mul_f32_e32 v187, v139, v187
	v_bfe_u32 v188, v186, 16, 1
	v_add3_u32 v186, v186, v188, 0x7fff
	v_bfe_u32 v188, v187, 16, 1
	v_add3_u32 v187, v187, v188, 0x7fff
	v_perm_b32 v93, v187, v186, 0x7060302
	v_lshlrev_b32_e32 v186, 16, v94
	v_mul_f32_e32 v186, v139, v186
	v_and_b32_e32 v187, 0xffff0000, v94
	v_mul_f32_e32 v187, v139, v187
	v_bfe_u32 v188, v186, 16, 1
	v_add3_u32 v186, v186, v188, 0x7fff
	v_bfe_u32 v188, v187, 16, 1
	v_add3_u32 v187, v187, v188, 0x7fff
	v_perm_b32 v94, v187, v186, 0x7060302
	v_lshlrev_b32_e32 v186, 16, v95
	v_mul_f32_e32 v186, v139, v186
	v_and_b32_e32 v187, 0xffff0000, v95
	v_mul_f32_e32 v187, v139, v187
	v_bfe_u32 v188, v186, 16, 1
	v_add3_u32 v186, v186, v188, 0x7fff
	v_bfe_u32 v188, v187, 16, 1
	v_add3_u32 v187, v187, v188, 0x7fff
	v_perm_b32 v95, v187, v186, 0x7060302
	v_lshlrev_b32_e32 v186, 16, v96
	v_mul_f32_e32 v186, v139, v186
	v_and_b32_e32 v187, 0xffff0000, v96
	v_mul_f32_e32 v187, v139, v187
	v_bfe_u32 v188, v186, 16, 1
	v_add3_u32 v186, v186, v188, 0x7fff
	v_bfe_u32 v188, v187, 16, 1
	v_add3_u32 v187, v187, v188, 0x7fff
	v_perm_b32 v96, v187, v186, 0x7060302
	v_mov_b32_e32 v1, 0
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v4, 0
	v_mov_b32_e32 v5, 0
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v7, 0
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v13, 0
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v135, 0xff7fffff
	v_mov_b32_e32 v136, 0
	s_mov_b32 s30, 0
	s_clause 0x3
	global_load_b128 v[113:116], v[129:130], off
	global_load_b128 v[117:120], v[131:132], off
	global_load_b128 v[121:124], v[129:130], off offset:32
	global_load_b128 v[125:128], v[131:132], off offset:32
.L_n_loop:
	v_dual_mov_b32 v97, 0 :: v_dual_mov_b32 v98, 0
	v_dual_mov_b32 v99, 0 :: v_dual_mov_b32 v100, 0
	v_dual_mov_b32 v101, 0 :: v_dual_mov_b32 v102, 0
	v_dual_mov_b32 v103, 0 :: v_dual_mov_b32 v104, 0
	v_dual_mov_b32 v105, 0 :: v_dual_mov_b32 v106, 0
	v_dual_mov_b32 v107, 0 :: v_dual_mov_b32 v108, 0
	v_dual_mov_b32 v109, 0 :: v_dual_mov_b32 v110, 0
	v_dual_mov_b32 v111, 0 :: v_dual_mov_b32 v112, 0
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x16_bf16 v[97:104], v[113:116], v[65:68], v[97:104]
	v_wmma_f32_16x16x16_bf16 v[105:112], v[117:120], v[65:68], v[105:112]
	s_clause 0x1
	global_load_b128 v[113:116], v[129:130], off offset:64
	global_load_b128 v[117:120], v[131:132], off offset:64
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x16_bf16 v[97:104], v[121:124], v[69:72], v[97:104]
	v_wmma_f32_16x16x16_bf16 v[105:112], v[125:128], v[69:72], v[105:112]
	s_clause 0x1
	global_load_b128 v[121:124], v[129:130], off offset:96
	global_load_b128 v[125:128], v[131:132], off offset:96
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x16_bf16 v[97:104], v[113:116], v[73:76], v[97:104]
	v_wmma_f32_16x16x16_bf16 v[105:112], v[117:120], v[73:76], v[105:112]
	s_clause 0x1
	global_load_b128 v[113:116], v[129:130], off offset:128
	global_load_b128 v[117:120], v[131:132], off offset:128
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x16_bf16 v[97:104], v[121:124], v[77:80], v[97:104]
	v_wmma_f32_16x16x16_bf16 v[105:112], v[125:128], v[77:80], v[105:112]
	s_clause 0x1
	global_load_b128 v[121:124], v[129:130], off offset:160
	global_load_b128 v[125:128], v[131:132], off offset:160
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x16_bf16 v[97:104], v[113:116], v[81:84], v[97:104]
	v_wmma_f32_16x16x16_bf16 v[105:112], v[117:120], v[81:84], v[105:112]
	s_clause 0x1
	global_load_b128 v[113:116], v[129:130], off offset:192
	global_load_b128 v[117:120], v[131:132], off offset:192
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x16_bf16 v[97:104], v[121:124], v[85:88], v[97:104]
	v_wmma_f32_16x16x16_bf16 v[105:112], v[125:128], v[85:88], v[105:112]
	s_clause 0x1
	global_load_b128 v[121:124], v[129:130], off offset:224
	global_load_b128 v[125:128], v[131:132], off offset:224
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x16_bf16 v[97:104], v[113:116], v[89:92], v[97:104]
	v_wmma_f32_16x16x16_bf16 v[105:112], v[117:120], v[89:92], v[105:112]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_bf16 v[97:104], v[121:124], v[93:96], v[97:104]
	v_wmma_f32_16x16x16_bf16 v[105:112], v[125:128], v[93:96], v[105:112]
	v_add_co_u32 v142, vcc_lo, v140, s29
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v143, null, v141, 0, vcc_lo
	s_clause 0x3
	global_load_b128 v[113:116], v[140:141], off
	global_load_b128 v[117:120], v[140:141], off offset:32
	global_load_b128 v[121:124], v[142:143], off
	global_load_b128 v[125:128], v[142:143], off offset:32
	s_mul_i32 s34, 2, s29
	s_mul_i32 s35, 3, s29
	s_lshl_b32 s36, s29, 2
	s_mul_i32 s37, 5, s29
	s_mul_i32 s38, 6, s29
	s_mul_i32 s39, 7, s29
	v_max3_num_f32 v186, v97, v98, v99
	v_max3_num_f32 v187, v100, v101, v102
	v_max3_num_f32 v188, v103, v104, v105
	v_max3_num_f32 v189, v106, v107, v108
	s_delay_alu instid0(VALU_DEP_4)
	v_max3_num_f32 v186, v186, v187, v188
	s_delay_alu instid0(VALU_DEP_2)
	v_max3_num_f32 v189, v189, v109, v110
	s_delay_alu instid0(VALU_DEP_2)
	v_max3_num_f32 v186, v186, v189, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_max_num_f32_e32 v186, v186, v112
	ds_bpermute_b32 v187, v138, v186
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v186, v186, v187
	s_delay_alu instid0(VALU_DEP_1)
	v_max_num_f32_e32 v188, v135, v186
	s_delay_alu instid0(VALU_DEP_1)
	v_sub_f32_e32 v137, v135, v188
	v_exp_f32_e32 v137, v137
	v_mov_b32_e32 v135, v188
	v_readfirstlane_b32 s32, v188
	s_wait_alu 0xfffe
	v_dual_subrev_f32 v97, s32, v97 :: v_dual_subrev_f32 v98, s32, v98
	v_dual_subrev_f32 v99, s32, v99 :: v_dual_subrev_f32 v100, s32, v100
	v_dual_subrev_f32 v101, s32, v101 :: v_dual_subrev_f32 v102, s32, v102
	v_dual_subrev_f32 v103, s32, v103 :: v_dual_subrev_f32 v104, s32, v104
	v_dual_subrev_f32 v105, s32, v105 :: v_dual_subrev_f32 v106, s32, v106
	v_dual_subrev_f32 v107, s32, v107 :: v_dual_subrev_f32 v108, s32, v108
	v_dual_subrev_f32 v109, s32, v109 :: v_dual_subrev_f32 v110, s32, v110
	v_dual_subrev_f32 v111, s32, v111 :: v_dual_subrev_f32 v112, s32, v112
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
	s_wait_alu 0x1
	v_mul_f32_e32 v136, v137, v136
	s_wait_alu 0x1
	v_perm_b32 v144, v98, v97, 0x7060302
	v_perm_b32 v145, v100, v99, 0x7060302
	v_add_f32_e32 v186, v97, v98
	v_add_f32_e32 v187, v99, v100
	v_perm_b32 v146, v102, v101, 0x7060302
	v_perm_b32 v147, v104, v103, 0x7060302
	v_add_f32_e32 v188, v101, v102
	v_add_f32_e32 v189, v103, v104
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v186, v186, v187
	v_add_f32_e32 v188, v188, v189
	v_perm_b32 v148, v106, v105, 0x7060302
	v_perm_b32 v149, v108, v107, 0x7060302
	v_add_f32_e32 v187, v105, v106
	v_add_f32_e32 v189, v107, v108
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v186, v186, v188
	v_add_f32_e32 v187, v187, v189
	v_perm_b32 v150, v110, v109, 0x7060302
	v_perm_b32 v151, v112, v111, 0x7060302
	v_add_f32_e32 v188, v109, v110
	v_add_f32_e32 v189, v111, v112
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v186, v186, v187
	v_add_f32_e32 v188, v188, v189
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v186, v186, v188
	ds_bpermute_b32 v187, v138, v186
	s_wait_dscnt 0x0
	v_add_f32_e32 v186, v186, v187
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v136, v136, v186
	v_cmp_neq_f32_e32 vcc_lo, 1.0, v137
	s_wait_alu 0xfffd
	s_and_saveexec_b32 s31, vcc_lo
	s_cbranch_execz .L_skip_rescale
	v_readfirstlane_b32 s33, v137
	s_wait_alu 0xfffe
	v_dual_mul_f32 v1, s33, v1 :: v_dual_mul_f32 v2, s33, v2
	v_dual_mul_f32 v3, s33, v3 :: v_dual_mul_f32 v4, s33, v4
	v_dual_mul_f32 v5, s33, v5 :: v_dual_mul_f32 v6, s33, v6
	v_dual_mul_f32 v7, s33, v7 :: v_dual_mul_f32 v8, s33, v8
	v_dual_mul_f32 v9, s33, v9 :: v_dual_mul_f32 v10, s33, v10
	v_dual_mul_f32 v11, s33, v11 :: v_dual_mul_f32 v12, s33, v12
	v_dual_mul_f32 v13, s33, v13 :: v_dual_mul_f32 v14, s33, v14
	v_dual_mul_f32 v15, s33, v15 :: v_dual_mul_f32 v16, s33, v16
	v_dual_mul_f32 v17, s33, v17 :: v_dual_mul_f32 v18, s33, v18
	v_dual_mul_f32 v19, s33, v19 :: v_dual_mul_f32 v20, s33, v20
	v_dual_mul_f32 v21, s33, v21 :: v_dual_mul_f32 v22, s33, v22
	v_dual_mul_f32 v23, s33, v23 :: v_dual_mul_f32 v24, s33, v24
	v_dual_mul_f32 v25, s33, v25 :: v_dual_mul_f32 v26, s33, v26
	v_dual_mul_f32 v27, s33, v27 :: v_dual_mul_f32 v28, s33, v28
	v_dual_mul_f32 v29, s33, v29 :: v_dual_mul_f32 v30, s33, v30
	v_dual_mul_f32 v31, s33, v31 :: v_dual_mul_f32 v32, s33, v32
	v_dual_mul_f32 v33, s33, v33 :: v_dual_mul_f32 v34, s33, v34
	v_dual_mul_f32 v35, s33, v35 :: v_dual_mul_f32 v36, s33, v36
	v_dual_mul_f32 v37, s33, v37 :: v_dual_mul_f32 v38, s33, v38
	v_dual_mul_f32 v39, s33, v39 :: v_dual_mul_f32 v40, s33, v40
	v_dual_mul_f32 v41, s33, v41 :: v_dual_mul_f32 v42, s33, v42
	v_dual_mul_f32 v43, s33, v43 :: v_dual_mul_f32 v44, s33, v44
	v_dual_mul_f32 v45, s33, v45 :: v_dual_mul_f32 v46, s33, v46
	v_dual_mul_f32 v47, s33, v47 :: v_dual_mul_f32 v48, s33, v48
	v_dual_mul_f32 v49, s33, v49 :: v_dual_mul_f32 v50, s33, v50
	v_dual_mul_f32 v51, s33, v51 :: v_dual_mul_f32 v52, s33, v52
	v_dual_mul_f32 v53, s33, v53 :: v_dual_mul_f32 v54, s33, v54
	v_dual_mul_f32 v55, s33, v55 :: v_dual_mul_f32 v56, s33, v56
	v_dual_mul_f32 v57, s33, v57 :: v_dual_mul_f32 v58, s33, v58
	v_dual_mul_f32 v59, s33, v59 :: v_dual_mul_f32 v60, s33, v60
	v_dual_mul_f32 v61, s33, v61 :: v_dual_mul_f32 v62, s33, v62
	v_dual_mul_f32 v63, s33, v63 :: v_dual_mul_f32 v64, s33, v64
.L_skip_rescale:
	s_or_b32 exec_lo, exec_lo, s31
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x16_bf16 v[1:8], v[113:116], v[144:147], v[1:8]
	v_wmma_f32_16x16x16_bf16 v[1:8], v[117:120], v[148:151], v[1:8]
	v_add_co_u32 v142, vcc_lo, v140, s34
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v143, null, v141, 0, vcc_lo
	s_clause 0x1
	global_load_b128 v[113:116], v[142:143], off
	global_load_b128 v[117:120], v[142:143], off offset:32
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x16_bf16 v[9:16], v[121:124], v[144:147], v[9:16]
	v_wmma_f32_16x16x16_bf16 v[9:16], v[125:128], v[148:151], v[9:16]
	v_add_co_u32 v142, vcc_lo, v140, s35
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v143, null, v141, 0, vcc_lo
	s_clause 0x1
	global_load_b128 v[121:124], v[142:143], off
	global_load_b128 v[125:128], v[142:143], off offset:32
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x16_bf16 v[17:24], v[113:116], v[144:147], v[17:24]
	v_wmma_f32_16x16x16_bf16 v[17:24], v[117:120], v[148:151], v[17:24]
	v_add_co_u32 v142, vcc_lo, v140, s36
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v143, null, v141, 0, vcc_lo
	s_clause 0x1
	global_load_b128 v[113:116], v[142:143], off
	global_load_b128 v[117:120], v[142:143], off offset:32
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x16_bf16 v[25:32], v[121:124], v[144:147], v[25:32]
	v_wmma_f32_16x16x16_bf16 v[25:32], v[125:128], v[148:151], v[25:32]
	v_add_co_u32 v142, vcc_lo, v140, s37
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v143, null, v141, 0, vcc_lo
	s_clause 0x1
	global_load_b128 v[121:124], v[142:143], off
	global_load_b128 v[125:128], v[142:143], off offset:32
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x16_bf16 v[33:40], v[113:116], v[144:147], v[33:40]
	v_wmma_f32_16x16x16_bf16 v[33:40], v[117:120], v[148:151], v[33:40]
	v_add_co_u32 v142, vcc_lo, v140, s38
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v143, null, v141, 0, vcc_lo
	s_clause 0x1
	global_load_b128 v[113:116], v[142:143], off
	global_load_b128 v[117:120], v[142:143], off offset:32
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x16_bf16 v[41:48], v[121:124], v[144:147], v[41:48]
	v_wmma_f32_16x16x16_bf16 v[41:48], v[125:128], v[148:151], v[41:48]
	v_add_co_u32 v142, vcc_lo, v140, s39
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v143, null, v141, 0, vcc_lo
	s_clause 0x1
	global_load_b128 v[121:124], v[142:143], off
	global_load_b128 v[125:128], v[142:143], off offset:32
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x16_bf16 v[49:56], v[113:116], v[144:147], v[49:56]
	v_wmma_f32_16x16x16_bf16 v[49:56], v[117:120], v[148:151], v[49:56]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_bf16 v[57:64], v[121:124], v[144:147], v[57:64]
	v_wmma_f32_16x16x16_bf16 v[57:64], v[125:128], v[148:151], v[57:64]
	v_add_co_u32 v129, vcc_lo, v129, 0x2000
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v130, null, v130, 0, vcc_lo
	v_add_co_u32 v131, vcc_lo, v131, 0x2000
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v132, null, v132, 0, vcc_lo
	v_add_co_u32 v140, vcc_lo, v140, 64
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v141, null, v141, 0, vcc_lo
	s_add_i32 s30, s30, 1
	s_wait_alu 0xfffe
	s_cmp_lt_u32 s30, s24
	s_cbranch_scc0 .L_n_done
	s_clause 0x3
	global_load_b128 v[113:116], v[129:130], off
	global_load_b128 v[117:120], v[131:132], off
	global_load_b128 v[121:124], v[129:130], off offset:32
	global_load_b128 v[125:128], v[131:132], off offset:32
	s_branch .L_n_loop
.L_n_done:
	v_rcp_f32_e32 v136, v136
	s_wait_alu 0x1
	v_readfirstlane_b32 s33, v136
	s_wait_alu 0xfffe
	v_dual_mul_f32 v1, s33, v1 :: v_dual_mul_f32 v2, s33, v2
	v_dual_mul_f32 v3, s33, v3 :: v_dual_mul_f32 v4, s33, v4
	v_dual_mul_f32 v5, s33, v5 :: v_dual_mul_f32 v6, s33, v6
	v_dual_mul_f32 v7, s33, v7 :: v_dual_mul_f32 v8, s33, v8
	v_bfe_u32 v186, v1, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v1, v1, v186, 0x7fff
	v_bfe_u32 v186, v2, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v2, v2, v186, 0x7fff
	s_delay_alu instid0(VALU_DEP_1)
	v_perm_b32 v97, v2, v1, 0x7060302
	v_bfe_u32 v186, v3, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v3, v3, v186, 0x7fff
	v_bfe_u32 v186, v4, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v4, v4, v186, 0x7fff
	s_delay_alu instid0(VALU_DEP_1)
	v_perm_b32 v98, v4, v3, 0x7060302
	v_bfe_u32 v186, v5, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v5, v5, v186, 0x7fff
	v_bfe_u32 v186, v6, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v6, v6, v186, 0x7fff
	s_delay_alu instid0(VALU_DEP_1)
	v_perm_b32 v99, v6, v5, 0x7060302
	v_bfe_u32 v186, v7, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v7, v7, v186, 0x7fff
	v_bfe_u32 v186, v8, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v8, v8, v186, 0x7fff
	s_delay_alu instid0(VALU_DEP_1)
	v_perm_b32 v100, v8, v7, 0x7060302
	global_store_b128 v[133:134], v[97:100], off offset:0
	v_dual_mul_f32 v9, s33, v9 :: v_dual_mul_f32 v10, s33, v10
	v_dual_mul_f32 v11, s33, v11 :: v_dual_mul_f32 v12, s33, v12
	v_dual_mul_f32 v13, s33, v13 :: v_dual_mul_f32 v14, s33, v14
	v_dual_mul_f32 v15, s33, v15 :: v_dual_mul_f32 v16, s33, v16
	v_bfe_u32 v186, v9, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v9, v9, v186, 0x7fff
	v_bfe_u32 v186, v10, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v10, v10, v186, 0x7fff
	s_delay_alu instid0(VALU_DEP_1)
	v_perm_b32 v97, v10, v9, 0x7060302
	v_bfe_u32 v186, v11, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v11, v11, v186, 0x7fff
	v_bfe_u32 v186, v12, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v12, v12, v186, 0x7fff
	s_delay_alu instid0(VALU_DEP_1)
	v_perm_b32 v98, v12, v11, 0x7060302
	v_bfe_u32 v186, v13, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v13, v13, v186, 0x7fff
	v_bfe_u32 v186, v14, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v14, v14, v186, 0x7fff
	s_delay_alu instid0(VALU_DEP_1)
	v_perm_b32 v99, v14, v13, 0x7060302
	v_bfe_u32 v186, v15, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v15, v15, v186, 0x7fff
	v_bfe_u32 v186, v16, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v16, v16, v186, 0x7fff
	s_delay_alu instid0(VALU_DEP_1)
	v_perm_b32 v100, v16, v15, 0x7060302
	global_store_b128 v[133:134], v[97:100], off offset:32
	v_dual_mul_f32 v17, s33, v17 :: v_dual_mul_f32 v18, s33, v18
	v_dual_mul_f32 v19, s33, v19 :: v_dual_mul_f32 v20, s33, v20
	v_dual_mul_f32 v21, s33, v21 :: v_dual_mul_f32 v22, s33, v22
	v_dual_mul_f32 v23, s33, v23 :: v_dual_mul_f32 v24, s33, v24
	v_bfe_u32 v186, v17, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v17, v17, v186, 0x7fff
	v_bfe_u32 v186, v18, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v18, v18, v186, 0x7fff
	s_delay_alu instid0(VALU_DEP_1)
	v_perm_b32 v97, v18, v17, 0x7060302
	v_bfe_u32 v186, v19, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v19, v19, v186, 0x7fff
	v_bfe_u32 v186, v20, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v20, v20, v186, 0x7fff
	s_delay_alu instid0(VALU_DEP_1)
	v_perm_b32 v98, v20, v19, 0x7060302
	v_bfe_u32 v186, v21, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v21, v21, v186, 0x7fff
	v_bfe_u32 v186, v22, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v22, v22, v186, 0x7fff
	s_delay_alu instid0(VALU_DEP_1)
	v_perm_b32 v99, v22, v21, 0x7060302
	v_bfe_u32 v186, v23, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v23, v23, v186, 0x7fff
	v_bfe_u32 v186, v24, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v24, v24, v186, 0x7fff
	s_delay_alu instid0(VALU_DEP_1)
	v_perm_b32 v100, v24, v23, 0x7060302
	global_store_b128 v[133:134], v[97:100], off offset:64
	v_dual_mul_f32 v25, s33, v25 :: v_dual_mul_f32 v26, s33, v26
	v_dual_mul_f32 v27, s33, v27 :: v_dual_mul_f32 v28, s33, v28
	v_dual_mul_f32 v29, s33, v29 :: v_dual_mul_f32 v30, s33, v30
	v_dual_mul_f32 v31, s33, v31 :: v_dual_mul_f32 v32, s33, v32
	v_bfe_u32 v186, v25, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v25, v25, v186, 0x7fff
	v_bfe_u32 v186, v26, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v26, v26, v186, 0x7fff
	s_delay_alu instid0(VALU_DEP_1)
	v_perm_b32 v97, v26, v25, 0x7060302
	v_bfe_u32 v186, v27, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v27, v27, v186, 0x7fff
	v_bfe_u32 v186, v28, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v28, v28, v186, 0x7fff
	s_delay_alu instid0(VALU_DEP_1)
	v_perm_b32 v98, v28, v27, 0x7060302
	v_bfe_u32 v186, v29, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v29, v29, v186, 0x7fff
	v_bfe_u32 v186, v30, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v30, v30, v186, 0x7fff
	s_delay_alu instid0(VALU_DEP_1)
	v_perm_b32 v99, v30, v29, 0x7060302
	v_bfe_u32 v186, v31, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v31, v31, v186, 0x7fff
	v_bfe_u32 v186, v32, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v32, v32, v186, 0x7fff
	s_delay_alu instid0(VALU_DEP_1)
	v_perm_b32 v100, v32, v31, 0x7060302
	global_store_b128 v[133:134], v[97:100], off offset:96
	v_dual_mul_f32 v33, s33, v33 :: v_dual_mul_f32 v34, s33, v34
	v_dual_mul_f32 v35, s33, v35 :: v_dual_mul_f32 v36, s33, v36
	v_dual_mul_f32 v37, s33, v37 :: v_dual_mul_f32 v38, s33, v38
	v_dual_mul_f32 v39, s33, v39 :: v_dual_mul_f32 v40, s33, v40
	v_bfe_u32 v186, v33, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v33, v33, v186, 0x7fff
	v_bfe_u32 v186, v34, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v34, v34, v186, 0x7fff
	s_delay_alu instid0(VALU_DEP_1)
	v_perm_b32 v97, v34, v33, 0x7060302
	v_bfe_u32 v186, v35, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v35, v35, v186, 0x7fff
	v_bfe_u32 v186, v36, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v36, v36, v186, 0x7fff
	s_delay_alu instid0(VALU_DEP_1)
	v_perm_b32 v98, v36, v35, 0x7060302
	v_bfe_u32 v186, v37, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v37, v37, v186, 0x7fff
	v_bfe_u32 v186, v38, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v38, v38, v186, 0x7fff
	s_delay_alu instid0(VALU_DEP_1)
	v_perm_b32 v99, v38, v37, 0x7060302
	v_bfe_u32 v186, v39, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v39, v39, v186, 0x7fff
	v_bfe_u32 v186, v40, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v40, v40, v186, 0x7fff
	s_delay_alu instid0(VALU_DEP_1)
	v_perm_b32 v100, v40, v39, 0x7060302
	global_store_b128 v[133:134], v[97:100], off offset:128
	v_dual_mul_f32 v41, s33, v41 :: v_dual_mul_f32 v42, s33, v42
	v_dual_mul_f32 v43, s33, v43 :: v_dual_mul_f32 v44, s33, v44
	v_dual_mul_f32 v45, s33, v45 :: v_dual_mul_f32 v46, s33, v46
	v_dual_mul_f32 v47, s33, v47 :: v_dual_mul_f32 v48, s33, v48
	v_bfe_u32 v186, v41, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v41, v41, v186, 0x7fff
	v_bfe_u32 v186, v42, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v42, v42, v186, 0x7fff
	s_delay_alu instid0(VALU_DEP_1)
	v_perm_b32 v97, v42, v41, 0x7060302
	v_bfe_u32 v186, v43, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v43, v43, v186, 0x7fff
	v_bfe_u32 v186, v44, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v44, v44, v186, 0x7fff
	s_delay_alu instid0(VALU_DEP_1)
	v_perm_b32 v98, v44, v43, 0x7060302
	v_bfe_u32 v186, v45, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v45, v45, v186, 0x7fff
	v_bfe_u32 v186, v46, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v46, v46, v186, 0x7fff
	s_delay_alu instid0(VALU_DEP_1)
	v_perm_b32 v99, v46, v45, 0x7060302
	v_bfe_u32 v186, v47, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v47, v47, v186, 0x7fff
	v_bfe_u32 v186, v48, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v48, v48, v186, 0x7fff
	s_delay_alu instid0(VALU_DEP_1)
	v_perm_b32 v100, v48, v47, 0x7060302
	global_store_b128 v[133:134], v[97:100], off offset:160
	v_dual_mul_f32 v49, s33, v49 :: v_dual_mul_f32 v50, s33, v50
	v_dual_mul_f32 v51, s33, v51 :: v_dual_mul_f32 v52, s33, v52
	v_dual_mul_f32 v53, s33, v53 :: v_dual_mul_f32 v54, s33, v54
	v_dual_mul_f32 v55, s33, v55 :: v_dual_mul_f32 v56, s33, v56
	v_bfe_u32 v186, v49, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v49, v49, v186, 0x7fff
	v_bfe_u32 v186, v50, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v50, v50, v186, 0x7fff
	s_delay_alu instid0(VALU_DEP_1)
	v_perm_b32 v97, v50, v49, 0x7060302
	v_bfe_u32 v186, v51, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v51, v51, v186, 0x7fff
	v_bfe_u32 v186, v52, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v52, v52, v186, 0x7fff
	s_delay_alu instid0(VALU_DEP_1)
	v_perm_b32 v98, v52, v51, 0x7060302
	v_bfe_u32 v186, v53, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v53, v53, v186, 0x7fff
	v_bfe_u32 v186, v54, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v54, v54, v186, 0x7fff
	s_delay_alu instid0(VALU_DEP_1)
	v_perm_b32 v99, v54, v53, 0x7060302
	v_bfe_u32 v186, v55, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v55, v55, v186, 0x7fff
	v_bfe_u32 v186, v56, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v56, v56, v186, 0x7fff
	s_delay_alu instid0(VALU_DEP_1)
	v_perm_b32 v100, v56, v55, 0x7060302
	global_store_b128 v[133:134], v[97:100], off offset:192
	v_dual_mul_f32 v57, s33, v57 :: v_dual_mul_f32 v58, s33, v58
	v_dual_mul_f32 v59, s33, v59 :: v_dual_mul_f32 v60, s33, v60
	v_dual_mul_f32 v61, s33, v61 :: v_dual_mul_f32 v62, s33, v62
	v_dual_mul_f32 v63, s33, v63 :: v_dual_mul_f32 v64, s33, v64
	v_bfe_u32 v186, v57, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v57, v57, v186, 0x7fff
	v_bfe_u32 v186, v58, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v58, v58, v186, 0x7fff
	s_delay_alu instid0(VALU_DEP_1)
	v_perm_b32 v97, v58, v57, 0x7060302
	v_bfe_u32 v186, v59, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v59, v59, v186, 0x7fff
	v_bfe_u32 v186, v60, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v60, v60, v186, 0x7fff
	s_delay_alu instid0(VALU_DEP_1)
	v_perm_b32 v98, v60, v59, 0x7060302
	v_bfe_u32 v186, v61, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v61, v61, v186, 0x7fff
	v_bfe_u32 v186, v62, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v62, v62, v186, 0x7fff
	s_delay_alu instid0(VALU_DEP_1)
	v_perm_b32 v99, v62, v61, 0x7060302
	v_bfe_u32 v186, v63, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v63, v63, v186, 0x7fff
	v_bfe_u32 v186, v64, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v64, v64, v186, 0x7fff
	s_delay_alu instid0(VALU_DEP_1)
	v_perm_b32 v100, v64, v63, 0x7060302
	global_store_b128 v[133:134], v[97:100], off offset:224
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v88_kernel
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
		.amdhsa_next_free_sgpr 40
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 8
		.amdhsa_round_robin_scheduling 1
	.end_amdhsa_kernel

	.amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name:           v88_kernel
    .symbol:         v88_kernel.kd
    .kernarg_segment_size: 56
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 32
    .sgpr_count:     40
    .vgpr_count:     190
    .max_flat_workgroup_size: 768
    .args:
      - { .offset: 0,  .size: 8, .value_kind: global_buffer }
      - { .offset: 8,  .size: 8, .value_kind: global_buffer }
      - { .offset: 16, .size: 8, .value_kind: global_buffer }
      - { .offset: 24, .size: 8, .value_kind: global_buffer }
      - { .offset: 32, .size: 4, .value_kind: by_value }
      - { .offset: 36, .size: 4, .value_kind: by_value }
      - { .offset: 40, .size: 4, .value_kind: by_value }
      - { .offset: 44, .size: 4, .value_kind: by_value }
      - { .offset: 48, .size: 4, .value_kind: by_value }
...
	.end_amdgpu_metadata
