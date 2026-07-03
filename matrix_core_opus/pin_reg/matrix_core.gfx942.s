	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.section	.text._Z27matrix_core_kernel_block_v2ILi256ELi192ELi128ELi16ELi2ELi2ELi1ELi16ELi16ELi16EEvPKvS1_Pviiii,"axG",@progbits,_Z27matrix_core_kernel_block_v2ILi256ELi192ELi128ELi16ELi2ELi2ELi1ELi16ELi16ELi16EEvPKvS1_Pviiii,comdat
	.protected	_Z27matrix_core_kernel_block_v2ILi256ELi192ELi128ELi16ELi2ELi2ELi1ELi16ELi16ELi16EEvPKvS1_Pviiii ; -- Begin function _Z27matrix_core_kernel_block_v2ILi256ELi192ELi128ELi16ELi2ELi2ELi1ELi16ELi16ELi16EEvPKvS1_Pviiii
	.globl	_Z27matrix_core_kernel_block_v2ILi256ELi192ELi128ELi16ELi2ELi2ELi1ELi16ELi16ELi16EEvPKvS1_Pviiii
	.p2align	8
	.type	_Z27matrix_core_kernel_block_v2ILi256ELi192ELi128ELi16ELi2ELi2ELi1ELi16ELi16ELi16EEvPKvS1_Pviiii,@function
_Z27matrix_core_kernel_block_v2ILi256ELi192ELi128ELi16ELi2ELi2ELi1ELi16ELi16ELi16EEvPKvS1_Pviiii: ; @_Z27matrix_core_kernel_block_v2ILi256ELi192ELi128ELi16ELi2ELi2ELi1ELi16ELi16ELi16EEvPKvS1_Pviiii
; %bb.0:                                ; %entry
	s_load_dwordx2 s[14:15], s[0:1], 0x10
	s_load_dwordx4 s[4:7], s[0:1], 0x18
	s_lshl_b32 s12, s3, 7
	v_and_b32_e32 v98, 15, v0
	v_bfe_u32 v96, v0, 4, 2
	v_lshrrev_b32_e32 v1, 3, v0
	v_lshrrev_b32_e32 v0, 2, v0
	s_mul_i32 s13, s2, 0xc0
	v_and_b32_e32 v99, 16, v1
	s_waitcnt lgkmcnt(0)
	s_cmp_lt_i32 s4, 1
	v_and_b32_e32 v97, 16, v0
	s_cbranch_scc1 .LBB0_4
; %bb.1:                                ; %for.body.preheader
	v_add_u32_e32 v0, v99, v98
	v_add_u32_e32 v1, 0xa0, v0
	v_mul_lo_u32 v1, s5, v1
	v_lshlrev_b32_e32 v100, 1, v1
	v_or_b32_e32 v1, 0x80, v0
	v_mul_lo_u32 v1, s5, v1
	v_lshlrev_b32_e32 v101, 1, v1
	v_add_u32_e32 v1, 0x60, v0
	v_mul_lo_u32 v1, s5, v1
	s_load_dwordx4 s[8:11], s[0:1], 0x0
	v_lshlrev_b32_e32 v102, 1, v1
	v_or_b32_e32 v1, 64, v0
	v_mul_lo_u32 v1, s5, v1
	s_mul_i32 s0, s5, s13
	v_lshlrev_b32_e32 v103, 1, v1
	v_add_u32_e32 v1, 32, v0
	v_mul_lo_u32 v0, s5, v0
	s_ashr_i32 s1, s0, 31
	v_mul_lo_u32 v1, s5, v1
	v_lshlrev_b32_e32 v105, 1, v0
	v_add_u32_e32 v0, v97, v98
	s_lshl_b64 s[0:1], s[0:1], 1
	v_lshlrev_b32_e32 v104, 1, v1
	v_add_u32_e32 v1, 0x60, v0
	s_waitcnt lgkmcnt(0)
	s_add_u32 s0, s8, s0
	s_mul_i32 s8, s6, s12
	v_mul_lo_u32 v1, s6, v1
	s_addc_u32 s1, s9, s1
	s_ashr_i32 s9, s8, 31
	v_lshlrev_b32_e32 v106, 1, v1
	v_or_b32_e32 v1, 64, v0
	s_and_b32 s1, s1, 0xffff
	s_lshl_b64 s[8:9], s[8:9], 1
	v_mul_lo_u32 v1, s6, v1
	s_add_u32 s8, s10, s8
	v_lshlrev_b32_e32 v107, 1, v1
	v_add_u32_e32 v1, 32, v0
	s_mov_b32 s3, 0x20000
	s_mov_b32 s2, -1
	s_addc_u32 s9, s11, s9
	s_add_i32 s4, s4, 15
	v_mul_lo_u32 v1, s6, v1
	v_mul_lo_u32 v0, s6, v0
	v_mov_b32_e32 v40, 0
	s_and_b32 s9, s9, 0xffff
	s_lshr_b32 s4, s4, 4
	v_lshlrev_b32_e32 v108, 1, v1
	v_lshlrev_b32_e32 v109, 1, v0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v13, 0
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v1, 0
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v7, 0
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v5, 0
	v_mov_b32_e32 v4, 0
	s_mov_b32 s10, s2
	s_mov_b32 s11, s3
	v_mov_b32_e32 v41, v40
	v_mov_b32_e32 v42, v40
	v_mov_b32_e32 v43, v40
	v_mov_b32_e32 v56, v40
	v_mov_b32_e32 v57, v40
	v_mov_b32_e32 v58, v40
	v_mov_b32_e32 v59, v40
	v_mov_b32_e32 v60, v40
	v_mov_b32_e32 v61, v40
	v_mov_b32_e32 v62, v40
	v_mov_b32_e32 v63, v40
	v_mov_b32_e32 v64, v40
	v_mov_b32_e32 v65, v40
	v_mov_b32_e32 v66, v40
	v_mov_b32_e32 v67, v40
	v_mov_b32_e32 v68, v40
	v_mov_b32_e32 v69, v40
	v_mov_b32_e32 v70, v40
	v_mov_b32_e32 v71, v40
	v_mov_b32_e32 v72, v40
	v_mov_b32_e32 v73, v40
	v_mov_b32_e32 v74, v40
	v_mov_b32_e32 v75, v40
	v_mov_b32_e32 v76, v40
	v_mov_b32_e32 v77, v40
	v_mov_b32_e32 v78, v40
	v_mov_b32_e32 v79, v40
	v_mov_b32_e32 v80, v40
	v_mov_b32_e32 v81, v40
	v_mov_b32_e32 v82, v40
	v_mov_b32_e32 v83, v40
	v_mov_b32_e32 v84, v40
	v_mov_b32_e32 v85, v40
	v_mov_b32_e32 v86, v40
	v_mov_b32_e32 v87, v40
	v_mov_b32_e32 v88, v40
	v_mov_b32_e32 v89, v40
	v_mov_b32_e32 v90, v40
	v_mov_b32_e32 v91, v40
	v_mov_b32_e32 v92, v40
	v_mov_b32_e32 v93, v40
	v_mov_b32_e32 v94, v40
	v_mov_b32_e32 v95, v40
	v_lshlrev_b32_e32 v110, 3, v96
.LBB0_2:                                ; %for.body
                                        ; =>This Inner Loop Header: Depth=1
	v_add_u32_e32 v111, v110, v105
	v_add_u32_e32 v112, v110, v104
	v_add_u32_e32 v113, v110, v103
	v_add_u32_e32 v114, v110, v102
	v_add_u32_e32 v115, v110, v109
	v_add_u32_e32 v116, v110, v108
	v_add_u32_e32 v117, v110, v107
	v_add_u32_e32 v118, v110, v106
	buffer_load_dwordx2 a[0:1], v111, s[0:3], 0 offen
	buffer_load_dwordx2 a[64:65], v115, s[8:11], 0 offen
	buffer_load_dwordx2 a[66:67], v116, s[8:11], 0 offen
	buffer_load_dwordx2 a[68:69], v117, s[8:11], 0 offen
                                        ; kill: killed $vgpr111
                                        ; kill: killed $vgpr116
                                        ; kill: killed $vgpr115
                                        ; kill: killed $vgpr117
	buffer_load_dwordx2 a[70:71], v118, s[8:11], 0 offen
	buffer_load_dwordx2 a[2:3], v112, s[0:3], 0 offen
	buffer_load_dwordx2 a[4:5], v113, s[0:3], 0 offen
	buffer_load_dwordx2 a[6:7], v114, s[0:3], 0 offen
	v_add_u32_e32 v111, v110, v101
	v_add_u32_e32 v112, v110, v100
	buffer_load_dwordx2 a[8:9], v111, s[0:3], 0 offen
	buffer_load_dwordx2 a[10:11], v112, s[0:3], 0 offen
	s_add_i32 s4, s4, -1
	v_add_u32_e32 v100, 32, v100
	v_add_u32_e32 v101, 32, v101
	v_add_u32_e32 v102, 32, v102
	v_add_u32_e32 v103, 32, v103
	v_add_u32_e32 v104, 32, v104
	v_add_u32_e32 v105, 32, v105
	v_add_u32_e32 v106, 32, v106
	v_add_u32_e32 v107, 32, v107
	v_add_u32_e32 v108, 32, v108
	s_cmp_eq_u32 s4, 0
	v_add_u32_e32 v109, 32, v109
	s_waitcnt vmcnt(8)
	v_mfma_f32_16x16x16_f16 v[4:7], a[64:65], a[0:1], v[4:7]
	s_waitcnt vmcnt(7)
	v_mfma_f32_16x16x16_f16 v[0:3], a[66:67], a[0:1], v[0:3]
	s_waitcnt vmcnt(6)
	v_mfma_f32_16x16x16_f16 v[8:11], a[68:69], a[0:1], v[8:11]
	s_waitcnt vmcnt(5)
	v_mfma_f32_16x16x16_f16 v[40:43], a[70:71], a[0:1], v[40:43]
	s_waitcnt vmcnt(4)
	v_mfma_f32_16x16x16_f16 v[12:15], a[64:65], a[2:3], v[12:15]
	v_mfma_f32_16x16x16_f16 v[56:59], a[66:67], a[2:3], v[56:59]
	v_mfma_f32_16x16x16_f16 v[16:19], a[68:69], a[2:3], v[16:19]
	v_mfma_f32_16x16x16_f16 v[60:63], a[70:71], a[2:3], v[60:63]
	s_waitcnt vmcnt(3)
	v_mfma_f32_16x16x16_f16 v[20:23], a[64:65], a[4:5], v[20:23]
	v_mfma_f32_16x16x16_f16 v[64:67], a[66:67], a[4:5], v[64:67]
	v_mfma_f32_16x16x16_f16 v[24:27], a[68:69], a[4:5], v[24:27]
	v_mfma_f32_16x16x16_f16 v[68:71], a[70:71], a[4:5], v[68:71]
	s_waitcnt vmcnt(2)
	v_mfma_f32_16x16x16_f16 v[28:31], a[64:65], a[6:7], v[28:31]
	v_mfma_f32_16x16x16_f16 v[72:75], a[66:67], a[6:7], v[72:75]
	v_mfma_f32_16x16x16_f16 v[32:35], a[68:69], a[6:7], v[32:35]
	v_mfma_f32_16x16x16_f16 v[76:79], a[70:71], a[6:7], v[76:79]
	s_waitcnt vmcnt(1)
	v_mfma_f32_16x16x16_f16 v[36:39], a[64:65], a[8:9], v[36:39]
	v_mfma_f32_16x16x16_f16 v[80:83], a[66:67], a[8:9], v[80:83]
	v_mfma_f32_16x16x16_f16 v[44:47], a[68:69], a[8:9], v[44:47]
	v_mfma_f32_16x16x16_f16 v[84:87], a[70:71], a[8:9], v[84:87]
	s_waitcnt vmcnt(0)
	v_mfma_f32_16x16x16_f16 v[48:51], a[64:65], a[10:11], v[48:51]
	v_mfma_f32_16x16x16_f16 v[88:91], a[66:67], a[10:11], v[88:91]
	v_mfma_f32_16x16x16_f16 v[52:55], a[68:69], a[10:11], v[52:55]
	v_mfma_f32_16x16x16_f16 v[92:95], a[70:71], a[10:11], v[92:95]
	s_cbranch_scc0 .LBB0_2
; %bb.3:                                ; %for.cond.cleanup.loopexit
	s_nop 5
	v_cvt_f16_f32_e32 v94, v94
	v_cvt_f16_f32_e32 v95, v95
	v_cvt_f16_f32_e32 v92, v92
	v_cvt_f16_f32_e32 v93, v93
	v_cvt_f16_f32_e32 v54, v54
	v_cvt_f16_f32_e32 v55, v55
	v_cvt_f16_f32_e32 v52, v52
	v_cvt_f16_f32_e32 v53, v53
	v_cvt_f16_f32_e32 v90, v90
	v_cvt_f16_f32_e32 v91, v91
	v_cvt_f16_f32_e32 v88, v88
	v_cvt_f16_f32_e32 v89, v89
	v_cvt_f16_f32_e32 v50, v50
	v_cvt_f16_f32_e32 v51, v51
	v_cvt_f16_f32_e32 v48, v48
	v_cvt_f16_f32_e32 v49, v49
	v_cvt_f16_f32_e32 v86, v86
	v_cvt_f16_f32_e32 v87, v87
	v_cvt_f16_f32_e32 v84, v84
	v_cvt_f16_f32_e32 v85, v85
	v_cvt_f16_f32_e32 v100, v46
	v_cvt_f16_f32_e32 v101, v47
	v_cvt_f16_f32_e32 v102, v44
	v_cvt_f16_f32_e32 v103, v45
	v_cvt_f16_f32_e32 v82, v82
	v_cvt_f16_f32_e32 v83, v83
	v_cvt_f16_f32_e32 v80, v80
	v_cvt_f16_f32_e32 v81, v81
	v_cvt_f16_f32_e32 v104, v38
	v_cvt_f16_f32_e32 v105, v39
	v_cvt_f16_f32_e32 v106, v36
	v_cvt_f16_f32_e32 v107, v37
	v_cvt_f16_f32_e32 v78, v78
	v_cvt_f16_f32_e32 v79, v79
	v_cvt_f16_f32_e32 v76, v76
	v_cvt_f16_f32_e32 v77, v77
	v_cvt_f16_f32_e32 v108, v34
	v_cvt_f16_f32_e32 v109, v35
	v_cvt_f16_f32_e32 v110, v32
	v_cvt_f16_f32_e32 v111, v33
	v_cvt_f16_f32_e32 v74, v74
	v_cvt_f16_f32_e32 v75, v75
	v_cvt_f16_f32_e32 v72, v72
	v_cvt_f16_f32_e32 v73, v73
	v_cvt_f16_f32_e32 v112, v30
	v_cvt_f16_f32_e32 v113, v31
	v_cvt_f16_f32_e32 v114, v28
	v_cvt_f16_f32_e32 v115, v29
	v_cvt_f16_f32_e32 v70, v70
	v_cvt_f16_f32_e32 v71, v71
	v_cvt_f16_f32_e32 v68, v68
	v_cvt_f16_f32_e32 v69, v69
	v_cvt_f16_f32_e32 v116, v26
	v_cvt_f16_f32_e32 v27, v27
	v_cvt_f16_f32_e32 v24, v24
	v_cvt_f16_f32_e32 v25, v25
	v_cvt_f16_f32_e32 v26, v66
	v_cvt_f16_f32_e32 v29, v67
	v_cvt_f16_f32_e32 v28, v64
	v_cvt_f16_f32_e32 v64, v65
	v_cvt_f16_f32_e32 v22, v22
	v_cvt_f16_f32_e32 v23, v23
	v_cvt_f16_f32_e32 v20, v20
	v_cvt_f16_f32_e32 v21, v21
	v_cvt_f16_f32_e32 v30, v62
	v_cvt_f16_f32_e32 v31, v63
	v_cvt_f16_f32_e32 v32, v60
	v_cvt_f16_f32_e32 v33, v61
	v_cvt_f16_f32_e32 v18, v18
	v_cvt_f16_f32_e32 v19, v19
	v_cvt_f16_f32_e32 v16, v16
	v_cvt_f16_f32_e32 v17, v17
	v_cvt_f16_f32_e32 v34, v58
	v_cvt_f16_f32_e32 v35, v59
	v_cvt_f16_f32_e32 v36, v56
	v_cvt_f16_f32_e32 v37, v57
	v_cvt_f16_f32_e32 v14, v14
	v_cvt_f16_f32_e32 v15, v15
	v_cvt_f16_f32_e32 v12, v12
	v_cvt_f16_f32_e32 v13, v13
	v_cvt_f16_f32_e32 v38, v42
	v_cvt_f16_f32_e32 v39, v43
	v_cvt_f16_f32_e32 v40, v40
	v_cvt_f16_f32_e32 v41, v41
	v_cvt_f16_f32_e32 v10, v10
	v_cvt_f16_f32_e32 v11, v11
	v_cvt_f16_f32_e32 v8, v8
	v_cvt_f16_f32_e32 v9, v9
	v_cvt_f16_f32_e32 v2, v2
	v_cvt_f16_f32_e32 v0, v0
	v_cvt_f16_f32_e32 v6, v6
	v_cvt_f16_f32_e32 v4, v4
	v_cvt_f16_f32_e32 v5, v5
	v_cvt_f16_f32_e32 v7, v7
	v_cvt_f16_f32_e32 v1, v1
	v_cvt_f16_f32_e32 v3, v3
	v_pack_b32_f16 v46, v4, v5
	v_pack_b32_f16 v47, v6, v7
	v_pack_b32_f16 v44, v0, v1
	v_pack_b32_f16 v45, v2, v3
	v_pack_b32_f16 v42, v8, v9
	v_pack_b32_f16 v43, v10, v11
	v_pack_b32_f16 v40, v40, v41
	v_pack_b32_f16 v41, v38, v39
	v_pack_b32_f16 v38, v12, v13
	v_pack_b32_f16 v39, v14, v15
	v_pack_b32_f16 v36, v36, v37
	v_pack_b32_f16 v37, v34, v35
	v_pack_b32_f16 v34, v16, v17
	v_pack_b32_f16 v35, v18, v19
	v_pack_b32_f16 v32, v32, v33
	v_pack_b32_f16 v33, v30, v31
	v_pack_b32_f16 v30, v20, v21
	v_pack_b32_f16 v31, v22, v23
	v_pack_b32_f16 v28, v28, v64
	v_pack_b32_f16 v29, v26, v29
	v_pack_b32_f16 v26, v24, v25
	v_pack_b32_f16 v27, v116, v27
	v_pack_b32_f16 v24, v68, v69
	v_pack_b32_f16 v25, v70, v71
	v_pack_b32_f16 v22, v114, v115
	v_pack_b32_f16 v23, v112, v113
	v_pack_b32_f16 v20, v72, v73
	v_pack_b32_f16 v21, v74, v75
	v_pack_b32_f16 v18, v110, v111
	v_pack_b32_f16 v19, v108, v109
	v_pack_b32_f16 v16, v76, v77
	v_pack_b32_f16 v17, v78, v79
	v_pack_b32_f16 v14, v106, v107
	v_pack_b32_f16 v15, v104, v105
	v_pack_b32_f16 v12, v80, v81
	v_pack_b32_f16 v13, v82, v83
	v_pack_b32_f16 v10, v102, v103
	v_pack_b32_f16 v11, v100, v101
	v_pack_b32_f16 v8, v84, v85
	v_pack_b32_f16 v9, v86, v87
	v_pack_b32_f16 v6, v48, v49
	v_pack_b32_f16 v7, v50, v51
	v_pack_b32_f16 v4, v88, v89
	v_pack_b32_f16 v5, v90, v91
	v_pack_b32_f16 v2, v52, v53
	v_pack_b32_f16 v3, v54, v55
	v_pack_b32_f16 v0, v92, v93
	v_pack_b32_f16 v1, v94, v95
	s_branch .LBB0_5
.LBB0_4:
	v_mov_b32_e32 v1, 0
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v5, 0
	v_mov_b32_e32 v4, 0
	v_mov_b32_e32 v7, 0
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v13, 0
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v46, 0
.LBB0_5:                                ; %Flow
	s_mul_i32 s0, s7, s13
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s2, s14, s0
	s_addc_u32 s3, s15, s1
	s_ashr_i32 s13, s12, 31
	s_lshl_b64 s[0:1], s[12:13], 1
	v_or_b32_e32 v49, v99, v98
	s_add_u32 s0, s2, s0
	v_lshl_or_b32 v48, v96, 2, v97
	s_addc_u32 s1, s3, s1
	s_lshl_b32 s6, s7, 5
	v_mad_u64_u32 v[48:49], s[4:5], v49, s7, v[48:49]
	s_and_b32 s1, s1, 0xffff
	s_mov_b32 s3, 0x20000
	s_mov_b32 s2, -1
	v_add_u32_e32 v49, 32, v48
	v_add_u32_e32 v50, 64, v48
	v_add_u32_e32 v51, 0x60, v48
	v_add_u32_e32 v52, s6, v48
	v_lshlrev_b32_e32 v48, 1, v48
	buffer_store_dwordx2 v[46:47], v48, s[0:3], 0 offen
	v_lshlrev_b32_e32 v46, 1, v49
	buffer_store_dwordx2 v[44:45], v46, s[0:3], 0 offen
	v_lshlrev_b32_e32 v44, 1, v50
	buffer_store_dwordx2 v[42:43], v44, s[0:3], 0 offen
	v_lshlrev_b32_e32 v42, 1, v51
	v_add_u32_e32 v53, s6, v49
	buffer_store_dwordx2 v[40:41], v42, s[0:3], 0 offen
	v_lshlrev_b32_e32 v40, 1, v52
	v_add_u32_e32 v54, s6, v50
	buffer_store_dwordx2 v[38:39], v40, s[0:3], 0 offen
	v_lshlrev_b32_e32 v38, 1, v53
	v_add_u32_e32 v55, s6, v51
	buffer_store_dwordx2 v[36:37], v38, s[0:3], 0 offen
	v_lshlrev_b32_e32 v36, 1, v54
	v_add_u32_e32 v56, s6, v52
	buffer_store_dwordx2 v[34:35], v36, s[0:3], 0 offen
	v_lshlrev_b32_e32 v34, 1, v55
	v_add_u32_e32 v57, s6, v53
	buffer_store_dwordx2 v[32:33], v34, s[0:3], 0 offen
	v_lshlrev_b32_e32 v32, 1, v56
	v_add_u32_e32 v58, s6, v54
	buffer_store_dwordx2 v[30:31], v32, s[0:3], 0 offen
	v_lshlrev_b32_e32 v30, 1, v57
	v_add_u32_e32 v59, s6, v55
	buffer_store_dwordx2 v[28:29], v30, s[0:3], 0 offen
	v_lshlrev_b32_e32 v28, 1, v58
	v_add_u32_e32 v60, s6, v56
	buffer_store_dwordx2 v[26:27], v28, s[0:3], 0 offen
	v_lshlrev_b32_e32 v26, 1, v59
	v_add_u32_e32 v61, s6, v57
	buffer_store_dwordx2 v[24:25], v26, s[0:3], 0 offen
	v_lshlrev_b32_e32 v24, 1, v60
	v_add_u32_e32 v62, s6, v58
	buffer_store_dwordx2 v[22:23], v24, s[0:3], 0 offen
	v_lshlrev_b32_e32 v22, 1, v61
	v_add_u32_e32 v63, s6, v59
	buffer_store_dwordx2 v[20:21], v22, s[0:3], 0 offen
	v_lshlrev_b32_e32 v20, 1, v62
	v_add_u32_e32 v64, s6, v60
	buffer_store_dwordx2 v[18:19], v20, s[0:3], 0 offen
	v_lshlrev_b32_e32 v18, 1, v63
	v_add_u32_e32 v65, s6, v61
	buffer_store_dwordx2 v[16:17], v18, s[0:3], 0 offen
	v_lshlrev_b32_e32 v16, 1, v64
	v_add_u32_e32 v66, s6, v62
	buffer_store_dwordx2 v[14:15], v16, s[0:3], 0 offen
	v_lshlrev_b32_e32 v14, 1, v65
	v_add_u32_e32 v67, s6, v63
	buffer_store_dwordx2 v[12:13], v14, s[0:3], 0 offen
	v_lshlrev_b32_e32 v12, 1, v66
	buffer_store_dwordx2 v[10:11], v12, s[0:3], 0 offen
	v_lshlrev_b32_e32 v10, 1, v67
	buffer_store_dwordx2 v[8:9], v10, s[0:3], 0 offen
	v_add_lshl_u32 v8, v64, s6, 1
	buffer_store_dwordx2 v[6:7], v8, s[0:3], 0 offen
	v_add_lshl_u32 v6, v65, s6, 1
	buffer_store_dwordx2 v[4:5], v6, s[0:3], 0 offen
	v_add_lshl_u32 v4, v66, s6, 1
	buffer_store_dwordx2 v[2:3], v4, s[0:3], 0 offen
	v_add_lshl_u32 v2, v67, s6, 1
	buffer_store_dwordx2 v[0:1], v2, s[0:3], 0 offen
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z27matrix_core_kernel_block_v2ILi256ELi192ELi128ELi16ELi2ELi2ELi1ELi16ELi16ELi16EEvPKvS1_Pviiii
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 40
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 192
		.amdhsa_next_free_sgpr 16
		.amdhsa_accum_offset 120
		.amdhsa_reserve_vcc 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z27matrix_core_kernel_block_v2ILi256ELi192ELi128ELi16ELi2ELi2ELi1ELi16ELi16ELi16EEvPKvS1_Pviiii,"axG",@progbits,_Z27matrix_core_kernel_block_v2ILi256ELi192ELi128ELi16ELi2ELi2ELi1ELi16ELi16ELi16EEvPKvS1_Pviiii,comdat
.Lfunc_end0:
	.size	_Z27matrix_core_kernel_block_v2ILi256ELi192ELi128ELi16ELi2ELi2ELi1ELi16ELi16ELi16EEvPKvS1_Pviiii, .Lfunc_end0-_Z27matrix_core_kernel_block_v2ILi256ELi192ELi128ELi16ELi2ELi2ELi1ELi16ELi16ELi16EEvPKvS1_Pviiii
                                        ; -- End function
	.set _Z27matrix_core_kernel_block_v2ILi256ELi192ELi128ELi16ELi2ELi2ELi1ELi16ELi16ELi16EEvPKvS1_Pviiii.num_vgpr, 119
	.set _Z27matrix_core_kernel_block_v2ILi256ELi192ELi128ELi16ELi2ELi2ELi1ELi16ELi16ELi16EEvPKvS1_Pviiii.num_agpr, 72
	.set _Z27matrix_core_kernel_block_v2ILi256ELi192ELi128ELi16ELi2ELi2ELi1ELi16ELi16ELi16EEvPKvS1_Pviiii.numbered_sgpr, 16
	.set _Z27matrix_core_kernel_block_v2ILi256ELi192ELi128ELi16ELi2ELi2ELi1ELi16ELi16ELi16EEvPKvS1_Pviiii.private_seg_size, 0
	.set _Z27matrix_core_kernel_block_v2ILi256ELi192ELi128ELi16ELi2ELi2ELi1ELi16ELi16ELi16EEvPKvS1_Pviiii.uses_vcc, 0
	.set _Z27matrix_core_kernel_block_v2ILi256ELi192ELi128ELi16ELi2ELi2ELi1ELi16ELi16ELi16EEvPKvS1_Pviiii.uses_flat_scratch, 0
	.set _Z27matrix_core_kernel_block_v2ILi256ELi192ELi128ELi16ELi2ELi2ELi1ELi16ELi16ELi16EEvPKvS1_Pviiii.has_dyn_sized_stack, 0
	.set _Z27matrix_core_kernel_block_v2ILi256ELi192ELi128ELi16ELi2ELi2ELi1ELi16ELi16ELi16EEvPKvS1_Pviiii.has_recursion, 0
	.set _Z27matrix_core_kernel_block_v2ILi256ELi192ELi128ELi16ELi2ELi2ELi1ELi16ELi16ELi16EEvPKvS1_Pviiii.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 2564
; TotalNumSgprs: 22
; NumVgprs: 119
; NumAgprs: 72
; TotalNumVgprs: 192
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 23
; NumSGPRsForWavesPerEU: 22
; NumVGPRsForWavesPerEU: 192
; AccumOffset: 120
; Occupancy: 2
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 29
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_6ae90eef49fee8d5,@object ; @__hip_cuid_6ae90eef49fee8d5
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_6ae90eef49fee8d5
__hip_cuid_6ae90eef49fee8d5:
	.byte	0                               ; 0x0
	.size	__hip_cuid_6ae90eef49fee8d5, 1

	.ident	"clang version 20.0.0git (https://github.com/ROCm/llvm-project.git 500de21e73b3ead1056df69392a8c0bfee8e696d)"
	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.1.1 25444 27682a16360e33e37c4f3cc6adf9a620733f8fe1)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_6ae90eef49fee8d5
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     72
    .args:
      - .actual_access:  read_only
        .address_space:  global
        .name:           ptr_a.coerce
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .name:           ptr_b.coerce
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .name:           ptr_c.coerce
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .name:           k
        .offset:         24
        .size:           4
        .value_kind:     by_value
      - .name:           stride_a
        .offset:         28
        .size:           4
        .value_kind:     by_value
      - .name:           stride_b
        .offset:         32
        .size:           4
        .value_kind:     by_value
      - .name:           stride_c
        .offset:         36
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 40
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z27matrix_core_kernel_block_v2ILi256ELi192ELi128ELi16ELi2ELi2ELi1ELi16ELi16ELi16EEvPKvS1_Pviiii
    .private_segment_fixed_size: 0
    .sgpr_count:     22
    .sgpr_spill_count: 0
    .symbol:         _Z27matrix_core_kernel_block_v2ILi256ELi192ELi128ELi16ELi2ELi2ELi1ELi16ELi16ELi16EEvPKvS1_Pviiii.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     192
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
