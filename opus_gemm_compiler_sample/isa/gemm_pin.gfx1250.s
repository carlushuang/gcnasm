	.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx1250"
	.amdhsa_code_object_version 6
	.text
	.weak	__cxa_pure_virtual              ; -- Begin function __cxa_pure_virtual
	.p2align	7
	.type	__cxa_pure_virtual,@function
__cxa_pure_virtual:                     ; @__cxa_pure_virtual
	.cfi_startproc
; %bb.0:                                ; %entry
	.cfi_escape 0x0f, 0x09, 0x90, 0x40, 0x94, 0x04, 0x35, 0x24, 0x36, 0xe9, 0x02 ; 
	.cfi_llvm_register_pair 16, 62, 32, 63, 32
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 2
	s_wait_loadcnt_dscnt 0x0
	s_wait_kmcnt 0x0
	s_trap 2
.Lfunc_end0:
	.size	__cxa_pure_virtual, .Lfunc_end0-__cxa_pure_virtual
	.cfi_endproc
                                        ; -- End function
	.set .L__cxa_pure_virtual.num_vgpr, 0
	.set .L__cxa_pure_virtual.num_agpr, 0
	.set .L__cxa_pure_virtual.numbered_sgpr, 0
	.set .L__cxa_pure_virtual.num_named_barrier, 0
	.set .L__cxa_pure_virtual.private_seg_size, 0
	.set .L__cxa_pure_virtual.uses_vcc, 0
	.set .L__cxa_pure_virtual.uses_flat_scratch, 0
	.set .L__cxa_pure_virtual.has_dyn_sized_stack, 0
	.set .L__cxa_pure_virtual.has_recursion, 0
	.set .L__cxa_pure_virtual.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Function info:
; codeLenInByte = 20
; TotalNumSgprs: 0
; NumVgprs: 0
; ScratchSize: 0
; MemoryBound: 0
	.text
	.weak	__cxa_deleted_virtual           ; -- Begin function __cxa_deleted_virtual
	.p2align	7
	.type	__cxa_deleted_virtual,@function
__cxa_deleted_virtual:                  ; @__cxa_deleted_virtual
	.cfi_startproc
; %bb.0:                                ; %entry
	.cfi_escape 0x0f, 0x09, 0x90, 0x40, 0x94, 0x04, 0x35, 0x24, 0x36, 0xe9, 0x02 ; 
	.cfi_llvm_register_pair 16, 62, 32, 63, 32
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 2
	s_wait_loadcnt_dscnt 0x0
	s_wait_kmcnt 0x0
	s_trap 2
.Lfunc_end1:
	.size	__cxa_deleted_virtual, .Lfunc_end1-__cxa_deleted_virtual
	.cfi_endproc
                                        ; -- End function
	.set .L__cxa_deleted_virtual.num_vgpr, 0
	.set .L__cxa_deleted_virtual.num_agpr, 0
	.set .L__cxa_deleted_virtual.numbered_sgpr, 0
	.set .L__cxa_deleted_virtual.num_named_barrier, 0
	.set .L__cxa_deleted_virtual.private_seg_size, 0
	.set .L__cxa_deleted_virtual.uses_vcc, 0
	.set .L__cxa_deleted_virtual.uses_flat_scratch, 0
	.set .L__cxa_deleted_virtual.has_dyn_sized_stack, 0
	.set .L__cxa_deleted_virtual.has_recursion, 0
	.set .L__cxa_deleted_virtual.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Function info:
; codeLenInByte = 20
; TotalNumSgprs: 0
; NumVgprs: 0
; ScratchSize: 0
; MemoryBound: 0
	.section	.text._Z32gemm_a16w16_4wave_compute_kernelI16opus_gemm_traitsILi128ELi128ELi256ELi128ELi3EDF16bDF16bDF16bfEEv15opus_gemm_kargs,"axG",@progbits,_Z32gemm_a16w16_4wave_compute_kernelI16opus_gemm_traitsILi128ELi128ELi256ELi128ELi3EDF16bDF16bDF16bfEEv15opus_gemm_kargs,comdat
	.protected	_Z32gemm_a16w16_4wave_compute_kernelI16opus_gemm_traitsILi128ELi128ELi256ELi128ELi3EDF16bDF16bDF16bfEEv15opus_gemm_kargs ; -- Begin function _Z32gemm_a16w16_4wave_compute_kernelI16opus_gemm_traitsILi128ELi128ELi256ELi128ELi3EDF16bDF16bDF16bfEEv15opus_gemm_kargs
	.globl	_Z32gemm_a16w16_4wave_compute_kernelI16opus_gemm_traitsILi128ELi128ELi256ELi128ELi3EDF16bDF16bDF16bfEEv15opus_gemm_kargs
	.p2align	8
	.type	_Z32gemm_a16w16_4wave_compute_kernelI16opus_gemm_traitsILi128ELi128ELi256ELi128ELi3EDF16bDF16bDF16bfEEv15opus_gemm_kargs,@function
_Z32gemm_a16w16_4wave_compute_kernelI16opus_gemm_traitsILi128ELi128ELi256ELi128ELi3EDF16bDF16bDF16bfEEv15opus_gemm_kargs: ; @_Z32gemm_a16w16_4wave_compute_kernelI16opus_gemm_traitsILi128ELi128ELi256ELi128ELi3EDF16bDF16bDF16bfEEv15opus_gemm_kargs
	.cfi_startproc
; %bb.0:                                ; %entry
	s_mov_b64 s[64:65], 0
	v_nop
	global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 2
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1 ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_load_b96 s[20:22], s[0:1], 0x18 nv
	;;#ASMSTART
	s_bfe_u32 s23, ttmp8, 0x50019
	;;#ASMEND
	s_wait_kmcnt 0x0
	s_add_co_i32 s35, s22, 0x7f
	s_cmp_lt_i32 s35, 0x80
	s_cbranch_scc1 .LBB2_12
; %bb.1:                                ; %if.end
	s_clause 0x3
	s_load_b64 s[24:25], s[0:1], 0x38 nv
	s_load_b128 s[16:19], s[0:1], 0x28 nv
	s_load_b128 s[8:11], s[0:1], 0x0 nv
	s_load_b64 s[4:5], s[0:1], 0x10 nv
	s_barrier_signal -1
	s_and_b32 s3, ttmp6, 15
	s_bfe_u32 s7, ttmp6, 0x40004
	s_and_b32 s0, ttmp7, 0xffff
	s_lshl_b32 s33, ttmp9, 9
	s_lshl_b32 s1, s3, 7
	s_lshl_b32 s0, s0, 10
	s_lshl_b32 s2, s7, 8
	s_mov_b32 s27, 0
	s_lshr_b32 s26, ttmp7, 16
	s_add_co_i32 s33, s33, s1
	s_add_co_i32 s6, s0, s2
	s_and_b32 s12, s23, 1
	s_lshl_b32 s13, s12, 6
	s_lshl_b32 s2, s12, 7
	s_wait_kmcnt 0x0
	s_ashr_i32 s1, s24, 31
	s_mov_b32 s0, s24
	s_ashr_i32 s15, s19, 31
	s_mov_b32 s14, s19
	s_mul_u64 s[0:1], s[0:1], s[26:27]
	s_mul_u64 s[14:15], s[14:15], s[26:27]
	s_lshl_b64 s[0:1], s[0:1], 1
	s_lshl_b64 s[28:29], s[14:15], 1
	s_add_nc_u64 s[0:1], s[10:11], s[0:1]
	s_or_b32 s14, s13, s33
	;;#ASMSTART
	s_sub_co_u32 s10, s20, s14
	;;#ASMEND
	s_or_b32 s2, s2, s6
	;;#ASMSTART
	s_sub_co_u32 s11, s21, s2
	;;#ASMEND
	;;#ASMSTART
	s_max_i32 s10, s10, 0
	;;#ASMEND
	;;#ASMSTART
	s_max_i32 s11, s11, 0
	;;#ASMEND
	s_cmp_lg_u32 s23, 0
	s_add_nc_u64 s[8:9], s[8:9], s[28:29]
	s_barrier_wait -1
	s_cbranch_scc1 .LBB2_3
; %bb.2:                                ; %if.then105
	s_barrier_signal -3
	s_barrier_wait -3
.LBB2_3:                                ; %if.end106
	s_barrier_signal -1
	s_cmp_ge_i32 s33, s20
	s_cselect_b32 s13, -1, 0
	s_cmp_ge_i32 s6, s21
	s_cselect_b32 s15, -1, 0
	s_or_b32 s13, s13, s15
	s_and_b32 vcc_lo, exec_lo, s13
	s_barrier_wait -1
	s_cbranch_vccnz .LBB2_12
; %bb.4:                                ; %_ZZ32gemm_a16w16_4wave_compute_kernelI16opus_gemm_traitsILi128ELi128ELi256ELi128ELi3EDF16bDF16bDF16bfEEv15opus_gemm_kargsENKUlvE0_clEv.exit10.i.i
	s_lshl_b32 s24, 0x1111, s3
	s_mov_b32 s15, 0
	s_mul_i32 s3, s12, 0x8800
	s_lshl_b32 s7, s7, 2
	v_mbcnt_lo_u32_b32 v64, -1, 0
	s_ashr_i32 s31, s16, 31
	s_add_co_i32 s19, s3, 0x19800
	s_ashr_i32 s37, s17, 31
	s_mov_b32 s30, s16
	s_mov_b32 s36, s17
	s_mov_b32 s3, s15
	s_lshl_b32 s7, 15, s7
	s_cmp_lt_i32 s23, 2
	s_mul_i32 s28, s12, 0x4400
	s_mul_u64 s[12:13], s[14:15], s[30:31]
	s_mul_u64 s[2:3], s[2:3], s[36:37]
	s_movk_i32 s34, 0x4400
	s_cselect_b32 s19, s28, s19
	s_cselect_b32 s29, s9, s1
	s_cselect_b32 s28, s8, s0
	s_cselect_b32 s1, s13, s3
	s_cselect_b32 s0, s12, s2
	s_mov_b32 s2, 0x11000
	v_dual_lshrrev_b32 v1, 1, v64 :: v_dual_bitop2_b32 v65, 15, v64 bitop3:0x40
	s_cselect_b32 s13, s16, s17
	s_cselect_b32 s8, s31, s37
	s_cselect_b32 s7, s24, s7
	s_cselect_b32 s12, 64, 0x80
	s_cselect_b32 s17, s10, s11
	s_cselect_b32 s39, s34, 0x8800
	s_cselect_b32 s24, 0x8800, s2
	s_lshl_b64 s[30:31], s[0:1], 1
	v_mul_u32_u24_e32 v0, 0x88, v65
	s_add_nc_u64 s[2:3], s[30:31], s[28:29]
	s_and_b32 s7, s7, 0xffff
	s_and_b32 s1, s3, 0x1ffffff
	s_lshr_b32 s14, s17, 16
	;;#ASMSTART
	s_sub_co_u32 s9, s22, s15
	;;#ASMEND
	;;#ASMSTART
	s_max_i32 s16, s9, 0
	;;#ASMEND
	s_lshr_b64 s[10:11], s[16:17], 16
	s_mov_b32 s0, 1
	s_or_b32 s3, s1, 0x80000000
	s_mov_b32 s1, s19
	s_lshl_b32 s9, s16, 16
	s_or_b32 s11, s14, 0x800000
	s_and_b32 s14, s8, 0xffff
	s_or_b32 s8, s7, 0x7510000
	s_add_nc_u64 s[30:31], s[30:31], 0x100
	tensor_load_to_lds s[0:3], s[8:15] scope:SCOPE_DEV
	s_add_nc_u64 s[2:3], s[30:31], s[28:29]
	s_lshl_b32 s34, s39, 1
	v_mad_u32 v0, 0x880, s23, v0
	s_movk_i32 s36, 0x80
	s_and_b32 s1, s3, 0x1ffffff
	s_add_co_i32 s7, s19, s34
	;;#ASMSTART
	s_sub_co_u32 s9, s22, s36
	;;#ASMEND
	;;#ASMSTART
	s_max_i32 s16, s9, 0
	;;#ASMEND
	s_lshr_b64 s[36:37], s[16:17], 16
	v_and_b32_e32 v66, 8, v1
	s_or_b32 s3, s1, 0x80000000
	s_mov_b32 s1, s7
	s_lshl_b32 s9, s16, 16
	s_mov_b32 s10, s36
	s_add_nc_u64 s[30:31], s[30:31], 0x100
	s_wait_tensorcnt 0xa
	tensor_load_to_lds s[0:3], s[8:15] scope:SCOPE_DEV
	s_add_nc_u64 s[2:3], s[30:31], s[28:29]
	s_movk_i32 s37, 0x100
	s_and_b32 s1, s3, 0x1ffffff
	;;#ASMSTART
	s_sub_co_u32 s9, s22, s37
	;;#ASMEND
	;;#ASMSTART
	s_max_i32 s16, s9, 0
	;;#ASMEND
	s_lshr_b64 s[40:41], s[16:17], 16
	v_add_lshl_u32 v212, v0, v66, 1
	s_or_b32 s3, s1, 0x80000000
	s_lshl1_add_u32 s1, s24, s19
	s_lshl_b32 s9, s16, 16
	s_mov_b32 s10, s40
	v_add_nc_u32_e32 v213, 64, v212
	s_wait_tensorcnt 0xa
	tensor_load_to_lds s[0:3], s[8:15] scope:SCOPE_DEV
	s_wait_tensorcnt 0x2
	s_barrier_signal -1
	v_mad_u32_u24 v67, 0x88, v65, v66
	v_add_nc_u32_e32 v214, 0x4400, v212
	v_add_nc_u32_e32 v215, 0x4440, v212
	s_lshr_b32 s35, s35, 7
	s_movk_i32 s38, 0x100
	v_lshlrev_b32_e32 v68, 1, v67
	v_add_nc_u32_e32 v233, 0x19800, v68
	v_add_nc_u32_e32 v247, 64, v233
	v_add_nc_u32_e32 v234, 0x19820, v68
	s_barrier_wait -1
	v_add_nc_u32_e32 v232, 0x19860, v68
	s_wait_alu depctr_va_vdst(9)
	ds_load_b128 v[48:51], v212 offset:17472
	ds_load_b128 v[52:55], v212 offset:17504
	s_wait_alu depctr_va_vdst(3)
	s_set_vgpr_msb 0x80                     ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[0:3] /*v[512:515]*/, v233
	ds_load_b128 v[8:11] /*v[520:523]*/, v233 offset:64
	s_wait_alu depctr_va_vdst(1)
	ds_load_b128 v[4:7] /*v[516:519]*/, v234
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[12:15] /*v[524:527]*/, v232
	ds_load_b128 v[16:19] /*v[528:531]*/, v233 offset:4352
	ds_load_b128 v[20:23] /*v[532:535]*/, v233 offset:4384
	ds_load_b128 v[24:27] /*v[536:539]*/, v233 offset:4416
	ds_load_b128 v[28:31] /*v[540:543]*/, v233 offset:4448
	ds_load_b128 v[32:35] /*v[544:547]*/, v233 offset:8704
	ds_load_b128 v[36:39] /*v[548:551]*/, v233 offset:8736
	ds_load_b128 v[40:43] /*v[552:555]*/, v233 offset:8768
	ds_load_b128 v[44:47] /*v[556:559]*/, v233 offset:8800
	ds_load_b128 v[48:51] /*v[560:563]*/, v233 offset:13056
	ds_load_b128 v[52:55] /*v[564:567]*/, v233 offset:13088
	ds_load_b128 v[56:59] /*v[568:571]*/, v233 offset:13120
	ds_load_b128 v[60:63] /*v[572:575]*/, v233 offset:13152
	s_set_vgpr_msb 0x8000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	ds_load_b128 v[24:27], v212 offset:128
	ds_load_b128 v[28:31], v212 offset:160
	ds_load_b128 v[8:11], v212 offset:192
	ds_load_b128 v[12:15], v212 offset:224
	ds_load_b128 v[16:19], v212 offset:17536
	ds_load_b128 v[20:23], v212 offset:17568
	ds_load_b128 v[0:3], v212 offset:17600
	ds_load_b128 v[4:7], v212 offset:17632
	ds_load_b128 v[40:43], v212
	ds_load_b128 v[44:47], v212 offset:32
	ds_load_b128 v[32:35], v212 offset:64
	ds_load_b128 v[36:39], v212 offset:96
	ds_load_b128 v[56:59], v212 offset:17408
	ds_load_b128 v[60:63], v212 offset:17440
	v_add_nc_u32_e32 v69, 32, v68
	v_add_nc_u32_e32 v68, 0x60, v68
	v_add_nc_u32_e32 v245, 0x1100, v233
	v_add_nc_u32_e32 v246, 0x1120, v233
	v_add_nc_u32_e32 v243, 0x1140, v233
	v_add_nc_u32_e32 v244, 0x1160, v233
	v_add_nc_u32_e32 v241, 0x2200, v233
	v_add_nc_u32_e32 v242, 0x2220, v233
	v_add_nc_u32_e32 v239, 0x2240, v233
	v_add_nc_u32_e32 v240, 0x2260, v233
	v_add_nc_u32_e32 v237, 0x3300, v233
	v_add_nc_u32_e32 v238, 0x3320, v233
	v_add_nc_u32_e32 v235, 0x3340, v233
	v_add_nc_u32_e32 v236, 0x3360, v233
	; sched_barrier mask(0x00000000)
	s_add_co_i32 s35, s35, -1
	v_add_nc_u32_e32 v230, 0x11000, v233
	s_mul_hi_u32 s1, s35, 0xaaaaaaab
	v_add_nc_u32_e32 v231, 0x11000, v234
	s_lshr_b32 s36, s1, 1
	v_add_nc_u32_e32 v222, 0x11000, v247
	v_add_nc_u32_e32 v225, 0x11000, v232
	v_add_nc_u32_e32 v226, 0x11000, v245
	v_add_nc_u32_e32 v227, 0x11000, v246
	v_add_nc_u32_e32 v228, 0x11000, v243
	v_add_nc_u32_e32 v229, 0x11000, v244
	v_add_nc_u32_e32 v223, 0x11000, v241
	v_add_nc_u32_e32 v224, 0x11000, v242
	v_add_nc_u32_e32 v220, 0x11000, v239
	v_add_nc_u32_e32 v221, 0x11000, v240
	v_add_nc_u32_e32 v218, 0x11000, v237
	v_add_nc_u32_e32 v219, 0x11000, v238
	v_add_nc_u32_e32 v216, 0x11000, v235
	v_add_nc_u32_e32 v217, 0x11000, v236
	v_add_nc_u32_e32 v210, 0x15400, v233
	v_add_nc_u32_e32 v211, 0x15400, v234
	v_add_nc_u32_e32 v208, 0x15400, v247
	v_add_nc_u32_e32 v209, 0x15400, v232
	v_add_nc_u32_e32 v206, 0x15400, v245
	v_add_nc_u32_e32 v207, 0x15400, v246
	v_add_nc_u32_e32 v204, 0x15400, v243
	v_add_nc_u32_e32 v205, 0x15400, v244
	v_add_nc_u32_e32 v202, 0x15400, v241
	v_add_nc_u32_e32 v203, 0x15400, v242
	v_add_nc_u32_e32 v200, 0x15400, v239
	v_add_nc_u32_e32 v201, 0x15400, v240
	v_add_nc_u32_e32 v198, 0x15400, v237
	v_add_nc_u32_e32 v199, 0x15400, v238
	v_add_nc_u32_e32 v196, 0x15400, v235
	v_add_nc_u32_e32 v197, 0x15400, v236
	v_add_nc_u32_e32 v194, 0x19800, v233
	v_add_nc_u32_e32 v195, 0x19800, v234
	v_add_nc_u32_e32 v192, 0x19800, v247
	v_add_nc_u32_e32 v193, 0x19800, v232
	v_add_nc_u32_e32 v190, 0x19800, v245
	v_add_nc_u32_e32 v191, 0x19800, v246
	v_add_nc_u32_e32 v188, 0x19800, v243
	v_add_nc_u32_e32 v189, 0x19800, v244
	v_add_nc_u32_e32 v186, 0x19800, v241
	v_add_nc_u32_e32 v187, 0x19800, v242
	v_add_nc_u32_e32 v184, 0x19800, v239
	v_add_nc_u32_e32 v185, 0x19800, v240
	v_add_nc_u32_e32 v182, 0x19800, v237
	v_add_nc_u32_e32 v183, 0x19800, v238
	v_add_nc_u32_e32 v180, 0x19800, v235
	v_add_nc_u32_e32 v181, 0x19800, v236
	v_add_nc_u32_e32 v178, 0x1dc00, v233
	v_add_nc_u32_e32 v179, 0x1dc00, v234
	v_add_nc_u32_e32 v176, 0x1dc00, v247
	v_add_nc_u32_e32 v177, 0x1dc00, v232
	v_add_nc_u32_e32 v174, 0x1dc00, v245
	v_add_nc_u32_e32 v175, 0x1dc00, v246
	v_add_nc_u32_e32 v172, 0x1dc00, v243
	v_add_nc_u32_e32 v173, 0x1dc00, v244
	v_add_nc_u32_e32 v170, 0x1dc00, v241
	v_add_nc_u32_e32 v171, 0x1dc00, v242
	v_add_nc_u32_e32 v168, 0x1dc00, v239
	v_add_nc_u32_e32 v169, 0x1dc00, v240
	v_add_nc_u32_e32 v166, 0x1dc00, v237
	v_add_nc_u32_e32 v167, 0x1dc00, v238
	v_add_nc_u32_e32 v164, 0x1dc00, v235
	v_add_nc_u32_e32 v165, 0x1dc00, v236
	v_add_nc_u32_e32 v162, 0x11080, v233
	v_add_nc_u32_e32 v163, 0x11080, v234
	v_add_nc_u32_e32 v160, 0x11080, v247
	v_add_nc_u32_e32 v161, 0x11080, v232
	v_add_nc_u32_e32 v158, 0x11080, v245
	v_add_nc_u32_e32 v159, 0x11080, v246
	v_add_nc_u32_e32 v156, 0x11080, v243
	v_add_nc_u32_e32 v157, 0x11080, v244
	v_add_nc_u32_e32 v154, 0x11080, v241
	v_add_nc_u32_e32 v155, 0x11080, v242
	v_add_nc_u32_e32 v152, 0x11080, v239
	v_add_nc_u32_e32 v153, 0x11080, v240
	v_add_nc_u32_e32 v150, 0x11080, v237
	v_add_nc_u32_e32 v151, 0x11080, v238
	v_add_nc_u32_e32 v148, 0x11080, v235
	v_add_nc_u32_e32 v149, 0x11080, v236
	v_add_nc_u32_e32 v146, 0x15480, v233
	v_add_nc_u32_e32 v147, 0x15480, v234
	v_add_nc_u32_e32 v144, 0x15480, v247
	v_add_nc_u32_e32 v145, 0x15480, v232
	v_add_nc_u32_e32 v142, 0x15480, v245
	v_add_nc_u32_e32 v143, 0x15480, v246
	v_add_nc_u32_e32 v140, 0x15480, v243
	v_add_nc_u32_e32 v141, 0x15480, v244
	v_add_nc_u32_e32 v138, 0x15480, v241
	v_add_nc_u32_e32 v139, 0x15480, v242
	v_add_nc_u32_e32 v136, 0x15480, v239
	v_add_nc_u32_e32 v137, 0x15480, v240
	v_add_nc_u32_e32 v134, 0x15480, v237
	v_add_nc_u32_e32 v135, 0x15480, v238
	v_add_nc_u32_e32 v132, 0x15480, v235
	v_add_nc_u32_e32 v133, 0x15480, v236
	v_add_nc_u32_e32 v130, 0x19880, v233
	v_add_nc_u32_e32 v131, 0x19880, v234
	v_add_nc_u32_e32 v128, 0x19880, v247
	v_add_nc_u32_e32 v129, 0x19880, v232
	v_add_nc_u32_e32 v126, 0x19880, v245
	v_add_nc_u32_e32 v127, 0x19880, v246
	v_add_nc_u32_e32 v124, 0x19880, v243
	v_add_nc_u32_e32 v125, 0x19880, v244
	v_add_nc_u32_e32 v122, 0x19880, v241
	v_add_nc_u32_e32 v123, 0x19880, v242
	v_add_nc_u32_e32 v120, 0x19880, v239
	v_add_nc_u32_e32 v121, 0x19880, v240
	v_add_nc_u32_e32 v118, 0x19880, v237
	v_add_nc_u32_e32 v119, 0x19880, v238
	v_add_nc_u32_e32 v116, 0x19880, v235
	v_add_nc_u32_e32 v117, 0x19880, v236
	v_add_nc_u32_e32 v114, 0x1dc80, v233
	v_add_nc_u32_e32 v115, 0x1dc80, v234
	v_add_nc_u32_e32 v112, 0x1dc80, v247
	v_add_nc_u32_e32 v113, 0x1dc80, v232
	v_add_nc_u32_e32 v110, 0x1dc80, v245
	v_add_nc_u32_e32 v111, 0x1dc80, v246
	v_add_nc_u32_e32 v108, 0x1dc80, v243
	v_add_nc_u32_e32 v109, 0x1dc80, v244
	v_add_nc_u32_e32 v106, 0x1dc80, v241
	v_add_nc_u32_e32 v107, 0x1dc80, v242
	v_add_nc_u32_e32 v104, 0x1dc80, v239
	v_add_nc_u32_e32 v105, 0x1dc80, v240
	v_add_nc_u32_e32 v102, 0x1dc80, v237
	v_add_nc_u32_e32 v103, 0x1dc80, v238
	v_add_nc_u32_e32 v100, 0x1dc80, v235
	v_add_nc_u32_e32 v101, 0x1dc80, v236
	v_add_nc_u32_e32 v96, 0x11000, v212
	v_add_nc_u32_e32 v97, 0x11020, v212
	v_add_nc_u32_e32 v94, 0x11000, v213
	v_add_nc_u32_e32 v95, 0x11060, v212
	v_add_nc_u32_e32 v91, 0x11000, v214
	v_add_nc_u32_e32 v92, 0x15420, v212
	v_add_nc_u32_e32 v82, 0x11000, v215
	v_add_nc_u32_e32 v83, 0x15460, v212
	v_add_nc_u32_e32 v98, 0x22000, v233
	v_or_b32_e32 v99, 0x22000, v69
	v_add_nc_u32_e32 v86, 0x22000, v247
	v_or_b32_e32 v93, 0x22000, v68
	v_add_nc_u32_e32 v87, 0x22000, v245
	v_add_nc_u32_e32 v88, 0x22000, v246
	v_add_nc_u32_e32 v89, 0x22000, v243
	v_add_nc_u32_e32 v90, 0x22000, v244
	v_add_nc_u32_e32 v84, 0x22000, v241
	v_add_nc_u32_e32 v85, 0x22000, v242
	v_add_nc_u32_e32 v80, 0x22000, v239
	v_add_nc_u32_e32 v81, 0x22000, v240
	v_add_nc_u32_e32 v78, 0x22000, v237
	v_add_nc_u32_e32 v79, 0x22000, v238
	v_add_nc_u32_e32 v76, 0x22000, v235
	v_add_nc_u32_e32 v77, 0x22000, v236
	v_add_nc_u32_e32 v68, 0x11080, v212
	v_add_nc_u32_e32 v69, 0x110a0, v212
	v_add_nc_u32_e32 v70, 0x11080, v213
	v_add_nc_u32_e32 v71, 0x110e0, v212
	v_add_nc_u32_e32 v72, 0x11080, v214
	v_add_nc_u32_e32 v73, 0x154a0, v212
	v_add_nc_u32_e32 v74, 0x11080, v215
	v_add_nc_u32_e32 v75, 0x154e0, v212
	s_mul_i32 s36, s36, 3
	s_cmp_lt_i32 s36, 1
	s_cbranch_scc1 .LBB2_13
; %bb.5:                                ; %for.body.lr.ph
	v_add_nc_u32_e32 v248, 0x26400, v233
	v_add_nc_u32_e32 v249, 0x26400, v234
	v_add_nc_u32_e32 v250, 0x26400, v247
	v_add_nc_u32_e32 v251, 0x26400, v232
	v_add_nc_u32_e32 v252, 0x26400, v245
	v_add_nc_u32_e32 v253, 0x26400, v246
	v_add_nc_u32_e32 v254, 0x26400, v243
	v_add_nc_u32_e32 v255, 0x26400, v244
	s_set_vgpr_msb 0x80                     ;  msbs: dst=2 src0=0 src1=0 src2=0
	v_add_nc_u32_e32 v128 /*v640*/, 0x26400, v241
	v_add_nc_u32_e32 v129 /*v641*/, 0x26400, v242
	v_add_nc_u32_e32 v130 /*v642*/, 0x26400, v239
	v_add_nc_u32_e32 v131 /*v643*/, 0x26400, v240
	v_add_nc_u32_e32 v132 /*v644*/, 0x26400, v237
	v_add_nc_u32_e32 v133 /*v645*/, 0x26400, v238
	v_add_nc_u32_e32 v134 /*v646*/, 0x26400, v235
	v_add_nc_u32_e32 v135 /*v647*/, 0x26400, v236
	v_add_nc_u32_e32 v136 /*v648*/, 0x2a800, v233
	v_add_nc_u32_e32 v137 /*v649*/, 0x2a800, v234
	v_add_nc_u32_e32 v138 /*v650*/, 0x2a800, v247
	v_add_nc_u32_e32 v139 /*v651*/, 0x2a800, v232
	v_add_nc_u32_e32 v140 /*v652*/, 0x2a800, v245
	v_add_nc_u32_e32 v141 /*v653*/, 0x2a800, v246
	v_add_nc_u32_e32 v142 /*v654*/, 0x2a800, v243
	v_add_nc_u32_e32 v143 /*v655*/, 0x2a800, v244
	v_add_nc_u32_e32 v144 /*v656*/, 0x2a800, v241
	v_add_nc_u32_e32 v145 /*v657*/, 0x2a800, v242
	v_add_nc_u32_e32 v146 /*v658*/, 0x2a800, v239
	v_add_nc_u32_e32 v147 /*v659*/, 0x2a800, v240
	v_add_nc_u32_e32 v148 /*v660*/, 0x2a800, v237
	v_add_nc_u32_e32 v149 /*v661*/, 0x2a800, v238
	v_add_nc_u32_e32 v150 /*v662*/, 0x2a800, v235
	v_add_nc_u32_e32 v151 /*v663*/, 0x2a800, v236
	v_add_nc_u32_e32 v152 /*v664*/, 0x2ec00, v233
	v_add_nc_u32_e32 v153 /*v665*/, 0x2ec00, v234
	v_add_nc_u32_e32 v154 /*v666*/, 0x2ec00, v247
	v_add_nc_u32_e32 v155 /*v667*/, 0x2ec00, v232
	v_add_nc_u32_e32 v156 /*v668*/, 0x2ec00, v245
	v_add_nc_u32_e32 v157 /*v669*/, 0x2ec00, v246
	v_add_nc_u32_e32 v158 /*v670*/, 0x2ec00, v243
	v_add_nc_u32_e32 v159 /*v671*/, 0x2ec00, v244
	v_add_nc_u32_e32 v160 /*v672*/, 0x2ec00, v241
	v_add_nc_u32_e32 v161 /*v673*/, 0x2ec00, v242
	v_add_nc_u32_e32 v162 /*v674*/, 0x2ec00, v239
	v_add_nc_u32_e32 v163 /*v675*/, 0x2ec00, v240
	v_add_nc_u32_e32 v164 /*v676*/, 0x2ec00, v237
	v_add_nc_u32_e32 v165 /*v677*/, 0x2ec00, v238
	v_add_nc_u32_e32 v166 /*v678*/, 0x2ec00, v235
	v_add_nc_u32_e32 v167 /*v679*/, 0x2ec00, v236
	v_add_nc_u32_e32 v168 /*v680*/, 0x22080, v233
	v_add_nc_u32_e32 v169 /*v681*/, 0x22080, v234
	v_add_nc_u32_e32 v170 /*v682*/, 0x22080, v247
	v_add_nc_u32_e32 v171 /*v683*/, 0x22080, v232
	v_add_nc_u32_e32 v172 /*v684*/, 0x22080, v245
	v_add_nc_u32_e32 v173 /*v685*/, 0x22080, v246
	v_add_nc_u32_e32 v174 /*v686*/, 0x22080, v243
	v_add_nc_u32_e32 v175 /*v687*/, 0x22080, v244
	v_add_nc_u32_e32 v176 /*v688*/, 0x22080, v241
	v_add_nc_u32_e32 v177 /*v689*/, 0x22080, v242
	v_add_nc_u32_e32 v178 /*v690*/, 0x22080, v239
	v_add_nc_u32_e32 v179 /*v691*/, 0x22080, v240
	v_add_nc_u32_e32 v180 /*v692*/, 0x22080, v237
	v_add_nc_u32_e32 v181 /*v693*/, 0x22080, v238
	v_add_nc_u32_e32 v182 /*v694*/, 0x22080, v235
	v_add_nc_u32_e32 v183 /*v695*/, 0x22080, v236
	v_add_nc_u32_e32 v184 /*v696*/, 0x26480, v233
	v_add_nc_u32_e32 v185 /*v697*/, 0x26480, v234
	v_add_nc_u32_e32 v186 /*v698*/, 0x26480, v247
	v_add_nc_u32_e32 v187 /*v699*/, 0x26480, v232
	v_add_nc_u32_e32 v188 /*v700*/, 0x26480, v245
	v_add_nc_u32_e32 v189 /*v701*/, 0x26480, v246
	v_add_nc_u32_e32 v190 /*v702*/, 0x26480, v243
	v_add_nc_u32_e32 v191 /*v703*/, 0x26480, v244
	v_add_nc_u32_e32 v192 /*v704*/, 0x26480, v241
	v_add_nc_u32_e32 v193 /*v705*/, 0x26480, v242
	v_add_nc_u32_e32 v194 /*v706*/, 0x26480, v239
	v_add_nc_u32_e32 v195 /*v707*/, 0x26480, v240
	v_add_nc_u32_e32 v196 /*v708*/, 0x26480, v237
	v_add_nc_u32_e32 v197 /*v709*/, 0x26480, v238
	v_add_nc_u32_e32 v198 /*v710*/, 0x26480, v235
	v_add_nc_u32_e32 v199 /*v711*/, 0x26480, v236
	v_add_nc_u32_e32 v200 /*v712*/, 0x2a880, v233
	v_add_nc_u32_e32 v201 /*v713*/, 0x2a880, v234
	v_add_nc_u32_e32 v202 /*v714*/, 0x2a880, v247
	v_add_nc_u32_e32 v203 /*v715*/, 0x2a880, v232
	v_add_nc_u32_e32 v204 /*v716*/, 0x2a880, v245
	v_add_nc_u32_e32 v205 /*v717*/, 0x2a880, v246
	v_add_nc_u32_e32 v206 /*v718*/, 0x2a880, v243
	v_add_nc_u32_e32 v207 /*v719*/, 0x2a880, v244
	v_add_nc_u32_e32 v208 /*v720*/, 0x2a880, v241
	v_add_nc_u32_e32 v209 /*v721*/, 0x2a880, v242
	v_add_nc_u32_e32 v210 /*v722*/, 0x2a880, v239
	v_add_nc_u32_e32 v211 /*v723*/, 0x2a880, v240
	v_add_nc_u32_e32 v212 /*v724*/, 0x2a880, v237
	v_add_nc_u32_e32 v213 /*v725*/, 0x2a880, v238
	v_add_nc_u32_e32 v214 /*v726*/, 0x2a880, v235
	v_add_nc_u32_e32 v215 /*v727*/, 0x2a880, v236
	v_add_nc_u32_e32 v216 /*v728*/, 0x2ec80, v233
	v_add_nc_u32_e32 v217 /*v729*/, 0x2ec80, v234
	v_add_nc_u32_e32 v218 /*v730*/, 0x2ec80, v247
	v_add_nc_u32_e32 v219 /*v731*/, 0x2ec80, v232
	v_add_nc_u32_e32 v220 /*v732*/, 0x2ec80, v245
	v_add_nc_u32_e32 v221 /*v733*/, 0x2ec80, v246
	v_add_nc_u32_e32 v222 /*v734*/, 0x2ec80, v243
	v_add_nc_u32_e32 v223 /*v735*/, 0x2ec80, v244
	v_add_nc_u32_e32 v224 /*v736*/, 0x2ec80, v241
	v_add_nc_u32_e32 v225 /*v737*/, 0x2ec80, v242
	v_add_nc_u32_e32 v226 /*v738*/, 0x2ec80, v239
	v_add_nc_u32_e32 v227 /*v739*/, 0x2ec80, v240
	v_add_nc_u32_e32 v228 /*v740*/, 0x2ec80, v237
	v_add_nc_u32_e32 v229 /*v741*/, 0x2ec80, v238
	v_add_nc_u32_e32 v230 /*v742*/, 0x2ec80, v235
	v_add_nc_u32_e32 v231 /*v743*/, 0x2ec80, v236
	s_set_vgpr_msb 0x8040                   ;  msbs: dst=1 src0=0 src1=0 src2=0
	v_dual_mov_b32 v0 /*v256*/, 0 :: v_dual_mov_b32 v1 /*v257*/, 0
	v_dual_mov_b32 v2 /*v258*/, 0 :: v_dual_mov_b32 v3 /*v259*/, 0
	v_dual_mov_b32 v4 /*v260*/, 0 :: v_dual_mov_b32 v5 /*v261*/, 0
	v_dual_mov_b32 v6 /*v262*/, 0 :: v_dual_mov_b32 v7 /*v263*/, 0
	v_dual_mov_b32 v8 /*v264*/, 0 :: v_dual_mov_b32 v9 /*v265*/, 0
	v_dual_mov_b32 v10 /*v266*/, 0 :: v_dual_mov_b32 v11 /*v267*/, 0
	v_dual_mov_b32 v12 /*v268*/, 0 :: v_dual_mov_b32 v13 /*v269*/, 0
	v_dual_mov_b32 v14 /*v270*/, 0 :: v_dual_mov_b32 v15 /*v271*/, 0
	v_dual_mov_b32 v16 /*v272*/, 0 :: v_dual_mov_b32 v17 /*v273*/, 0
	v_dual_mov_b32 v18 /*v274*/, 0 :: v_dual_mov_b32 v19 /*v275*/, 0
	v_dual_mov_b32 v20 /*v276*/, 0 :: v_dual_mov_b32 v21 /*v277*/, 0
	v_dual_mov_b32 v22 /*v278*/, 0 :: v_dual_mov_b32 v23 /*v279*/, 0
	v_dual_mov_b32 v24 /*v280*/, 0 :: v_dual_mov_b32 v25 /*v281*/, 0
	v_dual_mov_b32 v26 /*v282*/, 0 :: v_dual_mov_b32 v27 /*v283*/, 0
	v_dual_mov_b32 v28 /*v284*/, 0 :: v_dual_mov_b32 v29 /*v285*/, 0
	v_dual_mov_b32 v30 /*v286*/, 0 :: v_dual_mov_b32 v31 /*v287*/, 0
	v_dual_mov_b32 v32 /*v288*/, 0 :: v_dual_mov_b32 v33 /*v289*/, 0
	v_dual_mov_b32 v34 /*v290*/, 0 :: v_dual_mov_b32 v35 /*v291*/, 0
	v_dual_mov_b32 v36 /*v292*/, 0 :: v_dual_mov_b32 v37 /*v293*/, 0
	v_dual_mov_b32 v38 /*v294*/, 0 :: v_dual_mov_b32 v39 /*v295*/, 0
	v_dual_mov_b32 v40 /*v296*/, 0 :: v_dual_mov_b32 v41 /*v297*/, 0
	v_dual_mov_b32 v42 /*v298*/, 0 :: v_dual_mov_b32 v43 /*v299*/, 0
	v_dual_mov_b32 v44 /*v300*/, 0 :: v_dual_mov_b32 v45 /*v301*/, 0
	v_dual_mov_b32 v46 /*v302*/, 0 :: v_dual_mov_b32 v47 /*v303*/, 0
	v_dual_mov_b32 v48 /*v304*/, 0 :: v_dual_mov_b32 v49 /*v305*/, 0
	v_dual_mov_b32 v50 /*v306*/, 0 :: v_dual_mov_b32 v51 /*v307*/, 0
	v_dual_mov_b32 v52 /*v308*/, 0 :: v_dual_mov_b32 v53 /*v309*/, 0
	v_dual_mov_b32 v54 /*v310*/, 0 :: v_dual_mov_b32 v55 /*v311*/, 0
	v_dual_mov_b32 v56 /*v312*/, 0 :: v_dual_mov_b32 v57 /*v313*/, 0
	v_dual_mov_b32 v58 /*v314*/, 0 :: v_dual_mov_b32 v59 /*v315*/, 0
	v_dual_mov_b32 v60 /*v316*/, 0 :: v_dual_mov_b32 v61 /*v317*/, 0
	v_dual_mov_b32 v62 /*v318*/, 0 :: v_dual_mov_b32 v63 /*v319*/, 0
	v_dual_mov_b32 v64 /*v320*/, 0 :: v_dual_mov_b32 v65 /*v321*/, 0
	v_dual_mov_b32 v66 /*v322*/, 0 :: v_dual_mov_b32 v67 /*v323*/, 0
	v_dual_mov_b32 v68 /*v324*/, 0 :: v_dual_mov_b32 v69 /*v325*/, 0
	v_dual_mov_b32 v70 /*v326*/, 0 :: v_dual_mov_b32 v71 /*v327*/, 0
	v_dual_mov_b32 v72 /*v328*/, 0 :: v_dual_mov_b32 v73 /*v329*/, 0
	v_dual_mov_b32 v74 /*v330*/, 0 :: v_dual_mov_b32 v75 /*v331*/, 0
	v_dual_mov_b32 v76 /*v332*/, 0 :: v_dual_mov_b32 v77 /*v333*/, 0
	v_dual_mov_b32 v78 /*v334*/, 0 :: v_dual_mov_b32 v79 /*v335*/, 0
	v_dual_mov_b32 v80 /*v336*/, 0 :: v_dual_mov_b32 v81 /*v337*/, 0
	v_dual_mov_b32 v82 /*v338*/, 0 :: v_dual_mov_b32 v83 /*v339*/, 0
	v_dual_mov_b32 v84 /*v340*/, 0 :: v_dual_mov_b32 v85 /*v341*/, 0
	v_dual_mov_b32 v86 /*v342*/, 0 :: v_dual_mov_b32 v87 /*v343*/, 0
	v_dual_mov_b32 v88 /*v344*/, 0 :: v_dual_mov_b32 v89 /*v345*/, 0
	v_dual_mov_b32 v90 /*v346*/, 0 :: v_dual_mov_b32 v91 /*v347*/, 0
	v_dual_mov_b32 v92 /*v348*/, 0 :: v_dual_mov_b32 v93 /*v349*/, 0
	v_dual_mov_b32 v94 /*v350*/, 0 :: v_dual_mov_b32 v95 /*v351*/, 0
	v_dual_mov_b32 v96 /*v352*/, 0 :: v_dual_mov_b32 v97 /*v353*/, 0
	v_dual_mov_b32 v98 /*v354*/, 0 :: v_dual_mov_b32 v99 /*v355*/, 0
	v_dual_mov_b32 v100 /*v356*/, 0 :: v_dual_mov_b32 v101 /*v357*/, 0
	v_dual_mov_b32 v102 /*v358*/, 0 :: v_dual_mov_b32 v103 /*v359*/, 0
	v_dual_mov_b32 v104 /*v360*/, 0 :: v_dual_mov_b32 v105 /*v361*/, 0
	v_dual_mov_b32 v106 /*v362*/, 0 :: v_dual_mov_b32 v107 /*v363*/, 0
	v_dual_mov_b32 v108 /*v364*/, 0 :: v_dual_mov_b32 v109 /*v365*/, 0
	v_dual_mov_b32 v110 /*v366*/, 0 :: v_dual_mov_b32 v111 /*v367*/, 0
	v_dual_mov_b32 v112 /*v368*/, 0 :: v_dual_mov_b32 v113 /*v369*/, 0
	v_dual_mov_b32 v114 /*v370*/, 0 :: v_dual_mov_b32 v115 /*v371*/, 0
	v_dual_mov_b32 v116 /*v372*/, 0 :: v_dual_mov_b32 v117 /*v373*/, 0
	v_dual_mov_b32 v118 /*v374*/, 0 :: v_dual_mov_b32 v119 /*v375*/, 0
	v_dual_mov_b32 v120 /*v376*/, 0 :: v_dual_mov_b32 v121 /*v377*/, 0
	v_dual_mov_b32 v122 /*v378*/, 0 :: v_dual_mov_b32 v123 /*v379*/, 0
	v_dual_mov_b32 v124 /*v380*/, 0 :: v_dual_mov_b32 v125 /*v381*/, 0
	v_dual_mov_b32 v126 /*v382*/, 0 :: v_dual_mov_b32 v127 /*v383*/, 0
	v_dual_mov_b32 v128 /*v384*/, 0 :: v_dual_mov_b32 v129 /*v385*/, 0
	v_dual_mov_b32 v130 /*v386*/, 0 :: v_dual_mov_b32 v131 /*v387*/, 0
	v_dual_mov_b32 v132 /*v388*/, 0 :: v_dual_mov_b32 v133 /*v389*/, 0
	v_dual_mov_b32 v134 /*v390*/, 0 :: v_dual_mov_b32 v135 /*v391*/, 0
	v_dual_mov_b32 v136 /*v392*/, 0 :: v_dual_mov_b32 v137 /*v393*/, 0
	v_dual_mov_b32 v138 /*v394*/, 0 :: v_dual_mov_b32 v139 /*v395*/, 0
	v_dual_mov_b32 v140 /*v396*/, 0 :: v_dual_mov_b32 v141 /*v397*/, 0
	v_dual_mov_b32 v142 /*v398*/, 0 :: v_dual_mov_b32 v143 /*v399*/, 0
	v_dual_mov_b32 v144 /*v400*/, 0 :: v_dual_mov_b32 v145 /*v401*/, 0
	v_dual_mov_b32 v146 /*v402*/, 0 :: v_dual_mov_b32 v147 /*v403*/, 0
	v_dual_mov_b32 v148 /*v404*/, 0 :: v_dual_mov_b32 v149 /*v405*/, 0
	v_dual_mov_b32 v150 /*v406*/, 0 :: v_dual_mov_b32 v151 /*v407*/, 0
	v_dual_mov_b32 v152 /*v408*/, 0 :: v_dual_mov_b32 v153 /*v409*/, 0
	v_dual_mov_b32 v154 /*v410*/, 0 :: v_dual_mov_b32 v155 /*v411*/, 0
	v_dual_mov_b32 v156 /*v412*/, 0 :: v_dual_mov_b32 v157 /*v413*/, 0
	v_dual_mov_b32 v158 /*v414*/, 0 :: v_dual_mov_b32 v159 /*v415*/, 0
	v_dual_mov_b32 v160 /*v416*/, 0 :: v_dual_mov_b32 v161 /*v417*/, 0
	v_dual_mov_b32 v162 /*v418*/, 0 :: v_dual_mov_b32 v163 /*v419*/, 0
	v_dual_mov_b32 v164 /*v420*/, 0 :: v_dual_mov_b32 v165 /*v421*/, 0
	v_dual_mov_b32 v166 /*v422*/, 0 :: v_dual_mov_b32 v167 /*v423*/, 0
	v_dual_mov_b32 v168 /*v424*/, 0 :: v_dual_mov_b32 v169 /*v425*/, 0
	v_dual_mov_b32 v170 /*v426*/, 0 :: v_dual_mov_b32 v171 /*v427*/, 0
	v_dual_mov_b32 v172 /*v428*/, 0 :: v_dual_mov_b32 v173 /*v429*/, 0
	v_dual_mov_b32 v174 /*v430*/, 0 :: v_dual_mov_b32 v175 /*v431*/, 0
	v_dual_mov_b32 v176 /*v432*/, 0 :: v_dual_mov_b32 v177 /*v433*/, 0
	v_dual_mov_b32 v178 /*v434*/, 0 :: v_dual_mov_b32 v179 /*v435*/, 0
	v_dual_mov_b32 v180 /*v436*/, 0 :: v_dual_mov_b32 v181 /*v437*/, 0
	v_dual_mov_b32 v182 /*v438*/, 0 :: v_dual_mov_b32 v183 /*v439*/, 0
	v_dual_mov_b32 v184 /*v440*/, 0 :: v_dual_mov_b32 v185 /*v441*/, 0
	v_dual_mov_b32 v186 /*v442*/, 0 :: v_dual_mov_b32 v187 /*v443*/, 0
	v_dual_mov_b32 v188 /*v444*/, 0 :: v_dual_mov_b32 v189 /*v445*/, 0
	v_dual_mov_b32 v190 /*v446*/, 0 :: v_dual_mov_b32 v191 /*v447*/, 0
	v_dual_mov_b32 v192 /*v448*/, 0 :: v_dual_mov_b32 v193 /*v449*/, 0
	v_dual_mov_b32 v194 /*v450*/, 0 :: v_dual_mov_b32 v195 /*v451*/, 0
	v_dual_mov_b32 v196 /*v452*/, 0 :: v_dual_mov_b32 v197 /*v453*/, 0
	v_dual_mov_b32 v198 /*v454*/, 0 :: v_dual_mov_b32 v199 /*v455*/, 0
	v_dual_mov_b32 v200 /*v456*/, 0 :: v_dual_mov_b32 v201 /*v457*/, 0
	v_dual_mov_b32 v202 /*v458*/, 0 :: v_dual_mov_b32 v203 /*v459*/, 0
	v_dual_mov_b32 v204 /*v460*/, 0 :: v_dual_mov_b32 v205 /*v461*/, 0
	v_dual_mov_b32 v206 /*v462*/, 0 :: v_dual_mov_b32 v207 /*v463*/, 0
	v_dual_mov_b32 v208 /*v464*/, 0 :: v_dual_mov_b32 v209 /*v465*/, 0
	v_dual_mov_b32 v210 /*v466*/, 0 :: v_dual_mov_b32 v211 /*v467*/, 0
	v_dual_mov_b32 v212 /*v468*/, 0 :: v_dual_mov_b32 v213 /*v469*/, 0
	v_dual_mov_b32 v214 /*v470*/, 0 :: v_dual_mov_b32 v215 /*v471*/, 0
	v_dual_mov_b32 v216 /*v472*/, 0 :: v_dual_mov_b32 v217 /*v473*/, 0
	v_dual_mov_b32 v218 /*v474*/, 0 :: v_dual_mov_b32 v219 /*v475*/, 0
	v_dual_mov_b32 v220 /*v476*/, 0 :: v_dual_mov_b32 v221 /*v477*/, 0
	v_dual_mov_b32 v222 /*v478*/, 0 :: v_dual_mov_b32 v223 /*v479*/, 0
	v_dual_mov_b32 v224 /*v480*/, 0 :: v_dual_mov_b32 v225 /*v481*/, 0
	v_dual_mov_b32 v226 /*v482*/, 0 :: v_dual_mov_b32 v227 /*v483*/, 0
	v_dual_mov_b32 v228 /*v484*/, 0 :: v_dual_mov_b32 v229 /*v485*/, 0
	v_dual_mov_b32 v230 /*v486*/, 0 :: v_dual_mov_b32 v231 /*v487*/, 0
	v_dual_mov_b32 v232 /*v488*/, 0 :: v_dual_mov_b32 v233 /*v489*/, 0
	v_dual_mov_b32 v234 /*v490*/, 0 :: v_dual_mov_b32 v235 /*v491*/, 0
	v_dual_mov_b32 v236 /*v492*/, 0 :: v_dual_mov_b32 v237 /*v493*/, 0
	v_dual_mov_b32 v238 /*v494*/, 0 :: v_dual_mov_b32 v239 /*v495*/, 0
	v_dual_mov_b32 v240 /*v496*/, 0 :: v_dual_mov_b32 v241 /*v497*/, 0
	v_dual_mov_b32 v242 /*v498*/, 0 :: v_dual_mov_b32 v243 /*v499*/, 0
	v_dual_mov_b32 v244 /*v500*/, 0 :: v_dual_mov_b32 v245 /*v501*/, 0
	v_dual_mov_b32 v246 /*v502*/, 0 :: v_dual_mov_b32 v247 /*v503*/, 0
	v_dual_mov_b32 v248 /*v504*/, 0 :: v_dual_mov_b32 v249 /*v505*/, 0
	v_dual_mov_b32 v250 /*v506*/, 0 :: v_dual_mov_b32 v251 /*v507*/, 0
	v_dual_mov_b32 v252 /*v508*/, 0 :: v_dual_mov_b32 v253 /*v509*/, 0
	v_dual_mov_b32 v254 /*v510*/, 0 :: v_dual_mov_b32 v255 /*v511*/, 0
	s_set_vgpr_msb 0x4080                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	v_dual_mov_b32 v232 /*v744*/, s22 :: v_dual_add_nc_u32 v233 /*v745*/, 0x19800, v99
	v_add_nc_u32_e32 v234 /*v746*/, 0x19800, v93
	s_add_co_i32 s38, s39, s39
	s_mov_b32 s39, s15
	s_set_vgpr_msb 0x8000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_wait_dscnt 0x0
.LBB2_6:                                ; %for.body
                                        ; =>This Inner Loop Header: Depth=1
	s_add_nc_u64 s[30:31], s[30:31], 0x100
	s_set_vgpr_msb 0x52                     ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[256:263]*/, v[0:7] /*v[512:519]*/, v[40:47], v[0:7] /*v[256:263]*/
	s_wait_dscnt 0x8
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[112:115] /*v[624:627]*/, v237 offset:17408
	ds_load_b128 v[116:119] /*v[628:631]*/, v238 offset:17408
	s_sub_co_i32 s24, s24, s34
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[264:271]*/, v[16:23] /*v[528:535]*/, v[40:47], v[8:15] /*v[264:271]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[120:123] /*v[632:635]*/, v235 offset:17408
	ds_load_b128 v[124:127] /*v[636:639]*/, v236 offset:17408
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[0:7] /*v[512:519]*/, v[56:63], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[64:67] /*v[576:579]*/, v233 offset:17408
	ds_load_b128 v[68:71] /*v[580:583]*/, v234 offset:17408
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[296:303]*/, v[16:23] /*v[528:535]*/, v[56:63], v[40:47] /*v[296:303]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[72:75] /*v[584:587]*/, v247 offset:17408
	ds_load_b128 v[76:79] /*v[588:591]*/, v232 offset:17408
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[256:263]*/, v[8:15] /*v[520:527]*/, v[32:39], v[0:7] /*v[256:263]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[80:83] /*v[592:595]*/, v245 offset:17408
	ds_load_b128 v[84:87] /*v[596:599]*/, v246 offset:17408
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[264:271]*/, v[24:31] /*v[536:543]*/, v[32:39], v[8:15] /*v[264:271]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[88:91] /*v[600:603]*/, v243 offset:17408
	ds_load_b128 v[92:95] /*v[604:607]*/, v244 offset:17408
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[8:15] /*v[520:527]*/, v[48:55], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[96:99] /*v[608:611]*/, v241 offset:17408
	ds_load_b128 v[100:103] /*v[612:615]*/, v242 offset:17408
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[296:303]*/, v[24:31] /*v[536:543]*/, v[48:55], v[40:47] /*v[296:303]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[104:107] /*v[616:619]*/, v239 offset:17408
	ds_load_b128 v[108:111] /*v[620:623]*/, v240 offset:17408
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[272:279]*/, v[32:39] /*v[544:551]*/, v[40:47], v[16:23] /*v[272:279]*/
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[48:55] /*v[560:567]*/, v[40:47], v[24:31] /*v[280:287]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[304:311]*/, v[32:39] /*v[544:551]*/, v[56:63], v[48:55] /*v[304:311]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[56:63] /*v[312:319]*/, v[48:55] /*v[560:567]*/, v[56:63], v[56:63] /*v[312:319]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[272:279]*/, v[40:47] /*v[552:559]*/, v[32:39], v[16:23] /*v[272:279]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[56:63] /*v[568:575]*/, v[32:39], v[24:31] /*v[280:287]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[304:311]*/, v[40:47] /*v[552:559]*/, v[48:55], v[48:55] /*v[304:311]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[56:63] /*v[312:319]*/, v[56:63] /*v[568:575]*/, v[48:55], v[56:63] /*v[312:319]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[320:327]*/, v[64:71] /*v[576:583]*/, v[40:47], v[64:71] /*v[320:327]*/
	s_wait_alu depctr_va_vdst(14)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[0:3] /*v[512:515]*/, v233 offset:34816
	ds_load_b128 v[4:7] /*v[516:519]*/, v234 offset:34816
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[328:335]*/, v[80:87] /*v[592:599]*/, v[40:47], v[72:79] /*v[328:335]*/
	s_wait_alu depctr_va_vdst(11)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[8:11] /*v[520:523]*/, v247 offset:34816
	ds_load_b128 v[12:15] /*v[524:527]*/, v232 offset:34816
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[352:359]*/, v[64:71] /*v[576:583]*/, v[56:63], v[96:103] /*v[352:359]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[16:19] /*v[528:531]*/, v245 offset:34816
	ds_load_b128 v[20:23] /*v[532:535]*/, v246 offset:34816
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[360:367]*/, v[80:87] /*v[592:599]*/, v[56:63], v[104:111] /*v[360:367]*/
	s_wait_alu depctr_va_vdst(12)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[24:27] /*v[536:539]*/, v243 offset:34816
	ds_load_b128 v[28:31] /*v[540:543]*/, v244 offset:34816
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[320:327]*/, v[72:79] /*v[584:591]*/, v[32:39], v[64:71] /*v[320:327]*/
	s_wait_alu depctr_va_vdst(10)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[32:35] /*v[544:547]*/, v241 offset:34816
	ds_load_b128 v[36:39] /*v[548:551]*/, v242 offset:34816
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[328:335]*/, v[88:95] /*v[600:607]*/, v[32:39], v[72:79] /*v[328:335]*/
	s_wait_alu depctr_va_vdst(7)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[40:43] /*v[552:555]*/, v239 offset:34816
	ds_load_b128 v[44:47] /*v[556:559]*/, v240 offset:34816
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[352:359]*/, v[72:79] /*v[584:591]*/, v[48:55], v[96:103] /*v[352:359]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[48:51] /*v[560:563]*/, v237 offset:34816
	ds_load_b128 v[52:55] /*v[564:567]*/, v238 offset:34816
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[360:367]*/, v[88:95] /*v[600:607]*/, v[48:55], v[104:111] /*v[360:367]*/
	s_wait_alu depctr_va_vdst(8)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[56:59] /*v[568:571]*/, v235 offset:34816
	ds_load_b128 v[60:63] /*v[572:575]*/, v236 offset:34816
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[336:343]*/, v[96:103] /*v[608:615]*/, v[40:47], v[80:87] /*v[336:343]*/
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[344:351]*/, v[112:119] /*v[624:631]*/, v[40:47], v[88:95] /*v[344:351]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[368:375]*/, v[96:103] /*v[608:615]*/, v[56:63], v[112:119] /*v[368:375]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[376:383]*/, v[112:119] /*v[624:631]*/, v[56:63], v[120:127] /*v[376:383]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[336:343]*/, v[104:111] /*v[616:623]*/, v[32:39], v[80:87] /*v[336:343]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[344:351]*/, v[120:127] /*v[632:639]*/, v[32:39], v[88:95] /*v[344:351]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[368:375]*/, v[104:111] /*v[616:623]*/, v[48:55], v[112:119] /*v[368:375]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[376:383]*/, v[120:127] /*v[632:639]*/, v[48:55], v[120:127] /*v[376:383]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[384:391]*/, v[0:7] /*v[512:519]*/, v[40:47], v[128:135] /*v[384:391]*/
	s_wait_alu depctr_va_vdst(14)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[64:67] /*v[576:579]*/, v233 offset:52224
	ds_load_b128 v[68:71] /*v[580:583]*/, v234 offset:52224
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[392:399]*/, v[16:23] /*v[528:535]*/, v[40:47], v[136:143] /*v[392:399]*/
	s_wait_alu depctr_va_vdst(11)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[72:75] /*v[584:587]*/, v247 offset:52224
	ds_load_b128 v[76:79] /*v[588:591]*/, v232 offset:52224
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[416:423]*/, v[0:7] /*v[512:519]*/, v[56:63], v[160:167] /*v[416:423]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[80:83] /*v[592:595]*/, v245 offset:52224
	ds_load_b128 v[84:87] /*v[596:599]*/, v246 offset:52224
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[424:431]*/, v[16:23] /*v[528:535]*/, v[56:63], v[168:175] /*v[424:431]*/
	s_wait_alu depctr_va_vdst(12)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[88:91] /*v[600:603]*/, v243 offset:52224
	ds_load_b128 v[92:95] /*v[604:607]*/, v244 offset:52224
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[384:391]*/, v[8:15] /*v[520:527]*/, v[32:39], v[128:135] /*v[384:391]*/
	s_wait_alu depctr_va_vdst(10)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[96:99] /*v[608:611]*/, v241 offset:52224
	ds_load_b128 v[100:103] /*v[612:615]*/, v242 offset:52224
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[392:399]*/, v[24:31] /*v[536:543]*/, v[32:39], v[136:143] /*v[392:399]*/
	s_wait_alu depctr_va_vdst(7)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[104:107] /*v[616:619]*/, v239 offset:52224
	ds_load_b128 v[108:111] /*v[620:623]*/, v240 offset:52224
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[416:423]*/, v[8:15] /*v[520:527]*/, v[48:55], v[160:167] /*v[416:423]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[112:115] /*v[624:627]*/, v237 offset:52224
	ds_load_b128 v[116:119] /*v[628:631]*/, v238 offset:52224
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[424:431]*/, v[24:31] /*v[536:543]*/, v[48:55], v[168:175] /*v[424:431]*/
	s_wait_alu depctr_va_vdst(8)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[120:123] /*v[632:635]*/, v235 offset:52224
	ds_load_b128 v[124:127] /*v[636:639]*/, v236 offset:52224
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[400:407]*/, v[32:39] /*v[544:551]*/, v[40:47], v[144:151] /*v[400:407]*/
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[408:415]*/, v[48:55] /*v[560:567]*/, v[40:47], v[152:159] /*v[408:415]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[176:183] /*v[432:439]*/, v[32:39] /*v[544:551]*/, v[56:63], v[176:183] /*v[432:439]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[184:191] /*v[440:447]*/, v[48:55] /*v[560:567]*/, v[56:63], v[184:191] /*v[440:447]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[400:407]*/, v[40:47] /*v[552:559]*/, v[32:39], v[144:151] /*v[400:407]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[408:415]*/, v[56:63] /*v[568:575]*/, v[32:39], v[152:159] /*v[408:415]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[176:183] /*v[432:439]*/, v[40:47] /*v[552:559]*/, v[48:55], v[176:183] /*v[432:439]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[184:191] /*v[440:447]*/, v[56:63] /*v[568:575]*/, v[48:55], v[184:191] /*v[440:447]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[64:71] /*v[576:583]*/, v[40:47], v[192:199] /*v[448:455]*/
	s_wait_alu depctr_va_vdst(14)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[0:3] /*v[512:515]*/, v233 offset:128
	ds_load_b128 v[4:7] /*v[516:519]*/, v234 offset:128
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[80:87] /*v[592:599]*/, v[40:47], v[200:207] /*v[456:463]*/
	s_wait_alu depctr_va_vdst(11)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[8:11] /*v[520:523]*/, v247 offset:128
	ds_load_b128 v[12:15] /*v[524:527]*/, v232 offset:128
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[64:71] /*v[576:583]*/, v[56:63], v[224:231] /*v[480:487]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[16:19] /*v[528:531]*/, v245 offset:128
	ds_load_b128 v[20:23] /*v[532:535]*/, v246 offset:128
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[80:87] /*v[592:599]*/, v[56:63], v[232:239] /*v[488:495]*/
	s_wait_alu depctr_va_vdst(12)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[24:27] /*v[536:539]*/, v243 offset:128
	ds_load_b128 v[28:31] /*v[540:543]*/, v244 offset:128
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[72:79] /*v[584:591]*/, v[32:39], v[192:199] /*v[448:455]*/
	s_wait_alu depctr_va_vdst(10)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[32:35] /*v[544:547]*/, v241 offset:128
	ds_load_b128 v[36:39] /*v[548:551]*/, v242 offset:128
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[88:95] /*v[600:607]*/, v[32:39], v[200:207] /*v[456:463]*/
	s_wait_alu depctr_va_vdst(7)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[40:43] /*v[552:555]*/, v239 offset:128
	ds_load_b128 v[44:47] /*v[556:559]*/, v240 offset:128
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[72:79] /*v[584:591]*/, v[48:55], v[224:231] /*v[480:487]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[48:51] /*v[560:563]*/, v237 offset:128
	ds_load_b128 v[52:55] /*v[564:567]*/, v238 offset:128
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[88:95] /*v[600:607]*/, v[48:55], v[232:239] /*v[488:495]*/
	s_wait_alu depctr_va_vdst(8)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[56:59] /*v[568:571]*/, v235 offset:128
	ds_load_b128 v[60:63] /*v[572:575]*/, v236 offset:128
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[96:103] /*v[608:615]*/, v[40:47], v[208:215] /*v[464:471]*/
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[112:119] /*v[624:631]*/, v[40:47], v[216:223] /*v[472:479]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[96:103] /*v[608:615]*/, v[56:63], v[240:247] /*v[496:503]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[112:119] /*v[624:631]*/, v[56:63], v[248:255] /*v[504:511]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[104:111] /*v[616:623]*/, v[32:39], v[208:215] /*v[464:471]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[120:127] /*v[632:639]*/, v[32:39], v[216:223] /*v[472:479]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[104:111] /*v[616:623]*/, v[48:55], v[240:247] /*v[496:503]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[120:127] /*v[632:639]*/, v[48:55], v[248:255] /*v[504:511]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[256:263]*/, v[0:7] /*v[512:519]*/, v[24:31], v[0:7] /*v[256:263]*/
	s_wait_alu depctr_va_vdst(14)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[64:67] /*v[576:579]*/, v233 offset:17536
	ds_load_b128 v[68:71] /*v[580:583]*/, v234 offset:17536
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[264:271]*/, v[16:23] /*v[528:535]*/, v[24:31], v[8:15] /*v[264:271]*/
	s_wait_alu depctr_va_vdst(11)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[72:75] /*v[584:587]*/, v247 offset:17536
	ds_load_b128 v[76:79] /*v[588:591]*/, v232 offset:17536
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[0:7] /*v[512:519]*/, v[16:23], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[80:83] /*v[592:595]*/, v245 offset:17536
	ds_load_b128 v[84:87] /*v[596:599]*/, v246 offset:17536
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[296:303]*/, v[16:23] /*v[528:535]*/, v[16:23], v[40:47] /*v[296:303]*/
	s_wait_alu depctr_va_vdst(12)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[88:91] /*v[600:603]*/, v243 offset:17536
	ds_load_b128 v[92:95] /*v[604:607]*/, v244 offset:17536
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[256:263]*/, v[8:15] /*v[520:527]*/, v[8:15], v[0:7] /*v[256:263]*/
	s_wait_alu depctr_va_vdst(10)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[96:99] /*v[608:611]*/, v241 offset:17536
	ds_load_b128 v[100:103] /*v[612:615]*/, v242 offset:17536
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[264:271]*/, v[24:31] /*v[536:543]*/, v[8:15], v[8:15] /*v[264:271]*/
	s_wait_alu depctr_va_vdst(7)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[104:107] /*v[616:619]*/, v239 offset:17536
	ds_load_b128 v[108:111] /*v[620:623]*/, v240 offset:17536
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[8:15] /*v[520:527]*/, v[0:7], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[112:115] /*v[624:627]*/, v237 offset:17536
	ds_load_b128 v[116:119] /*v[628:631]*/, v238 offset:17536
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[296:303]*/, v[24:31] /*v[536:543]*/, v[0:7], v[40:47] /*v[296:303]*/
	s_wait_alu depctr_va_vdst(8)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[120:123] /*v[632:635]*/, v235 offset:17536
	ds_load_b128 v[124:127] /*v[636:639]*/, v236 offset:17536
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[272:279]*/, v[32:39] /*v[544:551]*/, v[24:31], v[16:23] /*v[272:279]*/
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[48:55] /*v[560:567]*/, v[24:31], v[24:31] /*v[280:287]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[304:311]*/, v[32:39] /*v[544:551]*/, v[16:23], v[48:55] /*v[304:311]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[56:63] /*v[312:319]*/, v[48:55] /*v[560:567]*/, v[16:23], v[56:63] /*v[312:319]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[272:279]*/, v[40:47] /*v[552:559]*/, v[8:15], v[16:23] /*v[272:279]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[56:63] /*v[568:575]*/, v[8:15], v[24:31] /*v[280:287]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[304:311]*/, v[40:47] /*v[552:559]*/, v[0:7], v[48:55] /*v[304:311]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[56:63] /*v[312:319]*/, v[56:63] /*v[568:575]*/, v[0:7], v[56:63] /*v[312:319]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[320:327]*/, v[64:71] /*v[576:583]*/, v[24:31], v[64:71] /*v[320:327]*/
	s_wait_alu depctr_va_vdst(14)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[0:3] /*v[512:515]*/, v233 offset:34944
	ds_load_b128 v[4:7] /*v[516:519]*/, v234 offset:34944
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[328:335]*/, v[80:87] /*v[592:599]*/, v[24:31], v[72:79] /*v[328:335]*/
	s_wait_alu depctr_va_vdst(11)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[8:11] /*v[520:523]*/, v247 offset:34944
	ds_load_b128 v[12:15] /*v[524:527]*/, v232 offset:34944
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[352:359]*/, v[64:71] /*v[576:583]*/, v[16:23], v[96:103] /*v[352:359]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[16:19] /*v[528:531]*/, v245 offset:34944
	ds_load_b128 v[20:23] /*v[532:535]*/, v246 offset:34944
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[360:367]*/, v[80:87] /*v[592:599]*/, v[16:23], v[104:111] /*v[360:367]*/
	s_wait_alu depctr_va_vdst(12)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[24:27] /*v[536:539]*/, v243 offset:34944
	ds_load_b128 v[28:31] /*v[540:543]*/, v244 offset:34944
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[320:327]*/, v[72:79] /*v[584:591]*/, v[8:15], v[64:71] /*v[320:327]*/
	s_wait_alu depctr_va_vdst(10)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[32:35] /*v[544:547]*/, v241 offset:34944
	ds_load_b128 v[36:39] /*v[548:551]*/, v242 offset:34944
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[328:335]*/, v[88:95] /*v[600:607]*/, v[8:15], v[72:79] /*v[328:335]*/
	s_wait_alu depctr_va_vdst(7)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[40:43] /*v[552:555]*/, v239 offset:34944
	ds_load_b128 v[44:47] /*v[556:559]*/, v240 offset:34944
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[352:359]*/, v[72:79] /*v[584:591]*/, v[0:7], v[96:103] /*v[352:359]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[48:51] /*v[560:563]*/, v237 offset:34944
	ds_load_b128 v[52:55] /*v[564:567]*/, v238 offset:34944
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[360:367]*/, v[88:95] /*v[600:607]*/, v[0:7], v[104:111] /*v[360:367]*/
	s_wait_alu depctr_va_vdst(8)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[56:59] /*v[568:571]*/, v235 offset:34944
	ds_load_b128 v[60:63] /*v[572:575]*/, v236 offset:34944
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[336:343]*/, v[96:103] /*v[608:615]*/, v[24:31], v[80:87] /*v[336:343]*/
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[344:351]*/, v[112:119] /*v[624:631]*/, v[24:31], v[88:95] /*v[344:351]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[368:375]*/, v[96:103] /*v[608:615]*/, v[16:23], v[112:119] /*v[368:375]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[376:383]*/, v[112:119] /*v[624:631]*/, v[16:23], v[120:127] /*v[376:383]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[336:343]*/, v[104:111] /*v[616:623]*/, v[8:15], v[80:87] /*v[336:343]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[344:351]*/, v[120:127] /*v[632:639]*/, v[8:15], v[88:95] /*v[344:351]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[368:375]*/, v[104:111] /*v[616:623]*/, v[0:7], v[112:119] /*v[368:375]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[376:383]*/, v[120:127] /*v[632:639]*/, v[0:7], v[120:127] /*v[376:383]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[384:391]*/, v[0:7] /*v[512:519]*/, v[24:31], v[128:135] /*v[384:391]*/
	s_wait_alu depctr_va_vdst(14)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[64:67] /*v[576:579]*/, v233 offset:52352
	ds_load_b128 v[68:71] /*v[580:583]*/, v234 offset:52352
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[392:399]*/, v[16:23] /*v[528:535]*/, v[24:31], v[136:143] /*v[392:399]*/
	s_wait_alu depctr_va_vdst(11)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[72:75] /*v[584:587]*/, v247 offset:52352
	ds_load_b128 v[76:79] /*v[588:591]*/, v232 offset:52352
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[416:423]*/, v[0:7] /*v[512:519]*/, v[16:23], v[160:167] /*v[416:423]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[80:83] /*v[592:595]*/, v245 offset:52352
	ds_load_b128 v[84:87] /*v[596:599]*/, v246 offset:52352
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[424:431]*/, v[16:23] /*v[528:535]*/, v[16:23], v[168:175] /*v[424:431]*/
	s_wait_alu depctr_va_vdst(12)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[88:91] /*v[600:603]*/, v243 offset:52352
	ds_load_b128 v[92:95] /*v[604:607]*/, v244 offset:52352
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[384:391]*/, v[8:15] /*v[520:527]*/, v[8:15], v[128:135] /*v[384:391]*/
	s_wait_alu depctr_va_vdst(10)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[96:99] /*v[608:611]*/, v241 offset:52352
	ds_load_b128 v[100:103] /*v[612:615]*/, v242 offset:52352
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[392:399]*/, v[24:31] /*v[536:543]*/, v[8:15], v[136:143] /*v[392:399]*/
	s_wait_alu depctr_va_vdst(7)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[104:107] /*v[616:619]*/, v239 offset:52352
	ds_load_b128 v[108:111] /*v[620:623]*/, v240 offset:52352
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[416:423]*/, v[8:15] /*v[520:527]*/, v[0:7], v[160:167] /*v[416:423]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[112:115] /*v[624:627]*/, v237 offset:52352
	ds_load_b128 v[116:119] /*v[628:631]*/, v238 offset:52352
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[424:431]*/, v[24:31] /*v[536:543]*/, v[0:7], v[168:175] /*v[424:431]*/
	s_wait_alu depctr_va_vdst(8)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[120:123] /*v[632:635]*/, v235 offset:52352
	ds_load_b128 v[124:127] /*v[636:639]*/, v236 offset:52352
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[400:407]*/, v[32:39] /*v[544:551]*/, v[24:31], v[144:151] /*v[400:407]*/
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[408:415]*/, v[48:55] /*v[560:567]*/, v[24:31], v[152:159] /*v[408:415]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[176:183] /*v[432:439]*/, v[32:39] /*v[544:551]*/, v[16:23], v[176:183] /*v[432:439]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[184:191] /*v[440:447]*/, v[48:55] /*v[560:567]*/, v[16:23], v[184:191] /*v[440:447]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[400:407]*/, v[40:47] /*v[552:559]*/, v[8:15], v[144:151] /*v[400:407]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[408:415]*/, v[56:63] /*v[568:575]*/, v[8:15], v[152:159] /*v[408:415]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[176:183] /*v[432:439]*/, v[40:47] /*v[552:559]*/, v[0:7], v[176:183] /*v[432:439]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[184:191] /*v[440:447]*/, v[56:63] /*v[568:575]*/, v[0:7], v[184:191] /*v[440:447]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[64:71] /*v[576:583]*/, v[24:31], v[192:199] /*v[448:455]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[80:87] /*v[592:599]*/, v[24:31], v[200:207] /*v[456:463]*/
	s_wait_dscnt 0x10
	s_wait_tensorcnt 0x1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[64:71] /*v[576:583]*/, v[16:23], v[224:231] /*v[480:487]*/
	s_barrier_signal -1
	s_barrier_wait -1
	s_wait_alu depctr_va_vdst(12)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[8:11] /*v[520:523]*/, v222
	ds_load_b128 v[12:15] /*v[524:527]*/, v225
	ds_load_b128 v[16:19] /*v[528:531]*/, v226
	ds_load_b128 v[20:23] /*v[532:535]*/, v227
	s_wait_alu depctr_va_vdst(11)
	ds_load_b128 v[24:27] /*v[536:539]*/, v228
	ds_load_b128 v[28:31] /*v[540:543]*/, v229
	s_wait_alu depctr_va_vdst(8)
	ds_load_b128 v[32:35] /*v[544:547]*/, v223
	ds_load_b128 v[36:39] /*v[548:551]*/, v224
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[80:87] /*v[592:599]*/, v[16:23], v[232:239] /*v[488:495]*/
	s_set_vgpr_msb 0x5200                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	ds_load_b128 v[48:51], v215 offset:34816
	ds_load_b128 v[52:55], v212 offset:52320
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(8) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x52                     ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[72:79] /*v[584:591]*/, v[8:15], v[192:199] /*v[448:455]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[0:3] /*v[512:515]*/, v230
	ds_load_b128 v[4:7] /*v[516:519]*/, v231
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x14
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[88:95] /*v[600:607]*/, v[8:15], v[200:207] /*v[456:463]*/
	s_set_vgpr_msb 0x5200                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	ds_load_b128 v[40:43], v212 offset:34816
	ds_load_b128 v[44:47], v212 offset:34848
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x52                     ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[72:79] /*v[584:591]*/, v[0:7], v[224:231] /*v[480:487]*/
	s_set_vgpr_msb 0x5200                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	ds_load_b128 v[32:35], v213 offset:34816
	ds_load_b128 v[36:39], v212 offset:34912
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x52                     ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[88:95] /*v[600:607]*/, v[0:7], v[232:239] /*v[488:495]*/
	s_set_vgpr_msb 0x5200                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	ds_load_b128 v[56:59], v214 offset:34816
	ds_load_b128 v[60:63], v212 offset:52256
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x52                     ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x18
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[96:103] /*v[608:615]*/, v[24:31], v[208:215] /*v[464:471]*/
	s_wait_alu depctr_va_vdst(10)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[40:43] /*v[552:555]*/, v220
	ds_load_b128 v[44:47] /*v[556:559]*/, v221
	s_add_nc_u64 s[2:3], s[30:31], s[28:29]
	s_add_co_i32 s9, s37, 0x80
	s_and_b32 s3, s3, 0x1ffffff
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_readfirstlane_b32 s40, v232 /*v744*/
	;;#ASMSTART
	s_sub_co_u32 s9, s40, s9
	;;#ASMEND
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[112:119] /*v[624:631]*/, v[24:31], v[216:223] /*v[472:479]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[48:51] /*v[560:563]*/, v218
	ds_load_b128 v[52:55] /*v[564:567]*/, v219
	;;#ASMSTART
	s_max_i32 s16, s9, 0
	;;#ASMEND
	s_lshr_b64 s[42:43], s[16:17], 16
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_lshl1_add_u32 s1, s24, s19
	s_bitset1_b32 s3, 31
	s_lshl_b32 s9, s16, 16
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[96:103] /*v[608:615]*/, v[16:23], v[240:247] /*v[496:503]*/
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[56:59] /*v[568:571]*/, v216
	ds_load_b128 v[60:63] /*v[572:575]*/, v217
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_mov_b32 s10, s42
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[112:119] /*v[624:631]*/, v[16:23], v[248:255] /*v[504:511]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[104:111] /*v[616:623]*/, v[8:15], v[208:215] /*v[464:471]*/
	; sched_barrier mask(0x00000000)
	tensor_load_to_lds s[0:3], s[8:15] scope:SCOPE_DEV
	s_wait_dscnt 0x18
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[120:127] /*v[632:639]*/, v[8:15], v[216:223] /*v[472:479]*/
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[104:111] /*v[616:623]*/, v[0:7], v[240:247] /*v[496:503]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[120:127] /*v[632:639]*/, v[0:7], v[248:255] /*v[504:511]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_add_nc_u64 s[30:31], s[30:31], 0x100
	s_set_vgpr_msb 0x5200                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	ds_load_b128 v[24:27], v212 offset:34944
	ds_load_b128 v[28:31], v212 offset:34976
	s_wait_alu depctr_va_vdst(2)
	ds_load_b128 v[8:11], v213 offset:34944
	ds_load_b128 v[12:15], v212 offset:35040
	ds_load_b128 v[16:19], v214 offset:34944
	ds_load_b128 v[20:23], v212 offset:52384
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[0:3], v215 offset:34944
	ds_load_b128 v[4:7], v212 offset:52448
	s_set_vgpr_msb 0x52                     ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[256:263]*/, v[0:7] /*v[512:519]*/, v[40:47], v[0:7] /*v[256:263]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[64:67] /*v[576:579]*/, v210
	ds_load_b128 v[68:71] /*v[580:583]*/, v211
	; sched_group_barrier mask(0x00000004) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(8) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[264:271]*/, v[16:23] /*v[528:535]*/, v[40:47], v[8:15] /*v[264:271]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[72:75] /*v[584:587]*/, v208
	ds_load_b128 v[76:79] /*v[588:591]*/, v209
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[0:7] /*v[512:519]*/, v[56:63], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[80:83] /*v[592:595]*/, v206
	ds_load_b128 v[84:87] /*v[596:599]*/, v207
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[296:303]*/, v[16:23] /*v[528:535]*/, v[56:63], v[40:47] /*v[296:303]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[88:91] /*v[600:603]*/, v204
	ds_load_b128 v[92:95] /*v[604:607]*/, v205
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[256:263]*/, v[8:15] /*v[520:527]*/, v[32:39], v[0:7] /*v[256:263]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[96:99] /*v[608:611]*/, v202
	ds_load_b128 v[100:103] /*v[612:615]*/, v203
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[264:271]*/, v[24:31] /*v[536:543]*/, v[32:39], v[8:15] /*v[264:271]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[104:107] /*v[616:619]*/, v200
	ds_load_b128 v[108:111] /*v[620:623]*/, v201
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[8:15] /*v[520:527]*/, v[48:55], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[112:115] /*v[624:627]*/, v198
	ds_load_b128 v[116:119] /*v[628:631]*/, v199
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[296:303]*/, v[24:31] /*v[536:543]*/, v[48:55], v[40:47] /*v[296:303]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[120:123] /*v[632:635]*/, v196
	ds_load_b128 v[124:127] /*v[636:639]*/, v197
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[272:279]*/, v[32:39] /*v[544:551]*/, v[40:47], v[16:23] /*v[272:279]*/
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[48:55] /*v[560:567]*/, v[40:47], v[24:31] /*v[280:287]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[304:311]*/, v[32:39] /*v[544:551]*/, v[56:63], v[48:55] /*v[304:311]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[56:63] /*v[312:319]*/, v[48:55] /*v[560:567]*/, v[56:63], v[56:63] /*v[312:319]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[272:279]*/, v[40:47] /*v[552:559]*/, v[32:39], v[16:23] /*v[272:279]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[56:63] /*v[568:575]*/, v[32:39], v[24:31] /*v[280:287]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[304:311]*/, v[40:47] /*v[552:559]*/, v[48:55], v[48:55] /*v[304:311]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[56:63] /*v[312:319]*/, v[56:63] /*v[568:575]*/, v[48:55], v[56:63] /*v[312:319]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[320:327]*/, v[64:71] /*v[576:583]*/, v[40:47], v[64:71] /*v[320:327]*/
	s_wait_alu depctr_va_vdst(14)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[0:3] /*v[512:515]*/, v194
	ds_load_b128 v[4:7] /*v[516:519]*/, v195
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[328:335]*/, v[80:87] /*v[592:599]*/, v[40:47], v[72:79] /*v[328:335]*/
	s_wait_alu depctr_va_vdst(11)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[8:11] /*v[520:523]*/, v192
	ds_load_b128 v[12:15] /*v[524:527]*/, v193
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[352:359]*/, v[64:71] /*v[576:583]*/, v[56:63], v[96:103] /*v[352:359]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[16:19] /*v[528:531]*/, v190
	ds_load_b128 v[20:23] /*v[532:535]*/, v191
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[360:367]*/, v[80:87] /*v[592:599]*/, v[56:63], v[104:111] /*v[360:367]*/
	s_wait_alu depctr_va_vdst(12)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[24:27] /*v[536:539]*/, v188
	ds_load_b128 v[28:31] /*v[540:543]*/, v189
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[320:327]*/, v[72:79] /*v[584:591]*/, v[32:39], v[64:71] /*v[320:327]*/
	s_wait_alu depctr_va_vdst(10)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[32:35] /*v[544:547]*/, v186
	ds_load_b128 v[36:39] /*v[548:551]*/, v187
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[328:335]*/, v[88:95] /*v[600:607]*/, v[32:39], v[72:79] /*v[328:335]*/
	s_wait_alu depctr_va_vdst(7)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[40:43] /*v[552:555]*/, v184
	ds_load_b128 v[44:47] /*v[556:559]*/, v185
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[352:359]*/, v[72:79] /*v[584:591]*/, v[48:55], v[96:103] /*v[352:359]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[48:51] /*v[560:563]*/, v182
	ds_load_b128 v[52:55] /*v[564:567]*/, v183
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[360:367]*/, v[88:95] /*v[600:607]*/, v[48:55], v[104:111] /*v[360:367]*/
	s_wait_alu depctr_va_vdst(8)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[56:59] /*v[568:571]*/, v180
	ds_load_b128 v[60:63] /*v[572:575]*/, v181
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[336:343]*/, v[96:103] /*v[608:615]*/, v[40:47], v[80:87] /*v[336:343]*/
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[344:351]*/, v[112:119] /*v[624:631]*/, v[40:47], v[88:95] /*v[344:351]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[368:375]*/, v[96:103] /*v[608:615]*/, v[56:63], v[112:119] /*v[368:375]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[376:383]*/, v[112:119] /*v[624:631]*/, v[56:63], v[120:127] /*v[376:383]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[336:343]*/, v[104:111] /*v[616:623]*/, v[32:39], v[80:87] /*v[336:343]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[344:351]*/, v[120:127] /*v[632:639]*/, v[32:39], v[88:95] /*v[344:351]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[368:375]*/, v[104:111] /*v[616:623]*/, v[48:55], v[112:119] /*v[368:375]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[376:383]*/, v[120:127] /*v[632:639]*/, v[48:55], v[120:127] /*v[376:383]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[384:391]*/, v[0:7] /*v[512:519]*/, v[40:47], v[128:135] /*v[384:391]*/
	s_wait_alu depctr_va_vdst(14)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[64:67] /*v[576:579]*/, v178
	ds_load_b128 v[68:71] /*v[580:583]*/, v179
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[392:399]*/, v[16:23] /*v[528:535]*/, v[40:47], v[136:143] /*v[392:399]*/
	s_wait_alu depctr_va_vdst(11)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[72:75] /*v[584:587]*/, v176
	ds_load_b128 v[76:79] /*v[588:591]*/, v177
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[416:423]*/, v[0:7] /*v[512:519]*/, v[56:63], v[160:167] /*v[416:423]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[80:83] /*v[592:595]*/, v174
	ds_load_b128 v[84:87] /*v[596:599]*/, v175
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[424:431]*/, v[16:23] /*v[528:535]*/, v[56:63], v[168:175] /*v[424:431]*/
	s_wait_alu depctr_va_vdst(12)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[88:91] /*v[600:603]*/, v172
	ds_load_b128 v[92:95] /*v[604:607]*/, v173
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[384:391]*/, v[8:15] /*v[520:527]*/, v[32:39], v[128:135] /*v[384:391]*/
	s_wait_alu depctr_va_vdst(10)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[96:99] /*v[608:611]*/, v170
	ds_load_b128 v[100:103] /*v[612:615]*/, v171
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[392:399]*/, v[24:31] /*v[536:543]*/, v[32:39], v[136:143] /*v[392:399]*/
	s_wait_alu depctr_va_vdst(7)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[104:107] /*v[616:619]*/, v168
	ds_load_b128 v[108:111] /*v[620:623]*/, v169
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[416:423]*/, v[8:15] /*v[520:527]*/, v[48:55], v[160:167] /*v[416:423]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[112:115] /*v[624:627]*/, v166
	ds_load_b128 v[116:119] /*v[628:631]*/, v167
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[424:431]*/, v[24:31] /*v[536:543]*/, v[48:55], v[168:175] /*v[424:431]*/
	s_wait_alu depctr_va_vdst(8)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[120:123] /*v[632:635]*/, v164
	ds_load_b128 v[124:127] /*v[636:639]*/, v165
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[400:407]*/, v[32:39] /*v[544:551]*/, v[40:47], v[144:151] /*v[400:407]*/
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[408:415]*/, v[48:55] /*v[560:567]*/, v[40:47], v[152:159] /*v[408:415]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[176:183] /*v[432:439]*/, v[32:39] /*v[544:551]*/, v[56:63], v[176:183] /*v[432:439]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[184:191] /*v[440:447]*/, v[48:55] /*v[560:567]*/, v[56:63], v[184:191] /*v[440:447]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[400:407]*/, v[40:47] /*v[552:559]*/, v[32:39], v[144:151] /*v[400:407]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[408:415]*/, v[56:63] /*v[568:575]*/, v[32:39], v[152:159] /*v[408:415]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[176:183] /*v[432:439]*/, v[40:47] /*v[552:559]*/, v[48:55], v[176:183] /*v[432:439]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[184:191] /*v[440:447]*/, v[56:63] /*v[568:575]*/, v[48:55], v[184:191] /*v[440:447]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[64:71] /*v[576:583]*/, v[40:47], v[192:199] /*v[448:455]*/
	s_wait_alu depctr_va_vdst(14)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[0:3] /*v[512:515]*/, v162
	ds_load_b128 v[4:7] /*v[516:519]*/, v163
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[80:87] /*v[592:599]*/, v[40:47], v[200:207] /*v[456:463]*/
	s_wait_alu depctr_va_vdst(11)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[8:11] /*v[520:523]*/, v160
	ds_load_b128 v[12:15] /*v[524:527]*/, v161
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[64:71] /*v[576:583]*/, v[56:63], v[224:231] /*v[480:487]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[16:19] /*v[528:531]*/, v158
	ds_load_b128 v[20:23] /*v[532:535]*/, v159
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[80:87] /*v[592:599]*/, v[56:63], v[232:239] /*v[488:495]*/
	s_wait_alu depctr_va_vdst(12)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[24:27] /*v[536:539]*/, v156
	ds_load_b128 v[28:31] /*v[540:543]*/, v157
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[72:79] /*v[584:591]*/, v[32:39], v[192:199] /*v[448:455]*/
	s_wait_alu depctr_va_vdst(10)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[32:35] /*v[544:547]*/, v154
	ds_load_b128 v[36:39] /*v[548:551]*/, v155
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[88:95] /*v[600:607]*/, v[32:39], v[200:207] /*v[456:463]*/
	s_wait_alu depctr_va_vdst(7)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[40:43] /*v[552:555]*/, v152
	ds_load_b128 v[44:47] /*v[556:559]*/, v153
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[72:79] /*v[584:591]*/, v[48:55], v[224:231] /*v[480:487]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[48:51] /*v[560:563]*/, v150
	ds_load_b128 v[52:55] /*v[564:567]*/, v151
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[88:95] /*v[600:607]*/, v[48:55], v[232:239] /*v[488:495]*/
	s_wait_alu depctr_va_vdst(8)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[56:59] /*v[568:571]*/, v148
	ds_load_b128 v[60:63] /*v[572:575]*/, v149
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[96:103] /*v[608:615]*/, v[40:47], v[208:215] /*v[464:471]*/
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[112:119] /*v[624:631]*/, v[40:47], v[216:223] /*v[472:479]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[96:103] /*v[608:615]*/, v[56:63], v[240:247] /*v[496:503]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[112:119] /*v[624:631]*/, v[56:63], v[248:255] /*v[504:511]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[104:111] /*v[616:623]*/, v[32:39], v[208:215] /*v[464:471]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[120:127] /*v[632:639]*/, v[32:39], v[216:223] /*v[472:479]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[104:111] /*v[616:623]*/, v[48:55], v[240:247] /*v[496:503]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[120:127] /*v[632:639]*/, v[48:55], v[248:255] /*v[504:511]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[256:263]*/, v[0:7] /*v[512:519]*/, v[24:31], v[0:7] /*v[256:263]*/
	s_wait_alu depctr_va_vdst(14)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[64:67] /*v[576:579]*/, v146
	ds_load_b128 v[68:71] /*v[580:583]*/, v147
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[264:271]*/, v[16:23] /*v[528:535]*/, v[24:31], v[8:15] /*v[264:271]*/
	s_wait_alu depctr_va_vdst(11)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[72:75] /*v[584:587]*/, v144
	ds_load_b128 v[76:79] /*v[588:591]*/, v145
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[0:7] /*v[512:519]*/, v[16:23], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[80:83] /*v[592:595]*/, v142
	ds_load_b128 v[84:87] /*v[596:599]*/, v143
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[296:303]*/, v[16:23] /*v[528:535]*/, v[16:23], v[40:47] /*v[296:303]*/
	s_wait_alu depctr_va_vdst(12)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[88:91] /*v[600:603]*/, v140
	ds_load_b128 v[92:95] /*v[604:607]*/, v141
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[256:263]*/, v[8:15] /*v[520:527]*/, v[8:15], v[0:7] /*v[256:263]*/
	s_wait_alu depctr_va_vdst(10)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[96:99] /*v[608:611]*/, v138
	ds_load_b128 v[100:103] /*v[612:615]*/, v139
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[264:271]*/, v[24:31] /*v[536:543]*/, v[8:15], v[8:15] /*v[264:271]*/
	s_wait_alu depctr_va_vdst(7)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[104:107] /*v[616:619]*/, v136
	ds_load_b128 v[108:111] /*v[620:623]*/, v137
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[8:15] /*v[520:527]*/, v[0:7], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[112:115] /*v[624:627]*/, v134
	ds_load_b128 v[116:119] /*v[628:631]*/, v135
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[296:303]*/, v[24:31] /*v[536:543]*/, v[0:7], v[40:47] /*v[296:303]*/
	s_wait_alu depctr_va_vdst(8)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[120:123] /*v[632:635]*/, v132
	ds_load_b128 v[124:127] /*v[636:639]*/, v133
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[272:279]*/, v[32:39] /*v[544:551]*/, v[24:31], v[16:23] /*v[272:279]*/
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[48:55] /*v[560:567]*/, v[24:31], v[24:31] /*v[280:287]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[304:311]*/, v[32:39] /*v[544:551]*/, v[16:23], v[48:55] /*v[304:311]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[56:63] /*v[312:319]*/, v[48:55] /*v[560:567]*/, v[16:23], v[56:63] /*v[312:319]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[272:279]*/, v[40:47] /*v[552:559]*/, v[8:15], v[16:23] /*v[272:279]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[56:63] /*v[568:575]*/, v[8:15], v[24:31] /*v[280:287]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[304:311]*/, v[40:47] /*v[552:559]*/, v[0:7], v[48:55] /*v[304:311]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[56:63] /*v[312:319]*/, v[56:63] /*v[568:575]*/, v[0:7], v[56:63] /*v[312:319]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[320:327]*/, v[64:71] /*v[576:583]*/, v[24:31], v[64:71] /*v[320:327]*/
	s_wait_alu depctr_va_vdst(14)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[0:3] /*v[512:515]*/, v130
	ds_load_b128 v[4:7] /*v[516:519]*/, v131
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[328:335]*/, v[80:87] /*v[592:599]*/, v[24:31], v[72:79] /*v[328:335]*/
	s_wait_alu depctr_va_vdst(11)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[8:11] /*v[520:523]*/, v128
	ds_load_b128 v[12:15] /*v[524:527]*/, v129
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[352:359]*/, v[64:71] /*v[576:583]*/, v[16:23], v[96:103] /*v[352:359]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[16:19] /*v[528:531]*/, v126
	ds_load_b128 v[20:23] /*v[532:535]*/, v127
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[360:367]*/, v[80:87] /*v[592:599]*/, v[16:23], v[104:111] /*v[360:367]*/
	s_wait_alu depctr_va_vdst(12)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[24:27] /*v[536:539]*/, v124
	ds_load_b128 v[28:31] /*v[540:543]*/, v125
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[320:327]*/, v[72:79] /*v[584:591]*/, v[8:15], v[64:71] /*v[320:327]*/
	s_wait_alu depctr_va_vdst(10)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[32:35] /*v[544:547]*/, v122
	ds_load_b128 v[36:39] /*v[548:551]*/, v123
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[328:335]*/, v[88:95] /*v[600:607]*/, v[8:15], v[72:79] /*v[328:335]*/
	s_wait_alu depctr_va_vdst(7)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[40:43] /*v[552:555]*/, v120
	ds_load_b128 v[44:47] /*v[556:559]*/, v121
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[352:359]*/, v[72:79] /*v[584:591]*/, v[0:7], v[96:103] /*v[352:359]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[48:51] /*v[560:563]*/, v118
	ds_load_b128 v[52:55] /*v[564:567]*/, v119
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[360:367]*/, v[88:95] /*v[600:607]*/, v[0:7], v[104:111] /*v[360:367]*/
	s_wait_alu depctr_va_vdst(8)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[56:59] /*v[568:571]*/, v116
	ds_load_b128 v[60:63] /*v[572:575]*/, v117
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[336:343]*/, v[96:103] /*v[608:615]*/, v[24:31], v[80:87] /*v[336:343]*/
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[344:351]*/, v[112:119] /*v[624:631]*/, v[24:31], v[88:95] /*v[344:351]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[368:375]*/, v[96:103] /*v[608:615]*/, v[16:23], v[112:119] /*v[368:375]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[376:383]*/, v[112:119] /*v[624:631]*/, v[16:23], v[120:127] /*v[376:383]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[336:343]*/, v[104:111] /*v[616:623]*/, v[8:15], v[80:87] /*v[336:343]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[344:351]*/, v[120:127] /*v[632:639]*/, v[8:15], v[88:95] /*v[344:351]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[368:375]*/, v[104:111] /*v[616:623]*/, v[0:7], v[112:119] /*v[368:375]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[376:383]*/, v[120:127] /*v[632:639]*/, v[0:7], v[120:127] /*v[376:383]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[384:391]*/, v[0:7] /*v[512:519]*/, v[24:31], v[128:135] /*v[384:391]*/
	s_wait_alu depctr_va_vdst(14)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[64:67] /*v[576:579]*/, v114
	ds_load_b128 v[68:71] /*v[580:583]*/, v115
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[392:399]*/, v[16:23] /*v[528:535]*/, v[24:31], v[136:143] /*v[392:399]*/
	s_wait_alu depctr_va_vdst(11)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[72:75] /*v[584:587]*/, v112
	ds_load_b128 v[76:79] /*v[588:591]*/, v113
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[416:423]*/, v[0:7] /*v[512:519]*/, v[16:23], v[160:167] /*v[416:423]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[80:83] /*v[592:595]*/, v110
	ds_load_b128 v[84:87] /*v[596:599]*/, v111
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[424:431]*/, v[16:23] /*v[528:535]*/, v[16:23], v[168:175] /*v[424:431]*/
	s_wait_alu depctr_va_vdst(12)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[88:91] /*v[600:603]*/, v108
	ds_load_b128 v[92:95] /*v[604:607]*/, v109
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[384:391]*/, v[8:15] /*v[520:527]*/, v[8:15], v[128:135] /*v[384:391]*/
	s_wait_alu depctr_va_vdst(10)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[96:99] /*v[608:611]*/, v106
	ds_load_b128 v[100:103] /*v[612:615]*/, v107
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[392:399]*/, v[24:31] /*v[536:543]*/, v[8:15], v[136:143] /*v[392:399]*/
	s_wait_alu depctr_va_vdst(7)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[104:107] /*v[616:619]*/, v104
	ds_load_b128 v[108:111] /*v[620:623]*/, v105
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[416:423]*/, v[8:15] /*v[520:527]*/, v[0:7], v[160:167] /*v[416:423]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[112:115] /*v[624:627]*/, v102
	ds_load_b128 v[116:119] /*v[628:631]*/, v103
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[424:431]*/, v[24:31] /*v[536:543]*/, v[0:7], v[168:175] /*v[424:431]*/
	s_wait_alu depctr_va_vdst(8)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[120:123] /*v[632:635]*/, v100
	ds_load_b128 v[124:127] /*v[636:639]*/, v101
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[400:407]*/, v[32:39] /*v[544:551]*/, v[24:31], v[144:151] /*v[400:407]*/
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[408:415]*/, v[48:55] /*v[560:567]*/, v[24:31], v[152:159] /*v[408:415]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[176:183] /*v[432:439]*/, v[32:39] /*v[544:551]*/, v[16:23], v[176:183] /*v[432:439]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[184:191] /*v[440:447]*/, v[48:55] /*v[560:567]*/, v[16:23], v[184:191] /*v[440:447]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[400:407]*/, v[40:47] /*v[552:559]*/, v[8:15], v[144:151] /*v[400:407]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[408:415]*/, v[56:63] /*v[568:575]*/, v[8:15], v[152:159] /*v[408:415]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[176:183] /*v[432:439]*/, v[40:47] /*v[552:559]*/, v[0:7], v[176:183] /*v[432:439]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[184:191] /*v[440:447]*/, v[56:63] /*v[568:575]*/, v[0:7], v[184:191] /*v[440:447]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[64:71] /*v[576:583]*/, v[24:31], v[192:199] /*v[448:455]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[80:87] /*v[592:599]*/, v[24:31], v[200:207] /*v[456:463]*/
	s_wait_dscnt 0x10
	s_wait_tensorcnt 0x1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[64:71] /*v[576:583]*/, v[16:23], v[224:231] /*v[480:487]*/
	s_barrier_signal -1
	s_barrier_wait -1
	s_wait_alu depctr_va_vdst(12)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[8:11] /*v[520:523]*/, v86
	s_set_vgpr_msb 0x8082                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[12:15] /*v[524:527]*/, v234 /*v746*/
	s_set_vgpr_msb 0x8280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[16:19] /*v[528:531]*/, v87
	ds_load_b128 v[20:23] /*v[532:535]*/, v88
	s_wait_alu depctr_va_vdst(11)
	ds_load_b128 v[24:27] /*v[536:539]*/, v89
	ds_load_b128 v[28:31] /*v[540:543]*/, v90
	s_wait_alu depctr_va_vdst(8)
	ds_load_b128 v[32:35] /*v[544:547]*/, v84
	ds_load_b128 v[36:39] /*v[548:551]*/, v85
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[80:87] /*v[592:599]*/, v[16:23], v[232:239] /*v[488:495]*/
	s_set_vgpr_msb 0x5200                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	ds_load_b128 v[48:51], v82
	ds_load_b128 v[52:55], v83
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(8) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x52                     ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[72:79] /*v[584:591]*/, v[8:15], v[192:199] /*v[448:455]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[0:3] /*v[512:515]*/, v98
	s_set_vgpr_msb 0x8082                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[4:7] /*v[516:519]*/, v233 /*v745*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x14
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[88:95] /*v[600:607]*/, v[8:15], v[200:207] /*v[456:463]*/
	s_set_vgpr_msb 0x5200                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	ds_load_b128 v[40:43], v96
	ds_load_b128 v[44:47], v97
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x52                     ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[72:79] /*v[584:591]*/, v[0:7], v[224:231] /*v[480:487]*/
	s_set_vgpr_msb 0x5200                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	ds_load_b128 v[32:35], v94
	ds_load_b128 v[36:39], v95
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x52                     ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[88:95] /*v[600:607]*/, v[0:7], v[232:239] /*v[488:495]*/
	s_set_vgpr_msb 0x5200                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	ds_load_b128 v[56:59], v91
	ds_load_b128 v[60:63], v92
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x52                     ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x18
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[96:103] /*v[608:615]*/, v[24:31], v[208:215] /*v[464:471]*/
	s_wait_alu depctr_va_vdst(10)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[40:43] /*v[552:555]*/, v80
	ds_load_b128 v[44:47] /*v[556:559]*/, v81
	s_add_nc_u64 s[2:3], s[30:31], s[28:29]
	s_add_co_i32 s9, s37, 0x100
	s_and_b32 s3, s3, 0x1ffffff
	;;#ASMSTART
	s_sub_co_u32 s9, s40, s9
	;;#ASMEND
	;;#ASMSTART
	s_max_i32 s16, s9, 0
	;;#ASMEND
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[112:119] /*v[624:631]*/, v[24:31], v[216:223] /*v[472:479]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[48:51] /*v[560:563]*/, v78
	ds_load_b128 v[52:55] /*v[564:567]*/, v79
	s_lshr_b64 s[42:43], s[16:17], 16
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_add_co_i32 s1, s1, s34
	s_bitset1_b32 s3, 31
	s_lshl_b32 s9, s16, 16
	s_mov_b32 s10, s42
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[96:103] /*v[608:615]*/, v[16:23], v[240:247] /*v[496:503]*/
	s_wait_alu depctr_va_vdst(11)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[56:59] /*v[568:571]*/, v76
	ds_load_b128 v[60:63] /*v[572:575]*/, v77
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[112:119] /*v[624:631]*/, v[16:23], v[248:255] /*v[504:511]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[104:111] /*v[616:623]*/, v[8:15], v[208:215] /*v[464:471]*/
	; sched_barrier mask(0x00000000)
	tensor_load_to_lds s[0:3], s[8:15] scope:SCOPE_DEV
	s_wait_dscnt 0x18
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[120:127] /*v[632:639]*/, v[8:15], v[216:223] /*v[472:479]*/
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[104:111] /*v[616:623]*/, v[0:7], v[240:247] /*v[496:503]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[120:127] /*v[632:639]*/, v[0:7], v[248:255] /*v[504:511]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_addk_co_i32 s37, 0x180
	s_add_nc_u64 s[30:31], s[30:31], 0x100
	s_add_co_i32 s24, s24, s38
	s_wait_alu depctr_va_vdst(6)
	s_set_vgpr_msb 0x5200                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	ds_load_b128 v[24:27], v68
	ds_load_b128 v[28:31], v69
	s_wait_alu depctr_va_vdst(2)
	ds_load_b128 v[8:11], v70
	ds_load_b128 v[12:15], v71
	ds_load_b128 v[16:19], v72
	ds_load_b128 v[20:23], v73
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[0:3], v74
	ds_load_b128 v[4:7], v75
	s_set_vgpr_msb 0x52                     ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[256:263]*/, v[0:7] /*v[512:519]*/, v[40:47], v[0:7] /*v[256:263]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[64:67] /*v[576:579]*/, v248
	ds_load_b128 v[68:71] /*v[580:583]*/, v249
	; sched_group_barrier mask(0x00000004) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(8) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[264:271]*/, v[16:23] /*v[528:535]*/, v[40:47], v[8:15] /*v[264:271]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[72:75] /*v[584:587]*/, v250
	ds_load_b128 v[76:79] /*v[588:591]*/, v251
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[0:7] /*v[512:519]*/, v[56:63], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[80:83] /*v[592:595]*/, v252
	ds_load_b128 v[84:87] /*v[596:599]*/, v253
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[296:303]*/, v[16:23] /*v[528:535]*/, v[56:63], v[40:47] /*v[296:303]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[88:91] /*v[600:603]*/, v254
	ds_load_b128 v[92:95] /*v[604:607]*/, v255
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[256:263]*/, v[8:15] /*v[520:527]*/, v[32:39], v[0:7] /*v[256:263]*/
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[96:99] /*v[608:611]*/, v128 /*v640*/
	ds_load_b128 v[100:103] /*v[612:615]*/, v129 /*v641*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[264:271]*/, v[24:31] /*v[536:543]*/, v[32:39], v[8:15] /*v[264:271]*/
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[104:107] /*v[616:619]*/, v130 /*v642*/
	ds_load_b128 v[108:111] /*v[620:623]*/, v131 /*v643*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[8:15] /*v[520:527]*/, v[48:55], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[112:115] /*v[624:627]*/, v132 /*v644*/
	ds_load_b128 v[116:119] /*v[628:631]*/, v133 /*v645*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[296:303]*/, v[24:31] /*v[536:543]*/, v[48:55], v[40:47] /*v[296:303]*/
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[120:123] /*v[632:635]*/, v134 /*v646*/
	ds_load_b128 v[124:127] /*v[636:639]*/, v135 /*v647*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[272:279]*/, v[32:39] /*v[544:551]*/, v[40:47], v[16:23] /*v[272:279]*/
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[48:55] /*v[560:567]*/, v[40:47], v[24:31] /*v[280:287]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[304:311]*/, v[32:39] /*v[544:551]*/, v[56:63], v[48:55] /*v[304:311]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[56:63] /*v[312:319]*/, v[48:55] /*v[560:567]*/, v[56:63], v[56:63] /*v[312:319]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[272:279]*/, v[40:47] /*v[552:559]*/, v[32:39], v[16:23] /*v[272:279]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[56:63] /*v[568:575]*/, v[32:39], v[24:31] /*v[280:287]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[304:311]*/, v[40:47] /*v[552:559]*/, v[48:55], v[48:55] /*v[304:311]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[56:63] /*v[312:319]*/, v[56:63] /*v[568:575]*/, v[48:55], v[56:63] /*v[312:319]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[320:327]*/, v[64:71] /*v[576:583]*/, v[40:47], v[64:71] /*v[320:327]*/
	s_wait_alu depctr_va_vdst(14)
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[0:3] /*v[512:515]*/, v136 /*v648*/
	ds_load_b128 v[4:7] /*v[516:519]*/, v137 /*v649*/
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[328:335]*/, v[80:87] /*v[592:599]*/, v[40:47], v[72:79] /*v[328:335]*/
	s_wait_alu depctr_va_vdst(11)
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[8:11] /*v[520:523]*/, v138 /*v650*/
	ds_load_b128 v[12:15] /*v[524:527]*/, v139 /*v651*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[352:359]*/, v[64:71] /*v[576:583]*/, v[56:63], v[96:103] /*v[352:359]*/
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[16:19] /*v[528:531]*/, v140 /*v652*/
	ds_load_b128 v[20:23] /*v[532:535]*/, v141 /*v653*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[360:367]*/, v[80:87] /*v[592:599]*/, v[56:63], v[104:111] /*v[360:367]*/
	s_wait_alu depctr_va_vdst(12)
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[24:27] /*v[536:539]*/, v142 /*v654*/
	ds_load_b128 v[28:31] /*v[540:543]*/, v143 /*v655*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[320:327]*/, v[72:79] /*v[584:591]*/, v[32:39], v[64:71] /*v[320:327]*/
	s_wait_alu depctr_va_vdst(10)
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[32:35] /*v[544:547]*/, v144 /*v656*/
	ds_load_b128 v[36:39] /*v[548:551]*/, v145 /*v657*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[328:335]*/, v[88:95] /*v[600:607]*/, v[32:39], v[72:79] /*v[328:335]*/
	s_wait_alu depctr_va_vdst(7)
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[40:43] /*v[552:555]*/, v146 /*v658*/
	ds_load_b128 v[44:47] /*v[556:559]*/, v147 /*v659*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[352:359]*/, v[72:79] /*v[584:591]*/, v[48:55], v[96:103] /*v[352:359]*/
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[48:51] /*v[560:563]*/, v148 /*v660*/
	ds_load_b128 v[52:55] /*v[564:567]*/, v149 /*v661*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[360:367]*/, v[88:95] /*v[600:607]*/, v[48:55], v[104:111] /*v[360:367]*/
	s_wait_alu depctr_va_vdst(8)
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[56:59] /*v[568:571]*/, v150 /*v662*/
	ds_load_b128 v[60:63] /*v[572:575]*/, v151 /*v663*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[336:343]*/, v[96:103] /*v[608:615]*/, v[40:47], v[80:87] /*v[336:343]*/
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[344:351]*/, v[112:119] /*v[624:631]*/, v[40:47], v[88:95] /*v[344:351]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[368:375]*/, v[96:103] /*v[608:615]*/, v[56:63], v[112:119] /*v[368:375]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[376:383]*/, v[112:119] /*v[624:631]*/, v[56:63], v[120:127] /*v[376:383]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[336:343]*/, v[104:111] /*v[616:623]*/, v[32:39], v[80:87] /*v[336:343]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[344:351]*/, v[120:127] /*v[632:639]*/, v[32:39], v[88:95] /*v[344:351]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[368:375]*/, v[104:111] /*v[616:623]*/, v[48:55], v[112:119] /*v[368:375]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[376:383]*/, v[120:127] /*v[632:639]*/, v[48:55], v[120:127] /*v[376:383]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[384:391]*/, v[0:7] /*v[512:519]*/, v[40:47], v[128:135] /*v[384:391]*/
	s_wait_alu depctr_va_vdst(14)
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[64:67] /*v[576:579]*/, v152 /*v664*/
	ds_load_b128 v[68:71] /*v[580:583]*/, v153 /*v665*/
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[392:399]*/, v[16:23] /*v[528:535]*/, v[40:47], v[136:143] /*v[392:399]*/
	s_wait_alu depctr_va_vdst(11)
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[72:75] /*v[584:587]*/, v154 /*v666*/
	ds_load_b128 v[76:79] /*v[588:591]*/, v155 /*v667*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[416:423]*/, v[0:7] /*v[512:519]*/, v[56:63], v[160:167] /*v[416:423]*/
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[80:83] /*v[592:595]*/, v156 /*v668*/
	ds_load_b128 v[84:87] /*v[596:599]*/, v157 /*v669*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[424:431]*/, v[16:23] /*v[528:535]*/, v[56:63], v[168:175] /*v[424:431]*/
	s_wait_alu depctr_va_vdst(12)
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[88:91] /*v[600:603]*/, v158 /*v670*/
	ds_load_b128 v[92:95] /*v[604:607]*/, v159 /*v671*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[384:391]*/, v[8:15] /*v[520:527]*/, v[32:39], v[128:135] /*v[384:391]*/
	s_wait_alu depctr_va_vdst(10)
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[96:99] /*v[608:611]*/, v160 /*v672*/
	ds_load_b128 v[100:103] /*v[612:615]*/, v161 /*v673*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[392:399]*/, v[24:31] /*v[536:543]*/, v[32:39], v[136:143] /*v[392:399]*/
	s_wait_alu depctr_va_vdst(7)
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[104:107] /*v[616:619]*/, v162 /*v674*/
	ds_load_b128 v[108:111] /*v[620:623]*/, v163 /*v675*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[416:423]*/, v[8:15] /*v[520:527]*/, v[48:55], v[160:167] /*v[416:423]*/
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[112:115] /*v[624:627]*/, v164 /*v676*/
	ds_load_b128 v[116:119] /*v[628:631]*/, v165 /*v677*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[424:431]*/, v[24:31] /*v[536:543]*/, v[48:55], v[168:175] /*v[424:431]*/
	s_wait_alu depctr_va_vdst(8)
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[120:123] /*v[632:635]*/, v166 /*v678*/
	ds_load_b128 v[124:127] /*v[636:639]*/, v167 /*v679*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[400:407]*/, v[32:39] /*v[544:551]*/, v[40:47], v[144:151] /*v[400:407]*/
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[408:415]*/, v[48:55] /*v[560:567]*/, v[40:47], v[152:159] /*v[408:415]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[176:183] /*v[432:439]*/, v[32:39] /*v[544:551]*/, v[56:63], v[176:183] /*v[432:439]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[184:191] /*v[440:447]*/, v[48:55] /*v[560:567]*/, v[56:63], v[184:191] /*v[440:447]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[400:407]*/, v[40:47] /*v[552:559]*/, v[32:39], v[144:151] /*v[400:407]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[408:415]*/, v[56:63] /*v[568:575]*/, v[32:39], v[152:159] /*v[408:415]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[176:183] /*v[432:439]*/, v[40:47] /*v[552:559]*/, v[48:55], v[176:183] /*v[432:439]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[184:191] /*v[440:447]*/, v[56:63] /*v[568:575]*/, v[48:55], v[184:191] /*v[440:447]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[64:71] /*v[576:583]*/, v[40:47], v[192:199] /*v[448:455]*/
	s_wait_alu depctr_va_vdst(14)
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[0:3] /*v[512:515]*/, v168 /*v680*/
	ds_load_b128 v[4:7] /*v[516:519]*/, v169 /*v681*/
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[80:87] /*v[592:599]*/, v[40:47], v[200:207] /*v[456:463]*/
	s_wait_alu depctr_va_vdst(11)
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[8:11] /*v[520:523]*/, v170 /*v682*/
	ds_load_b128 v[12:15] /*v[524:527]*/, v171 /*v683*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[64:71] /*v[576:583]*/, v[56:63], v[224:231] /*v[480:487]*/
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[16:19] /*v[528:531]*/, v172 /*v684*/
	ds_load_b128 v[20:23] /*v[532:535]*/, v173 /*v685*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[80:87] /*v[592:599]*/, v[56:63], v[232:239] /*v[488:495]*/
	s_wait_alu depctr_va_vdst(12)
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[24:27] /*v[536:539]*/, v174 /*v686*/
	ds_load_b128 v[28:31] /*v[540:543]*/, v175 /*v687*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[72:79] /*v[584:591]*/, v[32:39], v[192:199] /*v[448:455]*/
	s_wait_alu depctr_va_vdst(10)
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[32:35] /*v[544:547]*/, v176 /*v688*/
	ds_load_b128 v[36:39] /*v[548:551]*/, v177 /*v689*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[88:95] /*v[600:607]*/, v[32:39], v[200:207] /*v[456:463]*/
	s_wait_alu depctr_va_vdst(7)
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[40:43] /*v[552:555]*/, v178 /*v690*/
	ds_load_b128 v[44:47] /*v[556:559]*/, v179 /*v691*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[72:79] /*v[584:591]*/, v[48:55], v[224:231] /*v[480:487]*/
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[48:51] /*v[560:563]*/, v180 /*v692*/
	ds_load_b128 v[52:55] /*v[564:567]*/, v181 /*v693*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[88:95] /*v[600:607]*/, v[48:55], v[232:239] /*v[488:495]*/
	s_wait_alu depctr_va_vdst(8)
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[56:59] /*v[568:571]*/, v182 /*v694*/
	ds_load_b128 v[60:63] /*v[572:575]*/, v183 /*v695*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[96:103] /*v[608:615]*/, v[40:47], v[208:215] /*v[464:471]*/
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[112:119] /*v[624:631]*/, v[40:47], v[216:223] /*v[472:479]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[96:103] /*v[608:615]*/, v[56:63], v[240:247] /*v[496:503]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[112:119] /*v[624:631]*/, v[56:63], v[248:255] /*v[504:511]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[104:111] /*v[616:623]*/, v[32:39], v[208:215] /*v[464:471]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[120:127] /*v[632:639]*/, v[32:39], v[216:223] /*v[472:479]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[104:111] /*v[616:623]*/, v[48:55], v[240:247] /*v[496:503]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[120:127] /*v[632:639]*/, v[48:55], v[248:255] /*v[504:511]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[256:263]*/, v[0:7] /*v[512:519]*/, v[24:31], v[0:7] /*v[256:263]*/
	s_wait_alu depctr_va_vdst(14)
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[64:67] /*v[576:579]*/, v184 /*v696*/
	ds_load_b128 v[68:71] /*v[580:583]*/, v185 /*v697*/
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[264:271]*/, v[16:23] /*v[528:535]*/, v[24:31], v[8:15] /*v[264:271]*/
	s_wait_alu depctr_va_vdst(11)
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[72:75] /*v[584:587]*/, v186 /*v698*/
	ds_load_b128 v[76:79] /*v[588:591]*/, v187 /*v699*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[0:7] /*v[512:519]*/, v[16:23], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[80:83] /*v[592:595]*/, v188 /*v700*/
	ds_load_b128 v[84:87] /*v[596:599]*/, v189 /*v701*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[296:303]*/, v[16:23] /*v[528:535]*/, v[16:23], v[40:47] /*v[296:303]*/
	s_wait_alu depctr_va_vdst(12)
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[88:91] /*v[600:603]*/, v190 /*v702*/
	ds_load_b128 v[92:95] /*v[604:607]*/, v191 /*v703*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[256:263]*/, v[8:15] /*v[520:527]*/, v[8:15], v[0:7] /*v[256:263]*/
	s_wait_alu depctr_va_vdst(10)
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[96:99] /*v[608:611]*/, v192 /*v704*/
	ds_load_b128 v[100:103] /*v[612:615]*/, v193 /*v705*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[264:271]*/, v[24:31] /*v[536:543]*/, v[8:15], v[8:15] /*v[264:271]*/
	s_wait_alu depctr_va_vdst(7)
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[104:107] /*v[616:619]*/, v194 /*v706*/
	ds_load_b128 v[108:111] /*v[620:623]*/, v195 /*v707*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[8:15] /*v[520:527]*/, v[0:7], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[112:115] /*v[624:627]*/, v196 /*v708*/
	ds_load_b128 v[116:119] /*v[628:631]*/, v197 /*v709*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[296:303]*/, v[24:31] /*v[536:543]*/, v[0:7], v[40:47] /*v[296:303]*/
	s_wait_alu depctr_va_vdst(8)
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[120:123] /*v[632:635]*/, v198 /*v710*/
	ds_load_b128 v[124:127] /*v[636:639]*/, v199 /*v711*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[272:279]*/, v[32:39] /*v[544:551]*/, v[24:31], v[16:23] /*v[272:279]*/
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[48:55] /*v[560:567]*/, v[24:31], v[24:31] /*v[280:287]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[304:311]*/, v[32:39] /*v[544:551]*/, v[16:23], v[48:55] /*v[304:311]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[56:63] /*v[312:319]*/, v[48:55] /*v[560:567]*/, v[16:23], v[56:63] /*v[312:319]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[272:279]*/, v[40:47] /*v[552:559]*/, v[8:15], v[16:23] /*v[272:279]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[56:63] /*v[568:575]*/, v[8:15], v[24:31] /*v[280:287]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[304:311]*/, v[40:47] /*v[552:559]*/, v[0:7], v[48:55] /*v[304:311]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[56:63] /*v[312:319]*/, v[56:63] /*v[568:575]*/, v[0:7], v[56:63] /*v[312:319]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[320:327]*/, v[64:71] /*v[576:583]*/, v[24:31], v[64:71] /*v[320:327]*/
	s_wait_alu depctr_va_vdst(14)
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[0:3] /*v[512:515]*/, v200 /*v712*/
	ds_load_b128 v[4:7] /*v[516:519]*/, v201 /*v713*/
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[328:335]*/, v[80:87] /*v[592:599]*/, v[24:31], v[72:79] /*v[328:335]*/
	s_wait_alu depctr_va_vdst(11)
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[8:11] /*v[520:523]*/, v202 /*v714*/
	ds_load_b128 v[12:15] /*v[524:527]*/, v203 /*v715*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[352:359]*/, v[64:71] /*v[576:583]*/, v[16:23], v[96:103] /*v[352:359]*/
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[16:19] /*v[528:531]*/, v204 /*v716*/
	ds_load_b128 v[20:23] /*v[532:535]*/, v205 /*v717*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[360:367]*/, v[80:87] /*v[592:599]*/, v[16:23], v[104:111] /*v[360:367]*/
	s_wait_alu depctr_va_vdst(12)
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[24:27] /*v[536:539]*/, v206 /*v718*/
	ds_load_b128 v[28:31] /*v[540:543]*/, v207 /*v719*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[320:327]*/, v[72:79] /*v[584:591]*/, v[8:15], v[64:71] /*v[320:327]*/
	s_wait_alu depctr_va_vdst(10)
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[32:35] /*v[544:547]*/, v208 /*v720*/
	ds_load_b128 v[36:39] /*v[548:551]*/, v209 /*v721*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[328:335]*/, v[88:95] /*v[600:607]*/, v[8:15], v[72:79] /*v[328:335]*/
	s_wait_alu depctr_va_vdst(7)
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[40:43] /*v[552:555]*/, v210 /*v722*/
	ds_load_b128 v[44:47] /*v[556:559]*/, v211 /*v723*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[352:359]*/, v[72:79] /*v[584:591]*/, v[0:7], v[96:103] /*v[352:359]*/
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[48:51] /*v[560:563]*/, v212 /*v724*/
	ds_load_b128 v[52:55] /*v[564:567]*/, v213 /*v725*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[360:367]*/, v[88:95] /*v[600:607]*/, v[0:7], v[104:111] /*v[360:367]*/
	s_wait_alu depctr_va_vdst(8)
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[56:59] /*v[568:571]*/, v214 /*v726*/
	ds_load_b128 v[60:63] /*v[572:575]*/, v215 /*v727*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[336:343]*/, v[96:103] /*v[608:615]*/, v[24:31], v[80:87] /*v[336:343]*/
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[344:351]*/, v[112:119] /*v[624:631]*/, v[24:31], v[88:95] /*v[344:351]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[368:375]*/, v[96:103] /*v[608:615]*/, v[16:23], v[112:119] /*v[368:375]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[376:383]*/, v[112:119] /*v[624:631]*/, v[16:23], v[120:127] /*v[376:383]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[336:343]*/, v[104:111] /*v[616:623]*/, v[8:15], v[80:87] /*v[336:343]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[344:351]*/, v[120:127] /*v[632:639]*/, v[8:15], v[88:95] /*v[344:351]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[368:375]*/, v[104:111] /*v[616:623]*/, v[0:7], v[112:119] /*v[368:375]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[376:383]*/, v[120:127] /*v[632:639]*/, v[0:7], v[120:127] /*v[376:383]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[384:391]*/, v[0:7] /*v[512:519]*/, v[24:31], v[128:135] /*v[384:391]*/
	s_wait_alu depctr_va_vdst(14)
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[64:67] /*v[576:579]*/, v216 /*v728*/
	ds_load_b128 v[68:71] /*v[580:583]*/, v217 /*v729*/
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[392:399]*/, v[16:23] /*v[528:535]*/, v[24:31], v[136:143] /*v[392:399]*/
	s_wait_alu depctr_va_vdst(11)
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[72:75] /*v[584:587]*/, v218 /*v730*/
	ds_load_b128 v[76:79] /*v[588:591]*/, v219 /*v731*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[416:423]*/, v[0:7] /*v[512:519]*/, v[16:23], v[160:167] /*v[416:423]*/
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[80:83] /*v[592:595]*/, v220 /*v732*/
	ds_load_b128 v[84:87] /*v[596:599]*/, v221 /*v733*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[424:431]*/, v[16:23] /*v[528:535]*/, v[16:23], v[168:175] /*v[424:431]*/
	s_wait_alu depctr_va_vdst(12)
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[88:91] /*v[600:603]*/, v222 /*v734*/
	ds_load_b128 v[92:95] /*v[604:607]*/, v223 /*v735*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[384:391]*/, v[8:15] /*v[520:527]*/, v[8:15], v[128:135] /*v[384:391]*/
	s_wait_alu depctr_va_vdst(10)
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[96:99] /*v[608:611]*/, v224 /*v736*/
	ds_load_b128 v[100:103] /*v[612:615]*/, v225 /*v737*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[392:399]*/, v[24:31] /*v[536:543]*/, v[8:15], v[136:143] /*v[392:399]*/
	s_wait_alu depctr_va_vdst(7)
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[104:107] /*v[616:619]*/, v226 /*v738*/
	ds_load_b128 v[108:111] /*v[620:623]*/, v227 /*v739*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[416:423]*/, v[8:15] /*v[520:527]*/, v[0:7], v[160:167] /*v[416:423]*/
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[112:115] /*v[624:627]*/, v228 /*v740*/
	ds_load_b128 v[116:119] /*v[628:631]*/, v229 /*v741*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[424:431]*/, v[24:31] /*v[536:543]*/, v[0:7], v[168:175] /*v[424:431]*/
	s_wait_alu depctr_va_vdst(8)
	s_set_vgpr_msb 0x5282                   ;  msbs: dst=2 src0=2 src1=0 src2=0
	ds_load_b128 v[120:123] /*v[632:635]*/, v230 /*v742*/
	ds_load_b128 v[124:127] /*v[636:639]*/, v231 /*v743*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x8252                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[400:407]*/, v[32:39] /*v[544:551]*/, v[24:31], v[144:151] /*v[400:407]*/
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[408:415]*/, v[48:55] /*v[560:567]*/, v[24:31], v[152:159] /*v[408:415]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[176:183] /*v[432:439]*/, v[32:39] /*v[544:551]*/, v[16:23], v[176:183] /*v[432:439]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[184:191] /*v[440:447]*/, v[48:55] /*v[560:567]*/, v[16:23], v[184:191] /*v[440:447]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[400:407]*/, v[40:47] /*v[552:559]*/, v[8:15], v[144:151] /*v[400:407]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[408:415]*/, v[56:63] /*v[568:575]*/, v[8:15], v[152:159] /*v[408:415]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[176:183] /*v[432:439]*/, v[40:47] /*v[552:559]*/, v[0:7], v[176:183] /*v[432:439]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[184:191] /*v[440:447]*/, v[56:63] /*v[568:575]*/, v[0:7], v[184:191] /*v[440:447]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[64:71] /*v[576:583]*/, v[24:31], v[192:199] /*v[448:455]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[80:87] /*v[592:599]*/, v[24:31], v[200:207] /*v[456:463]*/
	s_wait_dscnt 0x10
	s_wait_tensorcnt 0x1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[64:71] /*v[576:583]*/, v[16:23], v[224:231] /*v[480:487]*/
	s_barrier_signal -1
	s_barrier_wait -1
	s_wait_alu depctr_va_vdst(12)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[8:11] /*v[520:523]*/, v247
	ds_load_b128 v[12:15] /*v[524:527]*/, v232
	ds_load_b128 v[16:19] /*v[528:531]*/, v245
	ds_load_b128 v[20:23] /*v[532:535]*/, v246
	s_wait_alu depctr_va_vdst(11)
	ds_load_b128 v[24:27] /*v[536:539]*/, v243
	ds_load_b128 v[28:31] /*v[540:543]*/, v244
	s_wait_alu depctr_va_vdst(8)
	ds_load_b128 v[32:35] /*v[544:547]*/, v241
	ds_load_b128 v[36:39] /*v[548:551]*/, v242
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[80:87] /*v[592:599]*/, v[16:23], v[232:239] /*v[488:495]*/
	s_set_vgpr_msb 0x5200                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	ds_load_b128 v[48:51], v215
	ds_load_b128 v[52:55], v212 offset:17504
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(8) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x52                     ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[72:79] /*v[584:591]*/, v[8:15], v[192:199] /*v[448:455]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[0:3] /*v[512:515]*/, v233
	ds_load_b128 v[4:7] /*v[516:519]*/, v234
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x14
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[88:95] /*v[600:607]*/, v[8:15], v[200:207] /*v[456:463]*/
	s_set_vgpr_msb 0x5200                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	ds_load_b128 v[40:43], v212
	ds_load_b128 v[44:47], v212 offset:32
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x52                     ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[72:79] /*v[584:591]*/, v[0:7], v[224:231] /*v[480:487]*/
	s_set_vgpr_msb 0x5200                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	ds_load_b128 v[32:35], v213
	ds_load_b128 v[36:39], v212 offset:96
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x52                     ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[88:95] /*v[600:607]*/, v[0:7], v[232:239] /*v[488:495]*/
	s_set_vgpr_msb 0x5200                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	ds_load_b128 v[56:59], v214
	ds_load_b128 v[60:63], v212 offset:17440
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x52                     ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x18
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[96:103] /*v[608:615]*/, v[24:31], v[208:215] /*v[464:471]*/
	s_wait_alu depctr_va_vdst(10)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[40:43] /*v[552:555]*/, v239
	ds_load_b128 v[44:47] /*v[556:559]*/, v240
	s_add_nc_u64 s[2:3], s[30:31], s[28:29]
	;;#ASMSTART
	s_sub_co_u32 s9, s40, s37
	;;#ASMEND
	s_and_b32 s3, s3, 0x1ffffff
	;;#ASMSTART
	s_max_i32 s16, s9, 0
	;;#ASMEND
	s_lshr_b64 s[40:41], s[16:17], 16
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[112:119] /*v[624:631]*/, v[24:31], v[216:223] /*v[472:479]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[48:51] /*v[560:563]*/, v237
	ds_load_b128 v[52:55] /*v[564:567]*/, v238
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_add_co_i32 s1, s1, s34
	s_bitset1_b32 s3, 31
	s_lshl_b32 s9, s16, 16
	s_mov_b32 s10, s40
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[96:103] /*v[608:615]*/, v[16:23], v[240:247] /*v[496:503]*/
	s_wait_alu depctr_va_vdst(11)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[56:59] /*v[568:571]*/, v235
	ds_load_b128 v[60:63] /*v[572:575]*/, v236
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[112:119] /*v[624:631]*/, v[16:23], v[248:255] /*v[504:511]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[104:111] /*v[616:623]*/, v[8:15], v[208:215] /*v[464:471]*/
	; sched_barrier mask(0x00000000)
	tensor_load_to_lds s[0:3], s[8:15] scope:SCOPE_DEV
	s_wait_dscnt 0x18
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[120:127] /*v[632:639]*/, v[8:15], v[216:223] /*v[472:479]*/
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[104:111] /*v[616:623]*/, v[0:7], v[240:247] /*v[496:503]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[120:127] /*v[632:639]*/, v[0:7], v[248:255] /*v[504:511]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_add_co_i32 s39, s39, 3
	s_cmp_ge_i32 s39, s36
	s_wait_alu depctr_va_vdst(6)
	s_set_vgpr_msb 0x5200                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	ds_load_b128 v[24:27], v212 offset:128
	ds_load_b128 v[28:31], v212 offset:160
	s_wait_alu depctr_va_vdst(2)
	ds_load_b128 v[8:11], v213 offset:128
	ds_load_b128 v[12:15], v212 offset:224
	ds_load_b128 v[16:19], v214 offset:128
	ds_load_b128 v[20:23], v212 offset:17568
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[0:3], v215 offset:128
	ds_load_b128 v[4:7], v212 offset:17632
	; sched_group_barrier mask(0x00000004) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(8) SyncID(0)
	s_cbranch_scc0 .LBB2_6
; %bb.7:                                ; %Flow
	s_mov_b32 s38, s37
	s_sub_co_i32 s35, s35, s36
	s_cmp_lg_u32 s35, 0
	s_mov_b32 s0, 0
	s_cbranch_scc0 .LBB2_9
.LBB2_8:                                ; %if.then118
	s_addk_co_i32 s38, 0x80
	s_set_vgpr_msb 0x52                     ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[256:263]*/, v[0:7] /*v[512:519]*/, v[40:47], v[0:7] /*v[256:263]*/
	s_wait_dscnt 0x8
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[104:107] /*v[616:619]*/, v239 offset:17408
	ds_load_b128 v[108:111] /*v[620:623]*/, v240 offset:17408
	s_add_nc_u64 s[30:31], s[30:31], 0x100
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[264:271]*/, v[16:23] /*v[528:535]*/, v[40:47], v[8:15] /*v[264:271]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[112:115] /*v[624:627]*/, v237 offset:17408
	ds_load_b128 v[116:119] /*v[628:631]*/, v238 offset:17408
	s_sub_co_i32 s24, s24, s34
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[0:7] /*v[512:519]*/, v[56:63], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[120:123] /*v[632:635]*/, v235 offset:17408
	ds_load_b128 v[124:127] /*v[636:639]*/, v236 offset:17408
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[296:303]*/, v[16:23] /*v[528:535]*/, v[56:63], v[40:47] /*v[296:303]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[64:67] /*v[576:579]*/, v233 offset:17408
	ds_load_b128 v[68:71] /*v[580:583]*/, v234 offset:17408
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[256:263]*/, v[8:15] /*v[520:527]*/, v[32:39], v[0:7] /*v[256:263]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[72:75] /*v[584:587]*/, v247 offset:17408
	ds_load_b128 v[76:79] /*v[588:591]*/, v232 offset:17408
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[264:271]*/, v[24:31] /*v[536:543]*/, v[32:39], v[8:15] /*v[264:271]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[80:83] /*v[592:595]*/, v245 offset:17408
	ds_load_b128 v[84:87] /*v[596:599]*/, v246 offset:17408
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[8:15] /*v[520:527]*/, v[48:55], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[88:91] /*v[600:603]*/, v243 offset:17408
	ds_load_b128 v[92:95] /*v[604:607]*/, v244 offset:17408
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[296:303]*/, v[24:31] /*v[536:543]*/, v[48:55], v[40:47] /*v[296:303]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[96:99] /*v[608:611]*/, v241 offset:17408
	ds_load_b128 v[100:103] /*v[612:615]*/, v242 offset:17408
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[272:279]*/, v[32:39] /*v[544:551]*/, v[40:47], v[16:23] /*v[272:279]*/
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[48:55] /*v[560:567]*/, v[40:47], v[24:31] /*v[280:287]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[304:311]*/, v[32:39] /*v[544:551]*/, v[56:63], v[48:55] /*v[304:311]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[56:63] /*v[312:319]*/, v[48:55] /*v[560:567]*/, v[56:63], v[56:63] /*v[312:319]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[272:279]*/, v[40:47] /*v[552:559]*/, v[32:39], v[16:23] /*v[272:279]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[56:63] /*v[568:575]*/, v[32:39], v[24:31] /*v[280:287]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[304:311]*/, v[40:47] /*v[552:559]*/, v[48:55], v[48:55] /*v[304:311]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[56:63] /*v[312:319]*/, v[56:63] /*v[568:575]*/, v[48:55], v[56:63] /*v[312:319]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[320:327]*/, v[64:71] /*v[576:583]*/, v[40:47], v[64:71] /*v[320:327]*/
	s_wait_alu depctr_va_vdst(14)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[0:3] /*v[512:515]*/, v233 offset:34816
	ds_load_b128 v[4:7] /*v[516:519]*/, v234 offset:34816
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[328:335]*/, v[80:87] /*v[592:599]*/, v[40:47], v[72:79] /*v[328:335]*/
	s_wait_alu depctr_va_vdst(11)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[8:11] /*v[520:523]*/, v247 offset:34816
	ds_load_b128 v[12:15] /*v[524:527]*/, v232 offset:34816
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[352:359]*/, v[64:71] /*v[576:583]*/, v[56:63], v[96:103] /*v[352:359]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[16:19] /*v[528:531]*/, v245 offset:34816
	ds_load_b128 v[20:23] /*v[532:535]*/, v246 offset:34816
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[360:367]*/, v[80:87] /*v[592:599]*/, v[56:63], v[104:111] /*v[360:367]*/
	s_wait_alu depctr_va_vdst(12)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[24:27] /*v[536:539]*/, v243 offset:34816
	ds_load_b128 v[28:31] /*v[540:543]*/, v244 offset:34816
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[320:327]*/, v[72:79] /*v[584:591]*/, v[32:39], v[64:71] /*v[320:327]*/
	s_wait_alu depctr_va_vdst(10)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[32:35] /*v[544:547]*/, v241 offset:34816
	ds_load_b128 v[36:39] /*v[548:551]*/, v242 offset:34816
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[328:335]*/, v[88:95] /*v[600:607]*/, v[32:39], v[72:79] /*v[328:335]*/
	s_wait_alu depctr_va_vdst(7)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[40:43] /*v[552:555]*/, v239 offset:34816
	ds_load_b128 v[44:47] /*v[556:559]*/, v240 offset:34816
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[352:359]*/, v[72:79] /*v[584:591]*/, v[48:55], v[96:103] /*v[352:359]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[48:51] /*v[560:563]*/, v237 offset:34816
	ds_load_b128 v[52:55] /*v[564:567]*/, v238 offset:34816
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[360:367]*/, v[88:95] /*v[600:607]*/, v[48:55], v[104:111] /*v[360:367]*/
	s_wait_alu depctr_va_vdst(8)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[56:59] /*v[568:571]*/, v235 offset:34816
	ds_load_b128 v[60:63] /*v[572:575]*/, v236 offset:34816
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[336:343]*/, v[96:103] /*v[608:615]*/, v[40:47], v[80:87] /*v[336:343]*/
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[344:351]*/, v[112:119] /*v[624:631]*/, v[40:47], v[88:95] /*v[344:351]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[368:375]*/, v[96:103] /*v[608:615]*/, v[56:63], v[112:119] /*v[368:375]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[376:383]*/, v[112:119] /*v[624:631]*/, v[56:63], v[120:127] /*v[376:383]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[336:343]*/, v[104:111] /*v[616:623]*/, v[32:39], v[80:87] /*v[336:343]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[344:351]*/, v[120:127] /*v[632:639]*/, v[32:39], v[88:95] /*v[344:351]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[368:375]*/, v[104:111] /*v[616:623]*/, v[48:55], v[112:119] /*v[368:375]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[376:383]*/, v[120:127] /*v[632:639]*/, v[48:55], v[120:127] /*v[376:383]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[384:391]*/, v[0:7] /*v[512:519]*/, v[40:47], v[128:135] /*v[384:391]*/
	s_wait_alu depctr_va_vdst(14)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[64:67] /*v[576:579]*/, v233 offset:52224
	ds_load_b128 v[68:71] /*v[580:583]*/, v234 offset:52224
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[392:399]*/, v[16:23] /*v[528:535]*/, v[40:47], v[136:143] /*v[392:399]*/
	s_wait_alu depctr_va_vdst(11)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[72:75] /*v[584:587]*/, v247 offset:52224
	ds_load_b128 v[76:79] /*v[588:591]*/, v232 offset:52224
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[416:423]*/, v[0:7] /*v[512:519]*/, v[56:63], v[160:167] /*v[416:423]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[80:83] /*v[592:595]*/, v245 offset:52224
	ds_load_b128 v[84:87] /*v[596:599]*/, v246 offset:52224
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[424:431]*/, v[16:23] /*v[528:535]*/, v[56:63], v[168:175] /*v[424:431]*/
	s_wait_alu depctr_va_vdst(12)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[88:91] /*v[600:603]*/, v243 offset:52224
	ds_load_b128 v[92:95] /*v[604:607]*/, v244 offset:52224
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[384:391]*/, v[8:15] /*v[520:527]*/, v[32:39], v[128:135] /*v[384:391]*/
	s_wait_alu depctr_va_vdst(10)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[96:99] /*v[608:611]*/, v241 offset:52224
	ds_load_b128 v[100:103] /*v[612:615]*/, v242 offset:52224
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[392:399]*/, v[24:31] /*v[536:543]*/, v[32:39], v[136:143] /*v[392:399]*/
	s_wait_alu depctr_va_vdst(7)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[104:107] /*v[616:619]*/, v239 offset:52224
	ds_load_b128 v[108:111] /*v[620:623]*/, v240 offset:52224
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[416:423]*/, v[8:15] /*v[520:527]*/, v[48:55], v[160:167] /*v[416:423]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[112:115] /*v[624:627]*/, v237 offset:52224
	ds_load_b128 v[116:119] /*v[628:631]*/, v238 offset:52224
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[424:431]*/, v[24:31] /*v[536:543]*/, v[48:55], v[168:175] /*v[424:431]*/
	s_wait_alu depctr_va_vdst(8)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[120:123] /*v[632:635]*/, v235 offset:52224
	ds_load_b128 v[124:127] /*v[636:639]*/, v236 offset:52224
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[400:407]*/, v[32:39] /*v[544:551]*/, v[40:47], v[144:151] /*v[400:407]*/
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[408:415]*/, v[48:55] /*v[560:567]*/, v[40:47], v[152:159] /*v[408:415]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[176:183] /*v[432:439]*/, v[32:39] /*v[544:551]*/, v[56:63], v[176:183] /*v[432:439]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[184:191] /*v[440:447]*/, v[48:55] /*v[560:567]*/, v[56:63], v[184:191] /*v[440:447]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[400:407]*/, v[40:47] /*v[552:559]*/, v[32:39], v[144:151] /*v[400:407]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[408:415]*/, v[56:63] /*v[568:575]*/, v[32:39], v[152:159] /*v[408:415]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[176:183] /*v[432:439]*/, v[40:47] /*v[552:559]*/, v[48:55], v[176:183] /*v[432:439]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[184:191] /*v[440:447]*/, v[56:63] /*v[568:575]*/, v[48:55], v[184:191] /*v[440:447]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[64:71] /*v[576:583]*/, v[40:47], v[192:199] /*v[448:455]*/
	s_wait_alu depctr_va_vdst(14)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[0:3] /*v[512:515]*/, v233 offset:128
	ds_load_b128 v[4:7] /*v[516:519]*/, v234 offset:128
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[80:87] /*v[592:599]*/, v[40:47], v[200:207] /*v[456:463]*/
	s_wait_alu depctr_va_vdst(11)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[8:11] /*v[520:523]*/, v247 offset:128
	ds_load_b128 v[12:15] /*v[524:527]*/, v232 offset:128
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[64:71] /*v[576:583]*/, v[56:63], v[224:231] /*v[480:487]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[16:19] /*v[528:531]*/, v245 offset:128
	ds_load_b128 v[20:23] /*v[532:535]*/, v246 offset:128
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[80:87] /*v[592:599]*/, v[56:63], v[232:239] /*v[488:495]*/
	s_wait_alu depctr_va_vdst(12)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[24:27] /*v[536:539]*/, v243 offset:128
	ds_load_b128 v[28:31] /*v[540:543]*/, v244 offset:128
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[72:79] /*v[584:591]*/, v[32:39], v[192:199] /*v[448:455]*/
	s_wait_alu depctr_va_vdst(10)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[32:35] /*v[544:547]*/, v241 offset:128
	ds_load_b128 v[36:39] /*v[548:551]*/, v242 offset:128
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[88:95] /*v[600:607]*/, v[32:39], v[200:207] /*v[456:463]*/
	s_wait_alu depctr_va_vdst(7)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[40:43] /*v[552:555]*/, v239 offset:128
	ds_load_b128 v[44:47] /*v[556:559]*/, v240 offset:128
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[72:79] /*v[584:591]*/, v[48:55], v[224:231] /*v[480:487]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[48:51] /*v[560:563]*/, v237 offset:128
	ds_load_b128 v[52:55] /*v[564:567]*/, v238 offset:128
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[88:95] /*v[600:607]*/, v[48:55], v[232:239] /*v[488:495]*/
	s_wait_alu depctr_va_vdst(8)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[56:59] /*v[568:571]*/, v235 offset:128
	ds_load_b128 v[60:63] /*v[572:575]*/, v236 offset:128
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[96:103] /*v[608:615]*/, v[40:47], v[208:215] /*v[464:471]*/
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[112:119] /*v[624:631]*/, v[40:47], v[216:223] /*v[472:479]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[96:103] /*v[608:615]*/, v[56:63], v[240:247] /*v[496:503]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[112:119] /*v[624:631]*/, v[56:63], v[248:255] /*v[504:511]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[104:111] /*v[616:623]*/, v[32:39], v[208:215] /*v[464:471]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[120:127] /*v[632:639]*/, v[32:39], v[216:223] /*v[472:479]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[104:111] /*v[616:623]*/, v[48:55], v[240:247] /*v[496:503]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[120:127] /*v[632:639]*/, v[48:55], v[248:255] /*v[504:511]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[256:263]*/, v[0:7] /*v[512:519]*/, v[24:31], v[0:7] /*v[256:263]*/
	s_wait_alu depctr_va_vdst(14)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[64:67] /*v[576:579]*/, v233 offset:17536
	ds_load_b128 v[68:71] /*v[580:583]*/, v234 offset:17536
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[264:271]*/, v[16:23] /*v[528:535]*/, v[24:31], v[8:15] /*v[264:271]*/
	s_wait_alu depctr_va_vdst(11)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[72:75] /*v[584:587]*/, v247 offset:17536
	ds_load_b128 v[76:79] /*v[588:591]*/, v232 offset:17536
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[0:7] /*v[512:519]*/, v[16:23], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[80:83] /*v[592:595]*/, v245 offset:17536
	ds_load_b128 v[84:87] /*v[596:599]*/, v246 offset:17536
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[296:303]*/, v[16:23] /*v[528:535]*/, v[16:23], v[40:47] /*v[296:303]*/
	s_wait_alu depctr_va_vdst(12)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[88:91] /*v[600:603]*/, v243 offset:17536
	ds_load_b128 v[92:95] /*v[604:607]*/, v244 offset:17536
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[256:263]*/, v[8:15] /*v[520:527]*/, v[8:15], v[0:7] /*v[256:263]*/
	s_wait_alu depctr_va_vdst(10)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[96:99] /*v[608:611]*/, v241 offset:17536
	ds_load_b128 v[100:103] /*v[612:615]*/, v242 offset:17536
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[264:271]*/, v[24:31] /*v[536:543]*/, v[8:15], v[8:15] /*v[264:271]*/
	s_wait_alu depctr_va_vdst(7)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[104:107] /*v[616:619]*/, v239 offset:17536
	ds_load_b128 v[108:111] /*v[620:623]*/, v240 offset:17536
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[8:15] /*v[520:527]*/, v[0:7], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[112:115] /*v[624:627]*/, v237 offset:17536
	ds_load_b128 v[116:119] /*v[628:631]*/, v238 offset:17536
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[296:303]*/, v[24:31] /*v[536:543]*/, v[0:7], v[40:47] /*v[296:303]*/
	s_wait_alu depctr_va_vdst(8)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[120:123] /*v[632:635]*/, v235 offset:17536
	ds_load_b128 v[124:127] /*v[636:639]*/, v236 offset:17536
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[272:279]*/, v[32:39] /*v[544:551]*/, v[24:31], v[16:23] /*v[272:279]*/
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[48:55] /*v[560:567]*/, v[24:31], v[24:31] /*v[280:287]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[304:311]*/, v[32:39] /*v[544:551]*/, v[16:23], v[48:55] /*v[304:311]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[56:63] /*v[312:319]*/, v[48:55] /*v[560:567]*/, v[16:23], v[56:63] /*v[312:319]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[272:279]*/, v[40:47] /*v[552:559]*/, v[8:15], v[16:23] /*v[272:279]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[56:63] /*v[568:575]*/, v[8:15], v[24:31] /*v[280:287]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[304:311]*/, v[40:47] /*v[552:559]*/, v[0:7], v[48:55] /*v[304:311]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[56:63] /*v[312:319]*/, v[56:63] /*v[568:575]*/, v[0:7], v[56:63] /*v[312:319]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[320:327]*/, v[64:71] /*v[576:583]*/, v[24:31], v[64:71] /*v[320:327]*/
	s_wait_alu depctr_va_vdst(14)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[0:3] /*v[512:515]*/, v233 offset:34944
	ds_load_b128 v[4:7] /*v[516:519]*/, v234 offset:34944
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[328:335]*/, v[80:87] /*v[592:599]*/, v[24:31], v[72:79] /*v[328:335]*/
	s_wait_alu depctr_va_vdst(11)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[8:11] /*v[520:523]*/, v247 offset:34944
	ds_load_b128 v[12:15] /*v[524:527]*/, v232 offset:34944
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[352:359]*/, v[64:71] /*v[576:583]*/, v[16:23], v[96:103] /*v[352:359]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[16:19] /*v[528:531]*/, v245 offset:34944
	ds_load_b128 v[20:23] /*v[532:535]*/, v246 offset:34944
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[360:367]*/, v[80:87] /*v[592:599]*/, v[16:23], v[104:111] /*v[360:367]*/
	s_wait_alu depctr_va_vdst(12)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[24:27] /*v[536:539]*/, v243 offset:34944
	ds_load_b128 v[28:31] /*v[540:543]*/, v244 offset:34944
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[320:327]*/, v[72:79] /*v[584:591]*/, v[8:15], v[64:71] /*v[320:327]*/
	s_wait_alu depctr_va_vdst(10)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[32:35] /*v[544:547]*/, v241 offset:34944
	ds_load_b128 v[36:39] /*v[548:551]*/, v242 offset:34944
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[328:335]*/, v[88:95] /*v[600:607]*/, v[8:15], v[72:79] /*v[328:335]*/
	s_wait_alu depctr_va_vdst(7)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[40:43] /*v[552:555]*/, v239 offset:34944
	ds_load_b128 v[44:47] /*v[556:559]*/, v240 offset:34944
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[352:359]*/, v[72:79] /*v[584:591]*/, v[0:7], v[96:103] /*v[352:359]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[48:51] /*v[560:563]*/, v237 offset:34944
	ds_load_b128 v[52:55] /*v[564:567]*/, v238 offset:34944
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[360:367]*/, v[88:95] /*v[600:607]*/, v[0:7], v[104:111] /*v[360:367]*/
	s_wait_alu depctr_va_vdst(8)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[56:59] /*v[568:571]*/, v235 offset:34944
	ds_load_b128 v[60:63] /*v[572:575]*/, v236 offset:34944
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[336:343]*/, v[96:103] /*v[608:615]*/, v[24:31], v[80:87] /*v[336:343]*/
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[344:351]*/, v[112:119] /*v[624:631]*/, v[24:31], v[88:95] /*v[344:351]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[368:375]*/, v[96:103] /*v[608:615]*/, v[16:23], v[112:119] /*v[368:375]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[376:383]*/, v[112:119] /*v[624:631]*/, v[16:23], v[120:127] /*v[376:383]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[336:343]*/, v[104:111] /*v[616:623]*/, v[8:15], v[80:87] /*v[336:343]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[344:351]*/, v[120:127] /*v[632:639]*/, v[8:15], v[88:95] /*v[344:351]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[368:375]*/, v[104:111] /*v[616:623]*/, v[0:7], v[112:119] /*v[368:375]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[376:383]*/, v[120:127] /*v[632:639]*/, v[0:7], v[120:127] /*v[376:383]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[384:391]*/, v[0:7] /*v[512:519]*/, v[24:31], v[128:135] /*v[384:391]*/
	s_wait_alu depctr_va_vdst(14)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[64:67] /*v[576:579]*/, v233 offset:52352
	ds_load_b128 v[68:71] /*v[580:583]*/, v234 offset:52352
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[392:399]*/, v[16:23] /*v[528:535]*/, v[24:31], v[136:143] /*v[392:399]*/
	s_wait_alu depctr_va_vdst(11)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[72:75] /*v[584:587]*/, v247 offset:52352
	ds_load_b128 v[76:79] /*v[588:591]*/, v232 offset:52352
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[416:423]*/, v[0:7] /*v[512:519]*/, v[16:23], v[160:167] /*v[416:423]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[80:83] /*v[592:595]*/, v245 offset:52352
	ds_load_b128 v[84:87] /*v[596:599]*/, v246 offset:52352
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[424:431]*/, v[16:23] /*v[528:535]*/, v[16:23], v[168:175] /*v[424:431]*/
	s_wait_alu depctr_va_vdst(12)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[88:91] /*v[600:603]*/, v243 offset:52352
	ds_load_b128 v[92:95] /*v[604:607]*/, v244 offset:52352
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[384:391]*/, v[8:15] /*v[520:527]*/, v[8:15], v[128:135] /*v[384:391]*/
	s_wait_alu depctr_va_vdst(10)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[96:99] /*v[608:611]*/, v241 offset:52352
	ds_load_b128 v[100:103] /*v[612:615]*/, v242 offset:52352
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[392:399]*/, v[24:31] /*v[536:543]*/, v[8:15], v[136:143] /*v[392:399]*/
	s_wait_alu depctr_va_vdst(7)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[104:107] /*v[616:619]*/, v239 offset:52352
	ds_load_b128 v[108:111] /*v[620:623]*/, v240 offset:52352
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[416:423]*/, v[8:15] /*v[520:527]*/, v[0:7], v[160:167] /*v[416:423]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[112:115] /*v[624:627]*/, v237 offset:52352
	ds_load_b128 v[116:119] /*v[628:631]*/, v238 offset:52352
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[424:431]*/, v[24:31] /*v[536:543]*/, v[0:7], v[168:175] /*v[424:431]*/
	s_wait_alu depctr_va_vdst(8)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[120:123] /*v[632:635]*/, v235 offset:52352
	ds_load_b128 v[124:127] /*v[636:639]*/, v236 offset:52352
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[400:407]*/, v[32:39] /*v[544:551]*/, v[24:31], v[144:151] /*v[400:407]*/
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[408:415]*/, v[48:55] /*v[560:567]*/, v[24:31], v[152:159] /*v[408:415]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[176:183] /*v[432:439]*/, v[32:39] /*v[544:551]*/, v[16:23], v[176:183] /*v[432:439]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[184:191] /*v[440:447]*/, v[48:55] /*v[560:567]*/, v[16:23], v[184:191] /*v[440:447]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[400:407]*/, v[40:47] /*v[552:559]*/, v[8:15], v[144:151] /*v[400:407]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[408:415]*/, v[56:63] /*v[568:575]*/, v[8:15], v[152:159] /*v[408:415]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[176:183] /*v[432:439]*/, v[40:47] /*v[552:559]*/, v[0:7], v[176:183] /*v[432:439]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[184:191] /*v[440:447]*/, v[56:63] /*v[568:575]*/, v[0:7], v[184:191] /*v[440:447]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[64:71] /*v[576:583]*/, v[24:31], v[192:199] /*v[448:455]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[80:87] /*v[592:599]*/, v[24:31], v[200:207] /*v[456:463]*/
	s_wait_dscnt 0x10
	s_wait_tensorcnt 0x1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[64:71] /*v[576:583]*/, v[16:23], v[224:231] /*v[480:487]*/
	s_barrier_signal -1
	s_barrier_wait -1
	s_wait_alu depctr_va_vdst(12)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[8:11] /*v[520:523]*/, v222
	ds_load_b128 v[12:15] /*v[524:527]*/, v225
	ds_load_b128 v[16:19] /*v[528:531]*/, v226
	ds_load_b128 v[20:23] /*v[532:535]*/, v227
	s_wait_alu depctr_va_vdst(11)
	ds_load_b128 v[24:27] /*v[536:539]*/, v228
	ds_load_b128 v[28:31] /*v[540:543]*/, v229
	s_wait_alu depctr_va_vdst(8)
	ds_load_b128 v[32:35] /*v[544:547]*/, v223
	ds_load_b128 v[36:39] /*v[548:551]*/, v224
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[80:87] /*v[592:599]*/, v[16:23], v[232:239] /*v[488:495]*/
	s_set_vgpr_msb 0x5200                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	ds_load_b128 v[48:51], v215 offset:34816
	ds_load_b128 v[52:55], v212 offset:52320
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(8) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x52                     ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[72:79] /*v[584:591]*/, v[8:15], v[192:199] /*v[448:455]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[0:3] /*v[512:515]*/, v230
	ds_load_b128 v[4:7] /*v[516:519]*/, v231
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x14
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[88:95] /*v[600:607]*/, v[8:15], v[200:207] /*v[456:463]*/
	s_set_vgpr_msb 0x5200                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	ds_load_b128 v[40:43], v212 offset:34816
	ds_load_b128 v[44:47], v212 offset:34848
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x52                     ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[72:79] /*v[584:591]*/, v[0:7], v[224:231] /*v[480:487]*/
	s_set_vgpr_msb 0x5200                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	ds_load_b128 v[32:35], v213 offset:34816
	ds_load_b128 v[36:39], v212 offset:34912
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x52                     ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[88:95] /*v[600:607]*/, v[0:7], v[232:239] /*v[488:495]*/
	s_set_vgpr_msb 0x5200                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	ds_load_b128 v[56:59], v214 offset:34816
	ds_load_b128 v[60:63], v212 offset:52256
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x52                     ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x18
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[96:103] /*v[608:615]*/, v[24:31], v[208:215] /*v[464:471]*/
	s_wait_alu depctr_va_vdst(10)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[40:43] /*v[552:555]*/, v220
	ds_load_b128 v[44:47] /*v[556:559]*/, v221
	s_add_nc_u64 s[2:3], s[30:31], s[28:29]
	;;#ASMSTART
	s_sub_co_u32 s9, s22, s38
	;;#ASMEND
	s_and_b32 s0, s3, 0x1ffffff
	;;#ASMSTART
	s_max_i32 s16, s9, 0
	;;#ASMEND
	s_or_b32 s3, s0, 0x80000000
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[112:119] /*v[624:631]*/, v[24:31], v[216:223] /*v[472:479]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[48:51] /*v[560:563]*/, v218
	ds_load_b128 v[52:55] /*v[564:567]*/, v219
	s_mov_b32 s0, 1
	s_lshr_b64 s[36:37], s[16:17], 16
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_lshl1_add_u32 s1, s24, s19
	s_lshl_b32 s9, s16, 16
	s_mov_b32 s15, 0
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[96:103] /*v[608:615]*/, v[16:23], v[240:247] /*v[496:503]*/
	s_wait_alu depctr_va_vdst(11)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[56:59] /*v[568:571]*/, v216
	ds_load_b128 v[60:63] /*v[572:575]*/, v217
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_mov_b32 s10, s36
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[112:119] /*v[624:631]*/, v[16:23], v[248:255] /*v[504:511]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[104:111] /*v[616:623]*/, v[8:15], v[208:215] /*v[464:471]*/
	; sched_barrier mask(0x00000000)
	tensor_load_to_lds s[0:3], s[8:15] scope:SCOPE_DEV
	s_wait_dscnt 0x18
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[120:127] /*v[632:639]*/, v[8:15], v[216:223] /*v[472:479]*/
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[104:111] /*v[616:623]*/, v[0:7], v[240:247] /*v[496:503]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[120:127] /*v[632:639]*/, v[0:7], v[248:255] /*v[504:511]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_wait_alu depctr_va_vdst(6)
	s_set_vgpr_msb 0x5200                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	ds_load_b128 v[24:27], v212 offset:34944
	ds_load_b128 v[28:31], v212 offset:34976
	s_wait_alu depctr_va_vdst(2)
	ds_load_b128 v[8:11], v213 offset:34944
	ds_load_b128 v[12:15], v212 offset:35040
	ds_load_b128 v[16:19], v214 offset:34944
	ds_load_b128 v[20:23], v212 offset:52384
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[0:3], v215 offset:34944
	ds_load_b128 v[4:7], v212 offset:52448
	; sched_group_barrier mask(0x00000004) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(8) SyncID(0)
.LBB2_9:                                ; %if.end119
	s_cmp_lt_u32 s35, 2
	s_cbranch_scc1 .LBB2_11
; %bb.10:                               ; %if.then121
	s_add_nc_u64 s[0:1], s[30:31], s[28:29]
	s_set_vgpr_msb 0x52                     ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[256:263]*/, v[0:7] /*v[512:519]*/, v[40:47], v[0:7] /*v[256:263]*/
	s_wait_dscnt 0x8
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[120:123] /*v[632:635]*/, v196
	ds_load_b128 v[124:127] /*v[636:639]*/, v197
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[264:271]*/, v[16:23] /*v[528:535]*/, v[40:47], v[8:15] /*v[264:271]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[64:67] /*v[576:579]*/, v210
	ds_load_b128 v[68:71] /*v[580:583]*/, v211
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[0:7] /*v[512:519]*/, v[56:63], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[72:75] /*v[584:587]*/, v208
	ds_load_b128 v[76:79] /*v[588:591]*/, v209
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[296:303]*/, v[16:23] /*v[528:535]*/, v[56:63], v[40:47] /*v[296:303]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[80:83] /*v[592:595]*/, v206
	ds_load_b128 v[84:87] /*v[596:599]*/, v207
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[256:263]*/, v[8:15] /*v[520:527]*/, v[32:39], v[0:7] /*v[256:263]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[88:91] /*v[600:603]*/, v204
	ds_load_b128 v[92:95] /*v[604:607]*/, v205
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[264:271]*/, v[24:31] /*v[536:543]*/, v[32:39], v[8:15] /*v[264:271]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[96:99] /*v[608:611]*/, v202
	ds_load_b128 v[100:103] /*v[612:615]*/, v203
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[8:15] /*v[520:527]*/, v[48:55], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[104:107] /*v[616:619]*/, v200
	ds_load_b128 v[108:111] /*v[620:623]*/, v201
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[296:303]*/, v[24:31] /*v[536:543]*/, v[48:55], v[40:47] /*v[296:303]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[112:115] /*v[624:627]*/, v198
	ds_load_b128 v[116:119] /*v[628:631]*/, v199
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[272:279]*/, v[32:39] /*v[544:551]*/, v[40:47], v[16:23] /*v[272:279]*/
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[48:55] /*v[560:567]*/, v[40:47], v[24:31] /*v[280:287]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[304:311]*/, v[32:39] /*v[544:551]*/, v[56:63], v[48:55] /*v[304:311]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[56:63] /*v[312:319]*/, v[48:55] /*v[560:567]*/, v[56:63], v[56:63] /*v[312:319]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[272:279]*/, v[40:47] /*v[552:559]*/, v[32:39], v[16:23] /*v[272:279]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[56:63] /*v[568:575]*/, v[32:39], v[24:31] /*v[280:287]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[304:311]*/, v[40:47] /*v[552:559]*/, v[48:55], v[48:55] /*v[304:311]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[56:63] /*v[312:319]*/, v[56:63] /*v[568:575]*/, v[48:55], v[56:63] /*v[312:319]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[320:327]*/, v[64:71] /*v[576:583]*/, v[40:47], v[64:71] /*v[320:327]*/
	s_wait_alu depctr_va_vdst(14)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[0:3] /*v[512:515]*/, v194
	ds_load_b128 v[4:7] /*v[516:519]*/, v195
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[328:335]*/, v[80:87] /*v[592:599]*/, v[40:47], v[72:79] /*v[328:335]*/
	s_wait_alu depctr_va_vdst(11)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[8:11] /*v[520:523]*/, v192
	ds_load_b128 v[12:15] /*v[524:527]*/, v193
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[352:359]*/, v[64:71] /*v[576:583]*/, v[56:63], v[96:103] /*v[352:359]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[16:19] /*v[528:531]*/, v190
	ds_load_b128 v[20:23] /*v[532:535]*/, v191
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[360:367]*/, v[80:87] /*v[592:599]*/, v[56:63], v[104:111] /*v[360:367]*/
	s_wait_alu depctr_va_vdst(12)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[24:27] /*v[536:539]*/, v188
	ds_load_b128 v[28:31] /*v[540:543]*/, v189
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[320:327]*/, v[72:79] /*v[584:591]*/, v[32:39], v[64:71] /*v[320:327]*/
	s_wait_alu depctr_va_vdst(10)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[32:35] /*v[544:547]*/, v186
	ds_load_b128 v[36:39] /*v[548:551]*/, v187
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[328:335]*/, v[88:95] /*v[600:607]*/, v[32:39], v[72:79] /*v[328:335]*/
	s_wait_alu depctr_va_vdst(7)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[40:43] /*v[552:555]*/, v184
	ds_load_b128 v[44:47] /*v[556:559]*/, v185
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[352:359]*/, v[72:79] /*v[584:591]*/, v[48:55], v[96:103] /*v[352:359]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[48:51] /*v[560:563]*/, v182
	ds_load_b128 v[52:55] /*v[564:567]*/, v183
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[360:367]*/, v[88:95] /*v[600:607]*/, v[48:55], v[104:111] /*v[360:367]*/
	s_wait_alu depctr_va_vdst(8)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[56:59] /*v[568:571]*/, v180
	ds_load_b128 v[60:63] /*v[572:575]*/, v181
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x14
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[336:343]*/, v[96:103] /*v[608:615]*/, v[40:47], v[80:87] /*v[336:343]*/
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[344:351]*/, v[112:119] /*v[624:631]*/, v[40:47], v[88:95] /*v[344:351]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[368:375]*/, v[96:103] /*v[608:615]*/, v[56:63], v[112:119] /*v[368:375]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[376:383]*/, v[112:119] /*v[624:631]*/, v[56:63], v[120:127] /*v[376:383]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[336:343]*/, v[104:111] /*v[616:623]*/, v[32:39], v[80:87] /*v[336:343]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[344:351]*/, v[120:127] /*v[632:639]*/, v[32:39], v[88:95] /*v[344:351]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[368:375]*/, v[104:111] /*v[616:623]*/, v[48:55], v[112:119] /*v[368:375]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[376:383]*/, v[120:127] /*v[632:639]*/, v[48:55], v[120:127] /*v[376:383]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[384:391]*/, v[0:7] /*v[512:519]*/, v[40:47], v[128:135] /*v[384:391]*/
	s_wait_alu depctr_va_vdst(14)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[64:67] /*v[576:579]*/, v178
	ds_load_b128 v[68:71] /*v[580:583]*/, v179
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[392:399]*/, v[16:23] /*v[528:535]*/, v[40:47], v[136:143] /*v[392:399]*/
	s_wait_alu depctr_va_vdst(11)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[72:75] /*v[584:587]*/, v176
	ds_load_b128 v[76:79] /*v[588:591]*/, v177
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[416:423]*/, v[0:7] /*v[512:519]*/, v[56:63], v[160:167] /*v[416:423]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[80:83] /*v[592:595]*/, v174
	ds_load_b128 v[84:87] /*v[596:599]*/, v175
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[424:431]*/, v[16:23] /*v[528:535]*/, v[56:63], v[168:175] /*v[424:431]*/
	s_wait_alu depctr_va_vdst(12)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[88:91] /*v[600:603]*/, v172
	ds_load_b128 v[92:95] /*v[604:607]*/, v173
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[384:391]*/, v[8:15] /*v[520:527]*/, v[32:39], v[128:135] /*v[384:391]*/
	s_wait_alu depctr_va_vdst(10)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[96:99] /*v[608:611]*/, v170
	ds_load_b128 v[100:103] /*v[612:615]*/, v171
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[392:399]*/, v[24:31] /*v[536:543]*/, v[32:39], v[136:143] /*v[392:399]*/
	s_wait_alu depctr_va_vdst(7)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[104:107] /*v[616:619]*/, v168
	ds_load_b128 v[108:111] /*v[620:623]*/, v169
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[416:423]*/, v[8:15] /*v[520:527]*/, v[48:55], v[160:167] /*v[416:423]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[112:115] /*v[624:627]*/, v166
	ds_load_b128 v[116:119] /*v[628:631]*/, v167
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[424:431]*/, v[24:31] /*v[536:543]*/, v[48:55], v[168:175] /*v[424:431]*/
	s_wait_alu depctr_va_vdst(8)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[120:123] /*v[632:635]*/, v164
	ds_load_b128 v[124:127] /*v[636:639]*/, v165
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[400:407]*/, v[32:39] /*v[544:551]*/, v[40:47], v[144:151] /*v[400:407]*/
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[408:415]*/, v[48:55] /*v[560:567]*/, v[40:47], v[152:159] /*v[408:415]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[176:183] /*v[432:439]*/, v[32:39] /*v[544:551]*/, v[56:63], v[176:183] /*v[432:439]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[184:191] /*v[440:447]*/, v[48:55] /*v[560:567]*/, v[56:63], v[184:191] /*v[440:447]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[400:407]*/, v[40:47] /*v[552:559]*/, v[32:39], v[144:151] /*v[400:407]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[408:415]*/, v[56:63] /*v[568:575]*/, v[32:39], v[152:159] /*v[408:415]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[176:183] /*v[432:439]*/, v[40:47] /*v[552:559]*/, v[48:55], v[176:183] /*v[432:439]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[184:191] /*v[440:447]*/, v[56:63] /*v[568:575]*/, v[48:55], v[184:191] /*v[440:447]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[64:71] /*v[576:583]*/, v[40:47], v[192:199] /*v[448:455]*/
	s_wait_alu depctr_va_vdst(14)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[0:3] /*v[512:515]*/, v162
	ds_load_b128 v[4:7] /*v[516:519]*/, v163
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[80:87] /*v[592:599]*/, v[40:47], v[200:207] /*v[456:463]*/
	s_wait_alu depctr_va_vdst(11)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[8:11] /*v[520:523]*/, v160
	ds_load_b128 v[12:15] /*v[524:527]*/, v161
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[64:71] /*v[576:583]*/, v[56:63], v[224:231] /*v[480:487]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[16:19] /*v[528:531]*/, v158
	ds_load_b128 v[20:23] /*v[532:535]*/, v159
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[80:87] /*v[592:599]*/, v[56:63], v[232:239] /*v[488:495]*/
	s_wait_alu depctr_va_vdst(12)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[24:27] /*v[536:539]*/, v156
	ds_load_b128 v[28:31] /*v[540:543]*/, v157
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[72:79] /*v[584:591]*/, v[32:39], v[192:199] /*v[448:455]*/
	s_wait_alu depctr_va_vdst(10)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[32:35] /*v[544:547]*/, v154
	ds_load_b128 v[36:39] /*v[548:551]*/, v155
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[88:95] /*v[600:607]*/, v[32:39], v[200:207] /*v[456:463]*/
	s_wait_alu depctr_va_vdst(7)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[40:43] /*v[552:555]*/, v152
	ds_load_b128 v[44:47] /*v[556:559]*/, v153
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[72:79] /*v[584:591]*/, v[48:55], v[224:231] /*v[480:487]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[48:51] /*v[560:563]*/, v150
	ds_load_b128 v[52:55] /*v[564:567]*/, v151
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[88:95] /*v[600:607]*/, v[48:55], v[232:239] /*v[488:495]*/
	s_wait_alu depctr_va_vdst(8)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[56:59] /*v[568:571]*/, v148
	ds_load_b128 v[60:63] /*v[572:575]*/, v149
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[96:103] /*v[608:615]*/, v[40:47], v[208:215] /*v[464:471]*/
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[112:119] /*v[624:631]*/, v[40:47], v[216:223] /*v[472:479]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[96:103] /*v[608:615]*/, v[56:63], v[240:247] /*v[496:503]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[112:119] /*v[624:631]*/, v[56:63], v[248:255] /*v[504:511]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[104:111] /*v[616:623]*/, v[32:39], v[208:215] /*v[464:471]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[120:127] /*v[632:639]*/, v[32:39], v[216:223] /*v[472:479]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[104:111] /*v[616:623]*/, v[48:55], v[240:247] /*v[496:503]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[120:127] /*v[632:639]*/, v[48:55], v[248:255] /*v[504:511]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[256:263]*/, v[0:7] /*v[512:519]*/, v[24:31], v[0:7] /*v[256:263]*/
	s_wait_alu depctr_va_vdst(14)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[64:67] /*v[576:579]*/, v146
	ds_load_b128 v[68:71] /*v[580:583]*/, v147
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[264:271]*/, v[16:23] /*v[528:535]*/, v[24:31], v[8:15] /*v[264:271]*/
	s_wait_alu depctr_va_vdst(11)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[72:75] /*v[584:587]*/, v144
	ds_load_b128 v[76:79] /*v[588:591]*/, v145
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[0:7] /*v[512:519]*/, v[16:23], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[80:83] /*v[592:595]*/, v142
	ds_load_b128 v[84:87] /*v[596:599]*/, v143
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[296:303]*/, v[16:23] /*v[528:535]*/, v[16:23], v[40:47] /*v[296:303]*/
	s_wait_alu depctr_va_vdst(12)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[88:91] /*v[600:603]*/, v140
	ds_load_b128 v[92:95] /*v[604:607]*/, v141
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[256:263]*/, v[8:15] /*v[520:527]*/, v[8:15], v[0:7] /*v[256:263]*/
	s_wait_alu depctr_va_vdst(10)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[96:99] /*v[608:611]*/, v138
	ds_load_b128 v[100:103] /*v[612:615]*/, v139
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[264:271]*/, v[24:31] /*v[536:543]*/, v[8:15], v[8:15] /*v[264:271]*/
	s_wait_alu depctr_va_vdst(7)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[104:107] /*v[616:619]*/, v136
	ds_load_b128 v[108:111] /*v[620:623]*/, v137
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[8:15] /*v[520:527]*/, v[0:7], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[112:115] /*v[624:627]*/, v134
	ds_load_b128 v[116:119] /*v[628:631]*/, v135
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[296:303]*/, v[24:31] /*v[536:543]*/, v[0:7], v[40:47] /*v[296:303]*/
	s_wait_alu depctr_va_vdst(8)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[120:123] /*v[632:635]*/, v132
	ds_load_b128 v[124:127] /*v[636:639]*/, v133
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[272:279]*/, v[32:39] /*v[544:551]*/, v[24:31], v[16:23] /*v[272:279]*/
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[48:55] /*v[560:567]*/, v[24:31], v[24:31] /*v[280:287]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[304:311]*/, v[32:39] /*v[544:551]*/, v[16:23], v[48:55] /*v[304:311]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[56:63] /*v[312:319]*/, v[48:55] /*v[560:567]*/, v[16:23], v[56:63] /*v[312:319]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[272:279]*/, v[40:47] /*v[552:559]*/, v[8:15], v[16:23] /*v[272:279]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[56:63] /*v[568:575]*/, v[8:15], v[24:31] /*v[280:287]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[304:311]*/, v[40:47] /*v[552:559]*/, v[0:7], v[48:55] /*v[304:311]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[56:63] /*v[312:319]*/, v[56:63] /*v[568:575]*/, v[0:7], v[56:63] /*v[312:319]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[320:327]*/, v[64:71] /*v[576:583]*/, v[24:31], v[64:71] /*v[320:327]*/
	s_wait_alu depctr_va_vdst(14)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[0:3] /*v[512:515]*/, v130
	ds_load_b128 v[4:7] /*v[516:519]*/, v131
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[328:335]*/, v[80:87] /*v[592:599]*/, v[24:31], v[72:79] /*v[328:335]*/
	s_wait_alu depctr_va_vdst(11)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[8:11] /*v[520:523]*/, v128
	ds_load_b128 v[12:15] /*v[524:527]*/, v129
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[352:359]*/, v[64:71] /*v[576:583]*/, v[16:23], v[96:103] /*v[352:359]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[16:19] /*v[528:531]*/, v126
	ds_load_b128 v[20:23] /*v[532:535]*/, v127
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[360:367]*/, v[80:87] /*v[592:599]*/, v[16:23], v[104:111] /*v[360:367]*/
	s_wait_alu depctr_va_vdst(12)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[24:27] /*v[536:539]*/, v124
	ds_load_b128 v[28:31] /*v[540:543]*/, v125
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[320:327]*/, v[72:79] /*v[584:591]*/, v[8:15], v[64:71] /*v[320:327]*/
	s_wait_alu depctr_va_vdst(10)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[32:35] /*v[544:547]*/, v122
	ds_load_b128 v[36:39] /*v[548:551]*/, v123
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[328:335]*/, v[88:95] /*v[600:607]*/, v[8:15], v[72:79] /*v[328:335]*/
	s_wait_alu depctr_va_vdst(7)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[40:43] /*v[552:555]*/, v120
	ds_load_b128 v[44:47] /*v[556:559]*/, v121
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[352:359]*/, v[72:79] /*v[584:591]*/, v[0:7], v[96:103] /*v[352:359]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[48:51] /*v[560:563]*/, v118
	ds_load_b128 v[52:55] /*v[564:567]*/, v119
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[360:367]*/, v[88:95] /*v[600:607]*/, v[0:7], v[104:111] /*v[360:367]*/
	s_wait_alu depctr_va_vdst(8)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[56:59] /*v[568:571]*/, v116
	ds_load_b128 v[60:63] /*v[572:575]*/, v117
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[336:343]*/, v[96:103] /*v[608:615]*/, v[24:31], v[80:87] /*v[336:343]*/
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[344:351]*/, v[112:119] /*v[624:631]*/, v[24:31], v[88:95] /*v[344:351]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[368:375]*/, v[96:103] /*v[608:615]*/, v[16:23], v[112:119] /*v[368:375]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[376:383]*/, v[112:119] /*v[624:631]*/, v[16:23], v[120:127] /*v[376:383]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[336:343]*/, v[104:111] /*v[616:623]*/, v[8:15], v[80:87] /*v[336:343]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[344:351]*/, v[120:127] /*v[632:639]*/, v[8:15], v[88:95] /*v[344:351]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[368:375]*/, v[104:111] /*v[616:623]*/, v[0:7], v[112:119] /*v[368:375]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[376:383]*/, v[120:127] /*v[632:639]*/, v[0:7], v[120:127] /*v[376:383]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[384:391]*/, v[0:7] /*v[512:519]*/, v[24:31], v[128:135] /*v[384:391]*/
	s_wait_alu depctr_va_vdst(14)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[64:67] /*v[576:579]*/, v114
	ds_load_b128 v[68:71] /*v[580:583]*/, v115
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[392:399]*/, v[16:23] /*v[528:535]*/, v[24:31], v[136:143] /*v[392:399]*/
	s_wait_alu depctr_va_vdst(11)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[72:75] /*v[584:587]*/, v112
	ds_load_b128 v[76:79] /*v[588:591]*/, v113
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[416:423]*/, v[0:7] /*v[512:519]*/, v[16:23], v[160:167] /*v[416:423]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[80:83] /*v[592:595]*/, v110
	ds_load_b128 v[84:87] /*v[596:599]*/, v111
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[424:431]*/, v[16:23] /*v[528:535]*/, v[16:23], v[168:175] /*v[424:431]*/
	s_wait_alu depctr_va_vdst(12)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[88:91] /*v[600:603]*/, v108
	ds_load_b128 v[92:95] /*v[604:607]*/, v109
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[384:391]*/, v[8:15] /*v[520:527]*/, v[8:15], v[128:135] /*v[384:391]*/
	s_wait_alu depctr_va_vdst(10)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[96:99] /*v[608:611]*/, v106
	ds_load_b128 v[100:103] /*v[612:615]*/, v107
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[392:399]*/, v[24:31] /*v[536:543]*/, v[8:15], v[136:143] /*v[392:399]*/
	s_wait_alu depctr_va_vdst(7)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[104:107] /*v[616:619]*/, v104
	ds_load_b128 v[108:111] /*v[620:623]*/, v105
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[416:423]*/, v[8:15] /*v[520:527]*/, v[0:7], v[160:167] /*v[416:423]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[112:115] /*v[624:627]*/, v102
	ds_load_b128 v[116:119] /*v[628:631]*/, v103
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[424:431]*/, v[24:31] /*v[536:543]*/, v[0:7], v[168:175] /*v[424:431]*/
	s_wait_alu depctr_va_vdst(8)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[120:123] /*v[632:635]*/, v100
	ds_load_b128 v[124:127] /*v[636:639]*/, v101
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[400:407]*/, v[32:39] /*v[544:551]*/, v[24:31], v[144:151] /*v[400:407]*/
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[408:415]*/, v[48:55] /*v[560:567]*/, v[24:31], v[152:159] /*v[408:415]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[176:183] /*v[432:439]*/, v[32:39] /*v[544:551]*/, v[16:23], v[176:183] /*v[432:439]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[184:191] /*v[440:447]*/, v[48:55] /*v[560:567]*/, v[16:23], v[184:191] /*v[440:447]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[400:407]*/, v[40:47] /*v[552:559]*/, v[8:15], v[144:151] /*v[400:407]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[408:415]*/, v[56:63] /*v[568:575]*/, v[8:15], v[152:159] /*v[408:415]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[176:183] /*v[432:439]*/, v[40:47] /*v[552:559]*/, v[0:7], v[176:183] /*v[432:439]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[184:191] /*v[440:447]*/, v[56:63] /*v[568:575]*/, v[0:7], v[184:191] /*v[440:447]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[64:71] /*v[576:583]*/, v[24:31], v[192:199] /*v[448:455]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	s_set_vgpr_msb 0x5200                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_add_nc_u32_e32 v32, 0x19800, v99
	s_set_vgpr_msb 0x52                     ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[80:87] /*v[592:599]*/, v[24:31], v[200:207] /*v[456:463]*/
	s_wait_dscnt 0x10
	s_wait_tensorcnt 0x1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[64:71] /*v[576:583]*/, v[16:23], v[224:231] /*v[480:487]*/
	s_barrier_signal -1
	s_barrier_wait -1
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[8:11] /*v[520:523]*/, v86
	ds_load_b128 v[16:19] /*v[528:531]*/, v87
	ds_load_b128 v[20:23] /*v[532:535]*/, v88
	ds_load_b128 v[24:27] /*v[536:539]*/, v89
	ds_load_b128 v[28:31] /*v[540:543]*/, v90
	ds_load_b128 v[32:35] /*v[544:547]*/, v84
	ds_load_b128 v[36:39] /*v[548:551]*/, v85
	s_set_vgpr_msb 0x8000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	ds_load_b128 v[56:59], v91
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v91, 0x19800, v93
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x80                     ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[12:15] /*v[524:527]*/, v91
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[80:87] /*v[592:599]*/, v[16:23], v[232:239] /*v[488:495]*/
	s_set_vgpr_msb 0x5200                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	ds_load_b128 v[48:51], v82
	ds_load_b128 v[52:55], v83
	; sched_group_barrier mask(0x00000100) size(8) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x52                     ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[72:79] /*v[584:591]*/, v[8:15], v[192:199] /*v[448:455]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[0:3] /*v[512:515]*/, v98
	ds_load_b128 v[4:7] /*v[516:519]*/, v32
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x15
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[88:95] /*v[600:607]*/, v[8:15], v[200:207] /*v[456:463]*/
	s_set_vgpr_msb 0x5200                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	ds_load_b128 v[40:43], v96
	ds_load_b128 v[44:47], v97
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x52                     ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[72:79] /*v[584:591]*/, v[0:7], v[224:231] /*v[480:487]*/
	s_wait_alu depctr_vm_vsrc(2)
	s_set_vgpr_msb 0x5200                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	ds_load_b128 v[32:35], v94
	ds_load_b128 v[36:39], v95
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_set_vgpr_msb 0x52                     ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[88:95] /*v[600:607]*/, v[0:7], v[232:239] /*v[488:495]*/
	s_set_vgpr_msb 0x5200                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	ds_load_b128 v[60:63], v92
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x52                     ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x18
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[96:103] /*v[608:615]*/, v[24:31], v[208:215] /*v[464:471]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[40:43] /*v[552:555]*/, v80
	ds_load_b128 v[44:47] /*v[556:559]*/, v81
	s_add_nc_u64 s[2:3], s[0:1], 0x100
	s_lshl1_add_u32 s1, s24, s7
	s_and_b32 s0, s3, 0x1ffffff
	s_add_co_i32 s7, s38, 0x80
	;;#ASMSTART
	s_sub_co_u32 s7, s22, s7
	;;#ASMEND
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[112:119] /*v[624:631]*/, v[24:31], v[216:223] /*v[472:479]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[48:51] /*v[560:563]*/, v78
	ds_load_b128 v[52:55] /*v[564:567]*/, v79
	;;#ASMSTART
	s_max_i32 s16, s7, 0
	;;#ASMEND
	s_or_b32 s3, s0, 0x80000000
	s_mov_b32 s0, 1
	s_lshl_b32 s9, s16, 16
	s_lshr_b64 s[16:17], s[16:17], 16
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[96:103] /*v[608:615]*/, v[16:23], v[240:247] /*v[496:503]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[56:59] /*v[568:571]*/, v76
	ds_load_b128 v[60:63] /*v[572:575]*/, v77
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	s_mov_b32 s15, 0
	s_mov_b32 s10, s16
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[112:119] /*v[624:631]*/, v[16:23], v[248:255] /*v[504:511]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[104:111] /*v[616:623]*/, v[8:15], v[208:215] /*v[464:471]*/
	; sched_barrier mask(0x00000000)
	tensor_load_to_lds s[0:3], s[8:15] scope:SCOPE_DEV
	s_wait_dscnt 0x18
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[120:127] /*v[632:639]*/, v[8:15], v[216:223] /*v[472:479]*/
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[104:111] /*v[616:623]*/, v[0:7], v[240:247] /*v[496:503]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000004) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[120:127] /*v[632:639]*/, v[0:7], v[248:255] /*v[504:511]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_mov_b32 s0, 2
	s_wait_alu depctr_va_vdst(6)
	s_set_vgpr_msb 0x5200                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	ds_load_b128 v[24:27], v68
	ds_load_b128 v[28:31], v69
	s_wait_alu depctr_va_vdst(2)
	ds_load_b128 v[8:11], v70
	ds_load_b128 v[12:15], v71
	ds_load_b128 v[16:19], v72
	ds_load_b128 v[20:23], v73
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[0:3], v74
	ds_load_b128 v[4:7], v75
	; sched_group_barrier mask(0x00000004) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(8) SyncID(0)
.LBB2_11:                               ; %if.end122
	s_ashr_i32 s3, s25, 31
	s_mov_b32 s2, s25
	s_add_co_i32 s7, s0, 1
	s_mul_u64 s[2:3], s[2:3], s[26:27]
	s_lshl_b32 s1, s23, 12
	s_lshl_b64 s[2:3], s[2:3], 1
	s_mul_i32 s8, s7, 0x11000
	s_cmp_lg_u32 s7, 3
	s_mul_i32 s0, s0, 0x8800
	s_cselect_b32 s9, s8, 0
	s_addk_co_i32 s0, 0x2200
	s_set_vgpr_msb 0x52                     ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[256:263]*/, v[0:7] /*v[512:519]*/, v[40:47], v[0:7] /*v[256:263]*/
	s_set_vgpr_msb 0x5200                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_wait_dscnt 0x8
	v_add_lshl_u32 v67, s0, v67, 1
	v_lshlrev_b32_e32 v65, 8, v65
	s_mov_b32 s8, 1
	s_add_nc_u64 s[2:3], s[4:5], s[2:3]
	s_mov_b32 s7, 0
	s_wait_alu depctr_vm_vsrc(6)
	v_add_nc_u32_e32 v68, 0x19800, v67
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x80                     ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[64:67] /*v[576:579]*/, v68
	ds_load_b128 v[68:71] /*v[580:583]*/, v68 offset:32
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[264:271]*/, v[16:23] /*v[528:535]*/, v[40:47], v[8:15] /*v[264:271]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[72:75] /*v[584:587]*/, v68 offset:64
	ds_load_b128 v[76:79] /*v[588:591]*/, v68 offset:96
	s_set_vgpr_msb 0x8000                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_or3_b32 v65, v65, s1, v66
	s_add_co_i32 s1, s9, 0x19800
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x52                     ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[0:7] /*v[512:519]*/, v[56:63], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[80:83] /*v[592:595]*/, v68 offset:4352
	ds_load_b128 v[84:87] /*v[596:599]*/, v68 offset:4384
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[296:303]*/, v[16:23] /*v[528:535]*/, v[56:63], v[40:47] /*v[296:303]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[88:91] /*v[600:603]*/, v68 offset:4416
	ds_load_b128 v[92:95] /*v[604:607]*/, v68 offset:4448
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[256:263]*/, v[8:15] /*v[520:527]*/, v[32:39], v[0:7] /*v[256:263]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[96:99] /*v[608:611]*/, v68 offset:8704
	ds_load_b128 v[100:103] /*v[612:615]*/, v68 offset:8736
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[264:271]*/, v[24:31] /*v[536:543]*/, v[32:39], v[8:15] /*v[264:271]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[104:107] /*v[616:619]*/, v68 offset:8768
	ds_load_b128 v[108:111] /*v[620:623]*/, v68 offset:8800
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[8:15] /*v[520:527]*/, v[48:55], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[112:115] /*v[624:627]*/, v68 offset:13056
	ds_load_b128 v[116:119] /*v[628:631]*/, v68 offset:13088
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[296:303]*/, v[24:31] /*v[536:543]*/, v[48:55], v[40:47] /*v[296:303]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[120:123] /*v[632:635]*/, v68 offset:13120
	ds_load_b128 v[124:127] /*v[636:639]*/, v68 offset:13152
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[272:279]*/, v[32:39] /*v[544:551]*/, v[40:47], v[16:23] /*v[272:279]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[48:55] /*v[560:567]*/, v[40:47], v[24:31] /*v[280:287]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[304:311]*/, v[32:39] /*v[544:551]*/, v[56:63], v[48:55] /*v[304:311]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[56:63] /*v[312:319]*/, v[48:55] /*v[560:567]*/, v[56:63], v[56:63] /*v[312:319]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[272:279]*/, v[40:47] /*v[552:559]*/, v[32:39], v[16:23] /*v[272:279]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[56:63] /*v[568:575]*/, v[32:39], v[24:31] /*v[280:287]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[304:311]*/, v[40:47] /*v[552:559]*/, v[48:55], v[48:55] /*v[304:311]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[56:63] /*v[312:319]*/, v[56:63] /*v[568:575]*/, v[48:55], v[56:63] /*v[312:319]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[320:327]*/, v[64:71] /*v[576:583]*/, v[40:47], v[64:71] /*v[320:327]*/
	s_wait_dscnt 0x10
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[0:3] /*v[512:515]*/, v68 offset:17408
	ds_load_b128 v[4:7] /*v[516:519]*/, v68 offset:17440
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[328:335]*/, v[80:87] /*v[592:599]*/, v[40:47], v[72:79] /*v[328:335]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[8:11] /*v[520:523]*/, v68 offset:17472
	ds_load_b128 v[12:15] /*v[524:527]*/, v68 offset:17504
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[352:359]*/, v[64:71] /*v[576:583]*/, v[56:63], v[96:103] /*v[352:359]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[16:19] /*v[528:531]*/, v68 offset:21760
	ds_load_b128 v[20:23] /*v[532:535]*/, v68 offset:21792
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[360:367]*/, v[80:87] /*v[592:599]*/, v[56:63], v[104:111] /*v[360:367]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[24:27] /*v[536:539]*/, v68 offset:21824
	ds_load_b128 v[28:31] /*v[540:543]*/, v68 offset:21856
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[320:327]*/, v[72:79] /*v[584:591]*/, v[32:39], v[64:71] /*v[320:327]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[32:35] /*v[544:547]*/, v68 offset:26112
	ds_load_b128 v[36:39] /*v[548:551]*/, v68 offset:26144
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[328:335]*/, v[88:95] /*v[600:607]*/, v[32:39], v[72:79] /*v[328:335]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[40:43] /*v[552:555]*/, v68 offset:26176
	ds_load_b128 v[44:47] /*v[556:559]*/, v68 offset:26208
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[352:359]*/, v[72:79] /*v[584:591]*/, v[48:55], v[96:103] /*v[352:359]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[48:51] /*v[560:563]*/, v68 offset:30464
	ds_load_b128 v[52:55] /*v[564:567]*/, v68 offset:30496
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[360:367]*/, v[88:95] /*v[600:607]*/, v[48:55], v[104:111] /*v[360:367]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[56:59] /*v[568:571]*/, v68 offset:30528
	ds_load_b128 v[60:63] /*v[572:575]*/, v68 offset:30560
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[336:343]*/, v[96:103] /*v[608:615]*/, v[40:47], v[80:87] /*v[336:343]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[344:351]*/, v[112:119] /*v[624:631]*/, v[40:47], v[88:95] /*v[344:351]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[368:375]*/, v[96:103] /*v[608:615]*/, v[56:63], v[112:119] /*v[368:375]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[376:383]*/, v[112:119] /*v[624:631]*/, v[56:63], v[120:127] /*v[376:383]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[336:343]*/, v[104:111] /*v[616:623]*/, v[32:39], v[80:87] /*v[336:343]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[344:351]*/, v[120:127] /*v[632:639]*/, v[32:39], v[88:95] /*v[344:351]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[368:375]*/, v[104:111] /*v[616:623]*/, v[48:55], v[112:119] /*v[368:375]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[376:383]*/, v[120:127] /*v[632:639]*/, v[48:55], v[120:127] /*v[376:383]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[384:391]*/, v[0:7] /*v[512:519]*/, v[40:47], v[128:135] /*v[384:391]*/
	s_wait_dscnt 0x10
	s_wait_alu depctr_va_vdst(14)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[64:67] /*v[576:579]*/, v68 offset:34816
	ds_load_b128 v[68:71] /*v[580:583]*/, v68 offset:34848
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[392:399]*/, v[16:23] /*v[528:535]*/, v[40:47], v[136:143] /*v[392:399]*/
	s_wait_alu depctr_va_vdst(11)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[72:75] /*v[584:587]*/, v68 offset:34880
	ds_load_b128 v[76:79] /*v[588:591]*/, v68 offset:34912
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[416:423]*/, v[0:7] /*v[512:519]*/, v[56:63], v[160:167] /*v[416:423]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[80:83] /*v[592:595]*/, v68 offset:39168
	ds_load_b128 v[84:87] /*v[596:599]*/, v68 offset:39200
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[424:431]*/, v[16:23] /*v[528:535]*/, v[56:63], v[168:175] /*v[424:431]*/
	s_wait_alu depctr_va_vdst(12)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[88:91] /*v[600:603]*/, v68 offset:39232
	ds_load_b128 v[92:95] /*v[604:607]*/, v68 offset:39264
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[384:391]*/, v[8:15] /*v[520:527]*/, v[32:39], v[128:135] /*v[384:391]*/
	s_wait_alu depctr_va_vdst(10)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[96:99] /*v[608:611]*/, v68 offset:43520
	ds_load_b128 v[100:103] /*v[612:615]*/, v68 offset:43552
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[392:399]*/, v[24:31] /*v[536:543]*/, v[32:39], v[136:143] /*v[392:399]*/
	s_wait_alu depctr_va_vdst(7)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[104:107] /*v[616:619]*/, v68 offset:43584
	ds_load_b128 v[108:111] /*v[620:623]*/, v68 offset:43616
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[416:423]*/, v[8:15] /*v[520:527]*/, v[48:55], v[160:167] /*v[416:423]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[112:115] /*v[624:627]*/, v68 offset:47872
	ds_load_b128 v[116:119] /*v[628:631]*/, v68 offset:47904
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[424:431]*/, v[24:31] /*v[536:543]*/, v[48:55], v[168:175] /*v[424:431]*/
	s_wait_alu depctr_va_vdst(8)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[120:123] /*v[632:635]*/, v68 offset:47936
	ds_load_b128 v[124:127] /*v[636:639]*/, v68 offset:47968
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[400:407]*/, v[32:39] /*v[544:551]*/, v[40:47], v[144:151] /*v[400:407]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[408:415]*/, v[48:55] /*v[560:567]*/, v[40:47], v[152:159] /*v[408:415]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[176:183] /*v[432:439]*/, v[32:39] /*v[544:551]*/, v[56:63], v[176:183] /*v[432:439]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[184:191] /*v[440:447]*/, v[48:55] /*v[560:567]*/, v[56:63], v[184:191] /*v[440:447]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[400:407]*/, v[40:47] /*v[552:559]*/, v[32:39], v[144:151] /*v[400:407]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[408:415]*/, v[56:63] /*v[568:575]*/, v[32:39], v[152:159] /*v[408:415]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[176:183] /*v[432:439]*/, v[40:47] /*v[552:559]*/, v[48:55], v[176:183] /*v[432:439]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[184:191] /*v[440:447]*/, v[56:63] /*v[568:575]*/, v[48:55], v[184:191] /*v[440:447]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x5200                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_add_nc_u32_e32 v66, 0x15480, v67
	s_set_vgpr_msb 0x52                     ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[64:71] /*v[576:583]*/, v[40:47], v[192:199] /*v[448:455]*/
	s_wait_dscnt 0x10
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[0:3] /*v[512:515]*/, v66
	ds_load_b128 v[4:7] /*v[516:519]*/, v66 offset:32
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[80:87] /*v[592:599]*/, v[40:47], v[200:207] /*v[456:463]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[8:11] /*v[520:523]*/, v66 offset:64
	ds_load_b128 v[12:15] /*v[524:527]*/, v66 offset:96
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[64:71] /*v[576:583]*/, v[56:63], v[224:231] /*v[480:487]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[16:19] /*v[528:531]*/, v66 offset:4352
	ds_load_b128 v[20:23] /*v[532:535]*/, v66 offset:4384
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[80:87] /*v[592:599]*/, v[56:63], v[232:239] /*v[488:495]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[24:27] /*v[536:539]*/, v66 offset:4416
	ds_load_b128 v[28:31] /*v[540:543]*/, v66 offset:4448
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[72:79] /*v[584:591]*/, v[32:39], v[192:199] /*v[448:455]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[32:35] /*v[544:547]*/, v66 offset:8704
	ds_load_b128 v[36:39] /*v[548:551]*/, v66 offset:8736
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[88:95] /*v[600:607]*/, v[32:39], v[200:207] /*v[456:463]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[40:43] /*v[552:555]*/, v66 offset:8768
	ds_load_b128 v[44:47] /*v[556:559]*/, v66 offset:8800
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[72:79] /*v[584:591]*/, v[48:55], v[224:231] /*v[480:487]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[48:51] /*v[560:563]*/, v66 offset:13056
	ds_load_b128 v[52:55] /*v[564:567]*/, v66 offset:13088
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[88:95] /*v[600:607]*/, v[48:55], v[232:239] /*v[488:495]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[56:59] /*v[568:571]*/, v66 offset:13120
	ds_load_b128 v[60:63] /*v[572:575]*/, v66 offset:13152
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[96:103] /*v[608:615]*/, v[40:47], v[208:215] /*v[464:471]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[112:119] /*v[624:631]*/, v[40:47], v[216:223] /*v[472:479]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[96:103] /*v[608:615]*/, v[56:63], v[240:247] /*v[496:503]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[112:119] /*v[624:631]*/, v[56:63], v[248:255] /*v[504:511]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[104:111] /*v[616:623]*/, v[32:39], v[208:215] /*v[464:471]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[120:127] /*v[632:639]*/, v[32:39], v[216:223] /*v[472:479]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[104:111] /*v[616:623]*/, v[48:55], v[240:247] /*v[496:503]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[120:127] /*v[632:639]*/, v[48:55], v[248:255] /*v[504:511]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[256:263]*/, v[0:7] /*v[512:519]*/, v[24:31], v[0:7] /*v[256:263]*/
	s_wait_dscnt 0x10
	s_wait_alu depctr_va_vdst(14)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[64:67] /*v[576:579]*/, v68 offset:128
	ds_load_b128 v[68:71] /*v[580:583]*/, v68 offset:160
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[264:271]*/, v[16:23] /*v[528:535]*/, v[24:31], v[8:15] /*v[264:271]*/
	s_wait_alu depctr_va_vdst(11)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[72:75] /*v[584:587]*/, v68 offset:192
	ds_load_b128 v[76:79] /*v[588:591]*/, v68 offset:224
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[0:7] /*v[512:519]*/, v[16:23], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[80:83] /*v[592:595]*/, v68 offset:4480
	ds_load_b128 v[84:87] /*v[596:599]*/, v68 offset:4512
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[296:303]*/, v[16:23] /*v[528:535]*/, v[16:23], v[40:47] /*v[296:303]*/
	s_wait_alu depctr_va_vdst(12)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[88:91] /*v[600:603]*/, v68 offset:4544
	ds_load_b128 v[92:95] /*v[604:607]*/, v68 offset:4576
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[256:263]*/, v[8:15] /*v[520:527]*/, v[8:15], v[0:7] /*v[256:263]*/
	s_wait_alu depctr_va_vdst(10)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[96:99] /*v[608:611]*/, v68 offset:8832
	ds_load_b128 v[100:103] /*v[612:615]*/, v68 offset:8864
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[264:271]*/, v[24:31] /*v[536:543]*/, v[8:15], v[8:15] /*v[264:271]*/
	s_wait_alu depctr_va_vdst(7)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[104:107] /*v[616:619]*/, v68 offset:8896
	ds_load_b128 v[108:111] /*v[620:623]*/, v68 offset:8928
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[8:15] /*v[520:527]*/, v[0:7], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[112:115] /*v[624:627]*/, v68 offset:13184
	ds_load_b128 v[116:119] /*v[628:631]*/, v68 offset:13216
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[296:303]*/, v[24:31] /*v[536:543]*/, v[0:7], v[40:47] /*v[296:303]*/
	s_wait_alu depctr_va_vdst(8)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[120:123] /*v[632:635]*/, v68 offset:13248
	ds_load_b128 v[124:127] /*v[636:639]*/, v68 offset:13280
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[272:279]*/, v[32:39] /*v[544:551]*/, v[24:31], v[16:23] /*v[272:279]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[48:55] /*v[560:567]*/, v[24:31], v[24:31] /*v[280:287]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[304:311]*/, v[32:39] /*v[544:551]*/, v[16:23], v[48:55] /*v[304:311]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[56:63] /*v[312:319]*/, v[48:55] /*v[560:567]*/, v[16:23], v[56:63] /*v[312:319]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[272:279]*/, v[40:47] /*v[552:559]*/, v[8:15], v[16:23] /*v[272:279]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[56:63] /*v[568:575]*/, v[8:15], v[24:31] /*v[280:287]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[304:311]*/, v[40:47] /*v[552:559]*/, v[0:7], v[48:55] /*v[304:311]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[56:63] /*v[312:319]*/, v[56:63] /*v[568:575]*/, v[0:7], v[56:63] /*v[312:319]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[320:327]*/, v[64:71] /*v[576:583]*/, v[24:31], v[64:71] /*v[320:327]*/
	s_wait_dscnt 0x10
	s_wait_alu depctr_va_vdst(14)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[0:3] /*v[512:515]*/, v68 offset:17536
	ds_load_b128 v[4:7] /*v[516:519]*/, v68 offset:17568
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[328:335]*/, v[80:87] /*v[592:599]*/, v[24:31], v[72:79] /*v[328:335]*/
	s_wait_alu depctr_va_vdst(11)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[8:11] /*v[520:523]*/, v68 offset:17600
	ds_load_b128 v[12:15] /*v[524:527]*/, v68 offset:17632
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[352:359]*/, v[64:71] /*v[576:583]*/, v[16:23], v[96:103] /*v[352:359]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[16:19] /*v[528:531]*/, v68 offset:21888
	ds_load_b128 v[20:23] /*v[532:535]*/, v68 offset:21920
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[360:367]*/, v[80:87] /*v[592:599]*/, v[16:23], v[104:111] /*v[360:367]*/
	s_wait_alu depctr_va_vdst(12)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[24:27] /*v[536:539]*/, v68 offset:21952
	ds_load_b128 v[28:31] /*v[540:543]*/, v68 offset:21984
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[320:327]*/, v[72:79] /*v[584:591]*/, v[8:15], v[64:71] /*v[320:327]*/
	s_wait_alu depctr_va_vdst(10)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[32:35] /*v[544:547]*/, v68 offset:26240
	ds_load_b128 v[36:39] /*v[548:551]*/, v68 offset:26272
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[328:335]*/, v[88:95] /*v[600:607]*/, v[8:15], v[72:79] /*v[328:335]*/
	s_wait_alu depctr_va_vdst(7)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[40:43] /*v[552:555]*/, v68 offset:26304
	ds_load_b128 v[44:47] /*v[556:559]*/, v68 offset:26336
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[352:359]*/, v[72:79] /*v[584:591]*/, v[0:7], v[96:103] /*v[352:359]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[48:51] /*v[560:563]*/, v68 offset:30592
	ds_load_b128 v[52:55] /*v[564:567]*/, v68 offset:30624
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[360:367]*/, v[88:95] /*v[600:607]*/, v[0:7], v[104:111] /*v[360:367]*/
	s_wait_alu depctr_va_vdst(8)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[56:59] /*v[568:571]*/, v68 offset:30656
	ds_load_b128 v[60:63] /*v[572:575]*/, v68 offset:30688
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x8005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_cvt_pk_bf16_f32 v32, v6 /*v262*/, v7 /*v263*/
	v_cvt_pk_bf16_f32 v33, v4 /*v260*/, v5 /*v261*/
	v_cvt_pk_bf16_f32 v34, v2 /*v258*/, v3 /*v259*/
	v_cvt_pk_bf16_f32 v35, v0 /*v256*/, v1 /*v257*/
	s_set_vgpr_msb 0x552                    ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[336:343]*/, v[96:103] /*v[608:615]*/, v[24:31], v[80:87] /*v[336:343]*/
	s_mov_b32 s4, 32
	s_set_vgpr_msb 0x5205                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_cvt_pk_bf16_f32 v36, v14 /*v270*/, v15 /*v271*/
	v_cvt_pk_bf16_f32 v37, v12 /*v268*/, v13 /*v269*/
	; sched_group_barrier mask(0x00000002) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_cvt_pk_bf16_f32 v38, v10 /*v266*/, v11 /*v267*/
	v_cvt_pk_bf16_f32 v39, v8 /*v264*/, v9 /*v265*/
	v_cvt_pk_bf16_f32 v40, v22 /*v278*/, v23 /*v279*/
	v_cvt_pk_bf16_f32 v41, v20 /*v276*/, v21 /*v277*/
	v_cvt_pk_bf16_f32 v42, v18 /*v274*/, v19 /*v275*/
	v_cvt_pk_bf16_f32 v43, v16 /*v272*/, v17 /*v273*/
	v_cvt_pk_bf16_f32 v44, v30 /*v286*/, v31 /*v287*/
	v_cvt_pk_bf16_f32 v45, v28 /*v284*/, v29 /*v285*/
	v_cvt_pk_bf16_f32 v46, v26 /*v282*/, v27 /*v283*/
	v_cvt_pk_bf16_f32 v47, v24 /*v280*/, v25 /*v281*/
	s_set_vgpr_msb 0x500                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_lshlrev_b32 v60, 4, v64 :: v_dual_bitop2_b32 v48, 1, v64 bitop3:0x40
	v_cmp_eq_u32_e64 s0, 1, v48
	v_dual_cndmask_b32 v48, v35, v39, s0 :: v_dual_cndmask_b32 v49, v34, v38, s0
	v_dual_cndmask_b32 v50, v33, v37, s0 :: v_dual_cndmask_b32 v51, v32, v36, s0
	v_dual_cndmask_b32 v52, v39, v43, s0 :: v_dual_cndmask_b32 v53, v38, v42, s0
	v_cndmask_b32_e64 v55, v36, v40, s0
	v_dual_cndmask_b32 v54, v37, v41, s0 :: v_dual_bitop2_b32 v36, 2, v64 bitop3:0x40
	v_dual_cndmask_b32 v43, v43, v47, s0 :: v_dual_cndmask_b32 v42, v42, v46, s0
	v_cndmask_b32_e64 v56, v41, v45, s0
	v_cmp_eq_u32_e32 vcc_lo, 0, v36
	v_dual_cndmask_b32 v57, v40, v44, s0 :: v_dual_cndmask_b32 v47, v47, v35, s0
	v_dual_cndmask_b32 v58, v45, v33, s0 :: v_dual_cndmask_b32 v59, v44, v32, s0
	v_dual_cndmask_b32 v32, v43, v48 :: v_dual_cndmask_b32 v33, v42, v49
	v_dual_cndmask_b32 v40, v48, v43 :: v_dual_cndmask_b32 v41, v49, v42
	v_dual_add_nc_u32 v48, 16, v60 :: v_dual_bitop2_b32 v45, 48, v60 bitop3:0x40
	v_bitop3_b32 v49, v60, 32, 48 bitop3:0x6c
	v_dual_cndmask_b32 v46, v46, v34, s0 :: v_dual_cndmask_b32 v36, v47, v52, vcc_lo
	v_dual_cndmask_b32 v44, v52, v47, vcc_lo :: v_dual_bitop2_b32 v47, v65, v45 bitop3:0x54
	v_and_or_b32 v48, v48, 48, v65
	v_dual_cndmask_b32 v34, v56, v50, vcc_lo :: v_dual_bitop2_b32 v49, v65, v49 bitop3:0x54
	v_cndmask_b32_e32 v37, v46, v53, vcc_lo
	v_cndmask_b32_e32 v42, v50, v56, vcc_lo
	v_dual_cndmask_b32 v45, v53, v46 :: v_dual_add_nc_u32 v50, 48, v60
	v_dual_lshlrev_b32 v52, 1, v47 :: v_dual_lshlrev_b32 v53, 1, v48
	v_dual_cndmask_b32 v35, v57, v51 :: v_dual_lshlrev_b32 v56, 1, v49
	v_dual_cndmask_b32 v38, v58, v54 :: v_dual_cndmask_b32 v39, v59, v55
	v_cndmask_b32_e32 v43, v51, v57, vcc_lo
	v_and_or_b32 v47, v50, 48, v65
	v_dual_add_nc_u32 v48, s1, v52 :: v_dual_add_nc_u32 v49, s1, v53
	v_add_nc_u32_e32 v50, s1, v56
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_cvt_pk_bf16_f32 v51, v56 /*v312*/, v57 /*v313*/
	s_set_vgpr_msb 0x500                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_cndmask_b32_e32 v46, v54, v58, vcc_lo
	s_wait_alu depctr_va_vdst(0)
	ds_store_b128 v48, v[32:35]
	ds_store_b128 v49, v[36:39]
	ds_store_b128 v50, v[40:43]
	s_wait_alu depctr_vm_vsrc(2)
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_cvt_pk_bf16_f32 v35, v32 /*v288*/, v33 /*v289*/
	s_wait_alu depctr_vm_vsrc(1)
	v_cvt_pk_bf16_f32 v39, v40 /*v296*/, v41 /*v297*/
	s_wait_alu depctr_vm_vsrc(0)
	v_cvt_pk_bf16_f32 v43, v48 /*v304*/, v49 /*v305*/
	v_cvt_pk_bf16_f32 v32, v38 /*v294*/, v39 /*v295*/
	v_cvt_pk_bf16_f32 v33, v36 /*v292*/, v37 /*v293*/
	v_cvt_pk_bf16_f32 v34, v34 /*v290*/, v35 /*v291*/
	v_cvt_pk_bf16_f32 v36, v46 /*v302*/, v47 /*v303*/
	v_cvt_pk_bf16_f32 v37, v44 /*v300*/, v45 /*v301*/
	v_cvt_pk_bf16_f32 v38, v42 /*v298*/, v43 /*v299*/
	v_cvt_pk_bf16_f32 v40, v54 /*v310*/, v55 /*v311*/
	v_cvt_pk_bf16_f32 v41, v52 /*v308*/, v53 /*v309*/
	v_cvt_pk_bf16_f32 v42, v50 /*v306*/, v51 /*v307*/
	v_cvt_pk_bf16_f32 v48, v62 /*v318*/, v63 /*v319*/
	v_cvt_pk_bf16_f32 v49, v60 /*v316*/, v61 /*v317*/
	v_cvt_pk_bf16_f32 v50, v58 /*v314*/, v59 /*v315*/
	s_set_vgpr_msb 0x500                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_cndmask_b32_e64 v62, v39, v43, s0
	v_cndmask_b32_e64 v43, v43, v51, s0
	v_cndmask_b32_e64 v51, v51, v35, s0
	v_lshlrev_b32_e32 v54, 1, v47
	v_cndmask_b32_e32 v47, v55, v59, vcc_lo
	v_dual_cndmask_b32 v57, v35, v39, s0 :: v_dual_cndmask_b32 v58, v34, v38, s0
	v_dual_cndmask_b32 v59, v33, v37, s0 :: v_dual_cndmask_b32 v61, v32, v36, s0
	v_dual_cndmask_b32 v63, v38, v42, s0 :: v_dual_cndmask_b32 v66, v36, v40, s0
	v_dual_cndmask_b32 v42, v42, v50, s0 :: v_dual_cndmask_b32 v67, v41, v49, s0
	v_cndmask_b32_e64 v69, v40, v48, s0
	v_dual_cndmask_b32 v71, v48, v32, s0 :: v_dual_cndmask_b32 v36, v51, v62, vcc_lo
	v_cndmask_b32_e32 v48, v62, v51, vcc_lo
	v_bitop3_b32 v51, v60, 0x4020, 48 bitop3:0x6c
	v_dual_cndmask_b32 v64, v37, v41, s0 :: v_dual_cndmask_b32 v50, v50, v34, s0
	v_cndmask_b32_e64 v70, v49, v33, s0
	v_dual_cndmask_b32 v32, v43, v57 :: v_dual_cndmask_b32 v33, v42, v58
	v_cndmask_b32_e32 v34, v67, v59, vcc_lo
	v_dual_cndmask_b32 v40, v57, v43 :: v_dual_cndmask_b32 v41, v58, v42
	v_dual_cndmask_b32 v42, v59, v67 :: v_dual_add_nc_u32 v57, 0x8000, v52
	v_add_nc_u32_e32 v58, 0x8000, v53
	v_add_lshl_u32 v59, v65, v51, 1
	v_add_nc_u32_e32 v60, 0x8000, v54
	v_dual_add_nc_u32 v55, s1, v54 :: v_dual_cndmask_b32 v35, v69, v61, vcc_lo
	v_cndmask_b32_e32 v37, v50, v63, vcc_lo
	v_dual_cndmask_b32 v38, v70, v64 :: v_dual_cndmask_b32 v39, v71, v66
	v_dual_cndmask_b32 v43, v61, v69, vcc_lo :: v_dual_cndmask_b32 v49, v63, v50, vcc_lo
	v_dual_cndmask_b32 v50, v64, v70 :: v_dual_cndmask_b32 v51, v66, v71
	v_dual_add_nc_u32 v61, s1, v57 :: v_dual_add_nc_u32 v62, s1, v58
	v_dual_add_nc_u32 v63, s1, v59 :: v_dual_add_nc_u32 v64, s1, v60
	s_wait_alu depctr_va_vdst(6)
	ds_store_b128 v55, v[44:47]
	s_wait_alu depctr_va_vdst(1)
	ds_store_b128 v61, v[32:35]
	ds_store_b128 v62, v[36:39]
	s_wait_alu depctr_va_vdst(0)
	ds_store_b128 v63, v[40:43]
	ds_store_b128 v64, v[48:51]
	s_set_vgpr_msb 0x52                     ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[368:375]*/, v[96:103] /*v[608:615]*/, v[16:23], v[112:119] /*v[368:375]*/
	; sched_group_barrier mask(0x00000200) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000200) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(4) SyncID(0)
	s_wait_dscnt 0x1c
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[368:375]*/, v[104:111] /*v[616:623]*/, v[0:7], v[112:119] /*v[368:375]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000200) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(4) SyncID(0)
	s_wait_dscnt 0x1a
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[344:351]*/, v[112:119] /*v[624:631]*/, v[24:31], v[88:95] /*v[344:351]*/
	s_wait_dscnt 0x18
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[344:351]*/, v[120:127] /*v[632:639]*/, v[8:15], v[88:95] /*v[344:351]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000200) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000200) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000200) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(4) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[376:383]*/, v[112:119] /*v[624:631]*/, v[16:23], v[120:127] /*v[376:383]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000200) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(4) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[336:343]*/, v[104:111] /*v[616:623]*/, v[8:15], v[80:87] /*v[336:343]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000200) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[376:383]*/, v[120:127] /*v[632:639]*/, v[0:7], v[120:127] /*v[376:383]*/
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[384:391]*/, v[0:7] /*v[512:519]*/, v[24:31], v[128:135] /*v[384:391]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	s_wait_dscnt 0x10
	ds_load_b128 v[64:67] /*v[576:579]*/, v68 offset:34944
	ds_load_b128 v[68:71] /*v[580:583]*/, v68 offset:34976
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[392:399]*/, v[16:23] /*v[528:535]*/, v[24:31], v[136:143] /*v[392:399]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[72:75] /*v[584:587]*/, v68 offset:35008
	ds_load_b128 v[76:79] /*v[588:591]*/, v68 offset:35040
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[416:423]*/, v[0:7] /*v[512:519]*/, v[16:23], v[160:167] /*v[416:423]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[80:83] /*v[592:595]*/, v68 offset:39296
	ds_load_b128 v[84:87] /*v[596:599]*/, v68 offset:39328
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[424:431]*/, v[16:23] /*v[528:535]*/, v[16:23], v[168:175] /*v[424:431]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[88:91] /*v[600:603]*/, v68 offset:39360
	ds_load_b128 v[92:95] /*v[604:607]*/, v68 offset:39392
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[384:391]*/, v[8:15] /*v[520:527]*/, v[8:15], v[128:135] /*v[384:391]*/
	s_wait_alu depctr_va_vdst(11)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[96:99] /*v[608:611]*/, v68 offset:43648
	ds_load_b128 v[100:103] /*v[612:615]*/, v68 offset:43680
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[392:399]*/, v[24:31] /*v[536:543]*/, v[8:15], v[136:143] /*v[392:399]*/
	s_wait_alu depctr_va_vdst(7)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[104:107] /*v[616:619]*/, v68 offset:43712
	ds_load_b128 v[108:111] /*v[620:623]*/, v68 offset:43744
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[416:423]*/, v[8:15] /*v[520:527]*/, v[0:7], v[160:167] /*v[416:423]*/
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[112:115] /*v[624:627]*/, v68 offset:48000
	ds_load_b128 v[116:119] /*v[628:631]*/, v68 offset:48032
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	s_set_vgpr_msb 0x8052                   ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[424:431]*/, v[24:31] /*v[536:543]*/, v[0:7], v[168:175] /*v[424:431]*/
	s_wait_alu depctr_va_vdst(8)
	s_set_vgpr_msb 0x5280                   ;  msbs: dst=2 src0=0 src1=0 src2=0
	ds_load_b128 v[120:123] /*v[632:635]*/, v68 offset:48064
	ds_load_b128 v[124:127] /*v[636:639]*/, v68 offset:48096
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_wait_alu depctr_vm_vsrc(6)
	s_set_vgpr_msb 0x8005                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_cvt_pk_bf16_f32 v36, v70 /*v326*/, v71 /*v327*/
	v_cvt_pk_bf16_f32 v37, v68 /*v324*/, v69 /*v325*/
	v_cvt_pk_bf16_f32 v38, v66 /*v322*/, v67 /*v323*/
	v_cvt_pk_bf16_f32 v39, v64 /*v320*/, v65 /*v321*/
	s_set_vgpr_msb 0x552                    ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[400:407]*/, v[32:39] /*v[544:551]*/, v[24:31], v[144:151] /*v[400:407]*/
	s_add_co_i32 s5, s9, 0x19880
	s_set_vgpr_msb 0x5205                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_cvt_pk_bf16_f32 v32, v78 /*v334*/, v79 /*v335*/
	v_cvt_pk_bf16_f32 v33, v76 /*v332*/, v77 /*v333*/
	; sched_group_barrier mask(0x00000002) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_cvt_pk_bf16_f32 v34, v74 /*v330*/, v75 /*v331*/
	v_cvt_pk_bf16_f32 v35, v72 /*v328*/, v73 /*v329*/
	v_cvt_pk_bf16_f32 v40, v86 /*v342*/, v87 /*v343*/
	v_cvt_pk_bf16_f32 v41, v84 /*v340*/, v85 /*v341*/
	v_cvt_pk_bf16_f32 v42, v82 /*v338*/, v83 /*v339*/
	v_cvt_pk_bf16_f32 v43, v80 /*v336*/, v81 /*v337*/
	v_cvt_pk_bf16_f32 v44, v94 /*v350*/, v95 /*v351*/
	v_cvt_pk_bf16_f32 v45, v92 /*v348*/, v93 /*v349*/
	v_cvt_pk_bf16_f32 v46, v90 /*v346*/, v91 /*v347*/
	v_cvt_pk_bf16_f32 v47, v88 /*v344*/, v89 /*v345*/
	s_set_vgpr_msb 0x500                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_cndmask_b32 v48, v39, v35, s0 :: v_dual_cndmask_b32 v49, v38, v34, s0
	v_dual_cndmask_b32 v50, v37, v33, s0 :: v_dual_cndmask_b32 v51, v36, v32, s0
	v_dual_cndmask_b32 v55, v35, v43, s0 :: v_dual_cndmask_b32 v61, v34, v42, s0
	v_dual_cndmask_b32 v62, v33, v41, s0 :: v_dual_cndmask_b32 v63, v32, v40, s0
	v_dual_cndmask_b32 v43, v43, v47, s0 :: v_dual_cndmask_b32 v42, v42, v46, s0
	v_dual_cndmask_b32 v41, v41, v45, s0 :: v_dual_cndmask_b32 v40, v40, v44, s0
	v_add_nc_u32_e32 v64, s5, v52
	v_dual_cndmask_b32 v32, v43, v48 :: v_dual_cndmask_b32 v33, v42, v49
	v_dual_cndmask_b32 v34, v41, v50 :: v_dual_cndmask_b32 v35, v40, v51
	v_dual_cndmask_b32 v47, v47, v39, s0 :: v_dual_cndmask_b32 v46, v46, v38, s0
	v_dual_cndmask_b32 v45, v45, v37, s0 :: v_dual_cndmask_b32 v44, v44, v36, s0
	s_wait_alu depctr_va_vdst(0)
	ds_store_b128 v64, v[32:35]
	s_wait_alu depctr_vm_vsrc(0)
	v_dual_cndmask_b32 v32, v47, v55 :: v_dual_cndmask_b32 v33, v46, v61
	v_dual_cndmask_b32 v34, v45, v62 :: v_dual_cndmask_b32 v35, v44, v63
	v_dual_cndmask_b32 v36, v48, v43 :: v_dual_cndmask_b32 v37, v49, v42
	v_dual_cndmask_b32 v38, v50, v41 :: v_dual_cndmask_b32 v39, v51, v40
	v_dual_cndmask_b32 v40, v55, v47 :: v_dual_add_nc_u32 v47, s5, v53
	v_dual_cndmask_b32 v41, v61, v46 :: v_dual_add_nc_u32 v48, s5, v56
	v_dual_cndmask_b32 v42, v62, v45 :: v_dual_cndmask_b32 v43, v63, v44
	s_wait_alu depctr_va_vdst(2)
	ds_store_b128 v47, v[32:35]
	s_wait_alu depctr_va_vdst(1)
	ds_store_b128 v48, v[36:39]
	s_wait_alu depctr_vm_vsrc(1)
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_cvt_pk_bf16_f32 v32, v102 /*v358*/, v103 /*v359*/
	v_cvt_pk_bf16_f32 v33, v100 /*v356*/, v101 /*v357*/
	v_cvt_pk_bf16_f32 v34, v98 /*v354*/, v99 /*v355*/
	v_cvt_pk_bf16_f32 v35, v96 /*v352*/, v97 /*v353*/
	s_wait_alu depctr_vm_vsrc(0)
	v_cvt_pk_bf16_f32 v36, v110 /*v366*/, v111 /*v367*/
	v_cvt_pk_bf16_f32 v37, v108 /*v364*/, v109 /*v365*/
	v_cvt_pk_bf16_f32 v38, v106 /*v362*/, v107 /*v363*/
	v_cvt_pk_bf16_f32 v39, v104 /*v360*/, v105 /*v361*/
	v_cvt_pk_bf16_f32 v44, v118 /*v374*/, v119 /*v375*/
	v_cvt_pk_bf16_f32 v45, v116 /*v372*/, v117 /*v373*/
	v_cvt_pk_bf16_f32 v46, v114 /*v370*/, v115 /*v371*/
	v_cvt_pk_bf16_f32 v47, v112 /*v368*/, v113 /*v369*/
	v_cvt_pk_bf16_f32 v48, v126 /*v382*/, v127 /*v383*/
	v_cvt_pk_bf16_f32 v49, v124 /*v380*/, v125 /*v381*/
	v_cvt_pk_bf16_f32 v50, v122 /*v378*/, v123 /*v379*/
	v_cvt_pk_bf16_f32 v51, v120 /*v376*/, v121 /*v377*/
	s_set_vgpr_msb 0x500                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_cndmask_b32 v61, v35, v39, s0 :: v_dual_cndmask_b32 v62, v34, v38, s0
	v_dual_cndmask_b32 v63, v33, v37, s0 :: v_dual_cndmask_b32 v64, v32, v36, s0
	v_dual_cndmask_b32 v65, v39, v47, s0 :: v_dual_cndmask_b32 v66, v38, v46, s0
	v_dual_cndmask_b32 v67, v37, v45, s0 :: v_dual_cndmask_b32 v68, v36, v44, s0
	v_dual_cndmask_b32 v47, v47, v51, s0 :: v_dual_cndmask_b32 v46, v46, v50, s0
	v_dual_cndmask_b32 v69, v45, v49, s0 :: v_dual_cndmask_b32 v70, v44, v48, s0
	v_dual_cndmask_b32 v51, v51, v35, s0 :: v_dual_cndmask_b32 v50, v50, v34, s0
	v_dual_cndmask_b32 v71, v49, v33, s0 :: v_dual_cndmask_b32 v72, v48, v32, s0
	v_dual_cndmask_b32 v32, v47, v61 :: v_dual_add_nc_u32 v55, s5, v54
	v_dual_cndmask_b32 v33, v46, v62 :: v_dual_cndmask_b32 v34, v69, v63
	v_dual_cndmask_b32 v35, v70, v64 :: v_dual_cndmask_b32 v36, v51, v65
	v_dual_cndmask_b32 v37, v50, v66 :: v_dual_cndmask_b32 v38, v71, v67
	v_dual_cndmask_b32 v39, v72, v68 :: v_dual_cndmask_b32 v44, v61, v47
	v_dual_cndmask_b32 v45, v62, v46 :: v_dual_cndmask_b32 v46, v63, v69
	v_dual_cndmask_b32 v47, v64, v70 :: v_dual_cndmask_b32 v48, v65, v51
	v_dual_cndmask_b32 v49, v66, v50 :: v_dual_cndmask_b32 v50, v67, v71
	v_dual_cndmask_b32 v51, v68, v72, vcc_lo :: v_dual_add_nc_u32 v61, s5, v57
	v_dual_add_nc_u32 v62, s5, v58 :: v_dual_add_nc_u32 v63, s5, v59
	v_add_nc_u32_e32 v64, s5, v60
	s_wait_alu depctr_va_vdst(10)
	ds_store_b128 v55, v[40:43]
	s_wait_alu depctr_va_vdst(2)
	ds_store_b128 v61, v[32:35]
	s_wait_alu depctr_va_vdst(1)
	ds_store_b128 v62, v[36:39]
	ds_store_b128 v63, v[44:47]
	s_wait_alu depctr_va_vdst(0)
	ds_store_b128 v64, v[48:51]
	s_set_vgpr_msb 0x52                     ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[176:183] /*v[432:439]*/, v[32:39] /*v[544:551]*/, v[16:23], v[176:183] /*v[432:439]*/
	; sched_group_barrier mask(0x00000200) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000200) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(4) SyncID(0)
	s_wait_dscnt 0x24
	v_wmma_f32_16x16x32_bf16 v[176:183] /*v[432:439]*/, v[40:47] /*v[552:559]*/, v[0:7], v[176:183] /*v[432:439]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000200) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(4) SyncID(0)
	s_wait_dscnt 0x22
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[408:415]*/, v[48:55] /*v[560:567]*/, v[24:31], v[152:159] /*v[408:415]*/
	s_wait_dscnt 0x20
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[408:415]*/, v[56:63] /*v[568:575]*/, v[8:15], v[152:159] /*v[408:415]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000200) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000200) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000200) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(4) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[184:191] /*v[440:447]*/, v[48:55] /*v[560:567]*/, v[16:23], v[184:191] /*v[440:447]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000200) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(4) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[400:407]*/, v[40:47] /*v[552:559]*/, v[8:15], v[144:151] /*v[400:407]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000200) size(1) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[184:191] /*v[440:447]*/, v[56:63] /*v[568:575]*/, v[0:7], v[184:191] /*v[440:447]*/
	; sched_barrier mask(0x00000000)
	s_wait_alu depctr_vm_vsrc(3)
	s_set_vgpr_msb 0x5205                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_cvt_pk_bf16_f32 v32, v134 /*v390*/, v135 /*v391*/
	v_cvt_pk_bf16_f32 v33, v132 /*v388*/, v133 /*v389*/
	s_set_vgpr_msb 0x552                    ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[64:71] /*v[576:583]*/, v[24:31], v[192:199] /*v[448:455]*/
	s_add_co_i32 s5, s9, 0x19900
	s_set_vgpr_msb 0x5205                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_cvt_pk_bf16_f32 v34, v130 /*v386*/, v131 /*v387*/
	v_cvt_pk_bf16_f32 v35, v128 /*v384*/, v129 /*v385*/
	s_wait_dscnt 0x10
	; sched_group_barrier mask(0x00000002) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000200) size(0) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(2) SyncID(0)
	s_set_vgpr_msb 0x552                    ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[80:87] /*v[592:599]*/, v[24:31], v[200:207] /*v[456:463]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000200) size(0) SyncID(0)
	s_wait_alu depctr_vm_vsrc(2)
	s_set_vgpr_msb 0x5205                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_cvt_pk_bf16_f32 v36, v142 /*v398*/, v143 /*v399*/
	v_cvt_pk_bf16_f32 v37, v140 /*v396*/, v141 /*v397*/
	; sched_group_barrier mask(0x00000002) size(2) SyncID(0)
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x500                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_cndmask_b32_e64 v51, v32, v36, s0
	s_set_vgpr_msb 0x52                     ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[64:71] /*v[576:583]*/, v[16:23], v[224:231] /*v[480:487]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000200) size(0) SyncID(0)
	s_set_vgpr_msb 0x5205                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_cvt_pk_bf16_f32 v38, v138 /*v394*/, v139 /*v395*/
	v_cvt_pk_bf16_f32 v39, v136 /*v392*/, v137 /*v393*/
	; sched_group_barrier mask(0x00000002) size(2) SyncID(0)
	s_set_vgpr_msb 0x500                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_cndmask_b32 v50, v33, v37, s0 :: v_dual_cndmask_b32 v49, v34, v38, s0
	s_set_vgpr_msb 0x52                     ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[80:87] /*v[592:599]*/, v[16:23], v[232:239] /*v[488:495]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000200) size(0) SyncID(0)
	s_set_vgpr_msb 0x5205                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_cvt_pk_bf16_f32 v40, v150 /*v406*/, v151 /*v407*/
	v_cvt_pk_bf16_f32 v41, v148 /*v404*/, v149 /*v405*/
	; sched_group_barrier mask(0x00000002) size(2) SyncID(0)
	s_set_vgpr_msb 0x500                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_cndmask_b32 v48, v35, v39, s0 :: v_dual_cndmask_b32 v63, v36, v40, s0
	s_set_vgpr_msb 0x52                     ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[72:79] /*v[584:591]*/, v[8:15], v[192:199] /*v[448:455]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000200) size(0) SyncID(0)
	s_set_vgpr_msb 0x5205                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_cvt_pk_bf16_f32 v42, v146 /*v402*/, v147 /*v403*/
	v_cvt_pk_bf16_f32 v43, v144 /*v400*/, v145 /*v401*/
	; sched_group_barrier mask(0x00000002) size(2) SyncID(0)
	s_set_vgpr_msb 0x500                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_cndmask_b32 v62, v37, v41, s0 :: v_dual_cndmask_b32 v61, v38, v42, s0
	s_set_vgpr_msb 0x52                     ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[88:95] /*v[600:607]*/, v[8:15], v[200:207] /*v[456:463]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000200) size(0) SyncID(0)
	s_set_vgpr_msb 0x5205                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_cvt_pk_bf16_f32 v44, v158 /*v414*/, v159 /*v415*/
	v_cvt_pk_bf16_f32 v45, v156 /*v412*/, v157 /*v413*/
	; sched_group_barrier mask(0x00000002) size(2) SyncID(0)
	s_set_vgpr_msb 0x500                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_cndmask_b32 v55, v39, v43, s0 :: v_dual_cndmask_b32 v65, v40, v44, s0
	s_set_vgpr_msb 0x52                     ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[72:79] /*v[584:591]*/, v[0:7], v[224:231] /*v[480:487]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000200) size(0) SyncID(0)
	s_set_vgpr_msb 0x5205                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_cvt_pk_bf16_f32 v46, v154 /*v410*/, v155 /*v411*/
	v_cvt_pk_bf16_f32 v47, v152 /*v408*/, v153 /*v409*/
	; sched_group_barrier mask(0x00000002) size(2) SyncID(0)
	s_set_vgpr_msb 0x500                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_cndmask_b32 v64, v41, v45, s0 :: v_dual_cndmask_b32 v42, v42, v46, s0
	s_set_vgpr_msb 0x52                     ;  msbs: dst=1 src0=2 src1=0 src2=1
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[88:95] /*v[600:607]*/, v[0:7], v[232:239] /*v[488:495]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000200) size(0) SyncID(0)
	s_set_vgpr_msb 0x5200                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_cndmask_b32_e64 v43, v43, v47, s0
	v_dual_cndmask_b32 v47, v47, v35, s0 :: v_dual_cndmask_b32 v46, v46, v34, s0
	v_dual_cndmask_b32 v66, v45, v33, s0 :: v_dual_cndmask_b32 v67, v44, v32, s0
	v_dual_cndmask_b32 v32, v43, v48 :: v_dual_cndmask_b32 v33, v42, v49
	v_dual_cndmask_b32 v34, v64, v50 :: v_dual_cndmask_b32 v35, v65, v51
	v_dual_cndmask_b32 v36, v47, v55 :: v_dual_cndmask_b32 v37, v46, v61
	v_dual_cndmask_b32 v38, v66, v62 :: v_dual_cndmask_b32 v39, v67, v63
	v_dual_cndmask_b32 v40, v48, v43 :: v_dual_cndmask_b32 v41, v49, v42
	v_dual_cndmask_b32 v42, v50, v64 :: v_dual_cndmask_b32 v43, v51, v65
	v_dual_add_nc_u32 v48, s5, v52 :: v_dual_add_nc_u32 v49, s5, v53
	v_add_nc_u32_e32 v50, s5, v56
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_cvt_pk_bf16_f32 v51, v184 /*v440*/, v185 /*v441*/
	s_set_vgpr_msb 0x500                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_cndmask_b32_e32 v44, v55, v47, vcc_lo
	s_wait_alu depctr_va_vdst(0)
	ds_store_b128 v48, v[32:35]
	ds_store_b128 v49, v[36:39]
	ds_store_b128 v50, v[40:43]
	s_wait_alu depctr_vm_vsrc(2)
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_cvt_pk_bf16_f32 v32, v166 /*v422*/, v167 /*v423*/
	v_cvt_pk_bf16_f32 v33, v164 /*v420*/, v165 /*v421*/
	v_cvt_pk_bf16_f32 v34, v162 /*v418*/, v163 /*v419*/
	v_cvt_pk_bf16_f32 v35, v160 /*v416*/, v161 /*v417*/
	s_wait_alu depctr_vm_vsrc(1)
	v_cvt_pk_bf16_f32 v36, v174 /*v430*/, v175 /*v431*/
	v_cvt_pk_bf16_f32 v37, v172 /*v428*/, v173 /*v429*/
	v_cvt_pk_bf16_f32 v38, v170 /*v426*/, v171 /*v427*/
	v_cvt_pk_bf16_f32 v39, v168 /*v424*/, v169 /*v425*/
	s_wait_alu depctr_vm_vsrc(0)
	v_cvt_pk_bf16_f32 v40, v182 /*v438*/, v183 /*v439*/
	v_cvt_pk_bf16_f32 v41, v180 /*v436*/, v181 /*v437*/
	v_cvt_pk_bf16_f32 v42, v178 /*v434*/, v179 /*v435*/
	v_cvt_pk_bf16_f32 v43, v176 /*v432*/, v177 /*v433*/
	v_cvt_pk_bf16_f32 v48, v190 /*v446*/, v191 /*v447*/
	v_cvt_pk_bf16_f32 v49, v188 /*v444*/, v189 /*v445*/
	v_cvt_pk_bf16_f32 v50, v186 /*v442*/, v187 /*v443*/
	s_set_vgpr_msb 0x500                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_cndmask_b32_e32 v45, v61, v46, vcc_lo
	v_dual_cndmask_b32 v46, v62, v66 :: v_dual_cndmask_b32 v47, v63, v67
	v_dual_cndmask_b32 v61, v35, v39, s0 :: v_dual_cndmask_b32 v62, v34, v38, s0
	v_dual_cndmask_b32 v63, v33, v37, s0 :: v_dual_cndmask_b32 v64, v32, v36, s0
	v_dual_cndmask_b32 v65, v39, v43, s0 :: v_dual_cndmask_b32 v66, v38, v42, s0
	v_dual_cndmask_b32 v67, v37, v41, s0 :: v_dual_cndmask_b32 v68, v36, v40, s0
	v_dual_cndmask_b32 v43, v43, v51, s0 :: v_dual_cndmask_b32 v42, v42, v50, s0
	v_dual_cndmask_b32 v69, v41, v49, s0 :: v_dual_cndmask_b32 v70, v40, v48, s0
	v_dual_cndmask_b32 v51, v51, v35, s0 :: v_dual_cndmask_b32 v50, v50, v34, s0
	v_dual_cndmask_b32 v71, v49, v33, s0 :: v_dual_cndmask_b32 v72, v48, v32, s0
	v_dual_cndmask_b32 v32, v43, v61 :: v_dual_add_nc_u32 v55, s5, v54
	v_dual_cndmask_b32 v33, v42, v62 :: v_dual_cndmask_b32 v34, v69, v63
	v_dual_cndmask_b32 v35, v70, v64 :: v_dual_cndmask_b32 v36, v51, v65
	v_dual_cndmask_b32 v37, v50, v66 :: v_dual_cndmask_b32 v38, v71, v67
	v_dual_cndmask_b32 v39, v72, v68 :: v_dual_cndmask_b32 v40, v61, v43
	v_dual_cndmask_b32 v41, v62, v42 :: v_dual_cndmask_b32 v42, v63, v69
	v_dual_cndmask_b32 v43, v64, v70 :: v_dual_cndmask_b32 v48, v65, v51
	v_dual_cndmask_b32 v49, v66, v50 :: v_dual_cndmask_b32 v50, v67, v71
	v_dual_cndmask_b32 v51, v68, v72, vcc_lo :: v_dual_add_nc_u32 v61, s5, v57
	v_dual_add_nc_u32 v62, s5, v58 :: v_dual_add_nc_u32 v63, s5, v59
	v_add_nc_u32_e32 v64, s5, v60
	s_wait_alu depctr_va_vdst(10)
	ds_store_b128 v55, v[44:47]
	s_wait_alu depctr_va_vdst(2)
	ds_store_b128 v61, v[32:35]
	s_wait_alu depctr_va_vdst(1)
	ds_store_b128 v62, v[36:39]
	ds_store_b128 v63, v[40:43]
	s_wait_alu depctr_va_vdst(0)
	ds_store_b128 v64, v[48:51]
	; sched_barrier mask(0x00000000)
	s_set_vgpr_msb 0x52                     ;  msbs: dst=1 src0=2 src1=0 src2=1
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[96:103] /*v[608:615]*/, v[24:31], v[208:215] /*v[464:471]*/
	; sched_group_barrier mask(0x00000002) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000200) size(0) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(2) SyncID(0)
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[112:119] /*v[624:631]*/, v[24:31], v[216:223] /*v[472:479]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000200) size(0) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(2) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[96:103] /*v[608:615]*/, v[16:23], v[240:247] /*v[496:503]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000200) size(0) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(2) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[112:119] /*v[624:631]*/, v[16:23], v[248:255] /*v[504:511]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000200) size(0) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(2) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[104:111] /*v[616:623]*/, v[8:15], v[208:215] /*v[464:471]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000200) size(0) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(2) SyncID(0)
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[120:127] /*v[632:639]*/, v[8:15], v[216:223] /*v[472:479]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000200) size(0) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(2) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[104:111] /*v[616:623]*/, v[0:7], v[240:247] /*v[496:503]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000200) size(0) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(2) SyncID(0)
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[120:127] /*v[632:639]*/, v[0:7], v[248:255] /*v[504:511]*/
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000200) size(0) SyncID(0)
	; sched_barrier mask(0x00000000)
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x5205                   ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_cvt_pk_bf16_f32 v0, v198 /*v454*/, v199 /*v455*/
	v_cvt_pk_bf16_f32 v1, v196 /*v452*/, v197 /*v453*/
	v_cvt_pk_bf16_f32 v2, v194 /*v450*/, v195 /*v451*/
	v_cvt_pk_bf16_f32 v3, v192 /*v448*/, v193 /*v449*/
	v_cvt_pk_bf16_f32 v4, v206 /*v462*/, v207 /*v463*/
	v_cvt_pk_bf16_f32 v5, v204 /*v460*/, v205 /*v461*/
	v_cvt_pk_bf16_f32 v6, v202 /*v458*/, v203 /*v459*/
	v_cvt_pk_bf16_f32 v7, v200 /*v456*/, v201 /*v457*/
	v_cvt_pk_bf16_f32 v8, v214 /*v470*/, v215 /*v471*/
	v_cvt_pk_bf16_f32 v9, v212 /*v468*/, v213 /*v469*/
	v_cvt_pk_bf16_f32 v10, v210 /*v466*/, v211 /*v467*/
	v_cvt_pk_bf16_f32 v11, v208 /*v464*/, v209 /*v465*/
	v_cvt_pk_bf16_f32 v12, v222 /*v478*/, v223 /*v479*/
	v_cvt_pk_bf16_f32 v13, v220 /*v476*/, v221 /*v477*/
	v_cvt_pk_bf16_f32 v14, v218 /*v474*/, v219 /*v475*/
	v_cvt_pk_bf16_f32 v15, v216 /*v472*/, v217 /*v473*/
	s_set_vgpr_msb 0x500                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_cndmask_b32 v16, v3, v7, s0 :: v_dual_cndmask_b32 v17, v2, v6, s0
	v_dual_cndmask_b32 v18, v1, v5, s0 :: v_dual_cndmask_b32 v19, v0, v4, s0
	v_dual_cndmask_b32 v20, v7, v11, s0 :: v_dual_cndmask_b32 v21, v6, v10, s0
	v_dual_cndmask_b32 v22, v5, v9, s0 :: v_dual_cndmask_b32 v23, v4, v8, s0
	v_dual_cndmask_b32 v11, v11, v15, s0 :: v_dual_cndmask_b32 v10, v10, v14, s0
	v_dual_cndmask_b32 v24, v9, v13, s0 :: v_dual_cndmask_b32 v25, v8, v12, s0
	v_dual_cndmask_b32 v15, v15, v3, s0 :: v_dual_cndmask_b32 v14, v14, v2, s0
	v_dual_cndmask_b32 v26, v13, v1, s0 :: v_dual_cndmask_b32 v27, v12, v0, s0
	s_add_co_i32 s9, s9, 0x19980
	v_dual_cndmask_b32 v0, v11, v16 :: v_dual_cndmask_b32 v1, v10, v17
	v_dual_cndmask_b32 v2, v24, v18 :: v_dual_cndmask_b32 v3, v25, v19
	v_dual_cndmask_b32 v4, v15, v20 :: v_dual_cndmask_b32 v5, v14, v21
	v_dual_cndmask_b32 v6, v26, v22 :: v_dual_cndmask_b32 v7, v27, v23
	v_dual_cndmask_b32 v8, v16, v11 :: v_dual_cndmask_b32 v9, v17, v10
	v_dual_cndmask_b32 v10, v18, v24 :: v_dual_cndmask_b32 v11, v19, v25
	v_dual_add_nc_u32 v16, s9, v52 :: v_dual_add_nc_u32 v17, s9, v53
	v_add_nc_u32_e32 v18, s9, v56
	s_wait_tensorcnt 0x0
	v_dual_cndmask_b32 v12, v20, v15 :: v_dual_cndmask_b32 v13, v21, v14
	s_wait_alu depctr_va_vdst(0)
	ds_store_b128 v16, v[0:3]
	ds_store_b128 v17, v[4:7]
	ds_store_b128 v18, v[8:11]
	s_wait_alu depctr_vm_vsrc(2)
	s_set_vgpr_msb 5                        ;  msbs: dst=0 src0=1 src1=1 src2=0
	v_cvt_pk_bf16_f32 v0, v230 /*v486*/, v231 /*v487*/
	v_cvt_pk_bf16_f32 v1, v228 /*v484*/, v229 /*v485*/
	v_cvt_pk_bf16_f32 v2, v226 /*v482*/, v227 /*v483*/
	v_cvt_pk_bf16_f32 v3, v224 /*v480*/, v225 /*v481*/
	s_wait_alu depctr_vm_vsrc(1)
	v_cvt_pk_bf16_f32 v4, v238 /*v494*/, v239 /*v495*/
	v_cvt_pk_bf16_f32 v5, v236 /*v492*/, v237 /*v493*/
	v_cvt_pk_bf16_f32 v6, v234 /*v490*/, v235 /*v491*/
	v_cvt_pk_bf16_f32 v7, v232 /*v488*/, v233 /*v489*/
	s_wait_alu depctr_vm_vsrc(0)
	v_cvt_pk_bf16_f32 v8, v246 /*v502*/, v247 /*v503*/
	v_cvt_pk_bf16_f32 v9, v244 /*v500*/, v245 /*v501*/
	v_cvt_pk_bf16_f32 v10, v242 /*v498*/, v243 /*v499*/
	v_cvt_pk_bf16_f32 v11, v240 /*v496*/, v241 /*v497*/
	v_cvt_pk_bf16_f32 v16, v254 /*v510*/, v255 /*v511*/
	v_cvt_pk_bf16_f32 v17, v252 /*v508*/, v253 /*v509*/
	v_cvt_pk_bf16_f32 v18, v250 /*v506*/, v251 /*v507*/
	v_cvt_pk_bf16_f32 v19, v248 /*v504*/, v249 /*v505*/
	s_set_vgpr_msb 0x500                    ;  msbs: dst=0 src0=0 src1=0 src2=0
	v_dual_cndmask_b32 v14, v22, v26 :: v_dual_cndmask_b32 v15, v23, v27
	v_dual_cndmask_b32 v21, v3, v7, s0 :: v_dual_cndmask_b32 v22, v2, v6, s0
	v_dual_cndmask_b32 v23, v1, v5, s0 :: v_dual_cndmask_b32 v24, v0, v4, s0
	v_dual_cndmask_b32 v25, v7, v11, s0 :: v_dual_cndmask_b32 v26, v6, v10, s0
	v_dual_cndmask_b32 v27, v5, v9, s0 :: v_dual_cndmask_b32 v28, v4, v8, s0
	v_dual_cndmask_b32 v11, v11, v19, s0 :: v_dual_cndmask_b32 v10, v10, v18, s0
	v_dual_cndmask_b32 v29, v9, v17, s0 :: v_dual_cndmask_b32 v30, v8, v16, s0
	v_dual_cndmask_b32 v19, v19, v3, s0 :: v_dual_cndmask_b32 v18, v18, v2, s0
	v_dual_cndmask_b32 v31, v17, v1, s0 :: v_dual_cndmask_b32 v32, v16, v0, s0
	s_lshl_b32 s0, s23, 5
	s_ashr_i32 s19, s18, 31
	s_add_co_i32 s12, s0, s33
	s_mov_b32 s13, s7
	s_lshl_b32 s0, s23, 14
	s_mul_u64 s[10:11], s[18:19], s[12:13]
	v_add_nc_u32_e32 v20, s9, v54
	s_add_nc_u64 s[10:11], s[10:11], s[6:7]
	v_cndmask_b32_e32 v0, v11, v21, vcc_lo
	s_lshl_b64 s[10:11], s[10:11], 1
	v_cndmask_b32_e32 v1, v10, v22, vcc_lo
	s_add_nc_u64 s[10:11], s[10:11], s[2:3]
	v_dual_cndmask_b32 v2, v29, v23 :: v_dual_cndmask_b32 v3, v30, v24
	v_dual_cndmask_b32 v8, v21, v11 :: v_dual_cndmask_b32 v9, v22, v10
	v_dual_cndmask_b32 v10, v23, v29 :: v_dual_cndmask_b32 v11, v24, v30
	v_dual_add_nc_u32 v21, s9, v57 :: v_dual_add_nc_u32 v22, s9, v58
	v_dual_add_nc_u32 v23, s9, v59 :: v_dual_add_nc_u32 v24, s9, v60
	s_add_co_i32 s9, s0, s1
	;;#ASMSTART
	s_sub_co_u32 s0, s20, s12
	;;#ASMEND
	v_dual_cndmask_b32 v4, v19, v25 :: v_dual_cndmask_b32 v5, v18, v26
	v_dual_cndmask_b32 v6, v31, v27 :: v_dual_cndmask_b32 v7, v32, v28
	v_dual_cndmask_b32 v16, v25, v19 :: v_dual_cndmask_b32 v17, v26, v18
	v_dual_cndmask_b32 v18, v27, v31 :: v_dual_cndmask_b32 v19, v28, v32
	s_and_b32 s1, s11, 0x1ffffff
	;;#ASMSTART
	s_max_i32 s3, s0, 0
	;;#ASMEND
	;;#ASMSTART
	s_sub_co_u32 s0, s21, s6
	;;#ASMEND
	;;#ASMSTART
	s_max_i32 s2, s0, 0
	;;#ASMEND
	s_lshr_b32 s0, s3, 16
	s_or_b32 s11, s1, 0x80000000
	s_lshl_b32 s1, s2, 16
	s_lshr_b64 s[2:3], s[2:3], 16
	s_or_b32 s3, s0, 0x1000000
	s_and_b32 s6, s19, 0xffff
	s_mov_b32 s0, 0x10000
	s_mov_b32 s5, s18
	s_wait_alu depctr_va_vdst(11)
	ds_store_b128 v20, v[12:15]
	s_wait_alu depctr_va_vdst(5)
	ds_store_b128 v21, v[0:3]
	s_wait_alu depctr_va_vdst(2)
	ds_store_b128 v22, v[4:7]
	ds_store_b128 v23, v[8:11]
	s_wait_alu depctr_va_vdst(0)
	ds_store_b128 v24, v[16:19]
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	tensor_store_from_lds s[8:11], s[0:7] scope:SCOPE_DEV
	; sched_group_barrier mask(0x00000002) size(48) SyncID(0)
	; sched_group_barrier mask(0x00000200) size(8) SyncID(0)
	s_wait_tensorcnt 0x0
.LBB2_12:                               ; %cleanup284
	s_endpgm
.LBB2_13:
	s_set_vgpr_msb 0x41                     ;  msbs: dst=1 src0=1 src1=0 src2=0
	v_mov_b32_e32 v248 /*v504*/, 0
	s_sub_co_i32 s35, s35, s36
	v_dual_mov_b32 v249 /*v505*/, v248 /*v504*/ :: v_dual_mov_b32 v250 /*v506*/, v248 /*v504*/
	v_dual_mov_b32 v251 /*v507*/, v248 /*v504*/ :: v_dual_mov_b32 v252 /*v508*/, v248 /*v504*/
	v_dual_mov_b32 v253 /*v509*/, v248 /*v504*/ :: v_dual_mov_b32 v254 /*v510*/, v248 /*v504*/
	v_mov_b32_e32 v255 /*v511*/, v248 /*v504*/
	v_mov_b64_e32 v[240:241] /*v[496:497]*/, v[248:249] /*v[504:505]*/
	v_mov_b64_e32 v[242:243] /*v[498:499]*/, v[250:251] /*v[506:507]*/
	v_mov_b64_e32 v[244:245] /*v[500:501]*/, v[252:253] /*v[508:509]*/
	v_mov_b64_e32 v[232:233] /*v[488:489]*/, v[248:249] /*v[504:505]*/
	v_mov_b64_e32 v[246:247] /*v[502:503]*/, v[254:255] /*v[510:511]*/
	v_mov_b64_e32 v[234:235] /*v[490:491]*/, v[250:251] /*v[506:507]*/
	v_mov_b64_e32 v[236:237] /*v[492:493]*/, v[252:253] /*v[508:509]*/
	v_mov_b64_e32 v[238:239] /*v[494:495]*/, v[254:255] /*v[510:511]*/
	v_mov_b64_e32 v[224:225] /*v[480:481]*/, v[248:249] /*v[504:505]*/
	v_mov_b64_e32 v[226:227] /*v[482:483]*/, v[250:251] /*v[506:507]*/
	v_mov_b64_e32 v[228:229] /*v[484:485]*/, v[252:253] /*v[508:509]*/
	v_mov_b64_e32 v[230:231] /*v[486:487]*/, v[254:255] /*v[510:511]*/
	v_mov_b64_e32 v[216:217] /*v[472:473]*/, v[248:249] /*v[504:505]*/
	v_mov_b64_e32 v[218:219] /*v[474:475]*/, v[250:251] /*v[506:507]*/
	v_mov_b64_e32 v[220:221] /*v[476:477]*/, v[252:253] /*v[508:509]*/
	v_mov_b64_e32 v[222:223] /*v[478:479]*/, v[254:255] /*v[510:511]*/
	v_mov_b64_e32 v[208:209] /*v[464:465]*/, v[248:249] /*v[504:505]*/
	v_mov_b64_e32 v[210:211] /*v[466:467]*/, v[250:251] /*v[506:507]*/
	v_mov_b64_e32 v[212:213] /*v[468:469]*/, v[252:253] /*v[508:509]*/
	v_mov_b64_e32 v[214:215] /*v[470:471]*/, v[254:255] /*v[510:511]*/
	v_mov_b64_e32 v[200:201] /*v[456:457]*/, v[248:249] /*v[504:505]*/
	v_mov_b64_e32 v[202:203] /*v[458:459]*/, v[250:251] /*v[506:507]*/
	v_mov_b64_e32 v[204:205] /*v[460:461]*/, v[252:253] /*v[508:509]*/
	v_mov_b64_e32 v[206:207] /*v[462:463]*/, v[254:255] /*v[510:511]*/
	v_mov_b64_e32 v[192:193] /*v[448:449]*/, v[248:249] /*v[504:505]*/
	v_mov_b64_e32 v[194:195] /*v[450:451]*/, v[250:251] /*v[506:507]*/
	v_mov_b64_e32 v[196:197] /*v[452:453]*/, v[252:253] /*v[508:509]*/
	v_mov_b64_e32 v[198:199] /*v[454:455]*/, v[254:255] /*v[510:511]*/
	v_mov_b64_e32 v[184:185] /*v[440:441]*/, v[248:249] /*v[504:505]*/
	v_mov_b64_e32 v[186:187] /*v[442:443]*/, v[250:251] /*v[506:507]*/
	v_mov_b64_e32 v[188:189] /*v[444:445]*/, v[252:253] /*v[508:509]*/
	v_mov_b64_e32 v[190:191] /*v[446:447]*/, v[254:255] /*v[510:511]*/
	v_mov_b64_e32 v[176:177] /*v[432:433]*/, v[248:249] /*v[504:505]*/
	v_mov_b64_e32 v[178:179] /*v[434:435]*/, v[250:251] /*v[506:507]*/
	v_mov_b64_e32 v[180:181] /*v[436:437]*/, v[252:253] /*v[508:509]*/
	v_mov_b64_e32 v[182:183] /*v[438:439]*/, v[254:255] /*v[510:511]*/
	v_mov_b64_e32 v[168:169] /*v[424:425]*/, v[248:249] /*v[504:505]*/
	v_mov_b64_e32 v[170:171] /*v[426:427]*/, v[250:251] /*v[506:507]*/
	v_mov_b64_e32 v[172:173] /*v[428:429]*/, v[252:253] /*v[508:509]*/
	v_mov_b64_e32 v[174:175] /*v[430:431]*/, v[254:255] /*v[510:511]*/
	v_mov_b64_e32 v[160:161] /*v[416:417]*/, v[248:249] /*v[504:505]*/
	v_mov_b64_e32 v[162:163] /*v[418:419]*/, v[250:251] /*v[506:507]*/
	v_mov_b64_e32 v[164:165] /*v[420:421]*/, v[252:253] /*v[508:509]*/
	v_mov_b64_e32 v[166:167] /*v[422:423]*/, v[254:255] /*v[510:511]*/
	v_mov_b64_e32 v[152:153] /*v[408:409]*/, v[248:249] /*v[504:505]*/
	v_mov_b64_e32 v[154:155] /*v[410:411]*/, v[250:251] /*v[506:507]*/
	v_mov_b64_e32 v[156:157] /*v[412:413]*/, v[252:253] /*v[508:509]*/
	v_mov_b64_e32 v[158:159] /*v[414:415]*/, v[254:255] /*v[510:511]*/
	v_mov_b64_e32 v[144:145] /*v[400:401]*/, v[248:249] /*v[504:505]*/
	v_mov_b64_e32 v[146:147] /*v[402:403]*/, v[250:251] /*v[506:507]*/
	v_mov_b64_e32 v[148:149] /*v[404:405]*/, v[252:253] /*v[508:509]*/
	v_mov_b64_e32 v[150:151] /*v[406:407]*/, v[254:255] /*v[510:511]*/
	v_mov_b64_e32 v[136:137] /*v[392:393]*/, v[248:249] /*v[504:505]*/
	v_mov_b64_e32 v[138:139] /*v[394:395]*/, v[250:251] /*v[506:507]*/
	v_mov_b64_e32 v[140:141] /*v[396:397]*/, v[252:253] /*v[508:509]*/
	v_mov_b64_e32 v[142:143] /*v[398:399]*/, v[254:255] /*v[510:511]*/
	v_mov_b64_e32 v[128:129] /*v[384:385]*/, v[248:249] /*v[504:505]*/
	v_mov_b64_e32 v[130:131] /*v[386:387]*/, v[250:251] /*v[506:507]*/
	v_mov_b64_e32 v[132:133] /*v[388:389]*/, v[252:253] /*v[508:509]*/
	v_mov_b64_e32 v[134:135] /*v[390:391]*/, v[254:255] /*v[510:511]*/
	v_mov_b64_e32 v[120:121] /*v[376:377]*/, v[248:249] /*v[504:505]*/
	v_mov_b64_e32 v[122:123] /*v[378:379]*/, v[250:251] /*v[506:507]*/
	v_mov_b64_e32 v[124:125] /*v[380:381]*/, v[252:253] /*v[508:509]*/
	v_mov_b64_e32 v[126:127] /*v[382:383]*/, v[254:255] /*v[510:511]*/
	v_mov_b64_e32 v[112:113] /*v[368:369]*/, v[248:249] /*v[504:505]*/
	v_mov_b64_e32 v[114:115] /*v[370:371]*/, v[250:251] /*v[506:507]*/
	v_mov_b64_e32 v[116:117] /*v[372:373]*/, v[252:253] /*v[508:509]*/
	v_mov_b64_e32 v[118:119] /*v[374:375]*/, v[254:255] /*v[510:511]*/
	v_mov_b64_e32 v[104:105] /*v[360:361]*/, v[248:249] /*v[504:505]*/
	v_mov_b64_e32 v[106:107] /*v[362:363]*/, v[250:251] /*v[506:507]*/
	v_mov_b64_e32 v[108:109] /*v[364:365]*/, v[252:253] /*v[508:509]*/
	v_mov_b64_e32 v[110:111] /*v[366:367]*/, v[254:255] /*v[510:511]*/
	v_mov_b64_e32 v[96:97] /*v[352:353]*/, v[248:249] /*v[504:505]*/
	v_mov_b64_e32 v[98:99] /*v[354:355]*/, v[250:251] /*v[506:507]*/
	v_mov_b64_e32 v[100:101] /*v[356:357]*/, v[252:253] /*v[508:509]*/
	v_mov_b64_e32 v[102:103] /*v[358:359]*/, v[254:255] /*v[510:511]*/
	v_mov_b64_e32 v[88:89] /*v[344:345]*/, v[248:249] /*v[504:505]*/
	v_mov_b64_e32 v[90:91] /*v[346:347]*/, v[250:251] /*v[506:507]*/
	v_mov_b64_e32 v[92:93] /*v[348:349]*/, v[252:253] /*v[508:509]*/
	v_mov_b64_e32 v[94:95] /*v[350:351]*/, v[254:255] /*v[510:511]*/
	v_mov_b64_e32 v[80:81] /*v[336:337]*/, v[248:249] /*v[504:505]*/
	v_mov_b64_e32 v[82:83] /*v[338:339]*/, v[250:251] /*v[506:507]*/
	v_mov_b64_e32 v[84:85] /*v[340:341]*/, v[252:253] /*v[508:509]*/
	v_mov_b64_e32 v[86:87] /*v[342:343]*/, v[254:255] /*v[510:511]*/
	v_mov_b64_e32 v[72:73] /*v[328:329]*/, v[248:249] /*v[504:505]*/
	v_mov_b64_e32 v[74:75] /*v[330:331]*/, v[250:251] /*v[506:507]*/
	v_mov_b64_e32 v[76:77] /*v[332:333]*/, v[252:253] /*v[508:509]*/
	v_mov_b64_e32 v[78:79] /*v[334:335]*/, v[254:255] /*v[510:511]*/
	v_mov_b64_e32 v[64:65] /*v[320:321]*/, v[248:249] /*v[504:505]*/
	v_mov_b64_e32 v[66:67] /*v[322:323]*/, v[250:251] /*v[506:507]*/
	v_mov_b64_e32 v[68:69] /*v[324:325]*/, v[252:253] /*v[508:509]*/
	v_mov_b64_e32 v[70:71] /*v[326:327]*/, v[254:255] /*v[510:511]*/
	v_mov_b64_e32 v[56:57] /*v[312:313]*/, v[248:249] /*v[504:505]*/
	v_mov_b64_e32 v[58:59] /*v[314:315]*/, v[250:251] /*v[506:507]*/
	v_mov_b64_e32 v[60:61] /*v[316:317]*/, v[252:253] /*v[508:509]*/
	v_mov_b64_e32 v[62:63] /*v[318:319]*/, v[254:255] /*v[510:511]*/
	v_mov_b64_e32 v[48:49] /*v[304:305]*/, v[248:249] /*v[504:505]*/
	v_mov_b64_e32 v[50:51] /*v[306:307]*/, v[250:251] /*v[506:507]*/
	v_mov_b64_e32 v[52:53] /*v[308:309]*/, v[252:253] /*v[508:509]*/
	v_mov_b64_e32 v[54:55] /*v[310:311]*/, v[254:255] /*v[510:511]*/
	v_mov_b64_e32 v[40:41] /*v[296:297]*/, v[248:249] /*v[504:505]*/
	v_mov_b64_e32 v[42:43] /*v[298:299]*/, v[250:251] /*v[506:507]*/
	v_mov_b64_e32 v[44:45] /*v[300:301]*/, v[252:253] /*v[508:509]*/
	v_mov_b64_e32 v[46:47] /*v[302:303]*/, v[254:255] /*v[510:511]*/
	v_mov_b64_e32 v[32:33] /*v[288:289]*/, v[248:249] /*v[504:505]*/
	v_mov_b64_e32 v[34:35] /*v[290:291]*/, v[250:251] /*v[506:507]*/
	v_mov_b64_e32 v[36:37] /*v[292:293]*/, v[252:253] /*v[508:509]*/
	v_mov_b64_e32 v[38:39] /*v[294:295]*/, v[254:255] /*v[510:511]*/
	v_mov_b64_e32 v[24:25] /*v[280:281]*/, v[248:249] /*v[504:505]*/
	v_mov_b64_e32 v[26:27] /*v[282:283]*/, v[250:251] /*v[506:507]*/
	v_mov_b64_e32 v[28:29] /*v[284:285]*/, v[252:253] /*v[508:509]*/
	v_mov_b64_e32 v[30:31] /*v[286:287]*/, v[254:255] /*v[510:511]*/
	v_mov_b64_e32 v[16:17] /*v[272:273]*/, v[248:249] /*v[504:505]*/
	v_mov_b64_e32 v[18:19] /*v[274:275]*/, v[250:251] /*v[506:507]*/
	v_mov_b64_e32 v[20:21] /*v[276:277]*/, v[252:253] /*v[508:509]*/
	v_mov_b64_e32 v[22:23] /*v[278:279]*/, v[254:255] /*v[510:511]*/
	v_mov_b64_e32 v[8:9] /*v[264:265]*/, v[248:249] /*v[504:505]*/
	v_mov_b64_e32 v[10:11] /*v[266:267]*/, v[250:251] /*v[506:507]*/
	v_mov_b64_e32 v[12:13] /*v[268:269]*/, v[252:253] /*v[508:509]*/
	v_mov_b64_e32 v[14:15] /*v[270:271]*/, v[254:255] /*v[510:511]*/
	v_mov_b64_e32 v[0:1] /*v[256:257]*/, v[248:249] /*v[504:505]*/
	v_mov_b64_e32 v[2:3] /*v[258:259]*/, v[250:251] /*v[506:507]*/
	v_mov_b64_e32 v[4:5] /*v[260:261]*/, v[252:253] /*v[508:509]*/
	v_mov_b64_e32 v[6:7] /*v[262:263]*/, v[254:255] /*v[510:511]*/
	s_cmp_lg_u32 s35, 0
	s_mov_b32 s0, 0
	s_set_vgpr_msb 0x4100                   ;  msbs: dst=0 src0=0 src1=0 src2=0
	s_cbranch_scc1 .LBB2_8
	s_branch .LBB2_9
.Lfunc_end2:
	.size	_Z32gemm_a16w16_4wave_compute_kernelI16opus_gemm_traitsILi128ELi128ELi256ELi128ELi3EDF16bDF16bDF16bfEEv15opus_gemm_kargs, .Lfunc_end2-_Z32gemm_a16w16_4wave_compute_kernelI16opus_gemm_traitsILi128ELi128ELi256ELi128ELi3EDF16bDF16bDF16bfEEv15opus_gemm_kargs
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z32gemm_a16w16_4wave_compute_kernelI16opus_gemm_traitsILi128ELi128ELi256ELi128ELi3EDF16bDF16bDF16bfEEv15opus_gemm_kargs
		.amdhsa_group_segment_fixed_size 313344
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 64
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 747
		.amdhsa_next_free_sgpr 66
		.amdhsa_named_barrier_count 0
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end2-_Z32gemm_a16w16_4wave_compute_kernelI16opus_gemm_traitsILi128ELi128ELi256ELi128ELi3EDF16bDF16bDF16bfEEv15opus_gemm_kargs)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z32gemm_a16w16_4wave_compute_kernelI16opus_gemm_traitsILi128ELi128ELi256ELi128ELi3EDF16bDF16bDF16bfEEv15opus_gemm_kargs,"axG",@progbits,_Z32gemm_a16w16_4wave_compute_kernelI16opus_gemm_traitsILi128ELi128ELi256ELi128ELi3EDF16bDF16bDF16bfEEv15opus_gemm_kargs,comdat
                                        ; -- End function
	.set .L_Z32gemm_a16w16_4wave_compute_kernelI16opus_gemm_traitsILi128ELi128ELi256ELi128ELi3EDF16bDF16bDF16bfEEv15opus_gemm_kargs.num_vgpr, 747
	.set .L_Z32gemm_a16w16_4wave_compute_kernelI16opus_gemm_traitsILi128ELi128ELi256ELi128ELi3EDF16bDF16bDF16bfEEv15opus_gemm_kargs.num_agpr, 0
	.set .L_Z32gemm_a16w16_4wave_compute_kernelI16opus_gemm_traitsILi128ELi128ELi256ELi128ELi3EDF16bDF16bDF16bfEEv15opus_gemm_kargs.numbered_sgpr, 66
	.set .L_Z32gemm_a16w16_4wave_compute_kernelI16opus_gemm_traitsILi128ELi128ELi256ELi128ELi3EDF16bDF16bDF16bfEEv15opus_gemm_kargs.num_named_barrier, 0
	.set .L_Z32gemm_a16w16_4wave_compute_kernelI16opus_gemm_traitsILi128ELi128ELi256ELi128ELi3EDF16bDF16bDF16bfEEv15opus_gemm_kargs.private_seg_size, 0
	.set .L_Z32gemm_a16w16_4wave_compute_kernelI16opus_gemm_traitsILi128ELi128ELi256ELi128ELi3EDF16bDF16bDF16bfEEv15opus_gemm_kargs.uses_vcc, 1
	.set .L_Z32gemm_a16w16_4wave_compute_kernelI16opus_gemm_traitsILi128ELi128ELi256ELi128ELi3EDF16bDF16bDF16bfEEv15opus_gemm_kargs.uses_flat_scratch, 0
	.set .L_Z32gemm_a16w16_4wave_compute_kernelI16opus_gemm_traitsILi128ELi128ELi256ELi128ELi3EDF16bDF16bDF16bfEEv15opus_gemm_kargs.has_dyn_sized_stack, 0
	.set .L_Z32gemm_a16w16_4wave_compute_kernelI16opus_gemm_traitsILi128ELi128ELi256ELi128ELi3EDF16bDF16bDF16bfEEv15opus_gemm_kargs.has_recursion, 0
	.set .L_Z32gemm_a16w16_4wave_compute_kernelI16opus_gemm_traitsILi128ELi128ELi256ELi128ELi3EDF16bDF16bDF16bfEEv15opus_gemm_kargs.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 27392
; TotalNumSgprs: 68
; NumVgprs: 747
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 313344 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 46
; NumSGPRsForWavesPerEU: 68
; NumVGPRsForWavesPerEU: 747
; NamedBarCnt: 0
; Occupancy: 1
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.type	__hip_cuid_6c16eec2d59a5c04,@object ; @__hip_cuid_6c16eec2d59a5c04
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_6c16eec2d59a5c04
__hip_cuid_6c16eec2d59a5c04:
	.byte	0                               ; 0x0
	.size	__hip_cuid_6c16eec2d59a5c04, 1

	.ident	"clang version 24.0.0git (https://github.com/demonsan/llvm-project.git 7ae83c1ff266d690ea518114a023cd12d156227f)"
	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git aa451e1fe6a793394d6733051b1778633063ae96+PATCHED:440716f8b87be9d8e20ed910e10e5b6d14d57cf6)"
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .offset:         0
        .size:           64
        .value_kind:     by_value
    .cluster_dims:
      - 4
      - 4
      - 1
    .group_segment_fixed_size: 313344
    .kernarg_segment_align: 8
    .kernarg_segment_size: 64
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 128
    .name:           _Z32gemm_a16w16_4wave_compute_kernelI16opus_gemm_traitsILi128ELi128ELi256ELi128ELi3EDF16bDF16bDF16bfEEv15opus_gemm_kargs
    .private_segment_fixed_size: 0
    .sgpr_count:     68
    .sgpr_spill_count: 0
    .symbol:         _Z32gemm_a16w16_4wave_compute_kernelI16opus_gemm_traitsILi128ELi128ELi256ELi128ELi3EDF16bDF16bDF16bfEEv15opus_gemm_kargs.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     747
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
