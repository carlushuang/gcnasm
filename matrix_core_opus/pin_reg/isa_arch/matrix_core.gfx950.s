	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z18matrix_core_gfx950PKDhS0_Pfiiii ; -- Begin function _Z18matrix_core_gfx950PKDhS0_Pfiiii
	.globl	_Z18matrix_core_gfx950PKDhS0_Pfiiii
	.p2align	8
	.type	_Z18matrix_core_gfx950PKDhS0_Pfiiii,@function
_Z18matrix_core_gfx950PKDhS0_Pfiiii:    ; @_Z18matrix_core_gfx950PKDhS0_Pfiiii
; %bb.0:                                ; %entry
	s_load_dwordx4 s[8:11], s[0:1], 0x0
	s_load_dwordx2 s[12:13], s[0:1], 0x10
	s_load_dwordx4 s[4:7], s[0:1], 0x18
	s_lshl_b32 s15, s2, 4
	s_lshl_b32 s14, s3, 4
	v_lshrrev_b32_e32 v1, 3, v0
	v_and_b32_e32 v5, 15, v0
	v_and_b32_e32 v6, 0x70, v1
	s_waitcnt lgkmcnt(0)
	s_cmp_lt_i32 s4, 1
	v_lshrrev_b32_e32 v4, 2, v0
	s_cbranch_scc1 .LBB0_3
; %bb.1:                                ; %for.body.preheader.i
	s_mul_i32 s0, s5, s15
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s8, s0
	s_mul_i32 s8, s6, s14
	s_addc_u32 s1, s9, s1
	s_ashr_i32 s9, s8, 31
	v_add_u32_e32 v1, v6, v5
	v_lshrrev_b32_e32 v0, 1, v0
	s_and_b32 s1, s1, 0xffff
	s_lshl_b64 s[8:9], s[8:9], 1
	v_mul_lo_u32 v1, s5, v1
	v_and_b32_e32 v0, 24, v0
	s_add_u32 s8, s10, s8
	v_lshl_add_u32 v7, v1, 1, v0
	v_and_or_b32 v1, v4, 16, v5
	s_mov_b32 s3, 0x20000
	s_mov_b32 s2, -1
	s_addc_u32 s9, s11, s9
	s_add_i32 s4, s4, 15
	v_mul_lo_u32 v1, s6, v1
	s_and_b32 s9, s9, 0xffff
	s_lshr_b32 s4, s4, 4
	v_lshl_add_u32 v8, v1, 1, v0
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v1, 0
	v_mov_b32_e32 v0, 0
	s_mov_b32 s10, s2
	s_mov_b32 s11, s3
.LBB0_2:                                ; %for.body.i
                                        ; =>This Inner Loop Header: Depth=1
	buffer_load_dwordx2 a[8:9], v8, s[8:11], 0 offen
	buffer_load_dwordx2 a[0:1], v7, s[0:3], 0 offen
	s_add_i32 s4, s4, -1
	v_add_u32_e32 v7, 32, v7
	s_cmp_eq_u32 s4, 0
	v_add_u32_e32 v8, 32, v8
	s_waitcnt vmcnt(0)
	v_mfma_f32_16x16x16_f16 v[0:3], a[8:9], a[0:1], v[0:3]
	s_cbranch_scc0 .LBB0_2
	s_branch .LBB0_4
.LBB0_3:
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v1, 0
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v3, 0
.LBB0_4:                                ; %_Z9mfma_tileILi16ELi16ELi16EEvPKDhS1_Pfiiii.exit
	s_mul_i32 s0, s7, s15
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 2
	s_add_u32 s2, s12, s0
	s_addc_u32 s3, s13, s1
	s_ashr_i32 s15, s14, 31
	s_lshl_b64 s[0:1], s[14:15], 2
	v_or_b32_e32 v5, v6, v5
	s_add_u32 s0, s2, s0
	s_addc_u32 s1, s3, s1
	v_mul_lo_u32 v5, s7, v5
	v_and_b32_e32 v4, 28, v4
	s_and_b32 s1, s1, 0xffff
	s_mov_b32 s3, 0x20000
	s_mov_b32 s2, -1
	v_add_lshl_u32 v4, v5, v4, 2
	buffer_store_dwordx4 v[0:3], v4, s[0:3], 0 offen
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z18matrix_core_gfx950PKDhS0_Pfiiii
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
		.amdhsa_next_free_vgpr 22
		.amdhsa_next_free_sgpr 16
		.amdhsa_accum_offset 12
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
	.text
.Lfunc_end0:
	.size	_Z18matrix_core_gfx950PKDhS0_Pfiiii, .Lfunc_end0-_Z18matrix_core_gfx950PKDhS0_Pfiiii
                                        ; -- End function
	.set _Z18matrix_core_gfx950PKDhS0_Pfiiii.num_vgpr, 9
	.set _Z18matrix_core_gfx950PKDhS0_Pfiiii.num_agpr, 10
	.set _Z18matrix_core_gfx950PKDhS0_Pfiiii.numbered_sgpr, 16
	.set _Z18matrix_core_gfx950PKDhS0_Pfiiii.private_seg_size, 0
	.set _Z18matrix_core_gfx950PKDhS0_Pfiiii.uses_vcc, 0
	.set _Z18matrix_core_gfx950PKDhS0_Pfiiii.uses_flat_scratch, 0
	.set _Z18matrix_core_gfx950PKDhS0_Pfiiii.has_dyn_sized_stack, 0
	.set _Z18matrix_core_gfx950PKDhS0_Pfiiii.has_recursion, 0
	.set _Z18matrix_core_gfx950PKDhS0_Pfiiii.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 376
; TotalNumSgprs: 22
; NumVgprs: 9
; NumAgprs: 10
; TotalNumVgprs: 22
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 22
; NumVGPRsForWavesPerEU: 22
; AccumOffset: 12
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 2
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_246d2d2bcfa4e238,@object ; @__hip_cuid_246d2d2bcfa4e238
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_246d2d2bcfa4e238
__hip_cuid_246d2d2bcfa4e238:
	.byte	0                               ; 0x0
	.size	__hip_cuid_246d2d2bcfa4e238, 1

	.ident	"clang version 20.0.0git (https://github.com/ROCm/llvm-project.git 27682a16360e33e37c4f3cc6adf9a620733f8fe1)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_246d2d2bcfa4e238
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     10
    .args:
      - .address_space:  global
        .name:           a.coerce
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .name:           b.coerce
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .name:           c.coerce
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .name:           k
        .offset:         24
        .size:           4
        .value_kind:     by_value
      - .name:           sa
        .offset:         28
        .size:           4
        .value_kind:     by_value
      - .name:           sb
        .offset:         32
        .size:           4
        .value_kind:     by_value
      - .name:           sc
        .offset:         36
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 40
    .max_flat_workgroup_size: 1024
    .name:           _Z18matrix_core_gfx950PKDhS0_Pfiiii
    .private_segment_fixed_size: 0
    .sgpr_count:     22
    .sgpr_spill_count: 0
    .symbol:         _Z18matrix_core_gfx950PKDhS0_Pfiiii.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     22
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
