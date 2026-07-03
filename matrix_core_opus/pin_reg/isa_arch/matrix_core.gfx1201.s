	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z19matrix_core_gfx1201PKDhS0_Pfiiii ; -- Begin function _Z19matrix_core_gfx1201PKDhS0_Pfiiii
	.globl	_Z19matrix_core_gfx1201PKDhS0_Pfiiii
	.p2align	8
	.type	_Z19matrix_core_gfx1201PKDhS0_Pfiiii,@function
_Z19matrix_core_gfx1201PKDhS0_Pfiiii:   ; @_Z19matrix_core_gfx1201PKDhS0_Pfiiii
; %bb.0:                                ; %entry
	s_clause 0x2
	s_load_b128 s[4:7], s[0:1], 0x18
	s_load_b128 s[8:11], s[0:1], 0x0
	s_load_b64 s[12:13], s[0:1], 0x10
	v_and_b32_e32 v17, 15, v0
	v_bfe_u32 v16, v0, 4, 1
	s_lshl_b32 s15, ttmp9, 4
	s_lshl_b32 s14, ttmp7, 4
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 s4, 1
	s_cbranch_scc1 .LBB0_3
; %bb.1:                                ; %for.body.preheader
	v_mul_lo_u32 v1, s5, v17
	v_mul_lo_u32 v2, s6, v17
	s_mul_i32 s0, s5, s15
	s_mul_i32 s16, s6, s14
	s_ashr_i32 s1, s0, 31
	v_dual_mov_b32 v0, 0 :: v_dual_lshlrev_b32 v3, 4, v16
	s_lshl_b64 s[0:1], s[0:1], 1
	s_ashr_i32 s17, s16, 31
	s_add_nc_u64 s[0:1], s[8:9], s[0:1]
	s_lshl_b64 s[8:9], s[16:17], 1
	v_lshl_add_u32 v18, v1, 1, v3
	v_lshl_add_u32 v19, v2, 1, v3
	v_dual_mov_b32 v1, v0 :: v_dual_mov_b32 v2, v0
	v_dual_mov_b32 v3, v0 :: v_dual_mov_b32 v4, v0
	v_dual_mov_b32 v5, v0 :: v_dual_mov_b32 v6, v0
	v_mov_b32_e32 v7, v0
	s_mov_b32 s3, 0x31004000
	s_mov_b32 s2, -1
	s_add_nc_u64 s[8:9], s[10:11], s[8:9]
	s_add_co_i32 s4, s4, 15
	s_and_b32 s1, s1, 0xffff
	s_and_b32 s9, s9, 0xffff
	s_wait_alu 0xfffe
	s_lshr_b32 s4, s4, 4
	s_mov_b32 s10, s2
	s_mov_b32 s11, s3
.LBB0_2:                                ; %for.body
                                        ; =>This Inner Loop Header: Depth=1
	buffer_load_b128 v[8:11], v18, s[0:3], null offen
	buffer_load_b128 v[12:15], v19, s[8:11], null offen
	v_add_nc_u32_e32 v18, 32, v18
	v_add_nc_u32_e32 v19, 32, v19
	s_add_co_i32 s4, s4, -1
	s_wait_alu 0xfffe
	s_cmp_eq_u32 s4, 0
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_f16 v[0:7], v[12:15], v[8:11], v[0:7]
	s_cbranch_scc0 .LBB0_2
	s_branch .LBB0_4
.LBB0_3:
	v_dual_mov_b32 v7, 0 :: v_dual_mov_b32 v6, 0
	v_dual_mov_b32 v5, 0 :: v_dual_mov_b32 v4, 0
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v2, 0
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v0, 0
.LBB0_4:                                ; %Flow
	v_mul_lo_u32 v8, v17, s7
	s_mul_i32 s0, s7, s15
	v_lshlrev_b32_e32 v9, 5, v16
	s_ashr_i32 s1, s0, 31
	s_ashr_i32 s15, s14, 31
	s_lshl_b64 s[0:1], s[0:1], 2
	s_lshl_b64 s[2:3], s[14:15], 2
	s_add_nc_u64 s[0:1], s[12:13], s[0:1]
	v_lshl_add_u32 v8, v8, 2, v9
	s_add_nc_u64 s[0:1], s[0:1], s[2:3]
	s_mov_b32 s3, 0x31004000
	s_mov_b32 s2, -1
	s_and_b32 s1, s1, 0xffff
	s_clause 0x1
	buffer_store_b128 v[0:3], v8, s[0:3], null offen
	buffer_store_b128 v[4:7], v8, s[0:3], null offen offset:16
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z19matrix_core_gfx1201PKDhS0_Pfiiii
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 40
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
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 20
		.amdhsa_next_free_sgpr 18
		.amdhsa_reserve_vcc 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_round_robin_scheduling 0
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
	.size	_Z19matrix_core_gfx1201PKDhS0_Pfiiii, .Lfunc_end0-_Z19matrix_core_gfx1201PKDhS0_Pfiiii
                                        ; -- End function
	.set _Z19matrix_core_gfx1201PKDhS0_Pfiiii.num_vgpr, 20
	.set _Z19matrix_core_gfx1201PKDhS0_Pfiiii.num_agpr, 0
	.set _Z19matrix_core_gfx1201PKDhS0_Pfiiii.numbered_sgpr, 18
	.set _Z19matrix_core_gfx1201PKDhS0_Pfiiii.private_seg_size, 0
	.set _Z19matrix_core_gfx1201PKDhS0_Pfiiii.uses_vcc, 0
	.set _Z19matrix_core_gfx1201PKDhS0_Pfiiii.uses_flat_scratch, 0
	.set _Z19matrix_core_gfx1201PKDhS0_Pfiiii.has_dyn_sized_stack, 0
	.set _Z19matrix_core_gfx1201PKDhS0_Pfiiii.has_recursion, 0
	.set _Z19matrix_core_gfx1201PKDhS0_Pfiiii.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 404
; TotalNumSgprs: 18
; NumVgprs: 20
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 18
; NumVGPRsForWavesPerEU: 20
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_5a2469d356522408,@object ; @__hip_cuid_5a2469d356522408
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_5a2469d356522408
__hip_cuid_5a2469d356522408:
	.byte	0                               ; 0x0
	.size	__hip_cuid_5a2469d356522408, 1

	.ident	"clang version 20.0.0git (https://github.com/ROCm/llvm-project.git 27682a16360e33e37c4f3cc6adf9a620733f8fe1)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_5a2469d356522408
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .name:           ptr_a.coerce
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .name:           ptr_b.coerce
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
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
    .max_flat_workgroup_size: 1024
    .name:           _Z19matrix_core_gfx1201PKDhS0_Pfiiii
    .private_segment_fixed_size: 0
    .sgpr_count:     18
    .sgpr_spill_count: 0
    .symbol:         _Z19matrix_core_gfx1201PKDhS0_Pfiiii.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     20
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
