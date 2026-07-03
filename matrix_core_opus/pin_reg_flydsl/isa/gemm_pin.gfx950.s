	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.globl	mfma_kernel_0                   ; -- Begin function mfma_kernel_0
	.p2align	8
	.type	mfma_kernel_0,@function
mfma_kernel_0:                          ; @mfma_kernel_0
; %bb.0:
	s_load_dwordx2 s[0:1], s[4:5], 0x0
	v_and_b32_e32 v0, 0x3ff, v0
	s_mov_b32 s3, 0x27000
	s_load_dwordx2 s[8:9], s[4:5], 0x10
	s_load_dwordx2 s[6:7], s[4:5], 0x20
	s_mov_b32 s2, -1
	s_waitcnt lgkmcnt(0)
	s_and_b32 s1, s1, 0xffff
	v_mul_u32_u24_e32 v1, 8, v0
	buffer_load_dwordx2 a[0:1], v1, s[0:3], 0 offen
	s_and_b32 s9, s9, 0xffff
	s_mov_b32 s10, s2
	s_mov_b32 s11, s3
	buffer_load_dwordx2 a[64:65], v1, s[8:11], 0 offen
	v_mul_u32_u24_e32 v0, 4, v0
	v_lshlrev_b32_e32 v4, 2, v0
	s_waitcnt vmcnt(0)
	v_mfma_f32_16x16x16_f16 v[0:3], a[0:1], a[64:65], 0
	s_nop 7
	global_store_dwordx4 v4, v[0:3], s[6:7]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel mfma_kernel_0
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 304
		.amdhsa_user_sgpr_count 8
		.amdhsa_user_sgpr_dispatch_ptr 1
		.amdhsa_user_sgpr_queue_ptr 1
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 1
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 2
		.amdhsa_next_free_vgpr 74
		.amdhsa_next_free_sgpr 12
		.amdhsa_accum_offset 8
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
	.size	mfma_kernel_0, .Lfunc_end0-mfma_kernel_0
                                        ; -- End function
	.set mfma_kernel_0.num_vgpr, 5
	.set mfma_kernel_0.num_agpr, 66
	.set mfma_kernel_0.numbered_sgpr, 12
	.set mfma_kernel_0.num_named_barrier, 0
	.set mfma_kernel_0.private_seg_size, 0
	.set mfma_kernel_0.uses_vcc, 0
	.set mfma_kernel_0.uses_flat_scratch, 0
	.set mfma_kernel_0.has_dyn_sized_stack, 0
	.set mfma_kernel_0.has_recursion, 0
	.set mfma_kernel_0.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 128
; TotalNumSgprs: 18
; NumVgprs: 5
; NumAgprs: 66
; TotalNumVgprs: 74
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 9
; NumSGPRsForWavesPerEU: 18
; NumVGPRsForWavesPerEU: 74
; AccumOffset: 8
; Occupancy: 6
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 8
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 2
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 1
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     66
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         48
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         52
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         56
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         60
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         62
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         64
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         66
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         68
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         70
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         88
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         96
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         104
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         112
        .size:           2
        .value_kind:     hidden_grid_dims
      - .offset:         128
        .size:           8
        .value_kind:     hidden_hostcall_buffer
      - .offset:         136
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
      - .offset:         144
        .size:           8
        .value_kind:     hidden_heap_v1
      - .offset:         152
        .size:           8
        .value_kind:     hidden_default_queue
      - .offset:         160
        .size:           8
        .value_kind:     hidden_completion_action
      - .offset:         248
        .size:           8
        .value_kind:     hidden_queue_ptr
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 304
    .max_flat_workgroup_size: 256
    .name:           mfma_kernel_0
    .private_segment_fixed_size: 0
    .sgpr_count:     18
    .sgpr_spill_count: 0
    .symbol:         mfma_kernel_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     74
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
