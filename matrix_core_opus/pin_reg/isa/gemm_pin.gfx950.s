	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z10gemm_plainPKDF16_S0_Pf ; -- Begin function _Z10gemm_plainPKDF16_S0_Pf
	.globl	_Z10gemm_plainPKDF16_S0_Pf
	.p2align	8
	.type	_Z10gemm_plainPKDF16_S0_Pf,@function
_Z10gemm_plainPKDF16_S0_Pf:             ; @_Z10gemm_plainPKDF16_S0_Pf
; %bb.0:                                ; %entry
	s_load_dwordx4 s[4:7], s[0:1], 0x0
	s_load_dwordx2 s[2:3], s[0:1], 0x10
	v_lshlrev_b32_e32 v1, 3, v0
	v_lshlrev_b32_e32 v0, 4, v0
	s_waitcnt lgkmcnt(0)
	global_load_dwordx2 v[2:3], v1, s[4:5]
	global_load_dwordx2 v[4:5], v1, s[6:7]
	s_waitcnt vmcnt(0)
	v_mfma_f32_16x16x16_f16 v[2:5], v[2:3], v[4:5], 0
	s_nop 7
	global_store_dwordx4 v0, v[2:5], s[2:3]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z10gemm_plainPKDF16_S0_Pf
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 24
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
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 6
		.amdhsa_next_free_sgpr 8
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
	.size	_Z10gemm_plainPKDF16_S0_Pf, .Lfunc_end0-_Z10gemm_plainPKDF16_S0_Pf
                                        ; -- End function
	.set _Z10gemm_plainPKDF16_S0_Pf.num_vgpr, 6
	.set _Z10gemm_plainPKDF16_S0_Pf.num_agpr, 0
	.set _Z10gemm_plainPKDF16_S0_Pf.numbered_sgpr, 8
	.set _Z10gemm_plainPKDF16_S0_Pf.private_seg_size, 0
	.set _Z10gemm_plainPKDF16_S0_Pf.uses_vcc, 0
	.set _Z10gemm_plainPKDF16_S0_Pf.uses_flat_scratch, 0
	.set _Z10gemm_plainPKDF16_S0_Pf.has_dyn_sized_stack, 0
	.set _Z10gemm_plainPKDF16_S0_Pf.has_recursion, 0
	.set _Z10gemm_plainPKDF16_S0_Pf.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 72
; TotalNumSgprs: 14
; NumVgprs: 6
; NumAgprs: 0
; TotalNumVgprs: 6
; ScratchSize: 0
; MemoryBound: 1
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 14
; NumVGPRsForWavesPerEU: 6
; AccumOffset: 8
; Occupancy: 8
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 1
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.protected	_Z11gemm_pinnedPKDF16_S0_Pf ; -- Begin function _Z11gemm_pinnedPKDF16_S0_Pf
	.globl	_Z11gemm_pinnedPKDF16_S0_Pf
	.p2align	8
	.type	_Z11gemm_pinnedPKDF16_S0_Pf,@function
_Z11gemm_pinnedPKDF16_S0_Pf:            ; @_Z11gemm_pinnedPKDF16_S0_Pf
; %bb.0:                                ; %entry
	s_load_dwordx4 s[4:7], s[0:1], 0x0
	s_load_dwordx2 s[2:3], s[0:1], 0x10
	v_mov_b32_e32 v4, v0
	v_lshlrev_b32_e32 v0, 3, v4
	v_lshlrev_b32_e32 v4, 4, v4
	s_waitcnt lgkmcnt(0)
	global_load_dwordx2 a[0:1], v0, s[4:5]
	global_load_dwordx2 a[2:3], v0, s[6:7]
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v1, v0
	v_mov_b32_e32 v2, v0
	v_mov_b32_e32 v3, v0
	s_waitcnt vmcnt(0)
	s_nop 0
	v_mfma_f32_16x16x16_f16 v[0:3], a[0:1], a[2:3], v[0:3]
	s_nop 7
	global_store_dwordx4 v4, v[0:3], s[2:3]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z11gemm_pinnedPKDF16_S0_Pf
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 24
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
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 12
		.amdhsa_next_free_sgpr 8
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
.Lfunc_end1:
	.size	_Z11gemm_pinnedPKDF16_S0_Pf, .Lfunc_end1-_Z11gemm_pinnedPKDF16_S0_Pf
                                        ; -- End function
	.set _Z11gemm_pinnedPKDF16_S0_Pf.num_vgpr, 5
	.set _Z11gemm_pinnedPKDF16_S0_Pf.num_agpr, 4
	.set _Z11gemm_pinnedPKDF16_S0_Pf.numbered_sgpr, 8
	.set _Z11gemm_pinnedPKDF16_S0_Pf.private_seg_size, 0
	.set _Z11gemm_pinnedPKDF16_S0_Pf.uses_vcc, 0
	.set _Z11gemm_pinnedPKDF16_S0_Pf.uses_flat_scratch, 0
	.set _Z11gemm_pinnedPKDF16_S0_Pf.has_dyn_sized_stack, 0
	.set _Z11gemm_pinnedPKDF16_S0_Pf.has_recursion, 0
	.set _Z11gemm_pinnedPKDF16_S0_Pf.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 96
; TotalNumSgprs: 14
; NumVgprs: 5
; NumAgprs: 4
; TotalNumVgprs: 12
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 1
; NumSGPRsForWavesPerEU: 14
; NumVGPRsForWavesPerEU: 12
; AccumOffset: 8
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 1
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_9cc4e50cfe255323,@object ; @__hip_cuid_9cc4e50cfe255323
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_9cc4e50cfe255323
__hip_cuid_9cc4e50cfe255323:
	.byte	0                               ; 0x0
	.size	__hip_cuid_9cc4e50cfe255323, 1

	.ident	"clang version 20.0.0git (https://github.com/ROCm/llvm-project.git 27682a16360e33e37c4f3cc6adf9a620733f8fe1)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_9cc4e50cfe255323
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .name:           A.coerce
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .name:           B.coerce
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .name:           C.coerce
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .max_flat_workgroup_size: 1024
    .name:           _Z10gemm_plainPKDF16_S0_Pf
    .private_segment_fixed_size: 0
    .sgpr_count:     14
    .sgpr_spill_count: 0
    .symbol:         _Z10gemm_plainPKDF16_S0_Pf.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     6
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .agpr_count:     4
    .args:
      - .address_space:  global
        .name:           A.coerce
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .name:           B.coerce
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .name:           C.coerce
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .max_flat_workgroup_size: 1024
    .name:           _Z11gemm_pinnedPKDF16_S0_Pf
    .private_segment_fixed_size: 0
    .sgpr_count:     14
    .sgpr_spill_count: 0
    .symbol:         _Z11gemm_pinnedPKDF16_S0_Pf.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     12
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
