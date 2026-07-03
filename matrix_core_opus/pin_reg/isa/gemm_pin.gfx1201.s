	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z10gemm_plainPKDF16_S0_Pf ; -- Begin function _Z10gemm_plainPKDF16_S0_Pf
	.globl	_Z10gemm_plainPKDF16_S0_Pf
	.p2align	8
	.type	_Z10gemm_plainPKDF16_S0_Pf,@function
_Z10gemm_plainPKDF16_S0_Pf:             ; @_Z10gemm_plainPKDF16_S0_Pf
; %bb.0:                                ; %entry
	s_load_b128 s[4:7], s[0:1], 0x0
	v_lshlrev_b32_e32 v1, 4, v0
	s_load_b64 s[0:1], s[0:1], 0x10
	v_lshlrev_b32_e32 v16, 5, v0
	s_wait_kmcnt 0x0
	s_clause 0x1
	global_load_b128 v[8:11], v1, s[4:5]
	global_load_b128 v[12:15], v1, s[6:7]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_f16 v[0:7], v[8:11], v[12:15], 0
	s_clause 0x1
	global_store_b128 v16, v[4:7], s[0:1] offset:16
	global_store_b128 v16, v[0:3], s[0:1]
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
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 17
		.amdhsa_next_free_sgpr 8
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
	.size	_Z10gemm_plainPKDF16_S0_Pf, .Lfunc_end0-_Z10gemm_plainPKDF16_S0_Pf
                                        ; -- End function
	.set _Z10gemm_plainPKDF16_S0_Pf.num_vgpr, 17
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
; codeLenInByte = 100
; TotalNumSgprs: 8
; NumVgprs: 17
; ScratchSize: 0
; MemoryBound: 1
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 8
; NumVGPRsForWavesPerEU: 17
; Occupancy: 16
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	_Z11gemm_pinnedPKDF16_S0_Pf ; -- Begin function _Z11gemm_pinnedPKDF16_S0_Pf
	.globl	_Z11gemm_pinnedPKDF16_S0_Pf
	.p2align	8
	.type	_Z11gemm_pinnedPKDF16_S0_Pf,@function
_Z11gemm_pinnedPKDF16_S0_Pf:            ; @_Z11gemm_pinnedPKDF16_S0_Pf
; %bb.0:                                ; %entry
	s_load_b128 s[4:7], s[0:1], 0x0
	v_mov_b32_e32 v16, v0
	s_load_b64 s[0:1], s[0:1], 0x10
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v0, 4, v16
	s_wait_kmcnt 0x0
	s_clause 0x1
	global_load_b128 v[8:11], v0, s[4:5]
	global_load_b128 v[12:15], v0, s[6:7]
	v_mov_b32_e32 v0, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v1, v0 :: v_dual_mov_b32 v2, v0
	v_dual_mov_b32 v3, v0 :: v_dual_mov_b32 v4, v0
	v_dual_mov_b32 v5, v0 :: v_dual_mov_b32 v6, v0
	v_mov_b32_e32 v7, v0
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x16_f16 v[0:7], v[8:11], v[12:15], v[0:7]
	v_lshlrev_b32_e32 v8, 5, v16
	s_clause 0x1
	global_store_b128 v8, v[4:7], s[0:1] offset:16
	global_store_b128 v8, v[0:3], s[0:1]
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
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 17
		.amdhsa_next_free_sgpr 8
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
.Lfunc_end1:
	.size	_Z11gemm_pinnedPKDF16_S0_Pf, .Lfunc_end1-_Z11gemm_pinnedPKDF16_S0_Pf
                                        ; -- End function
	.set _Z11gemm_pinnedPKDF16_S0_Pf.num_vgpr, 17
	.set _Z11gemm_pinnedPKDF16_S0_Pf.num_agpr, 0
	.set _Z11gemm_pinnedPKDF16_S0_Pf.numbered_sgpr, 8
	.set _Z11gemm_pinnedPKDF16_S0_Pf.private_seg_size, 0
	.set _Z11gemm_pinnedPKDF16_S0_Pf.uses_vcc, 0
	.set _Z11gemm_pinnedPKDF16_S0_Pf.uses_flat_scratch, 0
	.set _Z11gemm_pinnedPKDF16_S0_Pf.has_dyn_sized_stack, 0
	.set _Z11gemm_pinnedPKDF16_S0_Pf.has_recursion, 0
	.set _Z11gemm_pinnedPKDF16_S0_Pf.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 144
; TotalNumSgprs: 8
; NumVgprs: 17
; ScratchSize: 0
; MemoryBound: 1
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 8
; NumVGPRsForWavesPerEU: 17
; Occupancy: 16
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
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
	.type	__hip_cuid_6ec8f804a44238ec,@object ; @__hip_cuid_6ec8f804a44238ec
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_6ec8f804a44238ec
__hip_cuid_6ec8f804a44238ec:
	.byte	0                               ; 0x0
	.size	__hip_cuid_6ec8f804a44238ec, 1

	.ident	"clang version 20.0.0git (https://github.com/ROCm/llvm-project.git 27682a16360e33e37c4f3cc6adf9a620733f8fe1)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_6ec8f804a44238ec
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
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
    .sgpr_count:     8
    .sgpr_spill_count: 0
    .symbol:         _Z10gemm_plainPKDF16_S0_Pf.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     17
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
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
    .sgpr_count:     8
    .sgpr_spill_count: 0
    .symbol:         _Z11gemm_pinnedPKDF16_S0_Pf.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     17
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
