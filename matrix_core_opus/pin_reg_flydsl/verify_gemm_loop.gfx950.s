	.section	.AMDGPU.config,"",@progbits
	.long	47176
	.long	11468875
	.long	47180
	.long	5008
	.long	47200
	.long	0
	.long	4
	.long	0
	.long	8
	.long	0
	.text
	.globl	gemm_loop                       ; -- Begin function gemm_loop
	.p2align	8
	.type	gemm_loop,@function
gemm_loop:                              ; @gemm_loop
; %bb.0:                                ; %entry
	s_load_dwordx4 s[0:3], s[4:5], 0x24
	s_load_dwordx2 s[6:7], s[4:5], 0x34
	s_load_dword s8, s[4:5], 0x3c
	v_mov_b32_e32 v2, 0
	s_mov_b32 s4, 0
	v_and_b32_e32 v0, 0x3ff, v0
	v_mov_b32_e32 v3, v2
	v_mov_b32_e32 v4, v2
	v_mov_b32_e32 v5, v2
	v_mov_b32_e32 v6, v2
	v_mov_b32_e32 v7, v2
	v_mov_b32_e32 v8, v2
	v_mov_b32_e32 v9, v2
	v_mov_b32_e32 v10, v2
	v_mov_b32_e32 v11, v2
	v_mov_b32_e32 v12, v2
	v_mov_b32_e32 v13, v2
	v_mov_b32_e32 v14, v2
	v_mov_b32_e32 v15, v2
	v_mov_b32_e32 v16, v2
	v_mov_b32_e32 v17, v2
.LBB0_1:                                ; %loop
                                        ; =>This Inner Loop Header: Depth=1
	v_add_u32_e32 v18, s4, v0
	v_ashrrev_i32_e32 v19, 31, v18
	v_lshlrev_b64 v[18:19], 3, v[18:19]
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[20:21], s[0:1], 0, v[18:19]
	v_lshl_add_u64 v[18:19], s[2:3], 0, v[18:19]
	global_load_dwordx2 a[0:1], v[20:21], off
	global_load_dwordx2 a[64:65], v[18:19], off
	global_load_dwordx2 a[66:67], v[18:19], off offset:512
	global_load_dwordx2 a[68:69], v[18:19], off offset:1024
	global_load_dwordx2 a[70:71], v[18:19], off offset:1536
	s_add_i32 s4, s4, 1
	s_cmp_lt_i32 s4, s8
	s_waitcnt vmcnt(3)
	v_mfma_f32_16x16x16_f16 v[14:17], a[0:1], a[64:65], v[14:17]
	s_waitcnt vmcnt(2)
	v_mfma_f32_16x16x16_f16 v[10:13], a[0:1], a[66:67], v[10:13]
	s_waitcnt vmcnt(1)
	v_mfma_f32_16x16x16_f16 v[6:9], a[0:1], a[68:69], v[6:9]
	s_waitcnt vmcnt(0)
	v_mfma_f32_16x16x16_f16 v[2:5], a[0:1], a[70:71], v[2:5]
	s_cbranch_scc1 .LBB0_1
; %bb.2:                                ; %exit
	v_mov_b32_e32 v0, 0
	global_store_dwordx4 v0, v[14:17], s[6:7]
	s_nop 0
	global_store_dwordx4 v0, v[10:13], s[6:7] offset:16
	s_nop 0
	global_store_dwordx4 v0, v[6:9], s[6:7] offset:32
	s_nop 0
	global_store_dwordx4 v0, v[2:5], s[6:7] offset:48
	s_endpgm
.Lfunc_end0:
	.size	gemm_loop, .Lfunc_end0-gemm_loop
                                        ; -- End function
	.set gemm_loop.num_vgpr, 22
	.set gemm_loop.num_agpr, 72
	.set gemm_loop.numbered_sgpr, 9
	.set gemm_loop.num_named_barrier, 0
	.set gemm_loop.private_seg_size, 0
	.set gemm_loop.uses_vcc, 0
	.set gemm_loop.uses_flat_scratch, 0
	.set gemm_loop.has_dyn_sized_stack, 0
	.set gemm_loop.has_recursion, 0
	.set gemm_loop.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 288
; TotalNumSgprs: 15
; NumVgprs: 22
; NumAgprs: 72
; TotalNumVgprs: 96
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 11
; NumSGPRsForWavesPerEU: 15
; NumVGPRsForWavesPerEU: 96
; AccumOffset: 24
; Occupancy: 5
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 8
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 2
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 5
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.section	.AMDGPU.csdata,"",@progbits
	.section	".note.GNU-stack","",@progbits
	.amd_amdgpu_isa "amdgcn----gfx950"
