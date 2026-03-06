; ====================================================================
; NOP Loop Baseline (gfx942 / MI308)
;
; Measures loop overhead without any memory access.
; Two variants: LDS-style loop (3 VALU ops) and global-style loop
; (7 VALU ops including address reconstruction).
;
; Kernel arguments (24 bytes, 8-byte aligned):
;   offset  0: uint64_t* output       (8 bytes) - store results
;   offset  8: uint32_t  num_iters    (4 bytes) - iterations
;   offset 12: uint32_t  mode         (4 bytes) - 0=lds-style, 1=global-style
;   offset 16: padding                (8 bytes)
; ====================================================================

.text

; ====================================================================
; LDS-style overhead kernel (ds_read + wait + 3 ops)
; ====================================================================
.global nop_loop_lds_kernel
.p2align 8
.type nop_loop_lds_kernel,@function

.set s_karg,        0
.set s_bx,          2
.set s_out_ptr,     4
.set s_num_iters,   6
.set s_start_lo,    8
.set s_start_hi,    9
.set s_end_lo,      10
.set s_end_hi,      11
.set s_exec_save,   12  ; [12:13]

.set v_tid,         0
.set v_iter,        1
.set v_addr,        2   ; [2:3]
.set v_out,         4   ; [4:5]
.set v_dummy,       6

nop_loop_lds_kernel:
    s_load_dwordx2 s[s_out_ptr:s_out_ptr+1], s[s_karg:s_karg+1], 0
    s_load_dword   s[s_num_iters],           s[s_karg:s_karg+1], 8
    s_waitcnt lgkmcnt(0)

    v_cmp_eq_u32 vcc, 0, v[v_tid]
    s_and_saveexec_b64 s[s_exec_save:s_exec_save+1], vcc
    s_cbranch_execz L_nop_lds_exit

    v_mov_b32 v[v_iter], s[s_num_iters]
    v_mov_b32 v[v_dummy], 0

    s_waitcnt vmcnt(0) lgkmcnt(0)
    s_memrealtime s[s_start_lo:s_start_hi]
    s_waitcnt lgkmcnt(0)

    ; Loop body mimics LDS chase loop structure:
    ;   (no ds_read - just the overhead)
    ;   s_waitcnt lgkmcnt(0)  ; included as nop since nothing pending
    ;   v_sub_u32
    ;   v_cmp_gt_u32
    ;   s_cbranch_vccnz
L_nop_lds_loop:
    s_waitcnt lgkmcnt(0)
    v_sub_u32 v[v_iter], v[v_iter], 1
    v_cmp_gt_u32 vcc, v[v_iter], 0
    s_cbranch_vccnz L_nop_lds_loop

    s_waitcnt vmcnt(0) lgkmcnt(0)
    s_memrealtime s[s_end_lo:s_end_hi]
    s_waitcnt lgkmcnt(0)

    v_mov_b32 v[v_addr], s[s_out_ptr]
    v_mov_b32 v[v_addr+1], s[s_out_ptr+1]
    v_mov_b32 v[v_out], s[s_start_lo]
    v_mov_b32 v[v_out+1], s[s_start_hi]
    global_store_dwordx2 v[v_addr:v_addr+1], v[v_out:v_out+1], off
    s_waitcnt vmcnt(0)

    v_mov_b32 v[v_out], s[s_end_lo]
    v_mov_b32 v[v_out+1], s[s_end_hi]
    v_add_co_u32 v[v_addr], vcc, 8, v[v_addr]
    v_addc_co_u32 v[v_addr+1], vcc, 0, v[v_addr+1], vcc
    global_store_dwordx2 v[v_addr:v_addr+1], v[v_out:v_out+1], off
    s_waitcnt vmcnt(0)

L_nop_lds_exit:
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel nop_loop_lds_kernel
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 7
    .amdhsa_next_free_sgpr 14
    .amdhsa_accum_offset 8
    .amdhsa_ieee_mode 0
    .amdhsa_dx10_clamp 0
.end_amdhsa_kernel

; ====================================================================
; Global-style overhead kernel (load + wait + addr recon + 3 ops)
; ====================================================================
.text
.global nop_loop_global_kernel
.p2align 8
.type nop_loop_global_kernel,@function

nop_loop_global_kernel:
    s_load_dwordx2 s[s_out_ptr:s_out_ptr+1], s[s_karg:s_karg+1], 0
    s_load_dword   s[s_num_iters],           s[s_karg:s_karg+1], 8
    s_waitcnt lgkmcnt(0)

    v_cmp_eq_u32 vcc, 0, v[v_tid]
    s_and_saveexec_b64 s[s_exec_save:s_exec_save+1], vcc
    s_cbranch_execz L_nop_global_exit

    v_mov_b32 v[v_iter], s[s_num_iters]
    v_mov_b32 v[v_dummy], 0

    s_waitcnt vmcnt(0) lgkmcnt(0)
    s_memrealtime s[s_start_lo:s_start_hi]
    s_waitcnt lgkmcnt(0)

    ; Loop body mimics global chase loop structure:
    ;   (no global_load - just the overhead)
    ;   s_waitcnt vmcnt(0)
    ;   v_mov_b32 x2
    ;   v_add_co_u32
    ;   v_addc_co_u32
    ;   v_sub_u32
    ;   v_cmp_gt_u32
    ;   s_cbranch_vccnz
L_nop_global_loop:
    s_waitcnt vmcnt(0)
    v_mov_b32 v[v_addr], 0
    v_mov_b32 v[v_addr+1], 0
    v_add_co_u32 v[v_addr], vcc, v[v_dummy], v[v_addr]
    v_addc_co_u32 v[v_addr+1], vcc, 0, v[v_addr+1], vcc
    v_sub_u32 v[v_iter], v[v_iter], 1
    v_cmp_gt_u32 vcc, v[v_iter], 0
    s_cbranch_vccnz L_nop_global_loop

    s_waitcnt vmcnt(0) lgkmcnt(0)
    s_memrealtime s[s_end_lo:s_end_hi]
    s_waitcnt lgkmcnt(0)

    v_mov_b32 v[v_addr], s[s_out_ptr]
    v_mov_b32 v[v_addr+1], s[s_out_ptr+1]
    v_mov_b32 v[v_out], s[s_start_lo]
    v_mov_b32 v[v_out+1], s[s_start_hi]
    global_store_dwordx2 v[v_addr:v_addr+1], v[v_out:v_out+1], off
    s_waitcnt vmcnt(0)

    v_mov_b32 v[v_out], s[s_end_lo]
    v_mov_b32 v[v_out+1], s[s_end_hi]
    v_add_co_u32 v[v_addr], vcc, 8, v[v_addr]
    v_addc_co_u32 v[v_addr+1], vcc, 0, v[v_addr+1], vcc
    global_store_dwordx2 v[v_addr:v_addr+1], v[v_out:v_out+1], off
    s_waitcnt vmcnt(0)

L_nop_global_exit:
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel nop_loop_global_kernel
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 7
    .amdhsa_next_free_sgpr 14
    .amdhsa_accum_offset 8
    .amdhsa_ieee_mode 0
    .amdhsa_dx10_clamp 0
.end_amdhsa_kernel

; ====================================================================
; AMDGPU metadata
; ====================================================================
.amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: nop_loop_lds_kernel
    .symbol: nop_loop_lds_kernel.kd
    .kernarg_segment_size: 24
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 14
    .vgpr_count: 7
    .max_flat_workgroup_size: 64
    .args:
    - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
    - { .size: 4, .offset: 8, .value_kind: by_value }
    - { .size: 4, .offset: 12, .value_kind: by_value }
  - .name: nop_loop_global_kernel
    .symbol: nop_loop_global_kernel.kd
    .kernarg_segment_size: 24
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 14
    .vgpr_count: 7
    .max_flat_workgroup_size: 64
    .args:
    - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
    - { .size: 4, .offset: 8, .value_kind: by_value }
    - { .size: 4, .offset: 12, .value_kind: by_value }
...
.end_amdgpu_metadata
