; ====================================================================
; LDS Detailed Latency Benchmarks (gfx942 / MI308)
;
; 6 kernels measuring ds_read/write latency for b32, b64, b128.
;
; READs use pointer chasing:
;   b32:  ds_read_b32  v[chase], v[chase]
;   b64:  ds_read_b64  v[data:data+1], v[chase]  ->  v_mov chase, data
;   b128: ds_read_b128 v[data:data+3], v[chase]  ->  v_mov chase, data
;
; WRITEs measure write-and-wait latency to a fixed address:
;   ds_write_bXX v[addr], v[data...]
;   s_waitcnt lgkmcnt(0)
;
; All kernels: single lane 0, s_memrealtime timing, VGPR loop counter.
;
; Kernel arguments (24 bytes, 8-byte aligned):
;   offset  0: void*     chase_array  (8 bytes) - for reads; unused for writes
;   offset  8: uint64_t* output       (8 bytes) - [start, end] cycles
;   offset 16: uint32_t  num_entries  (4 bytes) - for reads
;   offset 20: uint32_t  num_iters    (4 bytes)
; ====================================================================

; ===================== Common SGPR layout =====================
.set s_karg,        0       ; [0:1]
.set s_bx,          2
.set s_chase_ptr,   4       ; [4:5]
.set s_out_ptr,     6       ; [6:7]
.set s_num_entries, 8
.set s_num_iters,   9
.set s_start_lo,    10
.set s_start_hi,    11
.set s_end_lo,      12
.set s_end_hi,      13
.set s_cnt,         14
.set s_iter,        15
.set s_tmp,         16
.set s_exec_save,   18      ; [18:19]

; ===================== Macros =====================

; Copy chase array from global to LDS, dword-by-dword
; Uses v2:v3 as addr, v4 as data, v5 as lds_off
.macro COPY_CHASE_TO_LDS
    s_mov_b32 s[s_cnt], 0
.Lcopy_\@:
    s_lshl_b32 s[s_tmp], s[s_cnt], 2
    v_mov_b32 v2, s[s_chase_ptr]
    v_mov_b32 v3, s[s_chase_ptr+1]
    v_add_co_u32 v2, vcc, s[s_tmp], v2
    v_addc_co_u32 v3, vcc, 0, v3, vcc
    global_load_dword v4, v[2:3], off
    s_waitcnt vmcnt(0)
    v_mov_b32 v5, s[s_tmp]
    ds_write_b32 v5, v4
    s_waitcnt lgkmcnt(0)
    s_add_u32 s[s_cnt], s[s_cnt], 1
    s_cmp_lt_u32 s[s_cnt], s[s_num_entries]
    s_cbranch_scc1 .Lcopy_\@
    s_barrier
.endm

; Store start/end cycles to output buffer
; Uses v2:v3 as addr, v4:v5 as data
.macro STORE_RESULTS
    v_mov_b32 v2, s[s_out_ptr]
    v_mov_b32 v3, s[s_out_ptr+1]
    v_mov_b32 v4, s[s_start_lo]
    v_mov_b32 v5, s[s_start_hi]
    global_store_dwordx2 v[2:3], v[4:5], off
    s_waitcnt vmcnt(0)
    v_mov_b32 v4, s[s_end_lo]
    v_mov_b32 v5, s[s_end_hi]
    v_add_co_u32 v2, vcc, 8, v2
    v_addc_co_u32 v3, vcc, 0, v3, vcc
    global_store_dwordx2 v[2:3], v[4:5], off
    s_waitcnt vmcnt(0)
.endm

; ====================================================================
; KERNEL 1: lds_read_b32_kernel
; ====================================================================
.text
.global lds_read_b32_kernel
.p2align 8
.type lds_read_b32_kernel,@function

; VGPR layout for read_b32:
; v0=tid, v1=iter, v2:v3=addr64, v4=data, v5=lds_off, v6=chase
lds_read_b32_kernel:
    s_load_dwordx2 s[s_chase_ptr:s_chase_ptr+1], s[s_karg:s_karg+1], 0
    s_load_dwordx2 s[s_out_ptr:s_out_ptr+1],     s[s_karg:s_karg+1], 8
    s_load_dword   s[s_num_entries],              s[s_karg:s_karg+1], 16
    s_load_dword   s[s_num_iters],                s[s_karg:s_karg+1], 20
    s_waitcnt lgkmcnt(0)

    v_cmp_eq_u32 vcc, 0, v0
    s_and_saveexec_b64 s[s_exec_save:s_exec_save+1], vcc
    s_cbranch_execz .Lexit_rb32

    COPY_CHASE_TO_LDS

    ; Warmup
    v_mov_b32 v6, 0
    s_mov_b32 s[s_iter], 64
.Lwarm_rb32:
    ds_read_b32 v6, v6
    s_waitcnt lgkmcnt(0)
    s_sub_u32 s[s_iter], s[s_iter], 1
    s_cbranch_scc1 .Lwarm_rb32

    ; Measure
    v_mov_b32 v6, 0
    v_mov_b32 v1, s[s_num_iters]
    s_waitcnt vmcnt(0) lgkmcnt(0)
    s_memrealtime s[s_start_lo:s_start_hi]
    s_waitcnt lgkmcnt(0)

.Lloop_rb32:
    ds_read_b32 v6, v6
    s_waitcnt lgkmcnt(0)
    v_sub_u32 v1, v1, 1
    v_cmp_gt_u32 vcc, v1, 0
    s_cbranch_vccnz .Lloop_rb32

    s_waitcnt vmcnt(0) lgkmcnt(0)
    s_memrealtime s[s_end_lo:s_end_hi]
    s_waitcnt lgkmcnt(0)

    STORE_RESULTS

.Lexit_rb32:
    s_endpgm

; ====================================================================
; KERNEL 2: lds_read_b64_kernel
; ====================================================================
.text
.global lds_read_b64_kernel
.p2align 8
.type lds_read_b64_kernel,@function

; VGPR layout for read_b64:
; v0=tid, v1=iter, v2:v3=addr64, v4:v5=data64(even-aligned), v6=chase, v7=unused
lds_read_b64_kernel:
    s_load_dwordx2 s[s_chase_ptr:s_chase_ptr+1], s[s_karg:s_karg+1], 0
    s_load_dwordx2 s[s_out_ptr:s_out_ptr+1],     s[s_karg:s_karg+1], 8
    s_load_dword   s[s_num_entries],              s[s_karg:s_karg+1], 16
    s_load_dword   s[s_num_iters],                s[s_karg:s_karg+1], 20
    s_waitcnt lgkmcnt(0)

    v_cmp_eq_u32 vcc, 0, v0
    s_and_saveexec_b64 s[s_exec_save:s_exec_save+1], vcc
    s_cbranch_execz .Lexit_rb64

    COPY_CHASE_TO_LDS

    ; Warmup
    v_mov_b32 v6, 0
    s_mov_b32 s[s_iter], 64
.Lwarm_rb64:
    ds_read_b64 v[4:5], v6
    s_waitcnt lgkmcnt(0)
    v_mov_b32 v6, v4
    s_sub_u32 s[s_iter], s[s_iter], 1
    s_cbranch_scc1 .Lwarm_rb64

    ; Measure
    v_mov_b32 v6, 0
    v_mov_b32 v1, s[s_num_iters]
    s_waitcnt vmcnt(0) lgkmcnt(0)
    s_memrealtime s[s_start_lo:s_start_hi]
    s_waitcnt lgkmcnt(0)

.Lloop_rb64:
    ds_read_b64 v[4:5], v6
    s_waitcnt lgkmcnt(0)
    v_mov_b32 v6, v4          ; chase = first dword of loaded pair
    v_sub_u32 v1, v1, 1
    v_cmp_gt_u32 vcc, v1, 0
    s_cbranch_vccnz .Lloop_rb64

    s_waitcnt vmcnt(0) lgkmcnt(0)
    s_memrealtime s[s_end_lo:s_end_hi]
    s_waitcnt lgkmcnt(0)

    STORE_RESULTS

.Lexit_rb64:
    s_endpgm

; ====================================================================
; KERNEL 3: lds_read_b128_kernel
; ====================================================================
.text
.global lds_read_b128_kernel
.p2align 8
.type lds_read_b128_kernel,@function

; VGPR layout for read_b128:
; v0=tid, v1=iter, v2:v3=addr64, v4:v7=data128(4-aligned), v8=chase
lds_read_b128_kernel:
    s_load_dwordx2 s[s_chase_ptr:s_chase_ptr+1], s[s_karg:s_karg+1], 0
    s_load_dwordx2 s[s_out_ptr:s_out_ptr+1],     s[s_karg:s_karg+1], 8
    s_load_dword   s[s_num_entries],              s[s_karg:s_karg+1], 16
    s_load_dword   s[s_num_iters],                s[s_karg:s_karg+1], 20
    s_waitcnt lgkmcnt(0)

    v_cmp_eq_u32 vcc, 0, v0
    s_and_saveexec_b64 s[s_exec_save:s_exec_save+1], vcc
    s_cbranch_execz .Lexit_rb128

    COPY_CHASE_TO_LDS

    ; Warmup
    v_mov_b32 v8, 0
    s_mov_b32 s[s_iter], 64
.Lwarm_rb128:
    ds_read_b128 v[4:7], v8
    s_waitcnt lgkmcnt(0)
    v_mov_b32 v8, v4
    s_sub_u32 s[s_iter], s[s_iter], 1
    s_cbranch_scc1 .Lwarm_rb128

    ; Measure
    v_mov_b32 v8, 0
    v_mov_b32 v1, s[s_num_iters]
    s_waitcnt vmcnt(0) lgkmcnt(0)
    s_memrealtime s[s_start_lo:s_start_hi]
    s_waitcnt lgkmcnt(0)

.Lloop_rb128:
    ds_read_b128 v[4:7], v8
    s_waitcnt lgkmcnt(0)
    v_mov_b32 v8, v4          ; chase = first dword of loaded quad
    v_sub_u32 v1, v1, 1
    v_cmp_gt_u32 vcc, v1, 0
    s_cbranch_vccnz .Lloop_rb128

    s_waitcnt vmcnt(0) lgkmcnt(0)
    s_memrealtime s[s_end_lo:s_end_hi]
    s_waitcnt lgkmcnt(0)

    STORE_RESULTS

.Lexit_rb128:
    s_endpgm

; ====================================================================
; KERNEL 4: lds_write_b32_kernel
; ====================================================================
.text
.global lds_write_b32_kernel
.p2align 8
.type lds_write_b32_kernel,@function

; VGPR: v0=tid, v1=iter, v2:v3=addr64(store results), v4:v5=out data, v6=lds_addr, v7=wdata
lds_write_b32_kernel:
    s_load_dwordx2 s[s_out_ptr:s_out_ptr+1], s[s_karg:s_karg+1], 8
    s_load_dword   s[s_num_iters],           s[s_karg:s_karg+1], 20
    s_waitcnt lgkmcnt(0)

    v_cmp_eq_u32 vcc, 0, v0
    s_and_saveexec_b64 s[s_exec_save:s_exec_save+1], vcc
    s_cbranch_execz .Lexit_wb32

    v_mov_b32 v6, 0       ; write to LDS offset 0
    v_mov_b32 v7, 0x42    ; write data (arbitrary)
    v_mov_b32 v1, s[s_num_iters]

    s_waitcnt vmcnt(0) lgkmcnt(0)
    s_memrealtime s[s_start_lo:s_start_hi]
    s_waitcnt lgkmcnt(0)

.Lloop_wb32:
    ds_write_b32 v6, v7
    s_waitcnt lgkmcnt(0)
    v_sub_u32 v1, v1, 1
    v_cmp_gt_u32 vcc, v1, 0
    s_cbranch_vccnz .Lloop_wb32

    s_waitcnt vmcnt(0) lgkmcnt(0)
    s_memrealtime s[s_end_lo:s_end_hi]
    s_waitcnt lgkmcnt(0)

    STORE_RESULTS

.Lexit_wb32:
    s_endpgm

; ====================================================================
; KERNEL 5: lds_write_b64_kernel
; ====================================================================
.text
.global lds_write_b64_kernel
.p2align 8
.type lds_write_b64_kernel,@function

; VGPR: v0=tid, v1=iter, v2:v3=addr64, v4:v5=out, v6=lds_addr, v8:v9=wdata64(even)
lds_write_b64_kernel:
    s_load_dwordx2 s[s_out_ptr:s_out_ptr+1], s[s_karg:s_karg+1], 8
    s_load_dword   s[s_num_iters],           s[s_karg:s_karg+1], 20
    s_waitcnt lgkmcnt(0)

    v_cmp_eq_u32 vcc, 0, v0
    s_and_saveexec_b64 s[s_exec_save:s_exec_save+1], vcc
    s_cbranch_execz .Lexit_wb64

    v_mov_b32 v6, 0
    v_mov_b32 v8, 0x42
    v_mov_b32 v9, 0x43
    v_mov_b32 v1, s[s_num_iters]

    s_waitcnt vmcnt(0) lgkmcnt(0)
    s_memrealtime s[s_start_lo:s_start_hi]
    s_waitcnt lgkmcnt(0)

.Lloop_wb64:
    ds_write_b64 v6, v[8:9]
    s_waitcnt lgkmcnt(0)
    v_sub_u32 v1, v1, 1
    v_cmp_gt_u32 vcc, v1, 0
    s_cbranch_vccnz .Lloop_wb64

    s_waitcnt vmcnt(0) lgkmcnt(0)
    s_memrealtime s[s_end_lo:s_end_hi]
    s_waitcnt lgkmcnt(0)

    STORE_RESULTS

.Lexit_wb64:
    s_endpgm

; ====================================================================
; KERNEL 6: lds_write_b128_kernel
; ====================================================================
.text
.global lds_write_b128_kernel
.p2align 8
.type lds_write_b128_kernel,@function

; VGPR: v0=tid, v1=iter, v2:v3=addr64, v4:v5=out, v6=lds_addr, v8:v11=wdata128(4-aligned)
lds_write_b128_kernel:
    s_load_dwordx2 s[s_out_ptr:s_out_ptr+1], s[s_karg:s_karg+1], 8
    s_load_dword   s[s_num_iters],           s[s_karg:s_karg+1], 20
    s_waitcnt lgkmcnt(0)

    v_cmp_eq_u32 vcc, 0, v0
    s_and_saveexec_b64 s[s_exec_save:s_exec_save+1], vcc
    s_cbranch_execz .Lexit_wb128

    v_mov_b32 v6, 0
    v_mov_b32 v8,  0x42
    v_mov_b32 v9,  0x43
    v_mov_b32 v10, 0x44
    v_mov_b32 v11, 0x45
    v_mov_b32 v1, s[s_num_iters]

    s_waitcnt vmcnt(0) lgkmcnt(0)
    s_memrealtime s[s_start_lo:s_start_hi]
    s_waitcnt lgkmcnt(0)

.Lloop_wb128:
    ds_write_b128 v6, v[8:11]
    s_waitcnt lgkmcnt(0)
    v_sub_u32 v1, v1, 1
    v_cmp_gt_u32 vcc, v1, 0
    s_cbranch_vccnz .Lloop_wb128

    s_waitcnt vmcnt(0) lgkmcnt(0)
    s_memrealtime s[s_end_lo:s_end_hi]
    s_waitcnt lgkmcnt(0)

    STORE_RESULTS

.Lexit_wb128:
    s_endpgm

; ====================================================================
; Kernel descriptors
; ====================================================================
.rodata

.p2align 6
.amdhsa_kernel lds_read_b32_kernel
    .amdhsa_group_segment_fixed_size 65536
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 8
    .amdhsa_next_free_sgpr 20
    .amdhsa_accum_offset 8
    .amdhsa_ieee_mode 0
    .amdhsa_dx10_clamp 0
.end_amdhsa_kernel

.p2align 6
.amdhsa_kernel lds_read_b64_kernel
    .amdhsa_group_segment_fixed_size 65536
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 8
    .amdhsa_next_free_sgpr 20
    .amdhsa_accum_offset 8
    .amdhsa_ieee_mode 0
    .amdhsa_dx10_clamp 0
.end_amdhsa_kernel

.p2align 6
.amdhsa_kernel lds_read_b128_kernel
    .amdhsa_group_segment_fixed_size 65536
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 9
    .amdhsa_next_free_sgpr 20
    .amdhsa_accum_offset 12
    .amdhsa_ieee_mode 0
    .amdhsa_dx10_clamp 0
.end_amdhsa_kernel

.p2align 6
.amdhsa_kernel lds_write_b32_kernel
    .amdhsa_group_segment_fixed_size 65536
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 8
    .amdhsa_next_free_sgpr 20
    .amdhsa_accum_offset 8
    .amdhsa_ieee_mode 0
    .amdhsa_dx10_clamp 0
.end_amdhsa_kernel

.p2align 6
.amdhsa_kernel lds_write_b64_kernel
    .amdhsa_group_segment_fixed_size 65536
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 10
    .amdhsa_next_free_sgpr 20
    .amdhsa_accum_offset 12
    .amdhsa_ieee_mode 0
    .amdhsa_dx10_clamp 0
.end_amdhsa_kernel

.p2align 6
.amdhsa_kernel lds_write_b128_kernel
    .amdhsa_group_segment_fixed_size 65536
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 12
    .amdhsa_next_free_sgpr 20
    .amdhsa_accum_offset 12
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
  - .name: lds_read_b32_kernel
    .symbol: lds_read_b32_kernel.kd
    .kernarg_segment_size: 24
    .group_segment_fixed_size: 65536
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 20
    .vgpr_count: 8
    .max_flat_workgroup_size: 64
    .args:
    - { .size: 8, .offset: 0,  .value_kind: global_buffer, .address_space: global }
    - { .size: 8, .offset: 8,  .value_kind: global_buffer, .address_space: global }
    - { .size: 4, .offset: 16, .value_kind: by_value }
    - { .size: 4, .offset: 20, .value_kind: by_value }
  - .name: lds_read_b64_kernel
    .symbol: lds_read_b64_kernel.kd
    .kernarg_segment_size: 24
    .group_segment_fixed_size: 65536
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 20
    .vgpr_count: 8
    .max_flat_workgroup_size: 64
    .args:
    - { .size: 8, .offset: 0,  .value_kind: global_buffer, .address_space: global }
    - { .size: 8, .offset: 8,  .value_kind: global_buffer, .address_space: global }
    - { .size: 4, .offset: 16, .value_kind: by_value }
    - { .size: 4, .offset: 20, .value_kind: by_value }
  - .name: lds_read_b128_kernel
    .symbol: lds_read_b128_kernel.kd
    .kernarg_segment_size: 24
    .group_segment_fixed_size: 65536
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 20
    .vgpr_count: 9
    .max_flat_workgroup_size: 64
    .args:
    - { .size: 8, .offset: 0,  .value_kind: global_buffer, .address_space: global }
    - { .size: 8, .offset: 8,  .value_kind: global_buffer, .address_space: global }
    - { .size: 4, .offset: 16, .value_kind: by_value }
    - { .size: 4, .offset: 20, .value_kind: by_value }
  - .name: lds_write_b32_kernel
    .symbol: lds_write_b32_kernel.kd
    .kernarg_segment_size: 24
    .group_segment_fixed_size: 65536
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 20
    .vgpr_count: 8
    .max_flat_workgroup_size: 64
    .args:
    - { .size: 8, .offset: 0,  .value_kind: global_buffer, .address_space: global }
    - { .size: 8, .offset: 8,  .value_kind: global_buffer, .address_space: global }
    - { .size: 4, .offset: 16, .value_kind: by_value }
    - { .size: 4, .offset: 20, .value_kind: by_value }
  - .name: lds_write_b64_kernel
    .symbol: lds_write_b64_kernel.kd
    .kernarg_segment_size: 24
    .group_segment_fixed_size: 65536
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 20
    .vgpr_count: 10
    .max_flat_workgroup_size: 64
    .args:
    - { .size: 8, .offset: 0,  .value_kind: global_buffer, .address_space: global }
    - { .size: 8, .offset: 8,  .value_kind: global_buffer, .address_space: global }
    - { .size: 4, .offset: 16, .value_kind: by_value }
    - { .size: 4, .offset: 20, .value_kind: by_value }
  - .name: lds_write_b128_kernel
    .symbol: lds_write_b128_kernel.kd
    .kernarg_segment_size: 24
    .group_segment_fixed_size: 65536
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 20
    .vgpr_count: 12
    .max_flat_workgroup_size: 64
    .args:
    - { .size: 8, .offset: 0,  .value_kind: global_buffer, .address_space: global }
    - { .size: 8, .offset: 8,  .value_kind: global_buffer, .address_space: global }
    - { .size: 4, .offset: 16, .value_kind: by_value }
    - { .size: 4, .offset: 20, .value_kind: by_value }
...
.end_amdgpu_metadata
