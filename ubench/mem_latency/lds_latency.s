; ====================================================================
; LDS Latency Micro-Benchmark (gfx942 / MI308)
;
; Measures LDS read latency using pointer chasing.
; Host prepares a random permutation in global memory.
; Kernel copies it to LDS, then chases pointers in a tight loop.
; Uses s_memrealtime to measure elapsed GPU cycles.
;
; Only lane 0 of a single wavefront performs the measurement.
;
; Kernel arguments (24 bytes, 8-byte aligned):
;   offset  0: uint32_t* chase_array  (8 bytes) - global ptr to chase data
;   offset  8: uint64_t* output       (8 bytes) - store [start, end] cycles
;   offset 16: uint32_t  num_entries  (4 bytes) - number of LDS entries
;   offset 20: uint32_t  num_iters    (4 bytes) - chase iterations
; ====================================================================

.text
.global lds_latency_kernel
.p2align 8
.type lds_latency_kernel,@function

; --- SGPR names ---
.set s_karg,        0       ; kernarg pointer [0:1]
.set s_bx,          2       ; workgroup id
.set s_chase_ptr,   4       ; chase_array base [4:5]
.set s_out_ptr,     6       ; output base [6:7]
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

; --- VGPR names (tuples must be even-aligned) ---
.set v_tid,         0
.set v_addr,        2       ; 64-bit address [2:3] (even-aligned)
.set v_data,        4
.set v_lds_off,     5
.set v_chase,       6
.set v_out,         8       ; 64-bit output [8:9] (even-aligned)
.set v_tmp,         10
.set v_iter,        11      ; vgpr iteration counter

lds_latency_kernel:
    ; ===========================================================
    ; Step 1: Load kernel arguments
    ; ===========================================================
    s_load_dwordx2 s[s_chase_ptr:s_chase_ptr+1], s[s_karg:s_karg+1], 0
    s_load_dwordx2 s[s_out_ptr:s_out_ptr+1],     s[s_karg:s_karg+1], 8
    s_load_dword   s[s_num_entries],              s[s_karg:s_karg+1], 16
    s_load_dword   s[s_num_iters],                s[s_karg:s_karg+1], 20

    s_waitcnt lgkmcnt(0)

    ; ===========================================================
    ; Step 2: Only lane 0 executes
    ; ===========================================================
    v_cmp_eq_u32 vcc, 0, v[v_tid]
    s_and_saveexec_b64 s[s_exec_save:s_exec_save+1], vcc
    s_cbranch_execz L_lds_exit

    ; ===========================================================
    ; Step 3: Copy chase_array from global to LDS
    ; ===========================================================
    s_mov_b32 s[s_cnt], 0

L_copy_loop:
    s_lshl_b32 s[s_tmp], s[s_cnt], 2
    v_mov_b32 v[v_addr], s[s_chase_ptr]
    v_mov_b32 v[v_addr+1], s[s_chase_ptr+1]
    v_add_co_u32 v[v_addr], vcc, s[s_tmp], v[v_addr]
    v_addc_co_u32 v[v_addr+1], vcc, 0, v[v_addr+1], vcc

    global_load_dword v[v_data], v[v_addr:v_addr+1], off
    s_waitcnt vmcnt(0)

    v_mov_b32 v[v_lds_off], s[s_tmp]
    ds_write_b32 v[v_lds_off], v[v_data]
    s_waitcnt lgkmcnt(0)

    s_add_u32 s[s_cnt], s[s_cnt], 1
    s_cmp_lt_u32 s[s_cnt], s[s_num_entries]
    s_cbranch_scc1 L_copy_loop

    s_barrier

    ; ===========================================================
    ; Step 4: Warm up
    ; ===========================================================
    v_mov_b32 v[v_chase], 0
    s_mov_b32 s[s_iter], 64
L_warmup:
    ds_read_b32 v[v_chase], v[v_chase]
    s_waitcnt lgkmcnt(0)
    s_sub_u32 s[s_iter], s[s_iter], 1
    s_cbranch_scc1 L_warmup

    ; ===========================================================
    ; Step 5: Measure - pointer chase loop
    ;   Use v_readfirstlane to feed chase result back to control
    ;   flow so SALU truly depends on the load result.
    ; ===========================================================
    v_mov_b32 v[v_chase], 0
    v_mov_b32 v[v_iter], s[s_num_iters]

    ; Synchronize before reading timer
    s_waitcnt vmcnt(0) lgkmcnt(0)
    s_memrealtime s[s_start_lo:s_start_hi]
    s_waitcnt lgkmcnt(0)

L_chase_loop:
    ds_read_b32 v[v_chase], v[v_chase]
    s_waitcnt lgkmcnt(0)
    v_sub_u32 v[v_iter], v[v_iter], 1
    v_cmp_gt_u32 vcc, v[v_iter], 0
    s_cbranch_vccnz L_chase_loop

    ; Synchronize before reading end timer
    s_waitcnt vmcnt(0) lgkmcnt(0)
    s_memrealtime s[s_end_lo:s_end_hi]
    s_waitcnt lgkmcnt(0)

    ; ===========================================================
    ; Step 6: Store results to output buffer
    ; ===========================================================
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

L_lds_exit:
    s_endpgm

; ====================================================================
; Kernel descriptor
; ====================================================================
.rodata
.p2align 6
.amdhsa_kernel lds_latency_kernel
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

.amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: lds_latency_kernel
    .symbol: lds_latency_kernel.kd
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
