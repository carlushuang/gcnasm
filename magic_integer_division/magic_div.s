

; s_numer, s_denom only support uint32 within 31bit, otherwise this magic-div might have branch
.macro .mdiv_u32_ss s_quot, s_numer, s_magic, s_shift, s_tmp
    s_mul_hi_u32 s[\s_tmp], s[\s_magic], s[\s_numer]
    s_add_u32 s[\s_tmp], s[\s_tmp], s[\s_numer]
    s_lshr_b32 s[\s_quot], s[\s_tmp], s[\s_shift]
.endm

.macro .mdiv_u32_rem_ss s_rem, s_quot, s_numer, s_magic, s_shift, s_denom, s_tmp
    .mdiv_u32_ss \s_quot, \s_numer, \s_magic, \s_shift, \s_tmp
    s_mul_i32 s[\s_tmp], s[\s_denom], s[\s_quot]
    s_sub_u32 s[\s_rem], s[\s_numer], s[\s_tmp]
.endm

; v_numer, s_denom only support uint32 within 31bit, otherwise this magic-div might have branch
.macro .mdiv_u32_vs v_quot, v_numer, s_magic, s_shift, v_tmp
    v_mul_hi_u32 v[\v_tmp], s[\s_magic], v[\v_numer]
    v_add_u32 v[\v_tmp], v[\v_tmp], v[\v_numer]
    v_lshrrev_b32 v[\v_quot], s[\s_shift], v[\v_tmp]
.endm

.macro .mdiv_u32_rem_vs v_rem, v_quot, v_numer, s_magic, s_shift, s_denom, v_tmp
    .mdiv_u32_vs \v_quot, \v_numer, \s_magic, \s_shift, \v_tmp
    v_mul_lo_u32 v[\v_tmp], s[\s_denom], v[\v_quot]
    v_sub_u32 v[\v_rem], v[\v_numer], v[\v_tmp]
.endm


.text
.global kernel_func
.p2align 8
.type kernel_func,@function



.set s_numerater_ptr,   4
.set s_quot_ptr,        8
.set s_rem_ptr,         12
.set s_denom,           16
.set s_magic,           17
.set s_shift,           18
.set s_total_size,      19
.set s_step,            20
.set s_tmp, 34

.set v_idx, 10
.set v_offset,     12
.set v_numer, 20
.set v_quot, 28
.set v_rem, 29
.set v_tmp, 30

.set inst_loop, 512


kernel_func:
    s_load_dwordx2        s[s_numerater_ptr:s_numerater_ptr+1], s[0:1], 0
    s_load_dwordx2        s[s_quot_ptr:s_quot_ptr+1],           s[0:1], 8
    s_load_dwordx2        s[s_rem_ptr:s_rem_ptr+1],             s[0:1], 16
    s_load_dword        s[s_denom],         s[0:1], 24
    s_load_dword        s[s_magic],         s[0:1], 28
    s_load_dword        s[s_shift],         s[0:1], 32
    s_load_dword        s[s_total_size],    s[0:1], 36

    s_lshl_b32 s[s_tmp], s2, 8      ; 256
    v_or_b32 v[v_idx], s[s_tmp], v0
    

    s_mov_b32 s[s_numerater_ptr+2], 0xffffffff
    s_mov_b32 s[s_numerater_ptr+3], 0x27000
    s_mov_b32 s[s_quot_ptr+2], 0xffffffff
    s_mov_b32 s[s_quot_ptr+3], 0x27000
    s_mov_b32 s[s_rem_ptr+2], 0xffffffff
    s_mov_b32 s[s_rem_ptr+3], 0x27000

    s_mov_b32 s0,   0
    s_mov_b32 s1,   inst_loop
    s_mov_b32 s[s_step], 256*256        ; 256 block_size, 256 grid_size, hardcode
    s_waitcnt             lgkmcnt(0)
L_kernel_start:
    v_lshlrev_b32 v[v_offset], 2, v[v_idx]
    buffer_load_dword v[v_numer], v[v_offset], s[s_numerater_ptr:s_numerater_ptr+3], 0 offen offset:0
    s_waitcnt vmcnt(0)


    .mdiv_u32_rem_vs v_rem, v_quot, v_numer, s_magic, s_shift, s_denom, v_tmp

    buffer_store_dword v[v_quot], v[v_offset], s[s_quot_ptr:s_quot_ptr+3], 0 offen offset:0
    buffer_store_dword v[v_rem], v[v_offset], s[s_rem_ptr:s_rem_ptr+3], 0 offen offset:0

    s_add_u32 s0, s0, 1
    v_add_u32 v[v_idx], s[s_step], v[v_idx]
    v_cmp_lt_u32 vcc, v[v_idx], s[s_total_size]
    s_cbranch_vccz L_end

    s_and_saveexec_b64 s[s_tmp+4:s_tmp+5], vcc

    s_cmp_lt_u32 s0, s1
    s_cbranch_scc1 L_kernel_start

L_end:
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel kernel_func
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 64
    .amdhsa_next_free_sgpr 48
    .amdhsa_ieee_mode 0
    .amdhsa_dx10_clamp 0
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [ 1, 0 ]
amdhsa.kernels:
  - .name: kernel_func
    .symbol: kernel_func.kd
    .sgpr_count: 48
    .vgpr_count: 64
    .kernarg_segment_align: 4
    .kernarg_segment_size: 40
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .reqd_workgroup_size : [256, 1, 1]
    .max_flat_workgroup_size: 256
    .args:
    - { .name: numerater_ptr,   .size: 8, .offset:   0,  .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: true}
    - { .name: quot_ptr,        .size: 8, .offset:   8,  .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: true}
    - { .name: rem_ptr,         .size: 8, .offset:   16, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: true}
    - { .name: denom,           .size: 4, .offset:   24, .value_kind: by_value, .value_type: i32}
    - { .name: magic,           .size: 4, .offset:   28, .value_kind: by_value, .value_type: i32}
    - { .name: shift,           .size: 4, .offset:   32, .value_kind: by_value, .value_type: i32}
    - { .name: total_size,      .size: 4, .offset:   36, .value_kind: by_value, .value_type: i32}
...
.end_amdgpu_metadata

