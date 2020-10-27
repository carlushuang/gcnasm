.text
.global kernel_func
.p2align 8
.type kernel_func,@function
kernel_func:
; This is just an example, not the optimal one
.set s_dptr,            0
.set s_karg,            2
.set s_bx,              4

.set s_ptr_in,          8
.set s_ptr_out,         12
.set s_total_number,    16
.set s_cnt,             17
.set s_tmp,             18
.set s_stride,          19
.set s_gdx,             20

.set v_buf,             0
.set v_offset,          16
.set v_tmp,             32

    ; http://www.hsafoundation.com/html/Content/Runtime/Topics/02_Core/hsa_kernel_dispatch_packet_t.htm
    s_load_dwordx2 s[s_ptr_in:s_ptr_in+1],      s[s_karg:s_karg+1],     0
    s_load_dwordx2 s[s_ptr_out:s_ptr_out+1],    s[s_karg:s_karg+1],     8
    s_load_dword   s[s_total_number],           s[s_karg:s_karg+1],     16

    s_cmp_eq_u32 s[s_bx], 0
    s_cbranch_scc0 kernel_end
    v_lshlrev_b32 v[v_offset], 2+1, v0

    s_waitcnt           lgkmcnt(0)

    s_mov_b32 s[s_ptr_in+2],    1020
    s_mov_b32 s[s_ptr_in+3],    0x27000
    s_mov_b32 s[s_ptr_out+2],   0xffffffff
    s_mov_b32 s[s_ptr_out+3],   0x27000

    s_mov_b32 s[s_cnt], 0
    s_mov_b32 s[s_stride], 256*2*4
    s_mov_b32 s[s_tmp], 0

kernel_start:
    buffer_load_dwordx2  v[2:3], v[v_offset], s[s_ptr_in:s_ptr_in+3],    s[s_tmp] offen offset:0
    s_waitcnt       vmcnt(0)

    buffer_store_dwordx2 v[2:3], v[v_offset], s[s_ptr_out:s_ptr_out+3],  s[s_tmp] offen offset:0


    s_waitcnt       vmcnt(0)
    s_add_u32 s[s_cnt], s[s_cnt], 256
    ;v_add_u32 v[v_offset], s[s_stride], v[v_offset]
    ;s_add_u32 s[s_tmp], s[s_tmp], s[s_stride]

    s_add_u32 s[s_ptr_in], s[s_ptr_in], s[s_stride]
    s_addc_u32 s[s_ptr_in+1], s[s_ptr_in+1], 0

    s_add_u32 s[s_ptr_out], s[s_ptr_out], s[s_stride]
    s_addc_u32 s[s_ptr_out+1], s[s_ptr_out+1], 0

    s_cmp_lt_u32 s[s_cnt], s[s_total_number]
    s_cbranch_scc1 kernel_start

kernel_end:
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel kernel_func
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_user_sgpr_dispatch_ptr 1
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 64
    .amdhsa_next_free_sgpr 32
    .amdhsa_ieee_mode 0
    .amdhsa_dx10_clamp 0
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [ 1, 0 ]
amdhsa.kernels:
  - .name: kernel_func
    .symbol: kernel_func.kd
    .sgpr_count: 32
    .vgpr_count: 64
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .reqd_workgroup_size : [256, 1, 1]
    .max_flat_workgroup_size: 256
    .args:
    - { .name: input,           .size: 8, .offset:   0, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: false}
    - { .name: output,          .size: 8, .offset:   8, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: false}
    - { .name: total_number,    .size: 4, .offset:  16, .value_kind: by_value, .value_type: i32}
    - { .name: __pack0,         .size: 4, .offset:  20, .value_kind: by_value, .value_type: i32}
...
.end_amdgpu_metadata
