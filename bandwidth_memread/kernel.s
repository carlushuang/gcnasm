.text
.global kernel_func
.p2align 8
.type kernel_func,@function

.set k_bdx,     256     ; should be 256 in bdx
.set k_end,     12
.set v_end,     128     ; hard code to this to let occupancy to be 1.  65536 / 256 = 256
.set s_A,       12
.set s_i_per_block,    14
.set s_iter, 15
.set s_tmp,     16
.set s_iter_2,  24
.set s_end,     31
.set a_end,     128

kernel_func:
    s_load_dwordx2         s[s_A:s_A+1], s[0:1], 0
    s_load_dword           s[s_i_per_block], s[0:1], 8
    s_load_dword           s[s_iter], s[0:1], 12
    s_waitcnt              lgkmcnt(0)

    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel kernel_func
    .amdhsa_group_segment_fixed_size 65536
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 64
    .amdhsa_next_free_sgpr 32
    .amdhsa_accum_offset 32
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
    .kernarg_segment_align: 4
    .kernarg_segment_size: 16
    .group_segment_fixed_size: 65536
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .reqd_workgroup_size : [1024, 1, 1]
    .max_flat_workgroup_size: 1024
    .args:
    - { .name: A,   .size: 8, .offset:   0, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: true}
    - { .name: issues_per_block, .size: 4, .offset:   8, .value_kind: by_value, .value_type: u32}
    - { .name: iter,  .size: 4, .offset:   12, .value_kind: by_value, .value_type: u32}
...
.end_amdgpu_metadata

