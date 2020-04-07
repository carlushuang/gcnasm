.include "common.inc"

.text
.globl transpose_32x32 ; in cov3, this is a must
.p2align 8
.type transpose_32x32,@function ; in cov3 this is a must

.set v_tid,     0
.set v_col,     1
.set v_row,     2
.set v_acc,     3
.set v_end,     v_acc+36-1

.set s_arg,     0
.set s_in,      4
.set s_out,     6
.set s_end,     7

transpose_32x32:
    // init state: s[0:1] kernarg segment, s[2] workgroup id
    s_load_dwordx4      s[s_in:s_in+3], s[s_arg:s_arg+1], 0x0       // kernarg, in
    v_lshlrev_b32       v[v_col], 2, v[v_tid]
    v_lshlrev_b32       v[v_row], 5, v[v_col]
    s_waitcnt           lgkmcnt(0)

    .g_load32_col       v_acc, v_col, s_in
    s_waitcnt           vmcnt(0)

    .ds_store32         v_col, v_acc, 32, 1
    s_waitcnt           lgkmcnt(0)
    ; s_barrier
    .ds_load32          v_acc, v_row, 1, 1
    s_waitcnt           lgkmcnt(0)
    ; s_barrier

    .g_store32_col      s_out, v_col, v_acc
    s_waitcnt           lgkmcnt(0)

    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel transpose_32x32
    .amdhsa_user_sgpr_kernarg_segment_ptr 1

    .amdhsa_next_free_vgpr 40
    .amdhsa_next_free_sgpr 7
    ;.amdhsa_next_free_vgpr .amdgcn.next_free_vgpr
    ;.amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
    .amdhsa_system_vgpr_workitem_id 0

    .amdhsa_ieee_mode 0
    .amdhsa_dx10_clamp 0
    .amdhsa_group_segment_fixed_size 32*32*4
.end_amdhsa_kernel

; a negative side for metadata is, it seems can't use arithmetic. other is use macro to wrap
.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 0
amdhsa.kernels:
  - .name: transpose_32x32
    .symbol: transpose_32x32.kd
    .sgpr_count: 7
    .vgpr_count: 40
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .group_segment_fixed_size: 4096
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .max_flat_workgroup_size: 512
    ;.args:
    ;- { .size: 8, .offset: 0, .value_kind: global_buffer, .value_type: f32, .name: in,  .address_space: global, .is_const: true }
    ;- { .size: 8, .offset: 8, .value_kind: global_buffer, .value_type: f32, .name: out, .address_space: global, .is_const: false }
...
.end_amdgpu_metadata
