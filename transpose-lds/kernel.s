.hsa_code_object_version 2,0
.hsa_code_object_isa 9, 0, 6, "AMD", "AMDGPU"

.include "common.inc"

.text
.p2align 8
.amdgpu_hsa_kernel transpose_32x32

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
    .amd_kernel_code_t
        ; enable_sgpr_private_segment_buffer  = 1     //(use 4 SGPR)
        ; enable_sgpr_dispatch_ptr            = 1     //(use 2 SGPR)
        enable_sgpr_kernarg_segment_ptr     = 1     //(use 2 SGPR) 64 bit address of Kernarg segment.
        ; enable_sgpr_workgroup_id_x          = 1     //(use 1 SGPR) 32 bit work group id in X dimension of grid for wavefront. Always present.
        ; ! user_sgpr is before enable_sgpr_workgroup_id_x(called system sgpr, alloced by SPI)
        ; SGPRs before the Work-Group Ids are set by CP using the 16 User Data registers.
        user_sgpr_count                     = 2     // total number of above enabled sgpr. does not contains enable_sgpr_* start from tgid_x

        enable_vgpr_workitem_id             = 0     // only vgpr0 used for workitem id x

        is_ptr64                            = 1     //
        float_mode                          = 192   // 0xc0

        wavefront_sgpr_count                = s_end+1+2*3    ; VCC, FLAT_SCRATCH and XNACK must be counted
        workitem_vgpr_count                 = v_end+1
        granulated_workitem_vgpr_count      = v_end/4  ; (workitem_vgpr_count-1)/4
        granulated_wavefront_sgpr_count     = (s_end+2*3)/8     ; (wavefront_sgpr_count-1)/8

        kernarg_segment_byte_size           = 16
        workgroup_group_segment_byte_size   = 32*32*4
    .end_amd_kernel_code_t

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
