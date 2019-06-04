.hsa_code_object_version 2,0
.hsa_code_object_isa 9, 0, 0, "AMD", "AMDGPU"

; (src_lane & (width-1)) + (self_lane_id&~(width-1))
.macro .__shfl val, src_lane, width_m1
    ; CAUSION! omited self_lane > width-1 case
    v_and_b32 v[\src_lane], \width_m1, v[\src_lane]
    v_lshlrev_b32 v[\src_lane], 2, v[\src_lane]
    ds_bpermute_b32 v[\val], v[\src_lane], v[\val]
.endm

.macro .__shfl_xor val, lane_mask, width, self_lane_id, idx_tmp
    v_xor_b32 v[\idx_tmp], \lane_mask, v[\self_lane_id]
    ;width_mask = width-1
    ;width_mask = ~width_mask
    ;v_xor_b32 v[\idx_tmp], width_mask, v[\idx_tmp]
    ;v_mov_b32 v[\val], v[\idx_tmp]
    v_lshlrev_b32 v[\idx_tmp], 2, v[\idx_tmp]
    ds_bpermute_b32 v[\val], v[\idx_tmp], v[\val]
.endm
.macro .s_cmul_imm a, br, bi, bi_neg, tmp
    v_mov_b32 v[\tmp], v[\a]
    v_mul_f32 v[\a], \bi_neg, v[\a+1]
    v_madmk_f32 v[\a], v[\tmp], \br, v[\a]
    v_mul_f32 v[\a+1], \br, v[\a+1]
    v_madmk_f32 v[\a+1], v[\tmp], \bi, v[\a+1]
.endm

.text
.p2align 8
.amdgpu_hsa_kernel shfl_xor_test

.set v_tid,     0
.set v_os,      1
.set v_val,     2
.set v_lane_id, 4
.set v_tmp,     5
.set v_end,     5

.set s_arg,     0
.set s_in,      4
.set s_out,     6
.set s_mask,    8
.set s_width,   9
.set s_end,     9

shfl_xor_test:
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
    s_load_dwordx2      s[s_mask:s_mask+1], s[s_arg:s_arg+1], 0x10
    v_lshlrev_b32       v[v_os], 2, v[v_tid]
    s_waitcnt           lgkmcnt(0)

    global_load_dword   v[v_val], v[v_os:v_os+1], s[s_in:s_in+1]
    s_waitcnt           vmcnt(0)

    ; __shfl_xor
    v_mbcnt_lo_u32_b32  v[v_lane_id], -1, 0
    v_mbcnt_hi_u32_b32  v[v_lane_id], -1, v[v_lane_id]

    ;.__shfl_xor         v_val, 2, 32, v_lane_id, v_tmp
    v_sub_u32           v[v_tmp], 32, v[v_lane_id]
    .__shfl v_val, v_tmp, 31

    s_waitcnt           lgkmcnt(0)

    global_store_dword  v[v_os:v_os+1], v[v_val], s[s_out:s_out+1]

    s_waitcnt           lgkmcnt(0)

    s_endpgm
