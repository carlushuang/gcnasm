.hsa_code_object_version 2,0
.hsa_code_object_isa 9, 0, 0, "AMD", "AMDGPU"

; Divisionby Invariant IntegersusingMultiplication, 7 Useof  oatingpoint
; CAUSION: val need not exeed 2**20-1, other wise need use fp64 to do the trick
.macro .v_uint_div_mod_sd dst, val, s_divider, mod
    v_cvt_f32_u32 v[\dst], v[\val]
    v_cvt_f32_u32 v[\mod], s[\s_divider]
    v_rcp_iflag_f32 v[\mod], v[\mod]
    v_mul_f32 v[\mod], 0x3f800004, v[\mod]
    v_mul_f32 v[\dst], v[\mod], v[\dst]
    v_cvt_u32_f32 v[\dst], v[\dst]
    v_mul_lo_u32 v[\mod], v[\dst], s[\s_divider]
    v_sub_u32 v[\mod], v[\val], v[\mod]
.endm

.text
.p2align 8
.amdgpu_hsa_kernel kernel_func

.set v_tid,     0
.set v_os,      1
.set v_val,     2
.set v_dst,     3
.set v_mod,     4
.set v_end,     4

.set s_arg,     0
.set s_in,      4
.set s_out,     6
.set s_divider, 8
.set s_end,     8

kernel_func:
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
        float_mode                          = 2   // 0xc0

        wavefront_sgpr_count                = s_end+1+2*3    ; VCC, FLAT_SCRATCH and XNACK must be counted
        workitem_vgpr_count                 = v_end+1
        granulated_workitem_vgpr_count      = v_end/4  ; (workitem_vgpr_count-1)/4
        granulated_wavefront_sgpr_count     = (s_end+2*3)/8     ; (wavefront_sgpr_count-1)/8

        kernarg_segment_byte_size           = 16
        workgroup_group_segment_byte_size   = 0
    .end_amd_kernel_code_t

    // init state: s[0:1] kernarg segment, s[2] workgroup id
    s_load_dwordx4      s[s_in:s_in+3], s[s_arg:s_arg+1], 0x0       // kernarg, in
    s_load_dword        s[s_divider], s[s_arg:s_arg+1], 0x10
    v_lshlrev_b32       v[v_os], 2, v[v_tid]
    s_waitcnt           lgkmcnt(0)

    global_load_dword   v[v_val], v[v_os:v_os+1], s[s_in:s_in+1]
    s_waitcnt           vmcnt(0)

    .v_uint_div_mod_sd v_dst, v_val, s_divider, v_mod

    global_store_dword  v[v_os:v_os+1], v[v_dst], s[s_out:s_out+1]

    s_waitcnt           lgkmcnt(0)

    s_endpgm
