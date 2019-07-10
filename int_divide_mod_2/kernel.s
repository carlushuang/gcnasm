.hsa_code_object_version 2,0
.hsa_code_object_isa 9, 0, 0, "AMD", "AMDGPU"

; Divisionby Invariant IntegersusingMultiplication, 7 Useof  oatingpoint
; CAUSION: val need not exeed 2**20-1, other wise need use fp64 to do the trick
.macro .v_u20_div_mod_sd dst, val, s_divider, mod
    v_cvt_f32_u32 v[\dst], v[\val]
    v_cvt_f32_u32 v[\mod], s[\s_divider]
    v_rcp_iflag_f32 v[\mod], v[\mod]
    v_mul_f32 v[\mod], 0x3f800004, v[\mod]
    v_mul_f32 v[\dst], v[\mod], v[\dst]
    v_cvt_u32_f32 v[\dst], v[\dst]
    v_mul_lo_u32 v[\mod], v[\dst], s[\s_divider]
    v_sub_u32 v[\mod], v[\val], v[\mod]
.endm

.macro .v_u32_div v_q, v_n, v_d, v_tmp4, s_tmp4
     v_cvt_f32_u32     v[\v_tmp4+0],   v[\v_d]
     v_rcp_f32         v[\v_tmp4+0],   v[\v_tmp4+0]
     v_mul_f32         v[\v_tmp4+0],   0x4f800000, v[\v_tmp4+0]
     v_cvt_u32_f32     v[\v_tmp4+0],   v[\v_tmp4+0]
     v_mul_lo_u32      v[\v_tmp4+1],   v[\v_d],      v[\v_tmp4+0]
     v_mul_hi_u32      v[\v_tmp4+2],   v[\v_d],      v[\v_tmp4+0]
     v_sub_co_u32      v[\v_tmp4+3],   vcc, 0,     v[\v_tmp4+1]
     v_cmp_ne_i32      s[\s_tmp4:\s_tmp4+1], 0,          v[\v_tmp4+2]
     v_cndmask_b32     v[\v_tmp4+1],   v[\v_tmp4+3],   v[\v_tmp4+1],   s[\s_tmp4:\s_tmp4+1]
     v_mul_hi_u32      v[\v_tmp4+1],   v[\v_tmp4+1],   v[\v_tmp4+0]
     v_sub_co_u32      v[\v_tmp4+2],   vcc,        v[\v_tmp4+0],   v[\v_tmp4+1]
     v_add_co_u32      v[\v_tmp4+0],   vcc,        v[\v_tmp4+0],   v[\v_tmp4+1]
     v_cndmask_b32     v[\v_tmp4+0],   v[\v_tmp4+0],   v[\v_tmp4+2],   s[\s_tmp4:\s_tmp4+1]
     v_mul_hi_u32      v[\v_tmp4+0],   v[\v_tmp4+0],   v[\v_n]
     v_mul_lo_u32      v[\v_tmp4+1],   v[\v_tmp4+0],   v[\v_d]
     v_sub_co_u32      v[\v_tmp4+2],   vcc,        v[\v_n],      v[\v_tmp4+1]
     v_cmp_ge_u32      s[\s_tmp4:\s_tmp4+1], v[\v_n],      v[\v_tmp4+1]
     v_cmp_ge_u32      s[\s_tmp4+2:\s_tmp4+3], v[\v_tmp4+2],   v[\v_d]
     v_add_co_u32      v[\v_tmp4+2],   vcc, 1, v[\v_tmp4+0]
     s_and_b64         s[\s_tmp4+2:\s_tmp4+3], s[\s_tmp4:\s_tmp4+1], s[\s_tmp4+2:\s_tmp4+3]
     v_add_co_u32      v[\v_tmp4+1],   vcc, -1,    v[\v_tmp4+0]
     v_cndmask_b32     v[\v_tmp4+2],   v[\v_tmp4+0],   v[\v_tmp4+2],      s[\s_tmp4+2:\s_tmp4+3]
     v_cndmask_b32     v[\v_tmp4+2],   v[\v_tmp4+1],   v[\v_tmp4+2],      s[\s_tmp4:\s_tmp4+1]
     v_cmp_ne_i32      vcc,          0,          v[\v_d]
     v_cndmask_b32     v[\v_q],      -1,         v[\v_tmp4+2],      vcc
.endm

.macro .v_u32_div_vs v_q, v_n, s_d, v_tmp4, s_tmp4
     v_cvt_f32_u32     v[\v_tmp4+0],   s[\s_d]
     v_rcp_f32         v[\v_tmp4+0],   v[\v_tmp4+0]
     v_mul_f32         v[\v_tmp4+0],   0x4f800000, v[\v_tmp4+0]
     v_cvt_u32_f32     v[\v_tmp4+0],   v[\v_tmp4+0]
     v_mul_lo_u32      v[\v_tmp4+1],   s[\s_d],      v[\v_tmp4+0]
     v_mul_hi_u32      v[\v_tmp4+2],   s[\s_d],      v[\v_tmp4+0]
     v_sub_co_u32      v[\v_tmp4+3],   vcc, 0,     v[\v_tmp4+1]
     v_cmp_ne_i32      s[\s_tmp4:\s_tmp4+1], 0,          v[\v_tmp4+2]
     v_cndmask_b32     v[\v_tmp4+1],   v[\v_tmp4+3],   v[\v_tmp4+1],   s[\s_tmp4:\s_tmp4+1]
     v_mul_hi_u32      v[\v_tmp4+1],   v[\v_tmp4+1],   v[\v_tmp4+0]
     v_sub_co_u32      v[\v_tmp4+2],   vcc,        v[\v_tmp4+0],   v[\v_tmp4+1]
     v_add_co_u32      v[\v_tmp4+0],   vcc,        v[\v_tmp4+0],   v[\v_tmp4+1]
     v_cndmask_b32     v[\v_tmp4+0],   v[\v_tmp4+0],   v[\v_tmp4+2],   s[\s_tmp4:\s_tmp4+1]
     v_mul_hi_u32      v[\v_tmp4+0],   v[\v_tmp4+0],   v[\v_n]
     v_mul_lo_u32      v[\v_tmp4+1],   s[\s_d],     v[\v_tmp4+0]
     v_sub_co_u32      v[\v_tmp4+2],   vcc,        v[\v_n],      v[\v_tmp4+1]
     v_cmp_ge_u32      s[\s_tmp4:\s_tmp4+1], v[\v_n],      v[\v_tmp4+1]
     v_cmp_le_u32      s[\s_tmp4+2:\s_tmp4+3],  s[\s_d],    v[\v_tmp4+2]
     v_add_co_u32      v[\v_tmp4+2],   vcc, 1, v[\v_tmp4+0]
     s_and_b64         s[\s_tmp4+2:\s_tmp4+3], s[\s_tmp4:\s_tmp4+1], s[\s_tmp4+2:\s_tmp4+3]
     v_add_co_u32      v[\v_tmp4+1],   vcc, -1,    v[\v_tmp4+0]
     v_cndmask_b32     v[\v_tmp4+2],   v[\v_tmp4+0],   v[\v_tmp4+2],      s[\s_tmp4+2:\s_tmp4+3]
     v_cndmask_b32     v[\v_tmp4+2],   v[\v_tmp4+1],   v[\v_tmp4+2],      s[\s_tmp4:\s_tmp4+1]
     v_cmp_ne_i32      vcc,          s[\s_d],   0
     v_cndmask_b32     v[\v_q],      -1,         v[\v_tmp4+2],      vcc
.endm

.macro .v_u32_div_ss v_q, s_n, s_d, v_tmp4, s_tmp4
     v_cvt_f32_u32     v[\v_tmp4+0],   s[\s_d]
     v_rcp_f32         v[\v_tmp4+0],   v[\v_tmp4+0]
     v_mul_f32         v[\v_tmp4+0],   0x4f800000, v[\v_tmp4+0]
     v_cvt_u32_f32     v[\v_tmp4+0],   v[\v_tmp4+0]
     v_mul_lo_u32      v[\v_tmp4+1],   s[\s_d],      v[\v_tmp4+0]
     v_mul_hi_u32      v[\v_tmp4+2],   s[\s_d],      v[\v_tmp4+0]
     v_sub_co_u32      v[\v_tmp4+3],   vcc, 0,     v[\v_tmp4+1]
     v_cmp_ne_i32      s[\s_tmp4:\s_tmp4+1], 0,          v[\v_tmp4+2]
     v_cndmask_b32     v[\v_tmp4+1],   v[\v_tmp4+3],   v[\v_tmp4+1],   s[\s_tmp4:\s_tmp4+1]
     v_mul_hi_u32      v[\v_tmp4+1],   v[\v_tmp4+1],   v[\v_tmp4+0]
     v_sub_co_u32      v[\v_tmp4+2],   vcc,        v[\v_tmp4+0],   v[\v_tmp4+1]
     v_add_co_u32      v[\v_tmp4+0],   vcc,        v[\v_tmp4+0],   v[\v_tmp4+1]
     v_cndmask_b32     v[\v_tmp4+0],   v[\v_tmp4+0],   v[\v_tmp4+2],   s[\s_tmp4:\s_tmp4+1]
     v_mul_hi_u32      v[\v_tmp4+0],   s[\s_n],   v[\v_tmp4+0]
     v_mul_lo_u32      v[\v_tmp4+1],   s[\s_d],     v[\v_tmp4+0]
     v_sub_co_u32      v[\v_tmp4+2],   vcc,        s[\s_n],      v[\v_tmp4+1]
     v_cmp_ge_u32      s[\s_tmp4:\s_tmp4+1], s[\s_n],      v[\v_tmp4+1]
     v_cmp_le_u32      s[\s_tmp4+2:\s_tmp4+3],  s[\s_d],    v[\v_tmp4+2]
     v_add_co_u32      v[\v_tmp4+2],   vcc, 1, v[\v_tmp4+0]
     s_and_b64         s[\s_tmp4+2:\s_tmp4+3], s[\s_tmp4:\s_tmp4+1], s[\s_tmp4+2:\s_tmp4+3]
     v_add_co_u32      v[\v_tmp4+1],   vcc, -1,    v[\v_tmp4+0]
     v_cndmask_b32     v[\v_tmp4+2],   v[\v_tmp4+0],   v[\v_tmp4+2],      s[\s_tmp4+2:\s_tmp4+3]
     v_cndmask_b32     v[\v_tmp4+2],   v[\v_tmp4+1],   v[\v_tmp4+2],      s[\s_tmp4:\s_tmp4+1]
     v_cmp_ne_i32      vcc,          s[\s_d],   0
     v_cndmask_b32     v[\v_q],      -1,         v[\v_tmp4+2],      vcc
.endm

.text
.p2align 8
.amdgpu_hsa_kernel kernel_func

.set v_tid,     0
.set v_os,      1
.set v_val,     2
.set v_dst,     3
.set v_mod,     4
.set v_div,     5
.set v_tmp,     6
.set v_end,     9

.set s_arg,     0
.set s_bx,      2
.set s_by,      3
.set s_in,      4
.set s_out,     6
.set s_divider, 8
.set s_os,      10
.set s_val,     11
.set s_tmp,     12
.set s_end,     15

kernel_func:
    .amd_kernel_code_t
        enable_sgpr_kernarg_segment_ptr     = 1     //(use 2 SGPR) 64 bit address of Kernarg segment.
        user_sgpr_count                     = 2     // total number of above enabled sgpr. does not contains enable_sgpr_* start from tgid_x
        enable_sgpr_workgroup_id_x = 1              ;        blockIdx.x
        enable_sgpr_workgroup_id_y = 1              ;        blockIdx.x

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
    s_lshl_b32          s[s_os], s[s_bx], 2
    s_waitcnt           lgkmcnt(0)

    s_load_dword        s[s_val], s[s_in:s_in+1], s[s_os]
    s_waitcnt           lgkmcnt(0)

    .v_u32_div_ss v_dst, s_val, s_divider, v_tmp, s_tmp
    v_mov_b32            v[v_os], s[s_os]
    v_cmp_eq_u32  vcc, 0, v0
    s_and_saveexec_b64 s[s_tmp:s_tmp+1], vcc
    ;s_cbranch_execz _L_end
    global_store_dword  v[v_os:v_os+1], v[v_dst], s[s_out:s_out+1]

    s_waitcnt           vmcnt(0)
_L_end:
    s_or_b64 exec, exec, s[s_tmp:s_tmp+1]
    s_waitcnt           vmcnt(0)
    s_endpgm
