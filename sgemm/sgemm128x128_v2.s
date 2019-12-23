
.hsa_code_object_version 2,1
.hsa_code_object_isa
.text
.p2align 8
.amdgpu_hsa_kernel sgemm_128x128

.macro .v_u32_div_ss v_q, s_n, s_d, v_tmp4, s_tmp4
     v_cvt_f32_u32     v[\v_tmp4+0], s[\s_d]
     v_rcp_f32         v[\v_tmp4+0], v[\v_tmp4+0]
     v_mul_f32         v[\v_tmp4+0], 0x4f800000, v[\v_tmp4+0]
     v_cvt_u32_f32     v[\v_tmp4+0], v[\v_tmp4+0]
     v_mul_lo_u32      v[\v_tmp4+1], s[\s_d], v[\v_tmp4+0]
     v_mul_hi_u32      v[\v_tmp4+2], s[\s_d], v[\v_tmp4+0]
     v_sub_co_u32      v[\v_tmp4+3], vcc, 0, v[\v_tmp4+1]
     v_cmp_ne_i32      s[\s_tmp4:\s_tmp4+1], 0, v[\v_tmp4+2]
     v_cndmask_b32     v[\v_tmp4+1], v[\v_tmp4+3], v[\v_tmp4+1], s[\s_tmp4:\s_tmp4+1]
     v_mul_hi_u32      v[\v_tmp4+1], v[\v_tmp4+1], v[\v_tmp4+0]
     v_sub_co_u32      v[\v_tmp4+2], vcc, v[\v_tmp4+0], v[\v_tmp4+1]
     v_add_co_u32      v[\v_tmp4+0], vcc, v[\v_tmp4+0], v[\v_tmp4+1]
     v_cndmask_b32     v[\v_tmp4+0], v[\v_tmp4+0], v[\v_tmp4+2], s[\s_tmp4:\s_tmp4+1]
     v_mul_hi_u32      v[\v_tmp4+0], s[\s_n], v[\v_tmp4+0]
     v_mul_lo_u32      v[\v_tmp4+1], s[\s_d], v[\v_tmp4+0]
     v_sub_co_u32      v[\v_tmp4+2], vcc, s[\s_n], v[\v_tmp4+1]
     v_cmp_ge_u32      s[\s_tmp4:\s_tmp4+1], s[\s_n], v[\v_tmp4+1]
     v_cmp_le_u32      s[\s_tmp4+2:\s_tmp4+3],  s[\s_d], v[\v_tmp4+2]
     v_add_co_u32      v[\v_tmp4+2], vcc, 1, v[\v_tmp4+0]
     s_and_b64         s[\s_tmp4+2:\s_tmp4+3], s[\s_tmp4:\s_tmp4+1], s[\s_tmp4+2:\s_tmp4+3]
     v_add_co_u32      v[\v_tmp4+1], vcc, -1, v[\v_tmp4+0]
     v_cndmask_b32     v[\v_tmp4+2], v[\v_tmp4+0], v[\v_tmp4+2], s[\s_tmp4+2:\s_tmp4+3]
     v_cndmask_b32     v[\v_tmp4+2], v[\v_tmp4+1], v[\v_tmp4+2], s[\s_tmp4:\s_tmp4+1]
     v_cmp_ne_i32      vcc, s[\s_d], 0
     v_cndmask_b32     v[\v_q], -1, v[\v_tmp4+2], vcc
.endm

.macro .s_fma4x4 c, a, b
    v_mac_f32 v[\c+0],  v[\a+0],  v[\b+0]
    v_mac_f32 v[\c+1],  v[\a+1],  v[\b+0]
    v_mac_f32 v[\c+2],  v[\a+2],  v[\b+0]
    v_mac_f32 v[\c+3],  v[\a+3],  v[\b+0]
    v_mac_f32 v[\c+4],  v[\a+0],  v[\b+1]
    v_mac_f32 v[\c+5],  v[\a+1],  v[\b+1]
    v_mac_f32 v[\c+6],  v[\a+2],  v[\b+1]
    v_mac_f32 v[\c+7],  v[\a+3],  v[\b+1]
    v_mac_f32 v[\c+8],  v[\a+0],  v[\b+2]
    v_mac_f32 v[\c+9],  v[\a+1],  v[\b+2]
    v_mac_f32 v[\c+10], v[\a+2],  v[\b+2]
    v_mac_f32 v[\c+11], v[\a+3],  v[\b+2]
    v_mac_f32 v[\c+12], v[\a+0],  v[\b+3]
    v_mac_f32 v[\c+13], v[\a+1],  v[\b+3]
    v_mac_f32 v[\c+14], v[\a+2],  v[\b+3]
    v_mac_f32 v[\c+15], v[\a+3],  v[\b+3]
.endm

.set k_ptr_c,           0
.set k_ptr_a,           8
.set k_ptr_b,           16
.set k_alpha,           24
.set k_m,               28
.set k_n,               32
.set k_k,               36
.set k_lda,             40
.set k_ldb,             44
.set k_ldc,             48
.set k_end,             52

.set s_ka,              0
.set s_bx,              2
.set s_by,              3
.set s_ptr_c,           4
.set s_ptr_a,           6
.set s_ptr_b,           8
.set s_bs_a,            10
.set s_bs_b,            11
.set s_alpha,           12
.set s_m,               13
.set s_n,               14
.set s_k,               15
.set s_lda,             16
.set s_ldb,             17
.set s_ldc,             18
.set s_m_blocks,        19
.set s_kitr,            20
.set s_wave_id,         21
.set s_m_idx,           22
.set s_n_idx,           23
.set s_tmp,             24
.set s_end,             s_tmp+3

.set v_c,               0
.set v_a0,              64
.set v_a1,              72
.set v_b0,              80
.set v_b1,              88
.set v_p0,              96
.set v_q0,              100
.set v_lane_id,         104
.set v_wave_p,          105
.set v_wave_q,          106
.set v_lane_lo,         107
.set v_lane_hi,         108
.set v_lane_w,          109
.set v_lane_u,          110
.set v_lane_v,          111
.set v_smem_store,      112
.set v_os_a,            113
.set v_os_b,            114
.set v_os_c,            115
.set v_smem_load_a,     116
.set v_smem_load_b,     117
.set v_smem_store_c,    118
.set v_smem_load_c,     119
.set v_tmp,             124
.set v_end,             v_tmp+3
.set v_p1,              104
.set v_q1,              108

sgemm_128x128:
    .amd_kernel_code_t
        enable_sgpr_kernarg_segment_ptr = 1         ;
        user_sgpr_count = 2
        enable_sgpr_workgroup_id_x = 1              ;        blockIdx.x
        enable_sgpr_workgroup_id_y = 1              ;        blockIdx.y

        enable_vgpr_workitem_id = 0
        is_ptr64 = 1
        float_mode = 192
        workgroup_group_segment_byte_size = 16384
        kernarg_segment_byte_size = k_end
        wavefront_sgpr_count = s_end+1+2*3          ; VCC, FLAT_SCRATCH and XNACK must be counted
        workitem_vgpr_count = v_end+1
        granulated_workitem_vgpr_count = v_end/4    ; (workitem_vgpr_count-1)/4
        granulated_wavefront_sgpr_count = (s_end+2*3)/8     ; (wavefront_sgpr_count-1)/8
    .end_amd_kernel_code_t

    s_load_dwordx4 s[s_ptr_c:s_ptr_c+3], s[s_ka:s_ka+1], 0+k_ptr_c
    s_load_dwordx2 s[s_ptr_b:s_ptr_b+1], s[s_ka:s_ka+1], 0+k_ptr_b
    s_load_dwordx4 s[s_alpha:s_alpha+3], s[s_ka:s_ka+1], 0+k_alpha
    s_load_dwordx2 s[s_lda:s_lda+1], s[s_ka:s_ka+1], 0+k_lda
    s_load_dword s[s_ldc], s[s_ka:s_ka+1], 0+k_ldc

    v_and_b32 v[v_lane_id], 63, v0
    v_lshrrev_b32 v[v_tmp+3], 6, v0
    v_lshrrev_b32 v[v_wave_p], 1, v[v_tmp+3]
    v_and_b32 v[v_wave_q], 1, v[v_tmp+3]
    v_readfirstlane_b32 s[s_wave_id], v[v_tmp+3]

    v_and_b32 v[v_lane_lo], 31, v[v_lane_id]
    v_lshrrev_b32 v[v_lane_hi], 5, v[v_lane_id]
    v_lshrrev_b32 v[v_lane_w], 4, v[v_lane_id]
    v_and_b32 v[v_tmp], 15, v[v_lane_id]
    v_lshrrev_b32 v[v_lane_u], 1, v[v_tmp]
    v_and_b32 v[v_lane_v], 1, v[v_tmp]

    v_lshlrev_b32 v[v_smem_store], 4, v0

    .cnt=0
    .rept 64
        v_mov_b32 v[.cnt], 0
        .cnt = .cnt + 1
    .endr

    s_waitcnt lgkmcnt(0)
    s_add_u32 s[s_tmp], 127, s[s_m]
    s_lshr_b32 s[s_m_blocks], s[s_tmp], 7
    .v_u32_div_ss v_os_a, s_bx, s_m_blocks, v_tmp, s_tmp
    v_readfirstlane_b32 s[s_n_idx], v[v_os_a]
    s_mul_i32 s[s_tmp], s[s_m_blocks], s[s_n_idx]
    s_sub_u32 s[s_m_idx], s[s_bx], s[s_tmp]
    s_lshl_b32 s[s_bs_a], s[s_lda], 3
    s_lshl_b32 s[s_bs_b], s[s_ldb], 3

    s_lshl_b32 s[s_tmp], s[s_m_idx], 9
    s_lshl_b32 s[s_tmp+1], s[s_n_idx], 9
    s_lshl_b32 s[s_tmp+3], s[s_wave_id], 1
    s_mul_i32 s[s_tmp+2], s[s_tmp+3], s[s_lda]
    s_mul_i32 s[s_tmp+3], s[s_tmp+3], s[s_ldb]
    s_add_u32 s[s_tmp], s[s_tmp], s[s_tmp+2]
    s_add_u32 s[s_tmp+1], s[s_tmp+1], s[s_tmp+3]

    s_add_u32 s[s_ptr_a], s[s_ptr_a], s[s_tmp]
    s_addc_u32 s[s_ptr_a+1], s[s_ptr_a+1], 0
    s_add_u32 s[s_ptr_b], s[s_ptr_b], s[s_tmp+1]
    s_addc_u32 s[s_ptr_b+1], s[s_ptr_b+1], 0

    v_mul_lo_u32 v[v_tmp], s[s_lda], v[v_lane_hi]
    v_mul_lo_u32 v[v_tmp+1], s[s_ldb], v[v_lane_hi]
    v_lshl_add_u32 v[v_os_a], v[v_lane_lo], 4, v[v_tmp]
    v_lshl_add_u32 v[v_os_b], v[v_lane_lo], 4, v[v_tmp+1]   

    global_load_dwordx4 v[v_p0:v_p0+3], v[v_os_a:v_os_a+1], s[s_ptr_a:s_ptr_a+1]
    global_load_dwordx4 v[v_q0:v_q0+3], v[v_os_b:v_os_b+1], s[s_ptr_b:s_ptr_b+1]

    v_lshlrev_b32 v[v_tmp], 4, v[v_lane_u]
    v_lshl_or_b32 v[v_smem_load_a], v[v_wave_p], 8, v[v_tmp]
    v_lshlrev_b32 v[v_tmp], 4, v[v_lane_v]
    v_lshl_or_b32 v[v_tmp+1], v[v_lane_w], 5, v[v_tmp]
    v_lshl_or_b32 v[v_smem_load_b], v[v_wave_q], 8, v[v_tmp+1]

    v_lshlrev_b32 v[v_tmp], 4, v[v_lane_u]
    v_lshl_or_b32 v[v_tmp+1], v[v_wave_p], 8, v[v_tmp]
    v_lshl_or_b32 v[v_tmp], v[v_lane_v], 10, v[v_tmp+1]
    v_lshl_or_b32 v[v_tmp+1], v[v_lane_w], 11, v[v_tmp]
    v_lshl_or_b32 v[v_smem_store_c], v[v_wave_q], 13, v[v_tmp+1]

    v_lshlrev_b32 v[v_tmp], 4, v[v_lane_lo]
    v_lshl_or_b32 v[v_tmp+1], v[v_lane_hi], 9, v[v_tmp]
    s_lshl_b32 s[s_tmp], s[s_wave_id], 10
    v_or_b32 v[v_smem_load_c], s[s_tmp], v[v_tmp+1]

    s_lshl_b32 s[s_tmp], s[s_n_idx], 7
    s_lshl_b32 s[s_tmp+1], s[s_wave_id], 2
    s_add_u32 s[s_tmp], s[s_tmp], s[s_tmp+1]
    s_mul_i32 s[s_tmp+3], s[s_tmp], s[s_ldc]
    s_lshl_b32 s[s_tmp+2], s[s_m_idx], 9
    s_add_u32 s[s_tmp], s[s_tmp+2], s[s_tmp+3]
    s_add_u32 s[s_ptr_c], s[s_ptr_c], s[s_tmp]
    s_addc_u32 s[s_ptr_c+1], s[s_ptr_c+1], 0
    v_mul_lo_u32 v[v_tmp], s[s_ldc], v[v_lane_hi]
    v_lshl_add_u32 v[v_os_c], v[v_lane_lo], 4, v[v_tmp]

    s_mov_b32 s[s_kitr], 8
    s_cmp_lt_u32 s[s_kitr], s[s_k]
    s_cbranch_scc0 L_sgemm128x128_k_loop_end
L_sgemm128x128_k_loop_start:
    s_waitcnt vmcnt(0)
    ds_write_b128 v[v_smem_store], v[v_p0:v_p0+3]
    ds_write_b128 v[v_smem_store], v[v_q0:v_q0+3], offset:0x1000

    s_add_u32 s[s_ptr_a], s[s_ptr_a], s[s_bs_a]
    s_addc_u32 s[s_ptr_a+1], s[s_ptr_a+1], 0
    s_add_u32 s[s_ptr_b], s[s_ptr_b], s[s_bs_b]
    s_addc_u32 s[s_ptr_b+1], s[s_ptr_b+1], 0

    s_waitcnt lgkmcnt(0)
    s_barrier

    global_load_dwordx4 v[v_p0:v_p0+3], v[v_os_a:v_os_a+1], s[s_ptr_a:s_ptr_a+1]
    global_load_dwordx4 v[v_q0:v_q0+3], v[v_os_b:v_os_b+1], s[s_ptr_b:s_ptr_b+1]

    ds_read_b128 v[v_a0+0:v_a0+3], v[v_smem_load_a], offset:0
    ds_read_b128 v[v_b0+0:v_b0+3], v[v_smem_load_b], offset:0x1000

    .cnt = 0
    .rept 4
        ds_read_b128 v[v_a0+4:v_a0+7], v[v_smem_load_a], offset:(.cnt)*0x200+0x80
        s_waitcnt lgkmcnt(1)
        .s_fma4x4 v_c+0 , v_a0+0, v_b0+0

        ds_read_b128 v[v_b0+4:v_b0+7], v[v_smem_load_b], offset:0x1000+(.cnt)*0x200+0x80
        s_waitcnt lgkmcnt(1)
        .s_fma4x4 v_c+16, v_a0+4, v_b0+0

        ds_read_b128 v[v_a1+0:v_a1+3], v[v_smem_load_a], offset:(.cnt+1)*0x200+0
        s_waitcnt lgkmcnt(1)
        .s_fma4x4 v_c+32, v_a0+0, v_b0+4

        ds_read_b128 v[v_b1+0:v_b1+3], v[v_smem_load_b], offset:0x1000+(.cnt+1)*0x200+0
        s_waitcnt lgkmcnt(1)
        .s_fma4x4 v_c+48, v_a0+4, v_b0+4

        ds_read_b128 v[v_a1+4:v_a1+7], v[v_smem_load_a], offset:(.cnt+1)*0x200+0x80
        s_waitcnt lgkmcnt(1)
        .s_fma4x4 v_c+0 , v_a1+0, v_b1+0

        ds_read_b128 v[v_b1+4:v_b1+7], v[v_smem_load_b], offset:0x1000+(.cnt+1)*0x200+0x80
        s_waitcnt lgkmcnt(1)
        .s_fma4x4 v_c+16, v_a1+4, v_b1+0
        .cnt = .cnt + 1

        .if .cnt != 7
            ds_read_b128 v[v_a0+0:v_a0+3], v[v_smem_load_a], offset:(.cnt+1)*0x200+0
            s_waitcnt lgkmcnt(1)
        .else
            s_waitcnt lgkmcnt(0)
        .endif
        .s_fma4x4 v_c+32, v_a1+0, v_b1+4

        .if .cnt != 7
            ds_read_b128 v[v_b0+0:v_b0+3], v[v_smem_load_b], offset:0x1000+(.cnt+1)*0x200+0
            s_waitcnt lgkmcnt(1)
        .endif
        .s_fma4x4 v_c+48, v_a1+4, v_b1+4
        .cnt = .cnt + 1
    .endr
    s_barrier
    s_add_u32 s[s_kitr], 8, s[s_kitr]
    s_cmp_lt_u32 s[s_kitr], s[s_k]
    s_cbranch_scc1 L_sgemm128x128_k_loop_start
L_sgemm128x128_k_loop_end:
    s_waitcnt vmcnt(0)
    ds_write_b128 v[v_smem_store], v[v_p0:v_p0+3]
    ds_write_b128 v[v_smem_store], v[v_q0:v_q0+3], offset:0x1000

    s_waitcnt lgkmcnt(0)
    s_barrier
    ds_read_b128 v[v_a0+0:v_a0+3], v[v_smem_load_a], offset:0
    ds_read_b128 v[v_b0+0:v_b0+3], v[v_smem_load_b], offset:0x1000

    .cnt = 0
    .rept 4
        ds_read_b128 v[v_a0+4:v_a0+7], v[v_smem_load_a], offset:(.cnt)*0x200+0x80
        s_waitcnt lgkmcnt(1)
        .s_fma4x4 v_c+0 , v_a0+0, v_b0+0

        ds_read_b128 v[v_b0+4:v_b0+7], v[v_smem_load_b], offset:0x1000+(.cnt)*0x200+0x80
        s_waitcnt lgkmcnt(1)
        .s_fma4x4 v_c+16, v_a0+4, v_b0+0

        ds_read_b128 v[v_a1+0:v_a1+3], v[v_smem_load_a], offset:(.cnt+1)*0x200+0
        s_waitcnt lgkmcnt(1)
        .s_fma4x4 v_c+32, v_a0+0, v_b0+4

        ds_read_b128 v[v_b1+0:v_b1+3], v[v_smem_load_b], offset:0x1000+(.cnt+1)*0x200+0
        s_waitcnt lgkmcnt(1)
        .s_fma4x4 v_c+48, v_a0+4, v_b0+4

        ds_read_b128 v[v_a1+4:v_a1+7], v[v_smem_load_a], offset:(.cnt+1)*0x200+0x80
        s_waitcnt lgkmcnt(1)
        .s_fma4x4 v_c+0 , v_a1+0, v_b1+0

        ds_read_b128 v[v_b1+4:v_b1+7], v[v_smem_load_b], offset:0x1000+(.cnt+1)*0x200+0x80
        s_waitcnt lgkmcnt(1)
        .s_fma4x4 v_c+16, v_a1+4, v_b1+0
        .cnt = .cnt + 1

        .if .cnt != 7
            ds_read_b128 v[v_a0+0:v_a0+3], v[v_smem_load_a], offset:(.cnt+1)*0x200+0
            s_waitcnt lgkmcnt(1)
        .else
            s_waitcnt lgkmcnt(0)
        .endif
        .s_fma4x4 v_c+32, v_a1+0, v_b1+4

        .if .cnt != 7
            ds_read_b128 v[v_b0+0:v_b0+3], v[v_smem_load_b], offset:0x1000+(.cnt+1)*0x200+0
            s_waitcnt lgkmcnt(1)
        .endif
        .s_fma4x4 v_c+48, v_a1+4, v_b1+4
        .cnt = .cnt + 1
    .endr
    s_barrier

    .cnt = 0
    .rept 4
        .set .cid, ((.cnt>>1)<<3)|((.cnt&1)<<1)
        .set .cof, ((.cnt>>1)<<5)|((.cnt&1)<<1)
        ds_write_b128 v[v_smem_store_c], v[v_c+4*.cid+0 :v_c+4*.cid+3 ], offset:0
        ds_write_b128 v[v_smem_store_c], v[v_c+4*.cid+16:v_c+4*.cid+19], offset:0x80
        ds_write_b128 v[v_smem_store_c], v[v_c+4*.cid+4 :v_c+4*.cid+7 ], offset:0x200
        ds_write_b128 v[v_smem_store_c], v[v_c+4*.cid+20:v_c+4*.cid+23], offset:0x280
        s_waitcnt lgkmcnt(0)
        s_barrier

        ds_read_b128 v[v_c+4*.cid+0 :v_c+4*.cid+3 ], v[v_smem_load_c], offset:0
        ds_read_b128 v[v_c+4*.cid+16:v_c+4*.cid+19], v[v_smem_load_c], offset:0x1000
        ds_read_b128 v[v_c+4*.cid+4 :v_c+4*.cid+7 ], v[v_smem_load_c], offset:0x2000
        ds_read_b128 v[v_c+4*.cid+20:v_c+4*.cid+23], v[v_smem_load_c], offset:0x3000

        s_mul_i32 s[s_tmp], .cof+0, s[s_ldc]
        v_add_co_u32 v[v_tmp], vcc, s[s_tmp], v[v_os_c]
        v_addc_co_u32  v[v_tmp+1], vcc, 0, v[v_os_c+1], vcc
        s_waitcnt lgkmcnt(3)
        global_store_dwordx4 v[v_tmp:v_tmp+1], v[v_c+4*.cid+0 :v_c+4*.cid+3 ], s[s_ptr_c:s_ptr_c+1]
        s_mul_i32 s[s_tmp], .cof+16, s[s_ldc]
        v_add_co_u32 v[v_tmp], vcc, s[s_tmp], v[v_os_c]
        v_addc_co_u32  v[v_tmp+1], vcc, 0, v[v_os_c+1], vcc
        s_waitcnt lgkmcnt(2)
        global_store_dwordx4 v[v_tmp:v_tmp+1], v[v_c+4*.cid+16:v_c+4*.cid+19], s[s_ptr_c:s_ptr_c+1]
        s_mul_i32 s[s_tmp], .cof+64, s[s_ldc]
        v_add_co_u32 v[v_tmp], vcc, s[s_tmp], v[v_os_c]
        v_addc_co_u32  v[v_tmp+1], vcc, 0, v[v_os_c+1], vcc
        s_waitcnt lgkmcnt(1)
        global_store_dwordx4 v[v_tmp:v_tmp+1], v[v_c+4*.cid+4 :v_c+4*.cid+7 ], s[s_ptr_c:s_ptr_c+1]
        s_mul_i32 s[s_tmp], .cof+80, s[s_ldc]
        v_add_co_u32 v[v_tmp], vcc, s[s_tmp], v[v_os_c]
        v_addc_co_u32  v[v_tmp+1], vcc, 0, v[v_os_c+1], vcc
        s_waitcnt lgkmcnt(0)
        global_store_dwordx4 v[v_tmp:v_tmp+1], v[v_c+4*.cid+20:v_c+4*.cid+23], s[s_ptr_c:s_ptr_c+1]
        s_barrier
        .cnt = .cnt + 1
    .endr

    s_endpgm