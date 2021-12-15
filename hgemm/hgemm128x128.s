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
.macro .s_fma8x8 c, a, b
      v_dot2_f32_f16 v[\c+0 ], v[\a+0],  v[\b+0], v[\c+0 ]
      v_dot2_f32_f16 v[\c+1 ], v[\a+1],  v[\b+0], v[\c+1 ]
      v_dot2_f32_f16 v[\c+2 ], v[\a+2],  v[\b+0], v[\c+2 ]
      v_dot2_f32_f16 v[\c+3 ], v[\a+3],  v[\b+0], v[\c+3 ]
      v_dot2_f32_f16 v[\c+4 ], v[\a+0],  v[\b+1], v[\c+4 ]
      v_dot2_f32_f16 v[\c+5 ], v[\a+1],  v[\b+1], v[\c+5 ]
      v_dot2_f32_f16 v[\c+6 ], v[\a+2],  v[\b+1], v[\c+6 ]
      v_dot2_f32_f16 v[\c+7 ], v[\a+3],  v[\b+1], v[\c+7 ]
      v_dot2_f32_f16 v[\c+8 ], v[\a+0],  v[\b+2], v[\c+8 ]
      v_dot2_f32_f16 v[\c+9 ], v[\a+1],  v[\b+2], v[\c+9 ]
      v_dot2_f32_f16 v[\c+10], v[\a+2],  v[\b+2], v[\c+10]
      v_dot2_f32_f16 v[\c+11], v[\a+3],  v[\b+2], v[\c+11]
      v_dot2_f32_f16 v[\c+12], v[\a+0],  v[\b+3], v[\c+12]
      v_dot2_f32_f16 v[\c+13], v[\a+1],  v[\b+3], v[\c+13]
      v_dot2_f32_f16 v[\c+14], v[\a+2],  v[\b+3], v[\c+14]
      v_dot2_f32_f16 v[\c+15], v[\a+3],  v[\b+3], v[\c+15]
      v_dot2_f32_f16 v[\c+16], v[\a+4],  v[\b+0], v[\c+16]
      v_dot2_f32_f16 v[\c+17], v[\a+5],  v[\b+0], v[\c+17]
      v_dot2_f32_f16 v[\c+18], v[\a+6],  v[\b+0], v[\c+18]
      v_dot2_f32_f16 v[\c+19], v[\a+7],  v[\b+0], v[\c+19]
      v_dot2_f32_f16 v[\c+20], v[\a+4],  v[\b+1], v[\c+20]
      v_dot2_f32_f16 v[\c+21], v[\a+5],  v[\b+1], v[\c+21]
      v_dot2_f32_f16 v[\c+22], v[\a+6],  v[\b+1], v[\c+22]
      v_dot2_f32_f16 v[\c+23], v[\a+7],  v[\b+1], v[\c+23]
      v_dot2_f32_f16 v[\c+24], v[\a+4],  v[\b+2], v[\c+24]
      v_dot2_f32_f16 v[\c+25], v[\a+5],  v[\b+2], v[\c+25]
      v_dot2_f32_f16 v[\c+26], v[\a+6],  v[\b+2], v[\c+26]
      v_dot2_f32_f16 v[\c+27], v[\a+7],  v[\b+2], v[\c+27]
      v_dot2_f32_f16 v[\c+28], v[\a+4],  v[\b+3], v[\c+28]
      v_dot2_f32_f16 v[\c+29], v[\a+5],  v[\b+3], v[\c+29]
      v_dot2_f32_f16 v[\c+30], v[\a+6],  v[\b+3], v[\c+30]
      v_dot2_f32_f16 v[\c+31], v[\a+7],  v[\b+3], v[\c+31]
      v_dot2_f32_f16 v[\c+32], v[\a+0],  v[\b+4], v[\c+32]
      v_dot2_f32_f16 v[\c+33], v[\a+1],  v[\b+4], v[\c+33]
      v_dot2_f32_f16 v[\c+34], v[\a+2],  v[\b+4], v[\c+34]
      v_dot2_f32_f16 v[\c+35], v[\a+3],  v[\b+4], v[\c+35]
      v_dot2_f32_f16 v[\c+36], v[\a+0],  v[\b+5], v[\c+36]
      v_dot2_f32_f16 v[\c+37], v[\a+1],  v[\b+5], v[\c+37]
      v_dot2_f32_f16 v[\c+38], v[\a+2],  v[\b+5], v[\c+38]
      v_dot2_f32_f16 v[\c+39], v[\a+3],  v[\b+5], v[\c+39]
      v_dot2_f32_f16 v[\c+40], v[\a+0],  v[\b+6], v[\c+40]
      v_dot2_f32_f16 v[\c+41], v[\a+1],  v[\b+6], v[\c+41]
      v_dot2_f32_f16 v[\c+42], v[\a+2],  v[\b+6], v[\c+42]
      v_dot2_f32_f16 v[\c+43], v[\a+3],  v[\b+6], v[\c+43]
      v_dot2_f32_f16 v[\c+44], v[\a+0],  v[\b+7], v[\c+44]
      v_dot2_f32_f16 v[\c+45], v[\a+1],  v[\b+7], v[\c+45]
      v_dot2_f32_f16 v[\c+46], v[\a+2],  v[\b+7], v[\c+46]
      v_dot2_f32_f16 v[\c+47], v[\a+3],  v[\b+7], v[\c+47]
      v_dot2_f32_f16 v[\c+48], v[\a+4],  v[\b+4], v[\c+48]
      v_dot2_f32_f16 v[\c+49], v[\a+5],  v[\b+4], v[\c+49]
      v_dot2_f32_f16 v[\c+50], v[\a+6],  v[\b+4], v[\c+50]
      v_dot2_f32_f16 v[\c+51], v[\a+7],  v[\b+4], v[\c+51]
      v_dot2_f32_f16 v[\c+52], v[\a+4],  v[\b+5], v[\c+52]
      v_dot2_f32_f16 v[\c+53], v[\a+5],  v[\b+5], v[\c+53]
      v_dot2_f32_f16 v[\c+54], v[\a+6],  v[\b+5], v[\c+54]
      v_dot2_f32_f16 v[\c+55], v[\a+7],  v[\b+5], v[\c+55]
      v_dot2_f32_f16 v[\c+56], v[\a+4],  v[\b+6], v[\c+56]
      v_dot2_f32_f16 v[\c+57], v[\a+5],  v[\b+6], v[\c+57]
      v_dot2_f32_f16 v[\c+58], v[\a+6],  v[\b+6], v[\c+58]
      v_dot2_f32_f16 v[\c+59], v[\a+7],  v[\b+6], v[\c+59]
      v_dot2_f32_f16 v[\c+60], v[\a+4],  v[\b+7], v[\c+60]
      v_dot2_f32_f16 v[\c+61], v[\a+5],  v[\b+7], v[\c+61]
      v_dot2_f32_f16 v[\c+62], v[\a+6],  v[\b+7], v[\c+62]
      v_dot2_f32_f16 v[\c+63], v[\a+7],  v[\b+7], v[\c+63]
.endm
.macro .print v_val, s_out, s_bx, v_tid, v_offset
    ;s_mov_b64 exec, -1
    s_cmp_eq_u32 s[\s_bx], 0
    ;s_cbranch_scc0 L_endhere
    ;v_cmpx_eq_u32 0, v0
    v_lshlrev_b32 v[\v_offset], 3, v[\v_tid]
    global_store_dword v[\v_offset], v[\v_tid], s[\s_out:\s_out+1], offset:0x0
    global_store_dword v[\v_offset], v[\v_val], s[\s_out:\s_out+1], offset:0x0004
    ;s_mov_b64 exec, -1
;L_endhere:
    s_endpgm  
.endm


;kernel arguments OFFSET
;k shift in 1 byte
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
.set k_print,           52
.set k_end,             60
;s shift in number, per 4 bytes
;arguload
;s_bx is block idx
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
.set s_print,           20
;workgroup-wave level dependent addressing index
.set s_m_blocks,        22
.set s_m_idx,           23
.set s_n_idx,           24
.set s_wave_id,         25
.set s_wave_p,          26
.set s_wave_q,          27
.set s_kitr,            28
;tmp register
.set s_tmp,             30
.set s_end,             39
;v shift in number, per 4 bytes
;v0 is tid
;fp32 res, conv to fp16, pack2 to fp32
.set v_c,               0
.set v_fp16_c,          0
.set v_out_c,           0
.set v_a,               64
.set v_b,               72
.set v_p0,              80
.set v_q0,              88
.set v_smem_store,      96
.set v_smem_load_a,     97
.set v_smem_load_b,     98
.set v_smem_store_c,    99
.set v_smem_load_c,     100
;thread-level dependent addressing index
.set v_laneid,          101
.set v_laneid_lo,       102
.set v_laneid_hi,       103
.set v_lane_lo,         104
.set v_lane_hi,         105
.set v_lane_w,          106
.set v_lane_u,          107
.set v_lane_v,          108
.set v_offset_a,        109
.set v_offset_b,        110
.set v_offset_c,        111
.set v_wave_p,          112
.set v_wave_q,          113
;tmp register
.set v_tmp,             114
.set v_tid,             127

.text
.globl hgemm_128x128_kpack2
.p2align 8
.type hgemm_128x128_kpack2,@function
hgemm_128x128_kpack2:
    s_load_dwordx4 s[s_ptr_c:s_ptr_c+3], s[s_ka:s_ka+1], 0+k_ptr_c
    s_load_dwordx2 s[s_ptr_b:s_ptr_b+1], s[s_ka:s_ka+1], 0+k_ptr_b
    s_load_dwordx4 s[s_alpha:s_alpha+3], s[s_ka:s_ka+1], 0+k_alpha
    s_load_dwordx2 s[s_lda:s_lda+1], s[s_ka:s_ka+1], 0+k_lda
    s_load_dword s[s_ldc], s[s_ka:s_ka+1], 0+k_ldc
    s_load_dwordx2 s[s_print:s_print+1], s[s_ka:s_ka+1], 0+k_print
    s_waitcnt lgkmcnt(0)
    v_mov_b32 v[v_tid], v0

    ;Thread Mapping
    s_add_u32 s[s_tmp], 127, s[s_m]
    s_lshr_b32 s[s_m_blocks], s[s_tmp], 7
    .v_u32_div_ss v_offset_a, s_bx, s_m_blocks, v_tmp, s_tmp
    v_readfirstlane_b32 s[s_n_idx], v[v_offset_a]
    s_mul_i32 s[s_tmp], s[s_n_idx], s[s_m_blocks]
    s_sub_i32 s[s_m_idx], s[s_bx], s[s_tmp]
    v_lshrrev_b32 v[v_tmp], 6, v[v_tid]

    ;wave_id, wave_p, wave_q
    v_readfirstlane_b32 s[s_wave_id], v[v_tmp]
    s_lshr_b32 s[s_wave_p], s[s_wave_id], 1
    s_and_b32  s[s_wave_q], s[s_wave_id], 1
    
    ;lane_id, laneid_lo, laneid_hi
    v_and_b32 v[v_laneid], 63, v[v_tid]
    v_and_b32 v[v_laneid_lo], 15, v[v_laneid]
    v_lshrrev_b32 v[v_laneid_hi], 4, v[v_laneid]
    v_and_b32 v[v_lane_lo], 31, v[v_laneid]
    v_lshrrev_b32 v[v_lane_hi], 5, v[v_laneid]
    v_lshrrev_b32 v[v_lane_w], 4, v[v_laneid]
    v_lshrrev_b32 v[v_lane_u], 1, v[v_laneid_lo]
    v_and_b32 v[v_lane_v], 1, v[v_laneid_lo]

    ;bs shift of a and b
    s_lshl_b32 s[s_bs_a], s[s_lda], 4
    s_lshl_b32 s[s_bs_b], s[s_ldb], 4
    
    ;C clear
    .cnt=0
    .rept 64
        v_mov_b32 v[.cnt], 0
        .cnt = .cnt + 1
    .endr

    ;ptr a addressing
    s_lshl_b32 s[s_tmp], s[s_m_idx], 9
    s_lshl_b32 s[s_tmp+1], s[s_wave_id], 2
    s_mul_i32  s[s_tmp+2], s[s_tmp+1], s[s_lda]
    s_add_u32  s[s_tmp+3], s[s_tmp], s[s_tmp+2]
    s_add_u32  s[s_ptr_a], s[s_ptr_a], s[s_tmp+3]
    s_addc_u32 s[s_ptr_a+1], s[s_ptr_a+1], 0
    v_mul_lo_u32 v[v_tmp], s[s_lda], v[v_laneid_hi]
    v_lshl_add_u32 v[v_offset_a], v[v_laneid_lo], 5, v[v_tmp]

    ;ptr b addressing
    s_lshl_b32 s[s_tmp], s[s_n_idx], 9
    s_lshl_b32 s[s_tmp+1], s[s_wave_id], 2
    s_mul_i32  s[s_tmp+2], s[s_tmp+1], s[s_ldb]
    s_add_u32  s[s_tmp+3], s[s_tmp], s[s_tmp+2]
    s_add_u32  s[s_ptr_b], s[s_ptr_b], s[s_tmp+3]
    s_addc_u32 s[s_ptr_b+1], s[s_ptr_b+1], 0
    v_mul_lo_u32 v[v_tmp], s[s_ldb], v[v_laneid_hi]
    v_lshl_add_u32 v[v_offset_b], v[v_laneid_lo], 5, v[v_tmp]

    ;ptr c addressing
    s_lshl_b32 s[s_tmp], s[s_n_idx], 7
    s_lshl_b32 s[s_tmp+1], s[s_wave_id], 2
    s_add_u32  s[s_tmp+2], s[s_tmp], s[s_tmp+1]
    s_lshr_b32 s[s_tmp+1], s[s_ldc], 1
    s_mul_i32  s[s_tmp+3], s[s_tmp+2], s[s_tmp+1]
    s_lshl_b32 s[s_tmp], s[s_m_idx], 8
    s_add_u32  s[s_tmp+2], s[s_tmp], s[s_tmp+3]
    s_add_u32  s[s_ptr_c], s[s_ptr_c], s[s_tmp+2]
    s_addc_u32 s[s_ptr_c+1], s[s_ptr_c+1], 0
    v_mul_lo_u32 v[v_tmp], s[s_tmp+1], v[v_lane_hi]
    v_lshl_add_u32 v[v_offset_c], v[v_lane_lo], 3, v[v_tmp]

    ;smem_store, smem_load_a, smem_load_b
    v_lshlrev_b32 v[v_smem_store], 5, v[v_tid]
    s_lshl_b32 s[s_tmp], s[s_wave_p], 8
    v_lshl_or_b32 v[v_smem_load_a], v[v_lane_u], 4, s[s_tmp]
    s_lshl_b32 s[s_tmp], s[s_wave_q], 8
    v_lshl_or_b32 v[v_tmp+1], v[v_lane_w], 5, s[s_tmp]
    v_lshl_or_b32 v[v_tmp], v[v_lane_v], 4, v[v_tmp+1]
    v_or_b32 v[v_smem_load_b], 0x2000, v[v_tmp]

    ;smem_store_c, smem_load_c
    v_lshlrev_b32 v[v_tmp], 3, v[v_lane_u]
    v_lshl_or_b32 v[v_tmp+1], s[s_wave_p], 7, v[v_tmp]
    v_lshl_or_b32 v[v_tmp], v[v_lane_v], 9, v[v_tmp+1]
    v_lshl_or_b32 v[v_tmp+1], v[v_lane_w], 10, v[v_tmp]
    v_lshl_or_b32 v[v_smem_store_c], s[s_wave_q], 12, v[v_tmp+1]
    v_lshlrev_b32 v[v_tmp], 3, v[v_lane_lo]
    v_lshl_or_b32 v[v_tmp+1], v[v_lane_hi], 8, v[v_tmp]
    v_lshl_or_b32 v[v_smem_load_c], s[s_wave_id], 9,  v[v_tmp+1]
    
    ;iterator clear
    s_mov_b32 s[s_kitr], 0
L_hgemm128x128_kpack2_loop_start:

    ;load from global, store to vgpr
    global_load_dwordx4 v[v_p0  :v_p0+3], v[v_offset_a], s[s_ptr_a:s_ptr_a+1]
    global_load_dwordx4 v[v_p0+4:v_p0+7], v[v_offset_a], s[s_ptr_a:s_ptr_a+1], offset:0x0010
    s_add_u32 s[s_ptr_a], s[s_ptr_a], s[s_bs_a]
    s_addc_u32 s[s_ptr_a+1], s[s_ptr_a+1], 0
    global_load_dwordx4 v[v_q0  :v_q0+3], v[v_offset_b], s[s_ptr_b:s_ptr_b+1]
    global_load_dwordx4 v[v_q0+4:v_q0+7], v[v_offset_b], s[s_ptr_b:s_ptr_b+1], offset:0x0010
    s_add_u32 s[s_ptr_b], s[s_ptr_b], s[s_bs_b]
    s_addc_u32 s[s_ptr_b+1], s[s_ptr_b+1], 0

    ;load from vgpr, store to lds
    s_waitcnt vmcnt(2)
    ds_write_b128 v[v_smem_store], v[v_p0:v_p0+3]
    ds_write_b128 v[v_smem_store], v[v_p0+4:v_p0+7], offset:0x0010
    s_waitcnt vmcnt(0)
    ds_write_b128 v[v_smem_store], v[v_q0:v_q0+3], offset:0x2000
    ds_write_b128 v[v_smem_store], v[v_q0+4:v_q0+7], offset:0x2010
    s_waitcnt lgkmcnt(0)
    s_barrier

    .cnt=0
    .rept 16
        ds_read_b128 v[v_a+0:v_a+3], v[v_smem_load_a], offset:(.cnt)*0x200+0
        ds_read_b128 v[v_a+4:v_a+7], v[v_smem_load_a], offset:(.cnt)*0x200+0x80
        ds_read_b128 v[v_b+0:v_b+3], v[v_smem_load_b], offset:(.cnt)*0x200+0
        ds_read_b128 v[v_b+4:v_b+7], v[v_smem_load_b], offset:(.cnt)*0x200+0x80
        s_waitcnt lgkmcnt(0)
        .s_fma8x8 v_c, v_a, v_b
        .cnt = .cnt + 1
    .endr
    s_add_u32 s[s_kitr], 16, s[s_kitr]
    s_lshr_b32 s[s_tmp], s[s_k], 1
    s_cmp_lt_u32 s[s_kitr], s[s_tmp]
    s_cbranch_scc1 L_hgemm128x128_kpack2_loop_start
L_hgemm128x128_kpack2_loop_end:
    s_barrier
    ;Packing 2FP16 to 1FP32
    .cnt=0
    .rept 32
        v_mul_f32 v[(.cnt)*2]  , s[s_alpha], v[(.cnt)*2]
        v_mul_f32 v[(.cnt)*2+1], s[s_alpha], v[(.cnt)*2+1]
        v_cvt_f16_f32 v[(.cnt)*2]  , v[(.cnt)*2]
        v_cvt_f16_f32 v[(.cnt)*2+1], v[(.cnt)*2+1]
        V_PACK_B32_F16 v[.cnt], v[(.cnt)*2], v[(.cnt)*2+1]
        .cnt = .cnt + 1
    .endr

    ;write to global c
    .cnt=0
    .rept 4
        .set .cid, ((.cnt>>1)<<3)|((.cnt&1)<<1)
        .set .cof, ((.cnt>>1)<<5)|((.cnt&1)<<1)
        ds_write_b64 v[v_smem_store_c], v[v_c+2*.cid+0 :v_c+2*.cid+1 ], offset:0
        ds_write_b64 v[v_smem_store_c], v[v_c+2*.cid+8 :v_c+2*.cid+9 ], offset:0x40
        ds_write_b64 v[v_smem_store_c], v[v_c+2*.cid+2 :v_c+2*.cid+3 ], offset:0x100
        ds_write_b64 v[v_smem_store_c], v[v_c+2*.cid+10:v_c+2*.cid+11], offset:0x140
        s_waitcnt lgkmcnt(0)
        s_barrier

        ;load shuffled data from lds, put into vgpr
        ds_read_b64 v[v_c+2*.cid+0 :v_c+2*.cid+1 ], v[v_smem_load_c], offset:0
        ds_read_b64 v[v_c+2*.cid+8 :v_c+2*.cid+9 ], v[v_smem_load_c], offset:0x0800
        ds_read_b64 v[v_c+2*.cid+2 :v_c+2*.cid+3 ], v[v_smem_load_c], offset:0x1000
        ds_read_b64 v[v_c+2*.cid+10:v_c+2*.cid+11], v[v_smem_load_c], offset:0x1800
        s_waitcnt lgkmcnt(0)
        s_barrier

        s_lshr_b32 s[s_tmp+2], s[s_ldc], 1
        s_mul_i32 s[s_tmp], .cof+0, s[s_tmp+2]
        v_add_co_u32 v[v_tmp], vcc, s[s_tmp], v[v_offset_c]
        v_addc_co_u32  v[v_tmp+1], vcc, 0, v[v_offset_c+1], vcc
        s_waitcnt lgkmcnt(3)
        global_store_dwordx2 v[v_tmp], v[v_c+2*.cid+0 :v_c+2*.cid+1 ], s[s_ptr_c:s_ptr_c+1]
        s_mul_i32 s[s_tmp], .cof+16, s[s_tmp+2]
        v_add_co_u32 v[v_tmp], vcc, s[s_tmp], v[v_offset_c]
        v_addc_co_u32  v[v_tmp+1], vcc, 0, v[v_offset_c+1], vcc
        s_waitcnt lgkmcnt(2)
        global_store_dwordx2 v[v_tmp], v[v_c+2*.cid+8 :v_c+2*.cid+9 ], s[s_ptr_c:s_ptr_c+1]
        s_mul_i32 s[s_tmp], .cof+64, s[s_tmp+2]
        v_add_co_u32 v[v_tmp], vcc, s[s_tmp], v[v_offset_c]
        v_addc_co_u32  v[v_tmp+1], vcc, 0, v[v_offset_c+1], vcc
        s_waitcnt lgkmcnt(1)
        global_store_dwordx2 v[v_tmp], v[v_c+2*.cid+2 :v_c+2*.cid+3 ], s[s_ptr_c:s_ptr_c+1]
        s_mul_i32 s[s_tmp], .cof+80, s[s_tmp+2]
        v_add_co_u32 v[v_tmp], vcc, s[s_tmp], v[v_offset_c]
        v_addc_co_u32  v[v_tmp+1], vcc, 0, v[v_offset_c+1], vcc
        s_waitcnt lgkmcnt(0)
        global_store_dwordx2 v[v_tmp], v[v_c+2*.cid+10:v_c+2*.cid+11], s[s_ptr_c:s_ptr_c+1]
        .if .cnt != 3
            s_waitcnt vmcnt(0)
            s_barrier
        .endif
        .cnt = .cnt + 1
    .endr

    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel hgemm_128x128_kpack2
    .amdhsa_group_segment_fixed_size 32768
    .amdhsa_user_sgpr_dispatch_ptr 0
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 128
    .amdhsa_next_free_sgpr 40
    .amdhsa_ieee_mode 0
    .amdhsa_dx10_clamp 0
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [ 1, 0 ]
amdhsa.kernels:
  - .name: hgemm_128x128_kpack2
    .symbol: hgemm_128x128_kpack2.kd
    .sgpr_count: 40
    .vgpr_count: 128
    .kernarg_segment_align: 8
    .kernarg_segment_size: 64
    .group_segment_fixed_size: 32768
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .reqd_workgroup_size : [256, 1, 1]
    .max_flat_workgroup_size: 256
    .args:
    - { .name: ptr_c,           .size: 8, .offset:   0, .value_kind: global_buffer, .value_type: f16, .address_space: global, .is_const: false}
    - { .name: ptr_a,           .size: 8, .offset:   8, .value_kind: global_buffer, .value_type: f16, .address_space: global, .is_const: true }
    - { .name: ptr_b,           .size: 8, .offset:  16, .value_kind: global_buffer, .value_type: f16, .address_space: global, .is_const: true }
    - { .name: alpha,           .size: 4, .offset:  24, .value_kind: by_value, .value_type: f32}
    - { .name: m,               .size: 4, .offset:  28, .value_kind: by_value, .value_type: i32}
    - { .name: n,               .size: 4, .offset:  32, .value_kind: by_value, .value_type: i32}
    - { .name: k,               .size: 4, .offset:  36, .value_kind: by_value, .value_type: i32}
    - { .name: lda,             .size: 4, .offset:  40, .value_kind: by_value, .value_type: i32}
    - { .name: ldb,             .size: 4, .offset:  44, .value_kind: by_value, .value_type: i32}
    - { .name: ldc,             .size: 4, .offset:  48, .value_kind: by_value, .value_type: i32}
    - { .name: print,           .size: 8, .offset:  52, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: false}
...
.end_amdgpu_metadata