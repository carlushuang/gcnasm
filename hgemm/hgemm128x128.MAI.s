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


;kernel arguments OFFSET, shift in 1 byte
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

;sgpr
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
.set s_m_blocks,        22
.set s_m_idx,           23
.set s_n_idx,           24
.set s_wave_id,         25
.set s_wave_p,          26
.set s_wave_q,          27
.set s_kitr,            28
.set s_tmp,             30
.set s_end,             39

;vgpr
.set v_c,               0
.set v_a0,              0
.set v_b0,              4
.set v_a1,              8
.set v_b1,              12
.set v_p0,              16
.set v_q0,              24
.set v_smem_store,      32
.set v_smem_load_a,     33
.set v_smem_load_b,     34
.set v_smem_store_c,    35
.set v_smem_load_c,     36
.set v_laneid,          37
.set v_lane_lo,         38
.set v_lane_hi,         39
.set v_offset_a0,       40
.set v_offset_a1,       41
.set v_offset_b0,       42
.set v_offset_b1,       43
.set v_offset_c,        44
.set v_wave_p,          45
.set v_wave_q,          46
.set v_lane_col,        47
.set v_lane_row,        48
.set v_tmp,             49
.set v_tid,             63

;Accvgpr
.set a_c,               0
.set a_end,             64

.text
.globl hgemm_128x128_kpack4
.p2align 8
.type hgemm_128x128_kpack4,@function
hgemm_128x128_kpack4:
    s_load_dwordx4 s[s_ptr_c:s_ptr_c+3], s[s_ka:s_ka+1], 0+k_ptr_c
    s_load_dwordx2 s[s_ptr_b:s_ptr_b+1], s[s_ka:s_ka+1], 0+k_ptr_b
    s_load_dwordx4 s[s_alpha:s_alpha+3], s[s_ka:s_ka+1], 0+k_alpha
    s_load_dwordx2 s[s_lda:s_lda+1], s[s_ka:s_ka+1], 0+k_lda
    s_load_dword s[s_ldc], s[s_ka:s_ka+1], 0+k_ldc
    s_load_dwordx2 s[s_print:s_print+1], s[s_ka:s_ka+1], 0+k_print
    s_waitcnt lgkmcnt(0)
    v_mov_b32 v[v_tid], v0
    ;pack4, 128*4fp16=128*2fp32
    s_lshl_b32 s[s_lda], s[s_lda], 1
    s_lshl_b32 s[s_ldb], s[s_ldb], 1
    ;128*1fp16=(128*fp32)/2
    s_lshr_b32 s[s_ldc], s[s_ldc], 1
    
    ;Thread Mapping
    s_add_u32 s[s_tmp], 127, s[s_m]
    s_lshr_b32 s[s_m_blocks], s[s_tmp], 7
    .v_u32_div_ss v_offset_a0, s_bx, s_m_blocks, v_tmp, s_tmp
    v_readfirstlane_b32 s[s_n_idx], v[v_offset_a0]
    s_mul_i32 s[s_tmp], s[s_n_idx], s[s_m_blocks]
    s_sub_i32 s[s_m_idx], s[s_bx], s[s_tmp]

    ;wave_id, wave_p, wave_q
    v_lshrrev_b32 v[v_tmp], 6, v[v_tid]
    v_readfirstlane_b32 s[s_wave_id], v[v_tmp]
    s_lshr_b32 s[s_wave_p], s[s_wave_id], 1
    s_and_b32  s[s_wave_q], s[s_wave_id], 1
    
    ;lane_id, lane_lo, lane_hi
    v_and_b32 v[v_laneid], 63, v[v_tid]
    v_and_b32 v[v_lane_lo], 31, v[v_laneid]
    v_lshrrev_b32 v[v_lane_hi], 5, v[v_laneid]
    
    ;lane_col, lane_row
    v_lshrrev_b32 v[v_lane_col], 4, v[v_laneid]
    v_and_b32     v[v_lane_row], 15, v[v_laneid]

    ;bs shift of a and b
    s_lshl_b32 s[s_bs_a], s[s_lda], 3
    s_lshl_b32 s[s_bs_b], s[s_ldb], 3


    ;ptr a addressing
    s_lshl_b32 s[s_tmp], s[s_m_idx], 10
    s_mul_i32  s[s_tmp+1], s[s_wave_id], s[s_lda]
    s_add_u32  s[s_tmp+2], s[s_tmp], s[s_tmp+1]
    s_add_u32  s[s_ptr_a], s[s_ptr_a], s[s_tmp+2]
    s_addc_u32 s[s_ptr_a+1], s[s_ptr_a+1], 0
    v_lshlrev_b32 v[v_offset_a0], 4, v[v_laneid]
    s_mul_i32 s[s_tmp], 4, s[s_lda]
    v_add_co_u32 v[v_offset_a1], s[s_tmp], v[v_offset_a0]
    ;.print v_offset_a1, s_print, s_bx, v_tid, v_tmp+4
    ;ptr b addressing
    s_lshl_b32 s[s_tmp], s[s_n_idx], 10
    s_mul_i32  s[s_tmp+1], s[s_wave_id], s[s_ldb]
    s_add_u32  s[s_tmp+2], s[s_tmp], s[s_tmp+1]
    s_add_u32  s[s_ptr_b], s[s_ptr_b], s[s_tmp+2]
    s_addc_u32 s[s_ptr_b+1], s[s_ptr_b+1], 0
    v_lshlrev_b32 v[v_offset_b0], 4, v[v_laneid]
    s_mul_i32 s[s_tmp], 4, s[s_ldb]
    v_add_co_u32 v[v_offset_b1], s[s_tmp], v[v_offset_b0]
    ;ptr c addressing
    s_lshl_b32 s[s_tmp], s[s_n_idx], 7
    s_lshl_b32 s[s_tmp+1], s[s_wave_id], 2
    s_add_u32  s[s_tmp+2], s[s_tmp], s[s_tmp+1]
    s_mul_i32  s[s_tmp+3], s[s_tmp+2], s[s_ldc]
    s_lshl_b32 s[s_tmp], s[s_m_idx], 8
    s_add_u32  s[s_tmp+2], s[s_tmp], s[s_tmp+3]
    s_add_u32  s[s_ptr_c], s[s_ptr_c], s[s_tmp+2]
    s_addc_u32 s[s_ptr_c+1], s[s_ptr_c+1], 0
    v_mul_lo_u32  v[v_tmp], v[v_lane_col], s[s_ldc]
    v_lshl_or_b32 v[v_offset_c], v[v_lane_row], 4, v[v_tmp] 

    ;smem_store
    v_lshlrev_b32 v[v_smem_store], 4, v[v_tid]
    ;smem_load_a
    s_lshl_b32 s[s_tmp], s[s_wave_p], 8
    v_lshl_or_b32 v[v_tmp], v[v_lane_hi], 10, s[s_tmp]
    v_lshl_or_b32 v[v_smem_load_a], v[v_lane_lo], 3, v[v_tmp]
    ;smem_load_b
    s_lshl_b32 s[s_tmp], s[s_wave_q], 8
    v_lshl_or_b32 v[v_tmp], v[v_lane_hi], 10, s[s_tmp]
    v_lshl_or_b32 v[v_tmp+1], v[v_lane_lo], 3, v[v_tmp]
    v_or_b32 v[v_smem_load_b], 0x2000, v[v_tmp+1]

    ;smem_store_c
    v_lshlrev_b32 v[v_tmp], 3, v[v_lane_hi]
    v_lshl_or_b32 v[v_tmp+1], s[s_wave_p], 6, v[v_tmp]
    v_lshl_or_b32 v[v_tmp], v[v_lane_lo], 8, v[v_tmp+1]
    v_lshl_or_b32 v[v_smem_store_c], s[s_wave_q], 13, v[v_tmp]
    ;smem_load_c
    v_lshlrev_b32 v[v_smem_load_c], 4,  v[v_tid]
    
    ;Mainloop 
    s_mov_b32 s[s_kitr], 32

    ;load from global, store to vgpr
    ;prefetch global
    global_load_dwordx4 v[v_p0  :v_p0+3], v[v_offset_a0], s[s_ptr_a:s_ptr_a+1]
    global_load_dwordx4 v[v_p0+4:v_p0+7], v[v_offset_a1], s[s_ptr_a:s_ptr_a+1]
    s_add_u32  s[s_ptr_a]  , s[s_ptr_a]  , s[s_bs_a]
    s_addc_u32 s[s_ptr_a+1], s[s_ptr_a+1], 0
    global_load_dwordx4 v[v_q0  :v_q0+3], v[v_offset_b0], s[s_ptr_b:s_ptr_b+1]
    global_load_dwordx4 v[v_q0+4:v_q0+7], v[v_offset_b1], s[s_ptr_b:s_ptr_b+1]
    s_add_u32  s[s_ptr_b]  , s[s_ptr_b]  , s[s_bs_b]
    s_addc_u32 s[s_ptr_b+1], s[s_ptr_b+1], 0
    
    ;C clear
    .cnt=0
    .rept 64
        v_accvgpr_write_b32 a[.cnt], 0
        .cnt = .cnt + 1
    .endr

    ;load from vgpr, store to lds
    s_waitcnt vmcnt(2)
    ds_write_b128 v[v_smem_store], v[v_p0  :v_p0+3]
    ds_write_b128 v[v_smem_store], v[v_p0+4:v_p0+7], offset:0x1000
    s_waitcnt vmcnt(0)
    ds_write_b128 v[v_smem_store], v[v_q0  :v_q0+3], offset:0x2000
    ds_write_b128 v[v_smem_store], v[v_q0+4:v_q0+7], offset:0x3000
    s_waitcnt lgkmcnt(0)
    s_barrier
L_hgemm128x128_kpack4_loop_start:
        .cnt =0
        ;Round0
        ;Pre-read ThisRoundIssue0.
            ds_read_b64 v[v_a0+0:v_a0+1], v[v_smem_load_a], offset:0*0x200+(.cnt)*0x800
            ds_read_b64 v[v_b0+0:v_b0+1], v[v_smem_load_b], offset:0*0x200+(.cnt)*0x800
            ds_read_b64 v[v_b0+2:v_b0+3], v[v_smem_load_b], offset:1*0x200+(.cnt)*0x800
            ds_read_b64 v[v_a0+2:v_a0+3], v[v_smem_load_a], offset:1*0x200+(.cnt)*0x800
        ;Issue0,(0,0)
        s_waitcnt lgkmcnt(2)
        v_mfma_f32_32x32x8f16 a[a_c+ 0:a_c+15], v[v_a0+0:v_a0+1], v[v_b0+0:v_b0+1], a[a_c+ 0:a_c+15]
        ;Pre-read NextLoopGlobal.
            global_load_dwordx4 v[v_p0  :v_p0+3], v[v_offset_a0], s[s_ptr_a:s_ptr_a+1]
            global_load_dwordx4 v[v_p0+4:v_p0+7], v[v_offset_a1], s[s_ptr_a:s_ptr_a+1]
            s_add_u32 s[s_ptr_a], s[s_ptr_a], s[s_bs_a]
            s_addc_u32 s[s_ptr_a+1], s[s_ptr_a+1], 0
            global_load_dwordx4 v[v_q0  :v_q0+3], v[v_offset_b0], s[s_ptr_b:s_ptr_b+1]
            global_load_dwordx4 v[v_q0+4:v_q0+7], v[v_offset_b1], s[s_ptr_b:s_ptr_b+1]
            s_add_u32 s[s_ptr_b], s[s_ptr_b], s[s_bs_b]
            s_addc_u32 s[s_ptr_b+1], s[s_ptr_b+1], 0
        ;Issue1,(0,1)
        s_waitcnt lgkmcnt(1)
        v_mfma_f32_32x32x8f16 a[a_c+16:a_c+31], v[v_a0+0:v_a0+1], v[v_b0+2:v_b0+3], a[a_c+16:a_c+31]     
        ;Pre-read NextRound.
            ds_read_b64 v[v_a1+0:v_a1+1], v[v_smem_load_a], offset:0*0x200+(.cnt+1)*0x800
            ds_read_b64 v[v_b1+0:v_b1+1], v[v_smem_load_b], offset:0*0x200+(.cnt+1)*0x800
        ;Issue2,(1,0)
        s_waitcnt lgkmcnt(2)
        v_mfma_f32_32x32x8f16 a[a_c+32:a_c+47], v[v_a0+2:v_a0+3], v[v_b0+0:v_b0+1], a[a_c+32:a_c+47]
        ;Pre-read NextRound.
            ds_read_b64 v[v_b1+2:v_b1+3], v[v_smem_load_b], offset:1*0x200+(.cnt+1)*0x800
            ds_read_b64 v[v_a1+2:v_a1+3], v[v_smem_load_a], offset:1*0x200+(.cnt+1)*0x800
        ;Issue3,(1,1)
        v_mfma_f32_32x32x8f16 a[a_c+48:a_c+63], v[v_a0+2:v_a0+3], v[v_b0+2:v_b0+3], a[a_c+48:a_c+63]
        
        .cnt =1
        ;Round1
        ;Issue0,(0,0)
        s_waitcnt lgkmcnt(2)
        v_mfma_f32_32x32x8f16 a[a_c+ 0:a_c+15], v[v_a1+0:v_a1+1], v[v_b1+0:v_b1+1], a[a_c+ 0:a_c+15]
        ;Pre-read NextRound.
            ds_read_b64 v[v_b0+0:v_b0+1], v[v_smem_load_b], offset:0*0x200+(.cnt+1)*0x800
            ds_read_b64 v[v_a0+0:v_a0+1], v[v_smem_load_a], offset:0*0x200+(.cnt+1)*0x800
        ;Issue1,(0,1)
        s_waitcnt lgkmcnt(3)
        v_mfma_f32_32x32x8f16 a[a_c+16:a_c+31], v[v_a1+0:v_a1+1], v[v_b1+2:v_b1+3], a[a_c+16:a_c+31]     
        ;Pre-read NextRound
            ds_read_b64 v[v_b0+2:v_b0+3], v[v_smem_load_b], offset:1*0x200+(.cnt+1)*0x800
            ds_read_b64 v[v_a0+2:v_a0+3], v[v_smem_load_a], offset:1*0x200+(.cnt+1)*0x800
        ;Issue2,(1,0)
        s_waitcnt lgkmcnt(4)
        v_mfma_f32_32x32x8f16 a[a_c+32:a_c+47], v[v_a1+2:v_a1+3], v[v_b1+0:v_b1+1], a[a_c+32:a_c+47]
        ;Issue3,(1,1)
        v_mfma_f32_32x32x8f16 a[a_c+48:a_c+63], v[v_a1+2:v_a1+3], v[v_b1+2:v_b1+3], a[a_c+48:a_c+63]


        .cnt =2
        ;Round2
        ;Pre-read NextRound.
            ds_read_b64 v[v_a1+0:v_a1+1], v[v_smem_load_a], offset:0*0x200+(.cnt+1)*0x800
            ds_read_b64 v[v_b1+0:v_b1+1], v[v_smem_load_b], offset:0*0x200+(.cnt+1)*0x800
        ;Issue0,(0,0)
        s_waitcnt lgkmcnt(4)
        v_mfma_f32_32x32x8f16 a[a_c+ 0:a_c+15], v[v_a0+0:v_a0+1], v[v_b0+0:v_b0+1], a[a_c+ 0:a_c+15]
        ;Pre-read NextRound.
            ds_read_b64 v[v_b1+2:v_b1+3], v[v_smem_load_b], offset:1*0x200+(.cnt+1)*0x800
            ds_read_b64 v[v_a1+2:v_a1+3], v[v_smem_load_a], offset:1*0x200+(.cnt+1)*0x800
        ;Issue1,(0,1)
        s_waitcnt lgkmcnt(5)
        v_mfma_f32_32x32x8f16 a[a_c+16:a_c+31], v[v_a0+0:v_a0+1], v[v_b0+2:v_b0+3], a[a_c+16:a_c+31]     
        ;Issue2,(1,0)
        s_waitcnt lgkmcnt(4)
        v_mfma_f32_32x32x8f16 a[a_c+32:a_c+47], v[v_a0+2:v_a0+3], v[v_b0+0:v_b0+1], a[a_c+32:a_c+47]
        ;Issue3,(1,1)
        v_mfma_f32_32x32x8f16 a[a_c+48:a_c+63], v[v_a0+2:v_a0+3], v[v_b0+2:v_b0+3], a[a_c+48:a_c+63]

        .cnt =3
        ;Round3
        ;Issue0,(0,0)
        s_waitcnt lgkmcnt(2)
        v_mfma_f32_32x32x8f16 a[a_c+ 0:a_c+15], v[v_a1+0:v_a1+1], v[v_b1+0:v_b1+1], a[a_c+ 0:a_c+15]
        ;Issue1,(0,1)
        s_waitcnt lgkmcnt(1)
        v_mfma_f32_32x32x8f16 a[a_c+16:a_c+31], v[v_a1+0:v_a1+1], v[v_b1+2:v_b1+3], a[a_c+16:a_c+31]
        ;Pre-Write NextLoop.
            s_waitcnt vmcnt(3)
            ds_write_b128 v[v_smem_store], v[v_p0  :v_p0+3]
            s_waitcnt vmcnt(2)
            ds_write_b128 v[v_smem_store], v[v_p0+4:v_p0+7], offset:0x1000  
        ;Issue2,(1,0)
        s_waitcnt lgkmcnt(2)
        v_mfma_f32_32x32x8f16 a[a_c+32:a_c+47], v[v_a1+2:v_a1+3], v[v_b1+0:v_b1+1], a[a_c+32:a_c+47]
        ;Pre-write NextLoop.
            s_waitcnt vmcnt(1)
            ds_write_b128 v[v_smem_store], v[v_q0  :v_q0+3], offset:0x2000   
            s_waitcnt vmcnt(0)
            ds_write_b128 v[v_smem_store], v[v_q0+4:v_q0+7], offset:0x3000
        ;Issue3,(1,1)
        v_mfma_f32_32x32x8f16 a[a_c+48:a_c+63], v[v_a1+2:v_a1+3], v[v_b1+2:v_b1+3], a[a_c+48:a_c+63]
        s_waitcnt lgkmcnt(0)
        s_barrier
    s_add_u32 s[s_kitr], 32, s[s_kitr]
    s_cmp_lt_u32 s[s_kitr], s[s_k]
    s_cbranch_scc1 L_hgemm128x128_kpack4_loop_start
L_hgemm128x128_kpack4_loop_end:
        .cnt =0
        ;Round0
        ;Pre-read ThisRoundIssue0.
            ds_read_b64 v[v_a0+0:v_a0+1], v[v_smem_load_a], offset:0*0x200+(.cnt)*0x800
            ds_read_b64 v[v_b0+0:v_b0+1], v[v_smem_load_b], offset:0*0x200+(.cnt)*0x800
            ds_read_b64 v[v_b0+2:v_b0+3], v[v_smem_load_b], offset:1*0x200+(.cnt)*0x800
            ds_read_b64 v[v_a0+2:v_a0+3], v[v_smem_load_a], offset:1*0x200+(.cnt)*0x800
        ;Issue0,(0,0)
        s_waitcnt lgkmcnt(2)
        v_mfma_f32_32x32x8f16 a[a_c+ 0:a_c+15], v[v_a0+0:v_a0+1], v[v_b0+0:v_b0+1], a[a_c+ 0:a_c+15]
        ;Issue1,(0,1)
        s_waitcnt lgkmcnt(1)
        v_mfma_f32_32x32x8f16 a[a_c+16:a_c+31], v[v_a0+0:v_a0+1], v[v_b0+2:v_b0+3], a[a_c+16:a_c+31]     
        ;Pre-read NextRound.
            ds_read_b64 v[v_a1+0:v_a1+1], v[v_smem_load_a], offset:0*0x200+(.cnt+1)*0x800
            ds_read_b64 v[v_b1+0:v_b1+1], v[v_smem_load_b], offset:0*0x200+(.cnt+1)*0x800
        ;Issue2,(1,0)
        s_waitcnt lgkmcnt(2)
        v_mfma_f32_32x32x8f16 a[a_c+32:a_c+47], v[v_a0+2:v_a0+3], v[v_b0+0:v_b0+1], a[a_c+32:a_c+47]
        ;Pre-read NextRound.
            ds_read_b64 v[v_b1+2:v_b1+3], v[v_smem_load_b], offset:1*0x200+(.cnt+1)*0x800
            ds_read_b64 v[v_a1+2:v_a1+3], v[v_smem_load_a], offset:1*0x200+(.cnt+1)*0x800
        ;Issue3,(1,1)
        v_mfma_f32_32x32x8f16 a[a_c+48:a_c+63], v[v_a0+2:v_a0+3], v[v_b0+2:v_b0+3], a[a_c+48:a_c+63]
        
        .cnt =1
        ;Round1
        ;Issue0,(0,0)
        s_waitcnt lgkmcnt(2)
        v_mfma_f32_32x32x8f16 a[a_c+ 0:a_c+15], v[v_a1+0:v_a1+1], v[v_b1+0:v_b1+1], a[a_c+ 0:a_c+15]
        ;Pre-read NextRound.
            ds_read_b64 v[v_b0+0:v_b0+1], v[v_smem_load_b], offset:0*0x200+(.cnt+1)*0x800
            ds_read_b64 v[v_a0+0:v_a0+1], v[v_smem_load_a], offset:0*0x200+(.cnt+1)*0x800
        ;Issue1,(0,1)
        s_waitcnt lgkmcnt(3)
        v_mfma_f32_32x32x8f16 a[a_c+16:a_c+31], v[v_a1+0:v_a1+1], v[v_b1+2:v_b1+3], a[a_c+16:a_c+31]     
        ;Pre-read NextRound
            ds_read_b64 v[v_b0+2:v_b0+3], v[v_smem_load_b], offset:1*0x200+(.cnt+1)*0x800
            ds_read_b64 v[v_a0+2:v_a0+3], v[v_smem_load_a], offset:1*0x200+(.cnt+1)*0x800
        ;Issue2,(1,0)
        s_waitcnt lgkmcnt(4)
        v_mfma_f32_32x32x8f16 a[a_c+32:a_c+47], v[v_a1+2:v_a1+3], v[v_b1+0:v_b1+1], a[a_c+32:a_c+47]
        ;Issue3,(1,1)
        v_mfma_f32_32x32x8f16 a[a_c+48:a_c+63], v[v_a1+2:v_a1+3], v[v_b1+2:v_b1+3], a[a_c+48:a_c+63]


        .cnt =2
        ;Round2
        ;Pre-read NextRound.
            ds_read_b64 v[v_a1+0:v_a1+1], v[v_smem_load_a], offset:0*0x200+(.cnt+1)*0x800
            ds_read_b64 v[v_b1+0:v_b1+1], v[v_smem_load_b], offset:0*0x200+(.cnt+1)*0x800
        ;Issue0,(0,0)
        s_waitcnt lgkmcnt(4)
        v_mfma_f32_32x32x8f16 a[a_c+ 0:a_c+15], v[v_a0+0:v_a0+1], v[v_b0+0:v_b0+1], a[a_c+ 0:a_c+15]
        ;Pre-read NextRound.
            ds_read_b64 v[v_b1+2:v_b1+3], v[v_smem_load_b], offset:1*0x200+(.cnt+1)*0x800
            ds_read_b64 v[v_a1+2:v_a1+3], v[v_smem_load_a], offset:1*0x200+(.cnt+1)*0x800
        ;Issue1,(0,1)
        s_waitcnt lgkmcnt(5)
        v_mfma_f32_32x32x8f16 a[a_c+16:a_c+31], v[v_a0+0:v_a0+1], v[v_b0+2:v_b0+3], a[a_c+16:a_c+31]     
        ;Issue2,(1,0)
        s_waitcnt lgkmcnt(4)
        v_mfma_f32_32x32x8f16 a[a_c+32:a_c+47], v[v_a0+2:v_a0+3], v[v_b0+0:v_b0+1], a[a_c+32:a_c+47]
        ;Issue3,(1,1)
        v_mfma_f32_32x32x8f16 a[a_c+48:a_c+63], v[v_a0+2:v_a0+3], v[v_b0+2:v_b0+3], a[a_c+48:a_c+63]

        .cnt =3
        ;Round3
        ;Issue0,(0,0)
        s_waitcnt lgkmcnt(2)
        v_mfma_f32_32x32x8f16 a[a_c+ 0:a_c+15], v[v_a1+0:v_a1+1], v[v_b1+0:v_b1+1], a[a_c+ 0:a_c+15]
        ;Issue1,(0,1)
        s_waitcnt lgkmcnt(1)
        v_mfma_f32_32x32x8f16 a[a_c+16:a_c+31], v[v_a1+0:v_a1+1], v[v_b1+2:v_b1+3], a[a_c+16:a_c+31]     
        ;Issue2,(1,0)
        s_waitcnt lgkmcnt(0)
        v_mfma_f32_32x32x8f16 a[a_c+32:a_c+47], v[v_a1+2:v_a1+3], v[v_b1+0:v_b1+1], a[a_c+32:a_c+47]
        ;Issue3,(1,1)
        v_mfma_f32_32x32x8f16 a[a_c+48:a_c+63], v[v_a1+2:v_a1+3], v[v_b1+2:v_b1+3], a[a_c+48:a_c+63]
    s_waitcnt lgkmcnt(0)
    s_barrier

    ;write to global c
    .cnt=0
    .rept 2
        .set .cid, 0
        .set .cof, (.cnt<<6)
        
        .in_cnt=0
        .rept 8
            v_accvgpr_read_b32 v[(.in_cnt)*2       ], a[(.in_cnt)*2  +(.cnt)*16     ]
            v_accvgpr_read_b32 v[(.in_cnt)*2+1     ], a[(.in_cnt)*2+1+(.cnt)*16     ]
            v_accvgpr_read_b32 v[(.in_cnt)*2   + 16], a[(.in_cnt)*2  +(.cnt)*16 + 32]
            v_accvgpr_read_b32 v[(.in_cnt)*2+1 + 16], a[(.in_cnt)*2+1+(.cnt)*16 + 32]
            v_mul_f32 v[(.in_cnt)*2       ], s[s_alpha], v[(.in_cnt)*2       ]
            v_mul_f32 v[(.in_cnt)*2+1     ], s[s_alpha], v[(.in_cnt)*2+1     ]
            v_mul_f32 v[(.in_cnt)*2   + 16], s[s_alpha], v[(.in_cnt)*2   + 16]
            v_mul_f32 v[(.in_cnt)*2+1 + 16], s[s_alpha], v[(.in_cnt)*2+1 + 16]
            v_cvt_f16_f32 v[(.in_cnt)*2       ], v[(.in_cnt)*2       ]
            v_cvt_f16_f32 v[(.in_cnt)*2+1     ], v[(.in_cnt)*2+1     ]
            v_cvt_f16_f32 v[(.in_cnt)*2   + 16], v[(.in_cnt)*2   + 16]
            v_cvt_f16_f32 v[(.in_cnt)*2+1 + 16], v[(.in_cnt)*2+1 + 16]
            V_PACK_B32_F16 v[.in_cnt     ], v[(.in_cnt)*2     ], v[(.in_cnt)*2+1   ]
            V_PACK_B32_F16 v[.in_cnt + 16], v[(.in_cnt)*2 + 16], v[(.in_cnt)*2+1+16]
            .in_cnt = .in_cnt + 1
        .endr
        
        ds_write_b64 v[v_smem_store_c], v[v_c+.cid+0 :v_c+.cid+1 ], offset:0x0
        ds_write_b64 v[v_smem_store_c], v[v_c+.cid+2 :v_c+.cid+3 ], offset:0x10
        ds_write_b64 v[v_smem_store_c], v[v_c+.cid+4 :v_c+.cid+5 ], offset:0x20
        ds_write_b64 v[v_smem_store_c], v[v_c+.cid+6 :v_c+.cid+7 ], offset:0x30
        ds_write_b64 v[v_smem_store_c], v[v_c+.cid+16:v_c+.cid+17], offset:0x0 +0x80
        ds_write_b64 v[v_smem_store_c], v[v_c+.cid+18:v_c+.cid+19], offset:0x10+0x80
        ds_write_b64 v[v_smem_store_c], v[v_c+.cid+20:v_c+.cid+21], offset:0x20+0x80
        ds_write_b64 v[v_smem_store_c], v[v_c+.cid+22:v_c+.cid+23], offset:0x30+0x80
        
        s_waitcnt lgkmcnt(0)
        s_barrier
        
        ;load shuffled data from lds, put into vgpr
        ds_read_b128 v[v_c+.cid+0 : v_c+.cid+3 ], v[v_smem_load_c], offset:0x0
        ds_read_b128 v[v_c+.cid+4 : v_c+.cid+7 ], v[v_smem_load_c], offset:0x1000
        ds_read_b128 v[v_c+.cid+16: v_c+.cid+19], v[v_smem_load_c], offset:0x2000
        ds_read_b128 v[v_c+.cid+20: v_c+.cid+23], v[v_smem_load_c], offset:0x3000
        s_waitcnt lgkmcnt(0)
        s_barrier
        s_mul_i32 s[s_tmp], .cof, s[s_ldc]
        v_add_co_u32 v[v_tmp], vcc, s[s_tmp], v[v_offset_c]
        v_addc_co_u32  v[v_tmp+1], vcc, 0, v[v_offset_c+1], vcc
        s_waitcnt lgkmcnt(3)
        global_store_dwordx4 v[v_tmp], v[v_c+.cid+0 : v_c+.cid+3 ], s[s_ptr_c:s_ptr_c+1]
        s_mul_i32 s[s_tmp], .cof+16, s[s_ldc]
        v_add_co_u32 v[v_tmp], vcc, s[s_tmp], v[v_offset_c]
        v_addc_co_u32  v[v_tmp+1], vcc, 0, v[v_offset_c+1], vcc
        s_waitcnt lgkmcnt(2)
        global_store_dwordx4 v[v_tmp], v[v_c+.cid+4 : v_c+.cid+7 ], s[s_ptr_c:s_ptr_c+1]
        s_mul_i32 s[s_tmp], .cof+32, s[s_ldc]
        v_add_co_u32 v[v_tmp], vcc, s[s_tmp], v[v_offset_c]
        v_addc_co_u32  v[v_tmp+1], vcc, 0, v[v_offset_c+1], vcc
        s_waitcnt lgkmcnt(1)
        global_store_dwordx4 v[v_tmp], v[v_c+.cid+16: v_c+.cid+19], s[s_ptr_c:s_ptr_c+1]
        s_mul_i32 s[s_tmp], .cof+48, s[s_ldc]
        v_add_co_u32 v[v_tmp], vcc, s[s_tmp], v[v_offset_c]
        v_addc_co_u32  v[v_tmp+1], vcc, 0, v[v_offset_c+1], vcc
        s_waitcnt lgkmcnt(0)
        global_store_dwordx4 v[v_tmp], v[v_c+.cid+20: v_c+.cid+23], s[s_ptr_c:s_ptr_c+1]
        
        .if .cnt != 1
            s_waitcnt vmcnt(0)
            s_barrier
        .endif
        .cnt = .cnt + 1
    .endr

    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel hgemm_128x128_kpack4
    .amdhsa_group_segment_fixed_size 16384
    .amdhsa_user_sgpr_dispatch_ptr 0
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 64
    .amdhsa_next_free_sgpr 40
    .amdhsa_ieee_mode 0
    .amdhsa_dx10_clamp 0
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [ 1, 0 ]
amdhsa.kernels:
  - .name: hgemm_128x128_kpack4
    .symbol: hgemm_128x128_kpack4.kd
    .sgpr_count: 40
    .vgpr_count: 64
    .kernarg_segment_align: 8
    .kernarg_segment_size: 64
    .group_segment_fixed_size: 16384
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