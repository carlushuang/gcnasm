.hsa_code_object_version 2,0
.hsa_code_object_isa 9, 0, 0, "AMD", "AMDGPU"

.text
.p2align 8
.amdgpu_hsa_kernel kernel_func

; this kernel can compete DMA copy used in rocm_bandwidth_test

.set v_tid,     0
.set v_os,      1
.set v_os_ptr,  2
.set v_x,       3
.set v_y,       4
.set v_tmp,     5
.set v_val,     7
.set v_end,     63

.set s_dptr,    0
.set s_arg,     2
.set s_bx,      4
.set s_by,      5
.set s_in,      6
.set s_out,     8
.set s_gdx,     10
.set s_tmp,     12
.set s_end,     20

.set p_dwords_per_unit, DWORD_PER_UNIT
.set p_bdx,     BLOCK_DIM_X
.set p_gdx,     GRID_DIM_X
.set p_gdy,     GRID_DIM_Y
.set p_unit_per_t, UNIT_PER_THRD
.set p_unit_stride, UNIT_STRIDE
.set p_unit_stride_shift, UNIT_STRIDE_SHIFT
.set p_loop,    P_LOOP

.if v_val+p_unit_per_t*p_dwords_per_unit-1 > v_end
    .error "register requested more than allocated. check v_end not enough"
.endif

kernel_func:
    .amd_kernel_code_t
        enable_sgpr_dispatch_ptr            = 1
        enable_sgpr_kernarg_segment_ptr     = 1
        user_sgpr_count                     = 4
        enable_sgpr_workgroup_id_x          = 1
        enable_sgpr_workgroup_id_y          = 1

        enable_vgpr_workitem_id             = 0

        is_ptr64                            = 1
        float_mode                          = 2

        wavefront_sgpr_count                = s_end+1+2*3       ; VCC, FLAT_SCRATCH and XNACK must be counted
        workitem_vgpr_count                 = v_end+1
        granulated_workitem_vgpr_count      = v_end/4           ; (workitem_vgpr_count-1)/4
        granulated_wavefront_sgpr_count     = (s_end+2*3)/8     ; (wavefront_sgpr_count-1)/8

        kernarg_segment_byte_size           = 16
        workgroup_group_segment_byte_size   = 0
    .end_amd_kernel_code_t

    s_load_dwordx2 s[s_in:s_in+1], s[s_arg:s_arg+1], 0
    s_load_dwordx2 s[s_out:s_out+1], s[s_arg:s_arg+1], 8

    ; performance is heavily affected by unit_stride!
    ;  x=tid&(unit_stride-1)
    ;  y=tid>>unit_stride_shift
    ;
    ;  by*gdx*bdx*unit_per_t*dpu + bx*bdx*unit_per_t*dpu + y*unit_stride*unit_per_t*dpu+x*dpu
    ; CAUSION! careful for the mad_u32_u24, carefully check not beyond 24bit
    v_and_b32 v[v_x], p_unit_stride-1, v[v_tid]
    v_lshrrev_b32 v[v_y], p_unit_stride_shift, v[v_tid]

    v_mul_u32_u24 v[v_x], p_dwords_per_unit, v[v_x]
    v_mov_b32 v[v_tmp], p_unit_stride*p_unit_per_t*p_dwords_per_unit
    v_mad_u32_u24 v[v_os], v[v_y], v[v_tmp], v[v_x]

    v_mov_b32 v[v_tmp], p_bdx*p_unit_per_t*p_dwords_per_unit
    v_mad_u32_u24 v[v_os], s[s_bx], v[v_tmp], v[v_os]
    v_mov_b32 v[v_tmp+1], p_gdx*p_bdx*p_unit_per_t*p_dwords_per_unit
    v_mad_u32_u24 v[v_os], s[s_by], v[v_tmp+1], v[v_os]

    v_lshlrev_b32 v[v_os], 2, v[v_os]

    s_waitcnt lgkmcnt(0)

    ; ---
    .rept p_loop

    .cnt=0
    v_mov_b32 v[v_os_ptr], v[v_os]
    .rept p_unit_per_t
        .if p_dwords_per_unit == 1
            ;global_load_dword v[v_val+.cnt], v[v_tmp+.cnt:v_tmp+.cnt+1], s[s_in:s_in+1]
            global_load_dword v[v_val+.cnt], v[v_os_ptr:v_os_ptr+1], s[s_in:s_in+1]
        .elseif p_dwords_per_unit == 2
            global_load_dwordx2 v[v_val+2*.cnt:v_val+2*.cnt+1],v[v_os_ptr:v_os_ptr+1], s[s_in:s_in+1]
        .elseif p_dwords_per_unit == 4
            global_load_dwordx4 v[v_val+4*.cnt:v_val+4*.cnt+3],v[v_os_ptr:v_os_ptr+1], s[s_in:s_in+1]
        .endif
        v_add_u32 v[v_os_ptr], p_unit_stride*p_dwords_per_unit*4, v[v_os_ptr]
        .cnt = .cnt+1
    .endr
    s_waitcnt vmcnt(0)

    .endr

    ; ---
    .rept p_loop

    .cnt=0
    v_mov_b32 v[v_os_ptr], v[v_os]
    .rept p_unit_per_t
        .if p_dwords_per_unit == 1
            ;global_store_dword v[v_tmp+.cnt:v_tmp+.cnt+1], v[v_val+.cnt], s[s_out:s_out+1]
            global_store_dword v[v_os_ptr:v_os_ptr+1], v[v_val+.cnt], s[s_out:s_out+1]
        .elseif p_dwords_per_unit == 2
            global_store_dwordx2 v[v_os_ptr:v_os_ptr+1], v[v_val+2*.cnt:v_val+2*.cnt+1], s[s_out:s_out+1]
        .elseif p_dwords_per_unit == 4
            global_store_dwordx4 v[v_os_ptr:v_os_ptr+1], v[v_val+4*.cnt:v_val+4*.cnt+3], s[s_out:s_out+1]
        .endif
        v_add_u32 v[v_os_ptr], p_unit_stride*p_dwords_per_unit*4, v[v_os_ptr]
        .cnt = .cnt+1
    .endr
    s_waitcnt vmcnt(0)

    .endr

    s_endpgm
