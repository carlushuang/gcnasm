.hsa_code_object_version 2,0
.hsa_code_object_isa 9, 0, 6, "AMD", "AMDGPU"

.text
.p2align 8
.amdgpu_hsa_kernel kernel_func

.set k_bdx,     256     ; should be 256 in bdx
.set k_end,     12

.set v_end,     255     ; hard code to this to let occupancy to be 1.  65536 / 256 = 256

.set s_blocks,  12
.set s_end,     31

.set inst_loop, 256

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

        kernarg_segment_byte_size           = k_end
        workgroup_group_segment_byte_size   = 0
    .end_amd_kernel_code_t

    s_load_dword        s[s_blocks], s[0:1], 8
    s_waitcnt           lgkmcnt(0)

L_kernel_start:
    s_sub_u32 s[s_blocks], s[s_blocks], 1
    .itr = 0
    .rept inst_loop
        ;v_fmac_f32 v[.itr], v[.itr+1], v[.itr+2]
        v_mac_f32 v[.itr], v[.itr+1], v[.itr+2]
        ;v_mac_f16 v[.itr], v[.itr+1], v[.itr+2]

        ;v_dot2_f32_f16 v[.itr], v[.itr+1], v[.itr+2], v[.itr+3]
        ;v_dot4_i32_i8 v[.itr], v[.itr+1], v[.itr+2], v[.itr+3]
        ;v_pk_fma_f16 v[.itr], v[.itr+1], v[.itr+2], v[.itr+3]
        .itr = .itr+4
        .if .itr > (v_end-4+1)
            .itr = 0
        .endif
    .endr

    s_cmp_gt_u32 s[s_blocks], 0
    s_cbranch_scc1 L_kernel_start

    s_endpgm
