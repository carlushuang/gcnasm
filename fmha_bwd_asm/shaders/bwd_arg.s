.text
.global kernel_func
.p2align 8
.type kernel_func,@function

kernel_func:

  s_and_b32     s1, s1, 0x0000ffff                      // 000000000000: 8601FF01 0000FFFF
  s_load_dwordx2  s[32:33], s[0:1], 0x00                // 000000000008: C0060800 00000000
  s_load_dwordx2  s[36:37], s[0:1], 0x10                // 000000000010: C0060900 00000010
  s_load_dwordx2  s[40:41], s[0:1], 0x20                // 000000000018: C0060A00 00000020
  s_load_dwordx2  s[8:9], s[0:1], 0x30                  // 000000000020: C0060200 00000030
  s_load_dwordx2  s[12:13], s[0:1], 0x40                // 000000000028: C0060300 00000040
  s_load_dwordx2  s[16:17], s[0:1], 0x50                // 000000000030: C0060400 00000050
  s_load_dwordx2  s[20:21], s[0:1], 0x60                // 000000000038: C0060500 00000060
  s_load_dwordx2  s[24:25], s[0:1], 0x70                // 000000000040: C0060600 00000070
  s_load_dwordx2  s[28:29], s[0:1], 0x80                // 000000000048: C0060700 00000080
  s_load_dword  s48, s[0:1], 0x90                       // 000000000050: C0020C00 00000090
  s_load_dword  s49, s[0:1], 0xa0                       // 000000000058: C0020C40 000000A0
  s_load_dword  s50, s[0:1], 0xb0                       // 000000000060: C0020C80 000000B0
  s_load_dword  s51, s[0:1], 0xc0                       // 000000000068: C0020CC0 000000C0
  s_load_dword  s52, s[0:1], 0xd0                       // 000000000070: C0020D00 000000D0
  s_load_dword  s53, s[0:1], 0xe0                       // 000000000078: C0020D40 000000E0
  v_lshrrev_b32  v1, 10, v0                             // 000000000080: 2002008A
  v_lshrrev_b32  v2, 10, v1                             // 000000000084: 2004028A
  v_and_b32     v2, 0x000003ff, v2                      // 000000000088: 260404FF 000003FF
  v_and_b32     v1, 0x000003ff, v1                      // 000000000090: 260202FF 000003FF
  v_and_b32     v0, 0x000003ff, v0                      // 000000000098: 260000FF 000003FF
  v_lshrrev_b32  v3, 6, v0                              // 0000000000A0: 20060086
  v_and_b32     v0, 63, v0                              // 0000000000A4: 260000BF
  s_mov_b32     s44, s2                                 // 0000000000A8: BEAC0002
  s_mov_b32     s45, s3                                 // 0000000000AC: BEAD0003
  s_mov_b32     s46, s4                                 // 0000000000B0: BEAE0004
  v_readfirstlane_b32  s47, v3                          // 0000000000B4: 7E5E0503
  s_waitcnt     lgkmcnt(0)                              // 0000000000B8: BF8CC07F
  s_mov_b32     s10, 0x80000000                         // 0000000000BC: BE8A00FF 80000000
  s_mov_b32     s14, 0x80000000                         // 0000000000C4: BE8E00FF 80000000
  s_mov_b32     s18, 0x80000000                         // 0000000000CC: BE9200FF 80000000
  s_mov_b32     s22, 0x80000000                         // 0000000000D4: BE9600FF 80000000
  s_mov_b32     s26, 0x80000000                         // 0000000000DC: BE9A00FF 80000000
  s_mov_b32     s30, 0x80000000                         // 0000000000E4: BE9E00FF 80000000
  s_mov_b32     s34, 0x80000000                         // 0000000000EC: BEA200FF 80000000
  s_mov_b32     s38, 0x80000000                         // 0000000000F4: BEA600FF 80000000
  s_mov_b32     s42, 0x80000000                         // 0000000000FC: BEAA00FF 80000000
  s_mov_b32     s11, 0x00020000                         // 000000000104: BE8B00FF 00020000
  s_mov_b32     s15, 0x00020000                         // 00000000010C: BE8F00FF 00020000
  s_mov_b32     s19, 0x00020000                         // 000000000114: BE9300FF 00020000
  s_mov_b32     s23, 0x00020000                         // 00000000011C: BE9700FF 00020000
  s_mov_b32     s27, 0x00020000                         // 000000000124: BE9B00FF 00020000
  s_mov_b32     s31, 0x00020000                         // 00000000012C: BE9F00FF 00020000
  s_mov_b32     s35, 0x00020000                         // 000000000134: BEA300FF 00020000
  s_mov_b32     s39, 0x00020000                         // 00000000013C: BEA700FF 00020000
  s_mov_b32     s43, 0x00020000                         // 000000000144: BEAB00FF 00020000
  s_and_b32     s9, s9, 0x0000ffff                      // 00000000014C: 8609FF09 0000FFFF
  s_and_b32     s13, s13, 0x0000ffff                    // 000000000154: 860DFF0D 0000FFFF
  s_and_b32     s17, s17, 0x0000ffff                    // 00000000015C: 8611FF11 0000FFFF
  s_and_b32     s21, s21, 0x0000ffff                    // 000000000164: 8615FF15 0000FFFF
  s_and_b32     s25, s25, 0x0000ffff                    // 00000000016C: 8619FF19 0000FFFF
  s_and_b32     s29, s29, 0x0000ffff                    // 000000000174: 861DFF1D 0000FFFF
  s_and_b32     s33, s33, 0x0000ffff                    // 00000000017C: 8621FF21 0000FFFF
  s_and_b32     s37, s37, 0x0000ffff                    // 000000000184: 8625FF25 0000FFFF
  s_and_b32     s41, s41, 0x0000ffff                    // 00000000018C: 8629FF29 0000FFFF
  s_or_b32      s9, s9, 0x00040000                      // 000000000194: 8709FF09 00040000
  s_or_b32      s13, s13, 0x00040000                    // 00000000019C: 870DFF0D 00040000
  s_or_b32      s17, s17, 0x00040000                    // 0000000001A4: 8711FF11 00040000
  s_or_b32      s21, s21, 0x00040000                    // 0000000001AC: 8715FF15 00040000
  s_or_b32      s25, s25, 0x00040000                    // 0000000001B4: 8719FF19 00040000
  s_or_b32      s29, s29, 0x00040000                    // 0000000001BC: 871DFF1D 00040000
  s_or_b32      s33, s33, 0x00040000                    // 0000000001C4: 8721FF21 00040000
  s_or_b32      s37, s37, 0x00040000                    // 0000000001CC: 8725FF25 00040000
  s_or_b32      s41, s41, 0x00040000                    // 0000000001D4: 8729FF29 00040000
  v_accvgpr_write  acc255, 0                            // 0000000001DC: D3D940FF 18000080
  v_mov_b32     v243, 0                                 // 0000000001E4: 7FE60280
  s_cmp_lt_i32  s47, 1                                  // 0000000001E8: BF04812F
  s_cbranch_scc0  label_00DD                            // 0000000001EC: BF840061
  v_mov_b32     v28, s8                                 // 0000000001F0: 7E380208
  v_mov_b32     v29, s9                                 // 0000000001F4: 7E3A0209
  v_mov_b32     v30, s12                                // 0000000001F8: 7E3C020C
  v_mov_b32     v31, s13                                // 0000000001FC: 7E3E020D
  v_mov_b32     v32, s16                                // 000000000200: 7E400210
  v_mov_b32     v33, s17                                // 000000000204: 7E420211
  v_mov_b32     v34, s20                                // 000000000208: 7E440214
  v_mov_b32     v35, s21                                // 00000000020C: 7E460215
  v_mov_b32     v36, s24                                // 000000000210: 7E480218
  v_mov_b32     v37, s25                                // 000000000214: 7E4A0219
  v_mov_b32     v38, s28                                // 000000000218: 7E4C021C
  v_mov_b32     v39, s29                                // 00000000021C: 7E4E021D
  v_mov_b32     v40, s32                                // 000000000220: 7E500220
  v_mov_b32     v41, s33                                // 000000000224: 7E520221
  v_mov_b32     v42, s36                                // 000000000228: 7E540224
  v_mov_b32     v43, s37                                // 00000000022C: 7E560225
  v_mov_b32     v44, s40                                // 000000000230: 7E580228
  v_mov_b32     v45, s41                                // 000000000234: 7E5A0229
  v_mov_b32     v46, s48                                // 000000000238: 7E5C0230
  v_mov_b32     v47, s49                                // 00000000023C: 7E5E0231
  v_mov_b32     v48, s50                                // 000000000240: 7E600232
  v_mov_b32     v49, s51                                // 000000000244: 7E620233
  v_mov_b32     v50, s52                                // 000000000248: 7E640234
  v_mov_b32     v51, s53                                // 00000000024C: 7E660235
  v_mov_b32     v243, v0                                // 000000000250: 7FE60300
  buffer_store_dword  v28, v243, s[32:35], 0 idxen      // 000000000254: E0702000 80081CF3
  v_add_u32     v243, 64, v243                          // 00000000025C: 69E7E6C0
  buffer_store_dword  v29, v243, s[32:35], 0 idxen      // 000000000260: E0702000 80081DF3
  v_add_u32     v243, 64, v243                          // 000000000268: 69E7E6C0
  buffer_store_dword  v30, v243, s[32:35], 0 idxen      // 00000000026C: E0702000 80081EF3
  v_add_u32     v243, 64, v243                          // 000000000274: 69E7E6C0
  buffer_store_dword  v31, v243, s[32:35], 0 idxen      // 000000000278: E0702000 80081FF3
  v_add_u32     v243, 64, v243                          // 000000000280: 69E7E6C0
  buffer_store_dword  v32, v243, s[32:35], 0 idxen      // 000000000284: E0702000 800820F3
  v_add_u32     v243, 64, v243                          // 00000000028C: 69E7E6C0
  buffer_store_dword  v33, v243, s[32:35], 0 idxen      // 000000000290: E0702000 800821F3
  v_add_u32     v243, 64, v243                          // 000000000298: 69E7E6C0
  buffer_store_dword  v34, v243, s[32:35], 0 idxen      // 00000000029C: E0702000 800822F3
  v_add_u32     v243, 64, v243                          // 0000000002A4: 69E7E6C0
  buffer_store_dword  v35, v243, s[32:35], 0 idxen      // 0000000002A8: E0702000 800823F3
  v_add_u32     v243, 64, v243                          // 0000000002B0: 69E7E6C0
  buffer_store_dword  v36, v243, s[32:35], 0 idxen      // 0000000002B4: E0702000 800824F3
  v_add_u32     v243, 64, v243                          // 0000000002BC: 69E7E6C0
  buffer_store_dword  v37, v243, s[32:35], 0 idxen      // 0000000002C0: E0702000 800825F3
  v_add_u32     v243, 64, v243                          // 0000000002C8: 69E7E6C0
  buffer_store_dword  v38, v243, s[32:35], 0 idxen      // 0000000002CC: E0702000 800826F3
  v_add_u32     v243, 64, v243                          // 0000000002D4: 69E7E6C0
  buffer_store_dword  v39, v243, s[32:35], 0 idxen      // 0000000002D8: E0702000 800827F3
  v_add_u32     v243, 64, v243                          // 0000000002E0: 69E7E6C0
  buffer_store_dword  v40, v243, s[32:35], 0 idxen      // 0000000002E4: E0702000 800828F3
  v_add_u32     v243, 64, v243                          // 0000000002EC: 69E7E6C0
  buffer_store_dword  v41, v243, s[32:35], 0 idxen      // 0000000002F0: E0702000 800829F3
  v_add_u32     v243, 64, v243                          // 0000000002F8: 69E7E6C0
  buffer_store_dword  v42, v243, s[32:35], 0 idxen      // 0000000002FC: E0702000 80082AF3
  v_add_u32     v243, 64, v243                          // 000000000304: 69E7E6C0
  buffer_store_dword  v43, v243, s[32:35], 0 idxen      // 000000000308: E0702000 80082BF3
  v_add_u32     v243, 64, v243                          // 000000000310: 69E7E6C0
  buffer_store_dword  v44, v243, s[32:35], 0 idxen      // 000000000314: E0702000 80082CF3
  v_add_u32     v243, 64, v243                          // 00000000031C: 69E7E6C0
  buffer_store_dword  v45, v243, s[32:35], 0 idxen      // 000000000320: E0702000 80082DF3
  v_add_u32     v243, 64, v243                          // 000000000328: 69E7E6C0
  buffer_store_dword  v46, v243, s[32:35], 0 idxen      // 00000000032C: E0702000 80082EF3
  v_add_u32     v243, 64, v243                          // 000000000334: 69E7E6C0
  buffer_store_dword  v47, v243, s[32:35], 0 idxen      // 000000000338: E0702000 80082FF3
  v_add_u32     v243, 64, v243                          // 000000000340: 69E7E6C0
  buffer_store_dword  v48, v243, s[32:35], 0 idxen      // 000000000344: E0702000 800830F3
  v_add_u32     v243, 64, v243                          // 00000000034C: 69E7E6C0
  buffer_store_dword  v49, v243, s[32:35], 0 idxen      // 000000000350: E0702000 800831F3
  v_add_u32     v243, 64, v243                          // 000000000358: 69E7E6C0
  buffer_store_dword  v50, v243, s[32:35], 0 idxen      // 00000000035C: E0702000 800832F3
  v_add_u32     v243, 64, v243                          // 000000000364: 69E7E6C0
  buffer_store_dword  v51, v243, s[32:35], 0 idxen      // 000000000368: E0702000 800833F3
  v_add_u32     v243, 64, v243                          // 000000000370: 69E7E6C0
label_00DD:
  s_waitcnt     0x0000                                  // 000000000374: BF8C0000
  s_endpgm                                              // 000000000378: BF810000

.rodata
.p2align 6
.amdhsa_kernel kernel_func
    .amdhsa_group_segment_fixed_size 65536
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 1
    .amdhsa_system_sgpr_workgroup_id_z 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 512 
    .amdhsa_next_free_sgpr 80
    .amdhsa_accum_offset 256
    .amdhsa_ieee_mode 0
    .amdhsa_dx10_clamp 0
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [ 1, 0 ]
amdhsa.kernels:
  - .name: kernel_func
    .symbol: kernel_func.kd
    .sgpr_count: 80
    .vgpr_count: 512
    .kernarg_segment_align: 4
    .kernarg_segment_size: 240
    .group_segment_fixed_size: 65536
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .reqd_workgroup_size : [256, 1, 1]
    .max_flat_workgroup_size: 256
    .args:
    - {.name: dQ, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global, .actual_access: read_write}
    - {.name: pad, .size: 8, .offset: 8, .value_kind: by_value, .value_type: i32}
    - {.name: dK, .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global, .actual_access: read_write}
    - {.name: pad, .size: 8, .offset: 24, .value_kind: by_value, .value_type: i32}
    - {.name: dV, .size: 8, .offset: 32, .value_kind: global_buffer, .address_space: global, .actual_access: read_write}
    - {.name: pad, .size: 8, .offset: 40, .value_kind: by_value, .value_type: i32}
    - {.name: Q, .size: 8, .offset: 48, .value_kind: global_buffer, .address_space: global, .actual_access: read_only}
    - {.name: pad, .size: 8, .offset: 56, .value_kind: by_value, .value_type: i32}
    - {.name: K, .size: 8, .offset: 64, .value_kind: global_buffer, .address_space: global, .actual_access: read_only}
    - {.name: pad, .size: 8, .offset: 72, .value_kind: by_value, .value_type: i32}
    - {.name: V, .size: 8, .offset: 80, .value_kind: global_buffer, .address_space: global, .actual_access: read_only}
    - {.name: pad, .size: 8, .offset: 88, .value_kind: by_value, .value_type: i32}
    - {.name: dO, .size: 8, .offset: 96, .value_kind: global_buffer, .address_space: global, .actual_access: read_only}
    - {.name: pad, .size: 8, .offset: 104, .value_kind: by_value, .value_type: i32}
    - {.name: Lse, .size: 8, .offset: 112, .value_kind: global_buffer, .address_space: global, .actual_access: read_only}
    - {.name: pad, .size: 8, .offset: 120, .value_kind: by_value, .value_type: i32}
    - {.name: D, .size: 8, .offset: 128, .value_kind: global_buffer, .address_space: global, .actual_access: read_only}
    - {.name: pad, .size: 8, .offset: 136, .value_kind: by_value, .value_type: i32}
    - {.name: scalar, .size: 4, .offset: 144, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 148, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 152, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 156, .value_kind: by_value, .value_type: i32}
    - {.name: log2e, .size: 4, .offset: 160, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 164, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 168, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 172, .value_kind: by_value, .value_type: i32}
    - {.name: seq_len, .size: 4, .offset: 176, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 180, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 184, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 188, .value_kind: by_value, .value_type: i32}
    - {.name: Ts, .size: 4, .offset: 192, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 196, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 200, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 204, .value_kind: by_value, .value_type: i32}
    - {.name: Hs, .size: 4, .offset: 208, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 212, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 216, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 220, .value_kind: by_value, .value_type: i32}
    - {.name: BAs, .size: 4, .offset: 224, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 228, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 232, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 236, .value_kind: by_value, .value_type: i32}
...
.end_amdgpu_metadata
