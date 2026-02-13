; ====================================================================
; Vector Add Assembly Kernel for gfx942 (MI300) -- PERSISTENT style
; Using buffer_load_dword ... offen lds (async global -> LDS via buffer)
;
; C[i] = A[i] + B[i]
;
; Persistent kernel: host launches gridDim = num_CUs, each workgroup
; loops over its share of the input via a grid-stride loop.
;   stride = num_CUs * BLOCK_SIZE (passed as kernel arg)
;   idx starts at global_id, increments by stride each iteration.
;
; Kernel arguments (32 bytes, 8-byte aligned):
;   offset  0: float* A       (8 bytes)
;   offset  8: float* B       (8 bytes)
;   offset 16: float* C       (8 bytes)
;   offset 24: uint32 N       (4 bytes)  -- number of elements
;   offset 28: uint32 stride  (4 bytes)  -- num_CUs * BLOCK_SIZE
;
; LDS layout (2048 bytes total, 256 threads * 4 bytes each):
;   [   0, 1024): A values  (one float per thread)
;   [1024, 2048): B values  (one float per thread)
;
; Buffer SRD (128-bit resource descriptor):
;   Word 0: base_address[31:0]
;   Word 1: base_address[47:32]  (stride=0, no swizzle)
;   Word 2: num_records = 0xFFFFFFFF
;   Word 3: 0x00020000  (DATA_FORMAT=32, gfx942 config)
;
; SGPR allocation:
;   s[0:1]   = kernarg_segment_ptr  (user SGPR)
;   s2       = workgroup_id_x       (system SGPR)
;   s[4:5]   = ptr_a
;   s[6:7]   = ptr_b
;   s[8:9]   = ptr_c
;   s10      = N
;   s11      = stride (elements)
;   s12      = m0 value for A region
;   s13      = m0 value for B region
;   s[14:15] = saved initial exec mask
;   s[16:19] = buffer SRD for A
;   s[20:23] = buffer SRD for B
;
; VGPR allocation:
;   v0 = threadIdx.x  (workitem_id, preserved)
;   v1 = current element index (idx), grid-stride loop variable
;   v2 = byte offset for current idx (idx * 4)
;   v3 = LDS byte address (threadIdx.x * 4, constant across iterations)
;   v4 = val_a / result
;   v5 = val_b
; ====================================================================

.text
.global vector_add_kernel
.p2align 8
.type vector_add_kernel,@function

; --- SGPR names ---
.set s_karg,        0       ; s[0:1] = kernarg segment pointer
.set s_bx,          2       ; s2     = workgroup_id_x
.set s_ptr_a,       4       ; s[4:5] = pointer to A
.set s_ptr_b,       6       ; s[6:7] = pointer to B
.set s_ptr_c,       8       ; s[8:9] = pointer to C
.set s_n,           10      ; s10    = number of elements
.set s_stride,      11      ; s11    = stride (elements) = num_CUs * 256
.set s_m0_a,        12      ; s12    = M0 value for A's LDS region
.set s_m0_b,        13      ; s13    = M0 value for B's LDS region
.set s_init_exec,   14      ; s[14:15] = saved initial exec mask
.set s_res_a,       16      ; s[16:19] = buffer SRD for A
.set s_res_b,       20      ; s[20:23] = buffer SRD for B

; --- VGPR names ---
.set v_tid,         0       ; v0 = threadIdx.x (preserved)
.set v_idx,         1       ; v1 = current element index (loop variable)
.set v_buf_off,     2       ; v2 = byte offset (idx * 4)
.set v_lds_addr,    3       ; v3 = LDS byte address (threadIdx.x * 4)
.set v_a,           4       ; v4 = value from A / result
.set v_b,           5       ; v5 = value from B

; --- LDS layout ---
.set LDS_SIZE,      2048
.set LDS_OFFSET_B,  1024

; --- Buffer SRD config ---
.set SRD_NUM_RECORDS,   0xFFFFFFFF
.set SRD_CONFIG_GFX942, 0x00020000

; ====================================================================
; Macro: emit buffer_load_dword ... offen lds  (manual MUBUF encoding)
; ====================================================================
.macro buffer_load_dword_offen_lds vdata, vaddr, srsrc_base
    .long 0xE0511000
    .long (0x80 << 24) | ((\srsrc_base / 4) << 16) | (\vdata << 8) | \vaddr
.endm

; ====================================================================
; Kernel code
; ====================================================================
vector_add_kernel:
    ; -----------------------------------------------------------
    ; Step 1: Load kernel arguments
    ; -----------------------------------------------------------
    s_load_dwordx2 s[s_ptr_a:s_ptr_a+1], s[s_karg:s_karg+1], 0
    s_load_dwordx2 s[s_ptr_b:s_ptr_b+1], s[s_karg:s_karg+1], 8
    s_load_dwordx2 s[s_ptr_c:s_ptr_c+1], s[s_karg:s_karg+1], 16
    s_load_dword   s[s_n],               s[s_karg:s_karg+1], 24
    s_load_dword   s[s_stride],          s[s_karg:s_karg+1], 28

    ; -----------------------------------------------------------
    ; Step 2: Compute initial idx = workgroup_id * 256 + threadIdx.x
    ; -----------------------------------------------------------
    s_lshl_b32 s[s_m0_a], s[s_bx], 8               ; tmp = workgroup_id * 256
    v_add_u32 v[v_idx], s[s_m0_a], v[v_tid]         ; v_idx = global thread ID (initial)

    ; -----------------------------------------------------------
    ; Step 3: Pre-compute loop-invariant values
    ; -----------------------------------------------------------
    ; LDS byte address = threadIdx.x * 4  (constant across iterations)
    v_lshlrev_b32 v[v_lds_addr], 2, v[v_tid]

    ; M0 values for LDS regions (constant across iterations)
    ;   M0_A = first_lane_threadIdx * 4  (wave's LDS base for A)
    ;   M0_B = M0_A + 1024              (wave's LDS base for B)
    v_readfirstlane_b32 s[s_m0_a], v[v_tid]
    s_lshl_b32 s[s_m0_a], s[s_m0_a], 2
    s_add_u32  s[s_m0_b], s[s_m0_a], LDS_OFFSET_B

    ; Wait for kernel arguments
    s_waitcnt lgkmcnt(0)

    ; -----------------------------------------------------------
    ; Step 4: Build buffer resource descriptors (once, outside loop)
    ; -----------------------------------------------------------
    s_mov_b32 s[s_res_a+0], s[s_ptr_a]
    s_and_b32 s[s_res_a+1], s[s_ptr_a+1], 0xFFFF
    s_mov_b32 s[s_res_a+2], SRD_NUM_RECORDS
    s_mov_b32 s[s_res_a+3], SRD_CONFIG_GFX942

    s_mov_b32 s[s_res_b+0], s[s_ptr_b]
    s_and_b32 s[s_res_b+1], s[s_ptr_b+1], 0xFFFF
    s_mov_b32 s[s_res_b+2], SRD_NUM_RECORDS
    s_mov_b32 s[s_res_b+3], SRD_CONFIG_GFX942

    ; Save initial exec mask (all lanes active at kernel entry)
    s_mov_b64 s[s_init_exec:s_init_exec+1], exec

    ; ===========================================================
    ; Grid-stride loop:  for (idx = global_id; idx < N; idx += stride)
    ; ===========================================================
L_loop:
    ; --- Bounds check: mask out lanes where idx >= N ---
    v_cmp_gt_u32 vcc, s[s_n], v[v_idx]
    s_and_b64 exec, exec, vcc
    s_cbranch_execz L_done

    ; --- Compute byte offset = idx * 4 ---
    v_lshlrev_b32 v[v_buf_off], 2, v[v_idx]

    ; --- Async load A[idx] -> LDS ---
    s_mov_b32 m0, s[s_m0_a]
    buffer_load_dword_offen_lds v_a, v_buf_off, s_res_a

    ; --- Async load B[idx] -> LDS ---
    s_mov_b32 m0, s[s_m0_b]
    buffer_load_dword_offen_lds v_b, v_buf_off, s_res_b

    ; --- Wait for global -> LDS transfers ---
    s_waitcnt vmcnt(0)

    ; --- Read A and B from LDS into VGPRs ---
    ds_read_b32 v[v_a], v[v_lds_addr]
    ds_read_b32 v[v_b], v[v_lds_addr] offset:LDS_OFFSET_B
    s_waitcnt lgkmcnt(0)

    ; --- Compute C[idx] = A[idx] + B[idx] ---
    v_add_f32 v[v_a], v[v_a], v[v_b]

    ; --- Store result ---
    global_store_dword v[v_buf_off], v[v_a], s[s_ptr_c:s_ptr_c+1]

    ; --- Advance idx by stride (restore exec so ALL lanes advance) ---
    s_mov_b64 exec, s[s_init_exec:s_init_exec+1]
    v_add_u32 v[v_idx], s[s_stride], v[v_idx]

    s_branch L_loop

L_done:
    ; Ensure last iteration's store completes
    s_mov_b64 exec, s[s_init_exec:s_init_exec+1]
    s_waitcnt vmcnt(0)
    s_endpgm

; ====================================================================
; Kernel descriptor
; ====================================================================
.rodata
.p2align 6
.amdhsa_kernel vector_add_kernel
    .amdhsa_group_segment_fixed_size 2048
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 8
    .amdhsa_next_free_sgpr 24
    .amdhsa_accum_offset 8
    .amdhsa_ieee_mode 0
    .amdhsa_dx10_clamp 0
.end_amdhsa_kernel

; ====================================================================
; AMDGPU metadata (code object v5)
; ====================================================================
.amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: vector_add_kernel
    .symbol: vector_add_kernel.kd
    .kernarg_segment_size: 32
    .group_segment_fixed_size: 2048
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 24
    .vgpr_count: 8
    .max_flat_workgroup_size: 256
    .args:
    - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
    - { .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global }
    - { .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global }
    - { .size: 4, .offset: 24, .value_kind: by_value }
    - { .size: 4, .offset: 28, .value_kind: by_value }
...
.end_amdgpu_metadata
