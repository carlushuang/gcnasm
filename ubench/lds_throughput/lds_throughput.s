; ====================================================================
; LDS Throughput Benchmarks (gfx942 / MI308)
;
; 6 kernels + 1 NOP baseline measuring LDS throughput (dwords/cycle).
;
; Unlike latency benchmarks (single lane, pointer chase), throughput
; benchmarks use ALL 64 lanes with many independent operations
; (no data dependency) to saturate the LDS pipeline.
;
; Bank-conflict-free addressing:
;   b32:  lane_id * 4   (each lane hits a unique bank)
;   b64:  lane_id * 8
;   b128: lane_id * 16
;
; Unroll factors: b32=32, b64=16, b128=8 (all use 32 dest VGPRs)
;
; Kernel arguments (12 bytes, 8-byte aligned):
;   offset  0: uint64_t* output       (8 bytes) - [start, end] ticks
;   offset  8: uint32_t  num_iters    (4 bytes)
; ====================================================================

; ===================== Common SGPR layout =====================
.set s_karg,        0       ; [0:1] kernarg pointer
.set s_bx,          2       ; workgroup id (unused but allocated by descriptor)
.set s_out_ptr,     4       ; [4:5] output pointer
.set s_num_iters,   6
.set s_start_lo,    8
.set s_start_hi,    9
.set s_end_lo,      10
.set s_end_hi,      11

; ===================== Common VGPR layout =====================
; v0  = lane_id (workitem_id_x)
; v1  = LDS address
; v2  = loop counter
; v3  = temp
; v4-v35 = destination/source VGPRs for unrolled operations

; ===================== Macros =====================

; Load kernargs and setup LDS address
; \shift: address shift (2 for b32, 3 for b64, 4 for b128)
.macro LOAD_KERNARGS shift
    s_load_dwordx2 s[s_out_ptr:s_out_ptr+1], s[s_karg:s_karg+1], 0
    s_load_dword   s[s_num_iters],           s[s_karg:s_karg+1], 8
    s_waitcnt lgkmcnt(0)
    v_lshlrev_b32 v1, \shift, v0    ; v1 = lane_id << shift
.endm

; Initialize LDS: each lane writes its lane_id to its address
.macro INIT_LDS
    ds_write_b32 v1, v0
    s_waitcnt lgkmcnt(0)
    s_barrier
.endm

; Start timing
.macro TIMING_START
    v_mov_b32 v2, s[s_num_iters]
    s_waitcnt vmcnt(0) lgkmcnt(0)
    s_memrealtime s[s_start_lo:s_start_hi]
    s_waitcnt lgkmcnt(0)
.endm

; End timing
.macro TIMING_END
    s_waitcnt vmcnt(0) lgkmcnt(0)
    s_memrealtime s[s_end_lo:s_end_hi]
    s_waitcnt lgkmcnt(0)
.endm

; Loop footer (VGPR-based counter to prevent SALU racing ahead)
.macro LOOP_FOOTER label
    s_waitcnt lgkmcnt(0)
    v_sub_u32 v2, v2, 1
    v_cmp_gt_u32 vcc, v2, 0
    s_cbranch_vccnz \label
.endm

; Store start/end ticks to output (lane 0 only)
; Uses v3 as temp, v36:v37 as addr, v38:v39 as data
.macro STORE_OUTPUT
    v_cmp_eq_u32 vcc, v0, 0
    s_cbranch_vccz .Lskip_store_\@
    v_mov_b32 v36, s[s_out_ptr]
    v_mov_b32 v37, s[s_out_ptr+1]
    v_mov_b32 v38, s[s_start_lo]
    v_mov_b32 v39, s[s_start_hi]
    global_store_dwordx2 v[36:37], v[38:39], off
    s_waitcnt vmcnt(0)
    v_mov_b32 v38, s[s_end_lo]
    v_mov_b32 v39, s[s_end_hi]
    v_add_co_u32 v36, vcc, 8, v36
    v_addc_co_u32 v37, vcc, 0, v37, vcc
    global_store_dwordx2 v[36:37], v[38:39], off
    s_waitcnt vmcnt(0)
.Lskip_store_\@:
.endm

; ====================================================================
; KERNEL 1: lds_tp_read_b32_kernel
; 32 independent ds_read_b32 per iteration, all 64 lanes
; ====================================================================
.text
.global lds_tp_read_b32_kernel
.p2align 8
.type lds_tp_read_b32_kernel,@function

lds_tp_read_b32_kernel:
    LOAD_KERNARGS 2     ; v1 = lane_id * 4
    INIT_LDS
    TIMING_START

.Lloop_tp_rb32:
    ds_read_b32 v4, v1
    ds_read_b32 v5, v1
    ds_read_b32 v6, v1
    ds_read_b32 v7, v1
    ds_read_b32 v8, v1
    ds_read_b32 v9, v1
    ds_read_b32 v10, v1
    ds_read_b32 v11, v1
    ds_read_b32 v12, v1
    ds_read_b32 v13, v1
    ds_read_b32 v14, v1
    ds_read_b32 v15, v1
    ds_read_b32 v16, v1
    ds_read_b32 v17, v1
    ds_read_b32 v18, v1
    ds_read_b32 v19, v1
    ds_read_b32 v20, v1
    ds_read_b32 v21, v1
    ds_read_b32 v22, v1
    ds_read_b32 v23, v1
    ds_read_b32 v24, v1
    ds_read_b32 v25, v1
    ds_read_b32 v26, v1
    ds_read_b32 v27, v1
    ds_read_b32 v28, v1
    ds_read_b32 v29, v1
    ds_read_b32 v30, v1
    ds_read_b32 v31, v1
    ds_read_b32 v32, v1
    ds_read_b32 v33, v1
    ds_read_b32 v34, v1
    ds_read_b32 v35, v1
    LOOP_FOOTER .Lloop_tp_rb32

    TIMING_END
    STORE_OUTPUT
    s_endpgm

; ====================================================================
; KERNEL 2: lds_tp_read_b64_kernel
; 16 independent ds_read_b64 per iteration, all 64 lanes
; ====================================================================
.text
.global lds_tp_read_b64_kernel
.p2align 8
.type lds_tp_read_b64_kernel,@function

lds_tp_read_b64_kernel:
    LOAD_KERNARGS 3     ; v1 = lane_id * 8
    INIT_LDS
    TIMING_START

.Lloop_tp_rb64:
    ds_read_b64 v[4:5], v1
    ds_read_b64 v[6:7], v1
    ds_read_b64 v[8:9], v1
    ds_read_b64 v[10:11], v1
    ds_read_b64 v[12:13], v1
    ds_read_b64 v[14:15], v1
    ds_read_b64 v[16:17], v1
    ds_read_b64 v[18:19], v1
    ds_read_b64 v[20:21], v1
    ds_read_b64 v[22:23], v1
    ds_read_b64 v[24:25], v1
    ds_read_b64 v[26:27], v1
    ds_read_b64 v[28:29], v1
    ds_read_b64 v[30:31], v1
    ds_read_b64 v[32:33], v1
    ds_read_b64 v[34:35], v1
    LOOP_FOOTER .Lloop_tp_rb64

    TIMING_END
    STORE_OUTPUT
    s_endpgm

; ====================================================================
; KERNEL 3: lds_tp_read_b128_kernel
; 8 independent ds_read_b128 per iteration, all 64 lanes
; ====================================================================
.text
.global lds_tp_read_b128_kernel
.p2align 8
.type lds_tp_read_b128_kernel,@function

lds_tp_read_b128_kernel:
    LOAD_KERNARGS 4     ; v1 = lane_id * 16
    INIT_LDS
    TIMING_START

.Lloop_tp_rb128:
    ds_read_b128 v[4:7], v1
    ds_read_b128 v[8:11], v1
    ds_read_b128 v[12:15], v1
    ds_read_b128 v[16:19], v1
    ds_read_b128 v[20:23], v1
    ds_read_b128 v[24:27], v1
    ds_read_b128 v[28:31], v1
    ds_read_b128 v[32:35], v1
    LOOP_FOOTER .Lloop_tp_rb128

    TIMING_END
    STORE_OUTPUT
    s_endpgm

; ====================================================================
; KERNEL 4: lds_tp_write_b32_kernel
; 32 independent ds_write_b32 per iteration, all 64 lanes
; ====================================================================
.text
.global lds_tp_write_b32_kernel
.p2align 8
.type lds_tp_write_b32_kernel,@function

lds_tp_write_b32_kernel:
    LOAD_KERNARGS 2     ; v1 = lane_id * 4
    TIMING_START

.Lloop_tp_wb32:
    ds_write_b32 v1, v0
    ds_write_b32 v1, v0
    ds_write_b32 v1, v0
    ds_write_b32 v1, v0
    ds_write_b32 v1, v0
    ds_write_b32 v1, v0
    ds_write_b32 v1, v0
    ds_write_b32 v1, v0
    ds_write_b32 v1, v0
    ds_write_b32 v1, v0
    ds_write_b32 v1, v0
    ds_write_b32 v1, v0
    ds_write_b32 v1, v0
    ds_write_b32 v1, v0
    ds_write_b32 v1, v0
    ds_write_b32 v1, v0
    ds_write_b32 v1, v0
    ds_write_b32 v1, v0
    ds_write_b32 v1, v0
    ds_write_b32 v1, v0
    ds_write_b32 v1, v0
    ds_write_b32 v1, v0
    ds_write_b32 v1, v0
    ds_write_b32 v1, v0
    ds_write_b32 v1, v0
    ds_write_b32 v1, v0
    ds_write_b32 v1, v0
    ds_write_b32 v1, v0
    ds_write_b32 v1, v0
    ds_write_b32 v1, v0
    ds_write_b32 v1, v0
    ds_write_b32 v1, v0
    LOOP_FOOTER .Lloop_tp_wb32

    TIMING_END
    STORE_OUTPUT
    s_endpgm

; ====================================================================
; KERNEL 5: lds_tp_write_b64_kernel
; 16 independent ds_write_b64 per iteration, all 64 lanes
; v4:v5 used as write data (even-aligned pair)
; ====================================================================
.text
.global lds_tp_write_b64_kernel
.p2align 8
.type lds_tp_write_b64_kernel,@function

lds_tp_write_b64_kernel:
    LOAD_KERNARGS 3     ; v1 = lane_id * 8
    v_mov_b32 v4, v0
    v_mov_b32 v5, v0
    TIMING_START

.Lloop_tp_wb64:
    ds_write_b64 v1, v[4:5]
    ds_write_b64 v1, v[4:5]
    ds_write_b64 v1, v[4:5]
    ds_write_b64 v1, v[4:5]
    ds_write_b64 v1, v[4:5]
    ds_write_b64 v1, v[4:5]
    ds_write_b64 v1, v[4:5]
    ds_write_b64 v1, v[4:5]
    ds_write_b64 v1, v[4:5]
    ds_write_b64 v1, v[4:5]
    ds_write_b64 v1, v[4:5]
    ds_write_b64 v1, v[4:5]
    ds_write_b64 v1, v[4:5]
    ds_write_b64 v1, v[4:5]
    ds_write_b64 v1, v[4:5]
    ds_write_b64 v1, v[4:5]
    LOOP_FOOTER .Lloop_tp_wb64

    TIMING_END
    STORE_OUTPUT
    s_endpgm

; ====================================================================
; KERNEL 6: lds_tp_write_b128_kernel
; 8 independent ds_write_b128 per iteration, all 64 lanes
; v4:v7 used as write data (4-aligned quad)
; ====================================================================
.text
.global lds_tp_write_b128_kernel
.p2align 8
.type lds_tp_write_b128_kernel,@function

lds_tp_write_b128_kernel:
    LOAD_KERNARGS 4     ; v1 = lane_id * 16
    v_mov_b32 v4, v0
    v_mov_b32 v5, v0
    v_mov_b32 v6, v0
    v_mov_b32 v7, v0
    TIMING_START

.Lloop_tp_wb128:
    ds_write_b128 v1, v[4:7]
    ds_write_b128 v1, v[4:7]
    ds_write_b128 v1, v[4:7]
    ds_write_b128 v1, v[4:7]
    ds_write_b128 v1, v[4:7]
    ds_write_b128 v1, v[4:7]
    ds_write_b128 v1, v[4:7]
    ds_write_b128 v1, v[4:7]
    LOOP_FOOTER .Lloop_tp_wb128

    TIMING_END
    STORE_OUTPUT
    s_endpgm

; ====================================================================
; KERNEL 7: lds_tp_nop_kernel
; 32 v_nop per iteration — baseline for loop overhead subtraction
; ====================================================================
.text
.global lds_tp_nop_kernel
.p2align 8
.type lds_tp_nop_kernel,@function

lds_tp_nop_kernel:
    s_load_dwordx2 s[s_out_ptr:s_out_ptr+1], s[s_karg:s_karg+1], 0
    s_load_dword   s[s_num_iters],           s[s_karg:s_karg+1], 8
    s_waitcnt lgkmcnt(0)

    v_mov_b32 v2, s[s_num_iters]
    s_waitcnt vmcnt(0) lgkmcnt(0)
    s_memrealtime s[s_start_lo:s_start_hi]
    s_waitcnt lgkmcnt(0)

.Lloop_tp_nop:
    ; Loop footer only — no payload, just overhead
    s_waitcnt lgkmcnt(0)
    v_sub_u32 v2, v2, 1
    v_cmp_gt_u32 vcc, v2, 0
    s_cbranch_vccnz .Lloop_tp_nop

    s_waitcnt vmcnt(0) lgkmcnt(0)
    s_memrealtime s[s_end_lo:s_end_hi]
    s_waitcnt lgkmcnt(0)

    STORE_OUTPUT
    s_endpgm

; ====================================================================
; Kernel descriptors
; ====================================================================
.rodata

; Helper macro for kernel descriptors
; \name: kernel name
; \lds_size: group segment size
; \vgpr_count: next free VGPR
; \accum_offset: accum offset (must be >= vgpr_count, 4-aligned)
.macro KERNEL_DESC name, lds_size, vgpr_count, accum_offset
.p2align 6
.amdhsa_kernel \name
    .amdhsa_group_segment_fixed_size \lds_size
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr \vgpr_count
    .amdhsa_next_free_sgpr 12
    .amdhsa_accum_offset \accum_offset
    .amdhsa_ieee_mode 0
    .amdhsa_dx10_clamp 0
.end_amdhsa_kernel
.endm

; v0-v3 control, v4-v35 unroll, v36-v39 output = 40 VGPRs
; LDS sizes: b32=256, b64=512, b128=1024
KERNEL_DESC lds_tp_read_b32_kernel,   256,  40, 40
KERNEL_DESC lds_tp_read_b64_kernel,   512,  40, 40
KERNEL_DESC lds_tp_read_b128_kernel,  1024, 40, 40
KERNEL_DESC lds_tp_write_b32_kernel,  256,  40, 40
KERNEL_DESC lds_tp_write_b64_kernel,  512,  40, 40
KERNEL_DESC lds_tp_write_b128_kernel, 1024, 40, 40
KERNEL_DESC lds_tp_nop_kernel,        0,    40, 40

; ====================================================================
; AMDGPU metadata
; ====================================================================
.amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: lds_tp_read_b32_kernel
    .symbol: lds_tp_read_b32_kernel.kd
    .kernarg_segment_size: 12
    .group_segment_fixed_size: 256
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 12
    .vgpr_count: 40
    .max_flat_workgroup_size: 64
    .args:
    - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
    - { .size: 4, .offset: 8, .value_kind: by_value }
  - .name: lds_tp_read_b64_kernel
    .symbol: lds_tp_read_b64_kernel.kd
    .kernarg_segment_size: 12
    .group_segment_fixed_size: 512
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 12
    .vgpr_count: 40
    .max_flat_workgroup_size: 64
    .args:
    - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
    - { .size: 4, .offset: 8, .value_kind: by_value }
  - .name: lds_tp_read_b128_kernel
    .symbol: lds_tp_read_b128_kernel.kd
    .kernarg_segment_size: 12
    .group_segment_fixed_size: 1024
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 12
    .vgpr_count: 40
    .max_flat_workgroup_size: 64
    .args:
    - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
    - { .size: 4, .offset: 8, .value_kind: by_value }
  - .name: lds_tp_write_b32_kernel
    .symbol: lds_tp_write_b32_kernel.kd
    .kernarg_segment_size: 12
    .group_segment_fixed_size: 256
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 12
    .vgpr_count: 40
    .max_flat_workgroup_size: 64
    .args:
    - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
    - { .size: 4, .offset: 8, .value_kind: by_value }
  - .name: lds_tp_write_b64_kernel
    .symbol: lds_tp_write_b64_kernel.kd
    .kernarg_segment_size: 12
    .group_segment_fixed_size: 512
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 12
    .vgpr_count: 40
    .max_flat_workgroup_size: 64
    .args:
    - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
    - { .size: 4, .offset: 8, .value_kind: by_value }
  - .name: lds_tp_write_b128_kernel
    .symbol: lds_tp_write_b128_kernel.kd
    .kernarg_segment_size: 12
    .group_segment_fixed_size: 1024
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 12
    .vgpr_count: 40
    .max_flat_workgroup_size: 64
    .args:
    - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
    - { .size: 4, .offset: 8, .value_kind: by_value }
  - .name: lds_tp_nop_kernel
    .symbol: lds_tp_nop_kernel.kd
    .kernarg_segment_size: 12
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 12
    .vgpr_count: 40
    .max_flat_workgroup_size: 64
    .args:
    - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
    - { .size: 4, .offset: 8, .value_kind: by_value }
...
.end_amdgpu_metadata
