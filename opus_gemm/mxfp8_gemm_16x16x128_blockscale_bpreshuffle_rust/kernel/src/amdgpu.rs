//! Thin wrappers over the gfx950 LLVM intrinsics the kernel needs and that
//! core::arch::amdgpu does not cover: buffer loads/stores (incl. direct-to-LDS),
//! ds_read_b64_tr_b8, scaled MFMA and s_waitcnt.
//!
//! Rust has only generic (flat) pointers and no bfloat type, so intrinsics that take an
//! LDS pointer (`ptr addrspace(3)`) or produce bfloat are reached through `rk.*` extern
//! placeholders that tools/relink.py defines in LLVM IR (and opt inlines).

use core::simd::{f32x4, f32x16, i32x2, i32x4, i32x8, simd_swizzle};

#[allow(improper_ctypes)]
unsafe extern "C" {
    #[link_name = "rk.buffer.load.lds.b128"]
    fn rk_buffer_load_lds_b128(rsrc: i32x4, lds: *mut u8, voffset: i32, soffset: i32);
    #[link_name = "rk.ds.read.tr8.b64"]
    fn rk_ds_read_tr8_b64(lds: *mut u8) -> i32x2;
    #[link_name = "rk.cvt.pk.bf16.f32"]
    safe fn rk_cvt_pk_bf16_f32(a: f32, b: f32) -> u32;
    #[link_name = "rk.pin.v16f32"]
    safe fn rk_pin_v16f32(x: f32x16) -> f32x16;
    #[link_name = "rk.pin.s.i32"]
    safe fn rk_pin_s_i32(x: i32) -> i32;
    #[link_name = "rk.asm.ds.read2st64.b32.o2"]
    fn rk_asm_ds_read2st64_b32_o2(lds: *mut u8) -> i32x2;
    #[link_name = "rk.asm.ds.write.b8"]
    fn rk_asm_ds_write_b8(lds: *mut u8, v: u32);
}

#[allow(improper_ctypes)]
unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.amdgcn.raw.buffer.load.i8"]
    fn llvm_raw_buffer_load_i8(rsrc: i32x4, voffset: i32, soffset: i32, aux: i32) -> u8;
    #[link_name = "llvm.amdgcn.raw.buffer.store.v4f32"]
    fn llvm_raw_buffer_store_v4f32(v: f32x4, rsrc: i32x4, voffset: i32, soffset: i32, aux: i32);
    #[link_name = "llvm.amdgcn.raw.buffer.store.v2i32"]
    fn llvm_raw_buffer_store_v2i32(v: i32x2, rsrc: i32x4, voffset: i32, soffset: i32, aux: i32);
    #[link_name = "llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4"]
    fn llvm_mfma_scale_16x16x128(a: i32x8, b: i32x8, c: f32x4, cbsz: i32, blgp: i32, opsel_a: i32, scale_a: i32, opsel_b: i32, scale_b: i32) -> f32x4;
    #[link_name = "llvm.amdgcn.s.waitcnt"]
    fn llvm_s_waitcnt(imm: i32);
}

// gfx9 buffer resource dword3, same as opus buffer_default_config()
const BUFFER_CONFIG: i32 = 0x0002_0000;
// cache-policy aux: on gfx940+ SLC encodes NT
pub const AUX_NT: i32 = 2;

pub type Rsrc = i32x4;

/// __builtin_amdgcn_make_buffer_rsrc(ptr, stride = 0, num_records, config)
#[inline(always)]
pub fn make_rsrc(ptr: *const u8, num_records: u32) -> Rsrc {
    let a = ptr as u64;
    i32x4::from_array([a as i32, ((a >> 32) as i32) & 0xffff, num_records as i32, BUFFER_CONFIG])
}

/// buffer_load_dwordx4 ... lds: 16 bytes per lane from rsrc+voffset+soffset into lds + 16*lane.
#[inline(always)]
pub unsafe fn buffer_load_lds_b128(rsrc: Rsrc, lds: *mut u8, voffset: i32, soffset: i32) {
    unsafe { rk_buffer_load_lds_b128(rsrc, lds, voffset, soffset) }
}

#[inline(always)]
pub unsafe fn buffer_load_u8(rsrc: Rsrc, voffset: i32, soffset: i32) -> u8 {
    unsafe { llvm_raw_buffer_load_i8(rsrc, voffset, soffset, 0) }
}

#[inline(always)]
pub unsafe fn buffer_store_b128<const AUX: i32>(v: f32x4, rsrc: Rsrc, voffset: i32, soffset: i32) {
    unsafe { llvm_raw_buffer_store_v4f32(v, rsrc, voffset, soffset, AUX) }
}

#[inline(always)]
pub unsafe fn buffer_store_b64<const AUX: i32>(v: i32x2, rsrc: Rsrc, voffset: i32, soffset: i32) {
    unsafe { llvm_raw_buffer_store_v2i32(v, rsrc, voffset, soffset, AUX) }
}

#[inline(always)]
pub unsafe fn ds_read_tr8_b64(lds: *mut u8) -> i32x2 {
    unsafe { rk_ds_read_tr8_b64(lds) }
}

/// Two floats -> packed bf16, round to nearest even (v_cvt_pk_bf16_f32).
#[inline(always)]
pub fn cvt_pk_bf16_f32(a: f32, b: f32) -> u32 {
    rk_cvt_pk_bf16_f32(a, b)
}

/// v_mfma_scale_f32_16x16x128_f8f6f4 with FP8(E4M3) A and B.
#[inline(always)]
pub fn mfma_scale_16x16x128_fp8<const OPSEL_A: i32, const OPSEL_B: i32>(
    a: i32x8, b: i32x8, c: f32x4, scale_a: i32, scale_b: i32) -> f32x4 {
    unsafe { llvm_mfma_scale_16x16x128(a, b, c, 0, 0, OPSEL_A, scale_a, OPSEL_B, scale_b) }
}

/// s_waitcnt vmcnt(0), other counters untouched (opus s_waitcnt_vmcnt(0)).
#[inline(always)]
pub fn s_waitcnt_vmcnt0() {
    unsafe { llvm_s_waitcnt(0x0f70) }
}

/// s_waitcnt lgkmcnt(0), other counters untouched (opus s_waitcnt_lgkmcnt(0)).
#[inline(always)]
pub fn s_waitcnt_lgkmcnt0() {
    unsafe { llvm_s_waitcnt(0xc07f) }
}

/// `asm volatile("" : "+v"(x))` on four accumulators: opaque to the optimizer, pinned to
/// one 16-VGPR tuple, and a scheduling fence (volatile asm).
#[inline(always)]
pub fn pin_acc(v: &mut [f32x4]) {
    let lo = simd_swizzle!(v[0], v[1], [0, 1, 2, 3, 4, 5, 6, 7]);
    let hi = simd_swizzle!(v[2], v[3], [0, 1, 2, 3, 4, 5, 6, 7]);
    let x = rk_pin_v16f32(simd_swizzle!(lo, hi, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]));
    v[0] = simd_swizzle!(x, [0, 1, 2, 3]);
    v[1] = simd_swizzle!(x, [4, 5, 6, 7]);
    v[2] = simd_swizzle!(x, [8, 9, 10, 11]);
    v[3] = simd_swizzle!(x, [12, 13, 14, 15]);
}

/// `asm volatile("" : "+s"(x))`: the optimizer can no longer see through a wave-uniform value.
#[inline(always)]
pub fn pin_sgpr(x: i32) -> i32 {
    rk_pin_s_i32(x)
}

/// `ds_read2st64_b32 $0, $1 offset0:0 offset1:2`: dwords at lds and lds + 512 bytes.
#[inline(always)]
pub unsafe fn ds_read2st64_b32_o2(lds: *mut u8) -> i32x2 {
    unsafe { rk_asm_ds_read2st64_b32_o2(lds) }
}

/// `ds_write_b8 $0, $1` as inline asm (memory clobber), like the C++ kernel.
#[inline(always)]
pub unsafe fn ds_write_b8(lds: *mut u8, v: u32) {
    unsafe { rk_asm_ds_write_b8(lds, v) }
}
