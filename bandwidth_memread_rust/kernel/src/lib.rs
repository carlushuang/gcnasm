#![no_std]
#![feature(abi_gpu_kernel, stdarch_amdgpu, link_llvm_intrinsics, portable_simd, simd_ffi)]
#![allow(internal_features, improper_ctypes)]

use core::arch::amdgpu::{readfirstlane_u32, workgroup_id_x, workitem_id_x};
use core::simd::{f32x4, i32x4};

const BLOCK_SIZE: u32 = 1024;
const UNROLL: u32 = 4;

// gfx9 buffer resource dword3 (DATA_FORMAT=32), same value CK uses for gfx9xx
const BUFFER_RSRC_DWORD3: i32 = 0x0002_0000;
// cache-policy aux bits: on gfx940+ SLC encodes NT
const AUX_NT: i32 = 2;

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.amdgcn.raw.buffer.load.v4f32"]
    fn raw_buffer_load_v4f32(rsrc: i32x4, voffset: i32, soffset: i32, aux: i32) -> f32x4;
    #[link_name = "llvm.amdgcn.raw.buffer.store.v4f32"]
    fn raw_buffer_store_v4f32(vdata: f32x4, rsrc: i32x4, voffset: i32, soffset: i32, aux: i32);
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    core::arch::amdgpu::endpgm()
}

// wave-uniform buffer descriptor covering [base, base + 4GiB)
#[inline(always)]
fn make_rsrc<T>(base: *const T) -> i32x4 {
    let a = base as u64;
    let lo = readfirstlane_u32(a as u32) as i32;
    let hi = readfirstlane_u32((a >> 32) as u32) as i32;
    i32x4::from_array([lo, hi & 0xffff, -1, BUFFER_RSRC_DWORD3])
}

// rustc drops `!nontemporal` on amdgpu (not in WELL_BEHAVED_NONTEMPORAL_ARCHS), so
// core::intrinsics::nontemporal_store is a plain store here; go through the buffer path
#[inline(always)]
unsafe fn nt_store(rsrc: i32x4, byte_offset: u32, v: f32x4) {
    unsafe { raw_buffer_store_v4f32(v, rsrc, byte_offset as i32, 0, AUX_NT) }
}

#[inline(always)]
unsafe fn nt_load(rsrc: i32x4, byte_offset: u32) -> f32x4 {
    unsafe { raw_buffer_load_v4f32(rsrc, byte_offset as i32, 0, AUX_NT) }
}

// same algorithm as gcnasm bandwidth_memread memread_kernel (float4, UNROLL=4, nt load)
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn memread_kernel(
    p_src: *const f32x4,
    p_dst: *mut f32x4,
    issues_per_block: i32,
    iters: i32,
) {
    let current = workgroup_id_x() as usize * issues_per_block as usize;
    let rsrc = make_rsrc(unsafe { p_src.add(current) });
    let tid = workitem_id_x();
    let mut v = f32x4::splat(0.0);
    for i in 0..iters as u32 {
        let mut offs = UNROLL * BLOCK_SIZE * i + tid;
        for _ in 0..UNROLL {
            v += unsafe { nt_load(rsrc, offs * 16) };
            offs += BLOCK_SIZE;
        }
    }
    if v == f32x4::splat(10000.0) {
        unsafe { *p_dst = v };
    }
}

// plain (temporal) global loads, for comparison
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn memread_plain_kernel(
    p_src: *const f32x4,
    p_dst: *mut f32x4,
    issues_per_block: i32,
    iters: i32,
) {
    let base = unsafe { p_src.add(workgroup_id_x() as usize * issues_per_block as usize) };
    let tid = workitem_id_x();
    let mut v = f32x4::splat(0.0);
    for i in 0..iters as u32 {
        let mut offs = UNROLL * BLOCK_SIZE * i + tid;
        for _ in 0..UNROLL {
            v += unsafe { base.add(offs as usize).read() };
            offs += BLOCK_SIZE;
        }
    }
    if v == f32x4::splat(10000.0) {
        unsafe { *p_dst = v };
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn memcpy_kernel(
    p_src: *const f32x4,
    p_dst: *mut f32x4,
    issues_per_block: i32,
    iters: i32,
) {
    let current = workgroup_id_x() as usize * issues_per_block as usize;
    let rsrc = make_rsrc(unsafe { p_src.add(current) });
    let dst_rsrc = make_rsrc(unsafe { p_dst.add(current) });
    let tid = workitem_id_x();
    for i in 0..iters as u32 {
        let offs = UNROLL * BLOCK_SIZE * i + tid;
        let mut tmp = [f32x4::splat(0.0); UNROLL as usize];
        for j in 0..UNROLL {
            tmp[j as usize] = unsafe { nt_load(rsrc, (offs + j * BLOCK_SIZE) * 16) };
        }
        for j in 0..UNROLL {
            unsafe { nt_store(dst_rsrc, (offs + j * BLOCK_SIZE) * 16, tmp[j as usize]) };
        }
    }
}
