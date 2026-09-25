//! Rust port of ../../mxfp8_gemm_16x16x128_blockscale_bpreshuffle/gemm_a8w8_mxfp8_scale_kernel_template.hpp
//!
//! gfx950 MXFP8 GEMM, C[M,N] = dequant(A[M,K]) @ dequant(B[N,K]).T:
//! A row-major FP8, B aiter shuffle_weight(layout=(16,16)), SFA E8M0 1x128 stored [K/128,M],
//! SFB E8M0 128x128 stored [N/128,K/128]. 256x256x128 tile, 8 waves, scaled 16x16x128 MFMA.
//! The instruction sequence, LDS layout and scheduling hints follow the C++ kernel
//! statement by statement; see the C++ source for the design rationale.
#![no_std]
#![feature(abi_gpu_kernel, stdarch_amdgpu, link_llvm_intrinsics, portable_simd, simd_ffi, gpu_intrinsics, gpu_launch_sized_workgroup_mem, core_intrinsics)]
#![allow(internal_features)]
// Kargs is passed by value exactly like the C++ kernel (byref kernarg, 96 bytes, same offsets).
#![allow(improper_gpu_kernel_arg)]

mod amdgpu;
pub mod layout;

use amdgpu::*;
use core::arch::amdgpu::{readfirstlane_u32, s_barrier, sched_barrier, sched_group_barrier, workgroup_id_x, workgroup_id_z, workitem_id_x};
use core::simd::{f32x4, i32x2, i32x4, i32x8, simd_swizzle};
use layout::*;

/// Same layout as opus_gemm_scale_kargs in gemm_a8w8_mxfp8_scale_common.h.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Kargs {
    pub ptr_a: *const u8,
    pub ptr_b: *const u8,
    pub ptr_c: *mut u8,
    pub m: i32,
    pub n: i32,
    pub k: i32,
    pub batch: i32,
    pub stride_a: i32,
    pub stride_b: i32,
    pub stride_c: i32,
    pub stride_a_batch: i32,
    pub stride_b_batch: i32,
    pub stride_c_batch: i32,
    pub ptr_sfa: *const u8,
    pub ptr_sfb: *const u8,
    pub stride_sfa: i32,
    pub stride_sfb: i32,
    pub stride_sfa_batch: i32,
    pub stride_sfb_batch: i32,
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    core::arch::amdgpu::endpgm()
}

const MAX_RECORDS: u32 = 0xffff_ffff;

type VA = [i32x4; 4]; // ra issues: (m_repeat, k_half)
type VB = [i32x4; 8]; // rb issues: two 16-byte K pieces per n_repeat
type VC = [f32x4; 8]; // m_repeat * 4 + n_repeat

#[inline(always)]
fn cat(lo: i32x4, hi: i32x4) -> i32x8 {
    simd_swizzle!(lo, hi, [0, 1, 2, 3, 4, 5, 6, 7])
}

#[inline(always)]
fn sched_barrier_pairs() {
    unsafe {
        sched_group_barrier::<0x08, 1, 0>();
        sched_group_barrier::<0x02, 2, 0>();
        sched_group_barrier::<0x004, 1, 0>();
        sched_group_barrier::<0x100, 1, 0>();
        sched_group_barrier::<0x08, 1, 0>();
        sched_group_barrier::<0x02, 2, 0>();
        sched_group_barrier::<0x004, 1, 0>();
        sched_group_barrier::<0x100, 1, 0>();
    }
}

#[inline(always)]
fn sched_barrier0() {
    unsafe { sched_barrier::<0>() }
}

// One scaled MFMA. opus mfma_adaptor_swap_ab feeds B as src0 and A as src1, so the
// B scale/op_sel goes with src0: op_sel_b = n_repeat, op_sel_a = half_m * E_M + m_repeat.
#[inline(always)]
fn mfma_one<const MREP: usize, const NR: usize, const OPSEL_SFB: i32, const OPSEL_SFA: i32>(
    va: &VA, vb: &VB, vc: &mut VC, sfa: u32, sfb: u32) {
    let a = cat(va[MREP * 2], va[MREP * 2 + 1]);
    let b = cat(vb[NR * 2], vb[NR * 2 + 1]);
    vc[MREP * 4 + NR] = mfma_scale_16x16x128_fp8::<OPSEL_SFB, OPSEL_SFA>(b, a, vc[MREP * 4 + NR], sfb as i32, sfa as i32);
}

/// mfma_scale_n_pair<T, HALF_TILE_M, M_REPEAT, N_GROUP>, then the C++ kernel's
/// `asm volatile("" : "+v"(v_c_pin[M_REPEAT]))` and sched_barrier_pairs_scale().
macro_rules! mfma_pair {
    ($half:literal, $mrep:literal, $ngroup:literal, $va:expr, $vb:expr, $vc:expr, $sfa:expr, $sfb:expr) => {{
        let vc: &mut VC = $vc;
        mfma_one::<$mrep, { $ngroup * 2 }, { $ngroup * 2 }, { $half * 2 + $mrep }>($va, $vb, vc, $sfa, $sfb);
        mfma_one::<$mrep, { $ngroup * 2 + 1 }, { $ngroup * 2 + 1 }, { $half * 2 + $mrep }>($va, $vb, vc, $sfa, $sfb);
        pin_acc(&mut vc[$mrep * 4..$mrep * 4 + 4]);
        sched_barrier_pairs();
    }};
}

#[inline(always)]
fn to_bf16x4(v: f32x4) -> i32x2 {
    i32x2::from_array([cvt_pk_bf16_f32(v[0], v[1]) as i32, cvt_pk_bf16_f32(v[2], v[3]) as i32])
}

#[inline(always)]
unsafe fn lds_read_b128(p: *mut u8) -> i32x4 {
    unsafe { (p as *const i32x4).read() }
}

struct Ctx {
    lds: *mut u8,
    lane_id: i32,
    wave_id_m: i32,
    wave_id_n: i32,
    stride_a: i32,
    stride_b: i32,
    stride_sfa: i32,
    g_sf: Rsrc,
    u_gsf: i32,
    u_ssf: i32,
    u_rsfa: i32,
    u_rsfb: i32,
}

impl Ctx {
    #[inline(always)]
    fn at(&self, off: i32) -> *mut u8 {
        unsafe { self.lds.add(off as usize) }
    }
    #[inline(always)]
    fn ga_offset(&self, half: i32, tile_k: i32) -> i32 {
        half * HALF_B_M * self.stride_a + tile_k * B_K
    }
    #[inline(always)]
    fn gb_offset(&self, half: i32, tile_k: i32) -> i32 {
        half * HALF_B_N * self.stride_b + tile_k * B_K * 16
    }
    #[inline(always)]
    fn sa_offset(stage: i32, half: i32) -> i32 {
        LDS_A + (stage * 2 + half) * SMEM_A_ELEM
    }
    #[inline(always)]
    fn sb_offset(stage: i32, half: i32) -> i32 {
        LDS_B + (stage * 2 + half) * SMEM_B_ELEM
    }
    /// async_load<VEC>(g, s.ptr, u_g, u_s + s_off, g_off): 2 x buffer_load_dwordx4 ... lds
    #[inline(always)]
    fn async_load(&self, g: Rsrc, u_g: &[i32; 2], u_s: &[i32; 2], s_off: i32, g_off: i32) {
        unsafe {
            buffer_load_lds_b128(g, self.at(u_s[0] + s_off), u_g[0], g_off);
            buffer_load_lds_b128(g, self.at(u_s[1] + s_off), u_g[1], g_off);
        }
    }
    #[inline(always)]
    fn load_a(&self, u_ra: &[i32; 4], s_off: i32) -> VA {
        unsafe { [0, 1, 2, 3].map(|i| lds_read_b128(self.at(u_ra[i] + s_off))) }
    }
    #[inline(always)]
    fn load_b_range<const BEGIN: usize, const END: usize>(&self, u_rb: &[i32; 8], s_off: i32, dst: &mut VB) {
        let mut i = BEGIN;
        while i < END {
            dst[i] = unsafe { lds_read_b128(self.at(u_rb[i] + s_off)) };
            i += 1;
        }
    }
    #[inline(always)]
    fn load_sfa_dword(&self, stage: i32) -> u32 {
        // TR8: low dword = scale bytes of the four M repeats; high dword unused.
        unsafe { ds_read_tr8_b64(self.at(LDS_SFA + self.u_rsfa + stage * SMEM_SF_ELEM))[0] as u32 }
    }
    #[inline(always)]
    fn load_sfb_pair(&self, stage: i32) -> [u32; 2] {
        // SFB half 1 sits exactly 512 bytes after half 0: offset1:2 in units of 256 bytes.
        let v = unsafe { ds_read2st64_b32_o2(self.at(LDS_SFB + self.u_rsfb + stage * SMEM_SF_ELEM)) };
        [v[0] as u32, v[1] as u32]
    }
    #[inline(always)]
    fn gsf_offset(&self, output_tile: i32, k_tile: i32) -> i32 {
        let local_role = pin_sgpr(self.wave_id_n);
        if local_role == 0 { output_tile * B_M + k_tile * self.stride_sfa } else { k_tile }
    }
    #[inline(always)]
    fn load_scale_raw(&self, output_tile: i32, k_tile: i32) -> u32 {
        unsafe { buffer_load_u8(self.g_sf, self.u_gsf, self.gsf_offset(output_tile, k_tile)) as u32 }
    }
    /// A producers publish the raw byte for TR8; B producers publish the byte replicated 4x.
    #[inline(always)]
    fn store_scale_dword(&self, stage: i32, raw: u32) {
        let local_role = pin_sgpr(self.wave_id_n);
        s_waitcnt_vmcnt0();
        unsafe {
            if local_role == 0 {
                ds_write_b8(self.at(LDS_SFA + stage * SMEM_SF_ELEM + self.wave_id_m * WARP_SIZE + self.lane_id), raw);
            } else {
                *(self.at(LDS_SFB + self.u_ssf + stage * SMEM_SF_ELEM) as *mut u32) = raw.wrapping_mul(0x0101_0101);
            }
        }
    }
}

/// store<VEC_C>(g_c, cast<D_C>(v), u_gc, c_off, aux): BF16 default policy, FP32 nt.
#[inline(always)]
unsafe fn store_c<const BF16: bool>(g_c: Rsrc, v: &VC, u_gc: &[i32; 8], c_off: i32) {
    let mut i = 0;
    while i < 8 {
        unsafe {
            if BF16 {
                buffer_store_b64::<0>(to_bf16x4(v[i]), g_c, u_gc[i] * 2, c_off * 2);
            } else {
                buffer_store_b128::<AUX_NT>(v[i], g_c, u_gc[i] * 4, c_off * 4);
            }
        }
        i += 1;
    }
}

#[inline(always)]
unsafe fn gemm<const OUTPUT_TILES: i32, const BF16: bool>(kargs: &Kargs) {
    let c_elem = if BF16 { 2 } else { 4 };
    let num_tiles_m = (kargs.m + B_M - 1) / B_M;
    let num_tiles_n = (kargs.n + B_N - 1) / B_N;
    let num_tiles_k = (kargs.k + B_K - 1) / B_K;
    let bid = workgroup_id_x() as i32;
    let block_n = bid % num_tiles_n;
    let first_block_m = (bid / num_tiles_n) * OUTPUT_TILES;
    let col = block_n * B_N;

    let batch_id = workgroup_id_z() as i32;
    let tid = workitem_id_x() as i32;
    let wave_id = readfirstlane_u32((tid / WARP_SIZE) as u32) as i32;
    let lane_id = tid % WARP_SIZE;
    let scale_producer_is_sfa = wave_id < T_M;

    let p_a = kargs.ptr_a.wrapping_offset((batch_id * kargs.stride_a_batch) as isize);
    let p_b = kargs.ptr_b.wrapping_offset((batch_id * kargs.stride_b_batch + col * kargs.stride_b) as isize);
    let p_c = kargs.ptr_c.wrapping_offset(((batch_id * kargs.stride_c_batch + col) * c_elem) as isize);
    let (p_sf, scale_bytes_remaining) = if scale_producer_is_sfa {
        let row_offset = first_block_m * B_M;
        (kargs.ptr_sfa.wrapping_offset((batch_id * kargs.stride_sfa_batch + row_offset) as isize),
         (kargs.stride_sfa_batch - row_offset) as u32)
    } else {
        let row_offset = block_n * (B_N / GROUP_N) * kargs.stride_sfb;
        (kargs.ptr_sfb.wrapping_offset((batch_id * kargs.stride_sfb_batch + row_offset) as isize),
         (kargs.stride_sfb_batch - row_offset) as u32)
    };

    let wave_id_m = wave_id % T_M;
    let wave_id_n = wave_id / T_M;
    let cx = Ctx {
        lds: core::intrinsics::gpu::gpu_launch_sized_workgroup_mem::<i32x4>() as *mut u8,
        lane_id,
        wave_id_m,
        wave_id_n,
        stride_a: kargs.stride_a,
        stride_b: kargs.stride_b,
        stride_sfa: kargs.stride_sfa,
        g_sf: make_rsrc(p_sf, scale_bytes_remaining),
        u_gsf: gsf(lane_id, wave_id_m, scale_producer_is_sfa, kargs.stride_sfb),
        u_ssf: ssf(lane_id, wave_id_m),
        u_rsfa: rsfa(lane_id, wave_id_m),
        u_rsfb: rsfb(lane_id, wave_id_n),
    };
    let u_ga = ga(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
    let u_sa = sa(wave_id_m, wave_id_n);
    let u_ra = ra(lane_id, wave_id_m);
    let u_gb = gb(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
    let u_gb_producer_0 = gb(lane_id, wave_id_m, 0, kargs.stride_b);
    let u_gb_producer_1 = gb(lane_id, wave_id_m, 1, kargs.stride_b);
    let u_sb = sa(wave_id_m, wave_id_n);
    let u_sb_producer_0 = sa(wave_id_m, 0);
    let u_sb_producer_1 = sa(wave_id_m, 1);
    let u_rb = rb(lane_id, wave_id_n);

    let mut v_scale_raw: u32 = 0;
    let mut first_stage = 0;
    let mut output_tile = 0;
    while output_tile < OUTPUT_TILES {
        let block_m = first_block_m + output_tile;
        if block_m >= num_tiles_m {
            break;
        }
        let row = block_m * B_M;
        let g_a = make_rsrc(p_a.wrapping_offset((row * kargs.stride_a) as isize), MAX_RECORDS);
        let g_b = make_rsrc(p_b, MAX_RECORDS);
        let g_c = make_rsrc(p_c.wrapping_offset((row * kargs.stride_c * c_elem) as isize), MAX_RECORDS);

        let zero_a = [i32x4::splat(0); 4];
        let mut v_a: [VA; 2] = [zero_a; 2];
        let mut v_b: VB = [i32x4::splat(0); 8];
        let mut v_b_second: VB = [i32x4::splat(0); 8];
        let mut v_c: [[VC; 2]; 2] = [[[f32x4::splat(0.0); 8]; 2]; 2];
        let mut v_sfa: u32 = 0;
        let mut v_sfb: [u32; 2] = [0; 2];

        let loops = num_tiles_k;
        // With one output tile per workgroup first_stage is always 0, so LLVM resolves every
        // unrolled iteration's LDS addresses to constants and keeps ~24 of them live in VGPRs
        // (spills). Keep the stage a runtime SGPR value there, as it is in the 4-tile kernel.
        let mut stage = if OUTPUT_TILES == 1 { pin_sgpr(readfirstlane_u32(first_stage as u32) as i32) } else { first_stage };
        let mut scale_stage = stage;

        // Prologue
        if output_tile == 0 {
            v_scale_raw = cx.load_scale_raw(output_tile, 0);
            cx.async_load(g_a, &u_ga, &u_sa, Ctx::sa_offset(stage, 0), cx.ga_offset(0, 0));
            cx.async_load(g_b, &u_gb, &u_sb, Ctx::sb_offset(stage, 0), cx.gb_offset(0, 0));
            cx.async_load(g_a, &u_ga, &u_sa, Ctx::sa_offset(stage, 1), cx.ga_offset(1, 0));
            cx.async_load(g_b, &u_gb, &u_sb, Ctx::sb_offset(stage, 1), cx.gb_offset(1, 0));

            s_waitcnt_vmcnt0();
            cx.store_scale_dword(stage, v_scale_raw);
            s_waitcnt_lgkmcnt0();
            s_barrier();
            sched_barrier0();
        }
        if loops > 1 && output_tile == 0 {
            cx.async_load(g_b, &u_gb, &u_sb, Ctx::sb_offset(stage ^ 1, 0), cx.gb_offset(0, 1));
            cx.async_load(g_b, &u_gb, &u_sb, Ctx::sb_offset(stage ^ 1, 1), cx.gb_offset(1, 1));
            sched_barrier0();
        }

        // Main loop body for K tile `tile` (the C++ loop runs under #pragma unroll 8).
        let mut main_iter = |tile: i32, stage: &mut i32, scale_stage: &mut i32| {
            let next_stage = *stage ^ 1;

            let v_scale_next_raw = cx.load_scale_raw(output_tile, tile + 1);
            sched_barrier0();

            v_sfa = cx.load_sfa_dword(*scale_stage);
            v_sfb = cx.load_sfb_pair(*scale_stage);
            v_a[0] = cx.load_a(&u_ra, Ctx::sa_offset(*stage, 0));
            sched_barrier0();

            cx.load_b_range::<0, 8>(&u_rb, Ctx::sb_offset(*stage, 0), &mut v_b);
            cx.load_b_range::<0, 4>(&u_rb, Ctx::sb_offset(*stage, 1), &mut v_b_second);
            sched_barrier0();

            cx.async_load(g_a, &u_ga, &u_sa, Ctx::sa_offset(next_stage, 0), cx.ga_offset(0, tile + 1));
            cx.async_load(g_a, &u_ga, &u_sa, Ctx::sa_offset(next_stage, 1), cx.ga_offset(1, tile + 1));
            sched_barrier0();

            s_waitcnt_lgkmcnt0();
            // A half 0 x B half 0 -> C[0][0]
            mfma_pair!(0, 0, 0, &v_a[0], &v_b, &mut v_c[0][0], v_sfa, v_sfb[0]);

            v_a[1] = cx.load_a(&u_ra, Ctx::sa_offset(*stage, 1));
            s_waitcnt_lgkmcnt0();

            mfma_pair!(0, 0, 1, &v_a[0], &v_b, &mut v_c[0][0], v_sfa, v_sfb[0]);
            mfma_pair!(0, 1, 0, &v_a[0], &v_b, &mut v_c[0][0], v_sfa, v_sfb[0]);
            mfma_pair!(0, 1, 1, &v_a[0], &v_b, &mut v_c[0][0], v_sfa, v_sfb[0]);
            // A half 1 x B half 0 -> C[1][0]
            mfma_pair!(1, 0, 0, &v_a[1], &v_b, &mut v_c[1][0], v_sfa, v_sfb[0]);
            mfma_pair!(1, 0, 1, &v_a[1], &v_b, &mut v_c[1][0], v_sfa, v_sfb[0]);
            mfma_pair!(1, 1, 0, &v_a[1], &v_b, &mut v_c[1][0], v_sfa, v_sfb[0]);
            mfma_pair!(1, 1, 1, &v_a[1], &v_b, &mut v_c[1][0], v_sfa, v_sfb[0]);

            cx.load_b_range::<4, 8>(&u_rb, Ctx::sb_offset(*stage, 1), &mut v_b_second);

            // A half 0 x B half 1 -> C[0][1]
            mfma_pair!(0, 0, 0, &v_a[0], &v_b_second, &mut v_c[0][1], v_sfa, v_sfb[1]);
            mfma_pair!(0, 0, 1, &v_a[0], &v_b_second, &mut v_c[0][1], v_sfa, v_sfb[1]);

            // Publish tile t+1's scales and release tile t's LDS stage, then start the
            // cold B path for tile t+2 while the final 12 MFMAs run.
            cx.store_scale_dword(next_stage, v_scale_next_raw);
            sched_barrier0();
            s_waitcnt_vmcnt0();
            s_waitcnt_lgkmcnt0();
            s_barrier();
            sched_barrier0();

            if tile + 2 < loops {
                if wave_id_n == 1 {
                    cx.async_load(g_b, &u_gb_producer_0, &u_sb_producer_0, Ctx::sb_offset(*stage, 0), cx.gb_offset(0, tile + 2));
                    cx.async_load(g_b, &u_gb_producer_1, &u_sb_producer_1, Ctx::sb_offset(*stage, 0), cx.gb_offset(0, tile + 2));
                    cx.async_load(g_b, &u_gb_producer_0, &u_sb_producer_0, Ctx::sb_offset(*stage, 1), cx.gb_offset(1, tile + 2));
                    cx.async_load(g_b, &u_gb_producer_1, &u_sb_producer_1, Ctx::sb_offset(*stage, 1), cx.gb_offset(1, tile + 2));
                }
                sched_barrier0();
            }

            sched_barrier0();
            mfma_pair!(0, 1, 0, &v_a[0], &v_b_second, &mut v_c[0][1], v_sfa, v_sfb[1]);
            mfma_pair!(0, 1, 1, &v_a[0], &v_b_second, &mut v_c[0][1], v_sfa, v_sfb[1]);
            // A half 1 x B half 1 -> C[1][1]
            mfma_pair!(1, 0, 0, &v_a[1], &v_b_second, &mut v_c[1][1], v_sfa, v_sfb[1]);
            mfma_pair!(1, 0, 1, &v_a[1], &v_b_second, &mut v_c[1][1], v_sfa, v_sfb[1]);
            mfma_pair!(1, 1, 0, &v_a[1], &v_b_second, &mut v_c[1][1], v_sfa, v_sfb[1]);
            mfma_pair!(1, 1, 1, &v_a[1], &v_b_second, &mut v_c[1][1], v_sfa, v_sfb[1]);
            *stage = next_stage;
            *scale_stage = next_stage;
        };

        // Rust has no #pragma unroll: unroll the K loop 8x by hand, then the remainder.
        let main_loops = loops - 1;
        let mut tile = 0;
        while tile + 8 <= main_loops {
            main_iter(tile, &mut stage, &mut scale_stage);
            main_iter(tile + 1, &mut stage, &mut scale_stage);
            main_iter(tile + 2, &mut stage, &mut scale_stage);
            main_iter(tile + 3, &mut stage, &mut scale_stage);
            main_iter(tile + 4, &mut stage, &mut scale_stage);
            main_iter(tile + 5, &mut stage, &mut scale_stage);
            main_iter(tile + 6, &mut stage, &mut scale_stage);
            main_iter(tile + 7, &mut stage, &mut scale_stage);
            tile += 8;
        }
        while tile < main_loops {
            main_iter(tile, &mut stage, &mut scale_stage);
            tile += 1;
        }

        // Epilogue
        v_sfa = cx.load_sfa_dword(scale_stage);
        v_sfb = cx.load_sfb_pair(scale_stage);
        v_a[0] = cx.load_a(&u_ra, Ctx::sa_offset(stage, 0));
        v_a[1] = cx.load_a(&u_ra, Ctx::sa_offset(stage, 1));
        cx.load_b_range::<0, 8>(&u_rb, Ctx::sb_offset(stage, 0), &mut v_b);
        s_waitcnt_lgkmcnt0();

        let has_next_output = output_tile + 1 < OUTPUT_TILES && block_m + 1 < num_tiles_m;
        let next_output_stage = stage ^ 1;

        if has_next_output {
            let next_row = (block_m + 1) * B_M;
            let g_a_next = make_rsrc(p_a.wrapping_offset((next_row * kargs.stride_a) as isize), MAX_RECORDS);
            v_scale_raw = cx.load_scale_raw(output_tile + 1, 0);
            cx.async_load(g_a_next, &u_ga, &u_sa, Ctx::sa_offset(next_output_stage, 0), cx.ga_offset(0, 0));
            cx.async_load(g_b, &u_gb, &u_sb, Ctx::sb_offset(next_output_stage, 0), cx.gb_offset(0, 0));
            cx.async_load(g_a_next, &u_ga, &u_sa, Ctx::sa_offset(next_output_stage, 1), cx.ga_offset(1, 0));
            cx.async_load(g_b, &u_gb, &u_sb, Ctx::sb_offset(next_output_stage, 1), cx.gb_offset(1, 0));
            sched_barrier0();
        }

        let u_gc = gc(lane_id, wave_id_m, wave_id_n, kargs.stride_c);
        let c_offset = |half_m: i32, half_n: i32| half_m * HALF_B_M * kargs.stride_c + half_n * HALF_B_N;

        mfma_pair!(0, 0, 0, &v_a[0], &v_b, &mut v_c[0][0], v_sfa, v_sfb[0]);
        mfma_pair!(0, 0, 1, &v_a[0], &v_b, &mut v_c[0][0], v_sfa, v_sfb[0]);
        mfma_pair!(0, 1, 0, &v_a[0], &v_b, &mut v_c[0][0], v_sfa, v_sfb[0]);
        mfma_pair!(0, 1, 1, &v_a[0], &v_b, &mut v_c[0][0], v_sfa, v_sfb[0]);
        mfma_pair!(1, 0, 0, &v_a[1], &v_b, &mut v_c[1][0], v_sfa, v_sfb[0]);
        mfma_pair!(1, 0, 1, &v_a[1], &v_b, &mut v_c[1][0], v_sfa, v_sfb[0]);
        mfma_pair!(1, 1, 0, &v_a[1], &v_b, &mut v_c[1][0], v_sfa, v_sfb[0]);
        mfma_pair!(1, 1, 1, &v_a[1], &v_b, &mut v_c[1][0], v_sfa, v_sfb[0]);

        // Drain the first two quadrants while the remaining MFMAs execute.
        unsafe {
            store_c::<BF16>(g_c, &v_c[0][0], &u_gc, c_offset(0, 0));
            store_c::<BF16>(g_c, &v_c[1][0], &u_gc, c_offset(1, 0));
        }
        sched_barrier0();

        cx.load_b_range::<0, 8>(&u_rb, Ctx::sb_offset(stage, 1), &mut v_b);

        mfma_pair!(0, 0, 0, &v_a[0], &v_b, &mut v_c[0][1], v_sfa, v_sfb[1]);

        // output_b1_handoff
        if has_next_output {
            s_waitcnt_vmcnt0();
            cx.store_scale_dword(next_output_stage, v_scale_raw);
            s_waitcnt_lgkmcnt0();
            s_barrier();
            sched_barrier0();
            if loops > 1 {
                cx.async_load(g_b, &u_gb, &u_sb, Ctx::sb_offset(stage, 0), cx.gb_offset(0, 1));
                cx.async_load(g_b, &u_gb, &u_sb, Ctx::sb_offset(stage, 1), cx.gb_offset(1, 1));
                sched_barrier0();
            }
            first_stage = next_output_stage;
        }

        mfma_pair!(0, 0, 1, &v_a[0], &v_b, &mut v_c[0][1], v_sfa, v_sfb[1]);
        mfma_pair!(0, 1, 0, &v_a[0], &v_b, &mut v_c[0][1], v_sfa, v_sfb[1]);
        mfma_pair!(0, 1, 1, &v_a[0], &v_b, &mut v_c[0][1], v_sfa, v_sfb[1]);
        mfma_pair!(1, 0, 0, &v_a[1], &v_b, &mut v_c[1][1], v_sfa, v_sfb[1]);
        mfma_pair!(1, 0, 1, &v_a[1], &v_b, &mut v_c[1][1], v_sfa, v_sfb[1]);
        mfma_pair!(1, 1, 0, &v_a[1], &v_b, &mut v_c[1][1], v_sfa, v_sfb[1]);
        mfma_pair!(1, 1, 1, &v_a[1], &v_b, &mut v_c[1][1], v_sfa, v_sfb[1]);

        unsafe {
            store_c::<BF16>(g_c, &v_c[0][1], &u_gc, c_offset(0, 1));
            store_c::<BF16>(g_c, &v_c[1][1], &u_gc, c_offset(1, 1));
        }
        output_tile += 1;
    }
}

// The four specializations of gemm_a8w8_mxfp8_scale_kernel.cc: {FP32, BF16} x {1, 4} output tiles.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn gemm_mxfp8_bpreshuffle_fp32_t4(kargs: Kargs) {
    unsafe { gemm::<4, false>(&kargs) }
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn gemm_mxfp8_bpreshuffle_fp32_t1(kargs: Kargs) {
    unsafe { gemm::<1, false>(&kargs) }
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn gemm_mxfp8_bpreshuffle_bf16_t4(kargs: Kargs) {
    unsafe { gemm::<4, true>(&kargs) }
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn gemm_mxfp8_bpreshuffle_bf16_t1(kargs: Kargs) {
    unsafe { gemm::<1, true>(&kargs) }
}
