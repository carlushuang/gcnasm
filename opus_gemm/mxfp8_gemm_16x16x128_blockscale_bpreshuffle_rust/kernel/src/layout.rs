//! Tile constants and per-lane offsets of every opus layout used by
//! ../../mxfp8_gemm_16x16x128_blockscale_bpreshuffle, written out in closed form
//! (opus builds them at compile time from shape/dim/stride tuples).
//!
//! Pure integer code: the device kernel calls these, and the host checks them
//! lane-for-lane against the opus originals (`--check-layouts`, oracle/layout_dump.cc).

pub const BLOCK_SIZE: i32 = 512;
pub const WARP_SIZE: i32 = 64;
pub const B_M: i32 = 256;
pub const B_N: i32 = 256;
pub const B_K: i32 = 128;
pub const T_M: i32 = 4;
pub const HALF_B_M: i32 = B_M / 2;
pub const HALF_B_N: i32 = B_N / 2;
pub const GROUP_N: i32 = 128;
pub const VEC_A: i32 = 16;
pub const VEC_B: i32 = 16;

// Each wave's 64 lanes x 16 bytes land contiguously in LDS, rows padded by 32 bytes.
pub const PITCH: i32 = WARP_SIZE * VEC_A + 32;
pub const SMEM_A_ELEM: i32 = (HALF_B_M / (WARP_SIZE * VEC_A / B_K)) * PITCH;
pub const SMEM_B_ELEM: i32 = (HALF_B_N / (WARP_SIZE * VEC_B / B_K)) * PITCH;
pub const SMEM_SF_ELEM: i32 = B_M * (B_K / 32);

// LDS map: A[2 stages][2 halves], B[2][2], SFA[2 stages], SFB[2 stages].
pub const LDS_A: i32 = 0;
pub const LDS_B: i32 = LDS_A + 4 * SMEM_A_ELEM;
pub const LDS_SFA: i32 = LDS_B + 4 * SMEM_B_ELEM;
pub const LDS_SFB: i32 = LDS_SFA + 2 * SMEM_SF_ELEM;
pub const LDS_BYTES: i32 = LDS_SFB + 2 * SMEM_SF_ELEM;

/// make_layout_ga_scale: A global -> LDS producer, 2 issues of 16 bytes.
pub fn ga(lane: i32, wave_m: i32, wave_n: i32, stride_a: i32) -> [i32; 2] {
    let base = (wave_n * 32 + (lane / 8) * 4 + wave_m) * stride_a + (lane % 8) * VEC_A;
    [base, base + 64 * stride_a]
}

/// make_layout_sa_scale / make_layout_sb_scale: wave-uniform LDS destination rows.
pub fn sa(wave_m: i32, wave_n: i32) -> [i32; 2] {
    let r = wave_n * 4 + wave_m;
    [r * PITCH, (r + 8) * PITCH]
}

/// make_layout_ra_scale: A LDS -> VGPR, issue (e_m, k_half) -> v_a[e_m * 2 + k_half].
pub fn ra(lane: i32, wave_m: i32) -> [i32; 4] {
    let lm = lane % 16;
    let base = (wave_m / 2) * 4 * PITCH + (lm % 4) * PITCH + (wave_m % 2) * 512 + (lm / 4) * 128 + (lane / 16) * 16;
    [base, base + 64, base + 8 * PITCH, base + 8 * PITCH + 64]
}

/// make_layout_gb_scale: B (16,16)-preshuffled global -> LDS producer.
pub fn gb(lane: i32, wave_m: i32, wave_n: i32, stride_b: i32) -> [i32; 2] {
    let base = wave_m * 16 * stride_b + wave_n * 1024 + lane * VEC_B;
    [base, base + 64 * stride_b]
}

/// make_layout_rb_scale: B LDS -> VGPR, issue i = 4a + 2b + c.
pub fn rb(lane: i32, wave_n: i32) -> [i32; 8] {
    let base = wave_n * PITCH + lane * VEC_B;
    let mut o = [0; 8];
    let mut i = 0;
    while i < 8 {
        o[i as usize] = base + (i >> 2) * 8 * PITCH + ((i >> 1) & 1) * 2 * PITCH + (i & 1) * 4 * PITCH;
        i += 1;
    }
    o
}

/// make_layout_rsfa_scale: ds_read_b64_tr_b8 source address (bytes).
pub fn rsfa(lane: i32, wave_m: i32) -> i32 {
    wave_m * WARP_SIZE + (lane & 15) * 8
}

/// make_layout_rsfb_scale(half_tile_n = 0): first dword of the ds_read2st64 pair.
pub fn rsfb(lane: i32, wave_n: i32) -> i32 {
    wave_n * 256 + lane * 4
}

/// make_layout_gsf_scale: compact E8M0 byte read by this producer lane.
pub fn gsf(lane: i32, wave_m: i32, is_sfa: bool, stride_sfb: i32) -> i32 {
    if is_sfa {
        wave_m * 16 + (lane & 15) + (lane >> 4) * (T_M * 16)
    } else {
        (wave_m >> 1) * stride_sfb
    }
}

/// make_layout_ssf_scale: packed SFB dword destination.
pub fn ssf(lane: i32, wave_m: i32) -> i32 {
    wave_m * 256 + lane * 4
}

/// partition_layout_c for the swap_ab 16x16 MFMA: issue i = e_m * 4 + e_n, 4 contiguous N elements.
pub fn gc(lane: i32, wave_m: i32, wave_n: i32, stride_c: i32) -> [i32; 8] {
    let base = (wave_m * 16 + lane % 16) * stride_c + wave_n * 16 + (lane / 16) * 4;
    let mut o = [0; 8];
    let mut i = 0;
    while i < 8 {
        o[i as usize] = base + (i / 4) * 64 * stride_c + (i % 4) * 32;
        i += 1;
    }
    o
}
