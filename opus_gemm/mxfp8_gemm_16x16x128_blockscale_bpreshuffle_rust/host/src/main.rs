//! Rust port of ../../mxfp8_gemm_16x16x128_blockscale_bpreshuffle/gemm_a8w8_mxfp8_scale_host.cc
//!
//! Generates the same inputs (counter-based, seed-reproducible), B preshuffle and
//! double-precision CPU reference, then launches either the Rust kernels
//! (build/mxfp8_gemm_rust.co) or the C++/opus kernels (a device-only .co of
//! gemm_a8w8_mxfp8_scale_kernel.cc) through one launch/timing path.
// Project style: snake_case type names.
#![allow(non_camel_case_types)]

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::ptr::null_mut;

#[path = "../../kernel/src/layout.rs"]
#[allow(dead_code)]
mod layout;

type hip_error = c_int;
type handle = *mut c_void;

const HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 63;
const H2D: c_int = 1;
const D2H: c_int = 2;

#[link(name = "amdhip64")]
unsafe extern "C" {
    fn hipSetDevice(dev: c_int) -> hip_error;
    fn hipGetDevice(dev: *mut c_int) -> hip_error;
    fn hipDeviceGetAttribute(v: *mut c_int, attr: c_int, dev: c_int) -> hip_error;
    fn hipGetErrorString(e: hip_error) -> *const c_char;
    fn hipMalloc(p: *mut *mut c_void, size: usize) -> hip_error;
    fn hipFree(p: *mut c_void) -> hip_error;
    fn hipMemcpy(dst: *mut c_void, src: *const c_void, size: usize, kind: c_int) -> hip_error;
    fn hipMemset(dst: *mut c_void, v: c_int, size: usize) -> hip_error;
    fn hipDeviceSynchronize() -> hip_error;
    fn hipModuleLoadData(m: *mut handle, image: *const c_void) -> hip_error;
    fn hipModuleGetFunction(f: *mut handle, m: handle, name: *const c_char) -> hip_error;
    fn hipModuleLaunchKernel(f: handle, gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32, shared: u32,
                             stream: handle, params: *mut *mut c_void, extra: *mut *mut c_void) -> hip_error;
    fn hipEventCreate(e: *mut handle) -> hip_error;
    fn hipEventDestroy(e: handle) -> hip_error;
    fn hipEventRecord(e: handle, s: handle) -> hip_error;
    fn hipEventSynchronize(e: handle) -> hip_error;
    fn hipEventElapsedTime(ms: *mut f32, a: handle, b: handle) -> hip_error;
}

macro_rules! check {
    ($e:expr) => {{
        let err = unsafe { $e };
        if err != 0 {
            let s = unsafe { CStr::from_ptr(hipGetErrorString(err)) };
            panic!("HIP error '{}' ({}) at {}:{}", s.to_string_lossy(), err, file!(), line!());
        }
    }};
}

/// Same layout as opus_gemm_scale_kargs.
#[repr(C)]
#[derive(Clone, Copy)]
struct opus_gemm_scale_kargs {
    ptr_a: *const c_void,
    ptr_b: *const c_void,
    ptr_c: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    batch: i32,
    stride_a: i32,
    stride_b: i32,
    stride_c: i32,
    stride_a_batch: i32,
    stride_b_batch: i32,
    stride_c_batch: i32,
    ptr_sfa: *const c_void,
    ptr_sfb: *const c_void,
    stride_sfa: i32,
    stride_sfb: i32,
    stride_sfa_batch: i32,
    stride_sfb_batch: i32,
}

// ---------------------------------------------------------------------------------------------
// FP8 E4M3FN / E8M0 / BF16 host helpers

/// float -> OCP E4M3FN, round to nearest even, saturate to +-448 (__hip_fp8_e4m3 conversion).
fn f32_to_e4m3(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7f;
    }
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
    let a = x.abs() as f64;
    if a >= 448.0 {
        return sign | 0x7e;
    }
    let min_normal = 2f64.powi(-6);
    let code = if a < min_normal {
        (a / 2f64.powi(-9)).round_ties_even() as u8 // 8 == the smallest normal, encoded the same way
    } else {
        let e = a.log2().floor() as i32;
        let e = if 2f64.powi(e) > a { e - 1 } else if 2f64.powi(e + 1) <= a { e + 1 } else { e };
        let q = ((a / 2f64.powi(e) - 1.0) * 8.0).round_ties_even() as i32;
        let (e, q) = if q == 8 { (e + 1, 0) } else { (e, q) };
        if e > 8 { 0x7e } else { (((e + 7) << 3) | q) as u8 }
    };
    sign | code
}

fn e4m3_to_f32(v: u8) -> f32 {
    let s = if v & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((v >> 3) & 0xf) as i32;
    let m = (v & 7) as f32;
    if e == 0xf && m == 7.0 {
        return f32::NAN;
    }
    if e == 0 { s * m * 2f32.powi(-9) } else { s * (1.0 + m / 8.0) * 2f32.powi(e - 7) }
}

fn e8m0_to_f32(e: u8) -> f32 {
    2f32.powi(e as i32 - 127)
}

fn f32_to_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    if bits & 0x7fff_ffff > 0x7f80_0000 {
        return ((bits >> 16) | 0x0040) as u16;
    }
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn bf16_to_f32(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}

fn round_bf16(v: f32) -> f32 {
    bf16_to_f32(f32_to_bf16_rne(v))
}

// ---------------------------------------------------------------------------------------------
// Input generation, identical to the C++ harness

fn mix_bits(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn par_fill<T: Send + Copy>(out: &mut [T], f: impl Fn(usize) -> T + Sync) {
    let threads = std::thread::available_parallelism().map_or(16, |n| n.get());
    let chunk = out.len().div_ceil(threads).max(1 << 16);
    std::thread::scope(|s| {
        for (ci, part) in out.chunks_mut(chunk).enumerate() {
            let f = &f;
            s.spawn(move || {
                for (j, o) in part.iter_mut().enumerate() {
                    *o = f(ci * chunk + j);
                }
            });
        }
    });
}

fn fill_fp8(out: &mut [u8], seed: u64) {
    par_fill(out, |i| {
        let unit = (mix_bits(seed.wrapping_add(i as u64)) >> 40) as f32 * 2f32.powi(-24);
        f32_to_e4m3(2.0 * unit - 1.0)
    });
}

fn fill_scales(out: &mut [u8], seed: u64) {
    par_fill(out, |i| (124 + mix_bits(seed.wrapping_add(i as u64)) % 7) as u8);
}

/// aiter shuffle_weight(layout=(16,16)): [N/16, K/16, 16, 16].
fn pack_weight_16x16(raw: &[u8], batches: usize, n: usize, k: usize) -> Vec<u8> {
    let mut packed = vec![0u8; raw.len()];
    for b in 0..batches {
        let base = b * n * k;
        for nb in 0..n / 16 {
            for kb in 0..k / 16 {
                let tile = base + (nb * (k / 16) + kb) * 256;
                for ni in 0..16 {
                    let src = base + (nb * 16 + ni) * k + kb * 16;
                    packed[tile + ni * 16..tile + ni * 16 + 16].copy_from_slice(&raw[src..src + 16]);
                }
            }
        }
    }
    packed
}

// ---------------------------------------------------------------------------------------------
// Reference + validation, same math and tolerance as the C++ harness

struct problem<'a> {
    a: &'a [u8],
    b: &'a [u8],
    sfa: &'a [u8],
    sfb: &'a [u8],
    m: usize,
    k: usize,
}

impl problem<'_> {
    /// (value, sum |term|) for one output, double accumulation.
    fn reference(&self, row: usize, col: usize) -> (f32, f64) {
        let groups_k = self.k / 128;
        let (mut sum, mut mag) = (0f64, 0f64);
        for kg in 0..groups_k {
            let scale = e8m0_to_f32(self.sfa[kg * self.m + row]) as f64
                * e8m0_to_f32(self.sfb[(col / 128) * groups_k + kg]) as f64;
            for p in kg * 128..(kg + 1) * 128 {
                let t = e4m3_to_f32(self.a[row * self.k + p]) as f64 * e4m3_to_f32(self.b[col * self.k + p]) as f64 * scale;
                sum += t;
                mag += t.abs();
            }
        }
        (sum as f32, mag)
    }
}

#[derive(Default)]
struct check_stats {
    checked: usize,
    errors: usize,
    max_diff: f64,
    max_ratio: f64,
}

fn check_one(st: &mut check_stats, raw_ref: f32, mag: f64, got: f32, bf16: bool, where_: (usize, usize)) {
    const REL_MAG: f64 = 5e-5;
    const ABS_FLOOR: f64 = 1e-4;
    let expected = if bf16 { round_bf16(raw_ref) } else { raw_ref };
    let err = ABS_FLOOR + REL_MAG * mag;
    let lo = if bf16 { round_bf16((raw_ref as f64 - err) as f32) as f64 } else { raw_ref as f64 - err };
    let hi = if bf16 { round_bf16((raw_ref as f64 + err) as f32) as f64 } else { raw_ref as f64 + err };
    let finite = raw_ref.is_finite() && got.is_finite() && mag.is_finite();
    let diff = if finite { (got as f64 - expected as f64).abs() } else { f64::INFINITY };
    let tol = (expected as f64 - lo).max(hi - expected as f64);
    let ratio = if tol > 0.0 { diff / tol } else if diff == 0.0 { 0.0 } else { f64::INFINITY };
    st.checked += 1;
    st.max_diff = st.max_diff.max(diff);
    st.max_ratio = st.max_ratio.max(ratio);
    if !finite || (got as f64) < lo || (got as f64) > hi {
        if st.errors < 10 {
            println!("Error at (row {}, col {}): ref={:.9}, result={:.9}, allowed=[{:.9},{:.9}]", where_.0, where_.1, expected, got, lo, hi);
        }
        st.errors += 1;
    }
}

// ---------------------------------------------------------------------------------------------
// kernel_module

#[derive(Clone, Copy, PartialEq)]
enum kernel_style {
    rust,
    opus,
}

struct kernel_module {
    label: String,
    style: kernel_style,
    module: handle,
}

impl kernel_module {
    fn load(spec: &str) -> kernel_module {
        let (style, path) = match spec.split_once(':') {
            Some(("rust", p)) => (kernel_style::rust, p),
            Some(("opus", p)) => (kernel_style::opus, p),
            _ => panic!("--co expects rust:<path> or opus:<path>, got {spec}"),
        };
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let image: Vec<u64> = bytes.chunks(8).map(|c| {
            let mut b = [0u8; 8];
            b[..c.len()].copy_from_slice(c);
            u64::from_le_bytes(b)
        }).collect();
        let mut module = null_mut();
        check!(hipModuleLoadData(&mut module, image.as_ptr() as *const c_void));
        let label = format!("{}:{}", if style == kernel_style::rust { "rust" } else { "opus" },
                            std::path::Path::new(path).file_name().unwrap().to_string_lossy());
        kernel_module { label, style, module }
    }

    fn function(&self, tiles: i32, bf16: bool) -> (handle, u32) {
        let (name, lds) = match self.style {
            kernel_style::rust => (format!("gemm_mxfp8_bpreshuffle_{}_t{}", if bf16 { "bf16" } else { "fp32" }, tiles),
                            layout::LDS_BYTES as u32),
            kernel_style::opus => (format!("_Z28gemm_a8w8_mxfp8_scale_kernelI28gemm_a8w8_mxfp8_scale_traitsILi256ELi256ELi128ELi1ELi128ELi128ELi{}ELb{}EEEv21opus_gemm_scale_kargs",
                                    tiles, bf16 as i32), 0),
        };
        let mut f = null_mut();
        let cname = CString::new(name).unwrap();
        check!(hipModuleGetFunction(&mut f, self.module, cname.as_ptr()));
        (f, lds)
    }
}

fn launch(f: handle, lds: u32, grid: (u32, u32), kargs: &opus_gemm_scale_kargs) {
    let mut args: [*mut c_void; 1] = [kargs as *const opus_gemm_scale_kargs as *mut c_void];
    check!(hipModuleLaunchKernel(f, grid.0, 1, grid.1, layout::BLOCK_SIZE as u32, 1, 1, lds, null_mut(),
                                 args.as_mut_ptr(), null_mut()));
}

fn time_ms(f: handle, lds: u32, grid: (u32, u32), kargs: &opus_gemm_scale_kargs, warmup: usize, iters: usize) -> f64 {
    for _ in 0..warmup {
        launch(f, lds, grid, kargs);
    }
    let (mut a, mut b) = (null_mut(), null_mut());
    check!(hipEventCreate(&mut a));
    check!(hipEventCreate(&mut b));
    check!(hipDeviceSynchronize());
    check!(hipEventRecord(a, null_mut()));
    for _ in 0..iters {
        launch(f, lds, grid, kargs);
    }
    check!(hipEventRecord(b, null_mut()));
    check!(hipEventSynchronize(b));
    let mut ms = 0f32;
    check!(hipEventElapsedTime(&mut ms, a, b));
    check!(hipEventDestroy(a));
    check!(hipEventDestroy(b));
    ms as f64 / iters as f64
}

// ---------------------------------------------------------------------------------------------

struct options {
    m: i32,
    n: i32,
    k: i32,
    batch: i32,
    verify: i32,
    warmup: usize,
    iters: usize,
    rounds: usize,
    bf16: bool,
    tiles: i32,
    seed: i32,
    samples: usize,
    co: Vec<String>,
}

fn parse_options() -> options {
    let mut o = options { m: 8192, n: 8192, k: 8192, batch: 1, verify: 1, warmup: 200, iters: 100, rounds: 1,
                          bf16: true, tiles: 0, seed: 1, samples: 65536, co: vec![] };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let (name, inline) = match args[i].split_once('=') {
            Some((n, v)) if n.starts_with("--") => (n.to_string(), Some(v.to_string())),
            _ => (args[i].clone(), None),
        };
        let mut value = || inline.clone().unwrap_or_else(|| { i += 1; args[i].clone() });
        match name.as_str() {
            "-m" => o.m = value().parse().unwrap(),
            "-n" => o.n = value().parse().unwrap(),
            "-k" => o.k = value().parse().unwrap(),
            "-b" | "--batch" => o.batch = value().parse().unwrap(),
            "-v" | "--verify" => o.verify = value().parse().unwrap(),
            "-w" | "--warmup" => o.warmup = value().parse().unwrap(),
            "-i" | "--iterations" => o.iters = value().parse().unwrap(),
            "--rounds" => o.rounds = value().parse().unwrap(),
            "--dtype" => o.bf16 = match value().as_str() { "bf16" => true, "fp32" => false, d => panic!("--dtype {d}") },
            "--tiles" => o.tiles = value().parse().unwrap(),
            "--seed" => o.seed = value().parse().unwrap(),
            "--samples" => o.samples = value().parse().unwrap(),
            "--co" => o.co.push(value()),
            "--check-layouts" => { check_layouts(&args[i + 1..]); std::process::exit(0); }
            _ => {
                eprintln!("usage: [-m M] [-n N] [-k K] [-b BATCH] [-v 0|1|2] [-w WARMUP] [-i ITERS] [--rounds R]\n\
                           \x20      [--dtype bf16|fp32] [--tiles 0|1|4] [--seed S] [--samples N] --co rust:<co>|opus:<co> ...\n\
                           \x20 -v 1 full CPU reference, -v 2 sampled (--samples outputs per batch)\n\
                           \x20 --check-layouts <dump.bin> <stride_a> <stride_b> <stride_c> <stride_sfb>");
                std::process::exit(1);
            }
        }
        i += 1;
    }
    assert!(o.m % 256 == 0 && o.n % 256 == 0 && o.k % 128 == 0, "M/N must be multiples of 256, K of 128");
    assert!(matches!(o.tiles, 0 | 1 | 4), "--tiles must be 0, 1 or 4");
    if o.co.is_empty() {
        let exe = std::env::current_exe().unwrap();
        let co = exe.parent().unwrap().join("mxfp8_gemm_rust.co");
        o.co.push(format!("rust:{}", co.display()));
    }
    o
}

/// Compare kernel/src/layout.rs against offsets dumped from the opus layouts (oracle/layout_dump.cc).
fn check_layouts(args: &[String]) {
    use layout::*;
    let bytes = std::fs::read(&args[0]).unwrap();
    let dump: Vec<i32> = bytes.chunks(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
    let p: Vec<i32> = args[1..5].iter().map(|s| s.parse().unwrap()).collect();
    let (sa_, sb_, sc_, ssfb) = (p[0], p[1], p[2], p[3]);
    let mut mismatches = 0;
    for tid in 0..BLOCK_SIZE {
        let (wave, lane) = (tid / WARP_SIZE, tid % WARP_SIZE);
        let (wm, wn) = (wave % T_M, wave / T_M);
        let mut v: Vec<i32> = vec![];
        v.extend(ga(lane, wm, wn, sa_));
        v.extend(sa(wm, wn));
        v.extend(ra(lane, wm));
        v.extend(gb(lane, wm, wn, sb_));
        v.extend(gb(lane, wm, 0, sb_));
        v.extend(gb(lane, wm, 1, sb_));
        v.extend(sa(wm, wn));
        v.extend(sa(wm, 0));
        v.extend(sa(wm, 1));
        v.extend(rb(lane, wn));
        v.push(rsfa(lane, wm));
        v.push(rsfb(lane, wn));
        v.push(gsf(lane, wm, true, ssfb));
        v.push(gsf(lane, wm, false, ssfb));
        v.push(ssf(lane, wm));
        v.extend(gc(lane, wm, wn, sc_));
        v.extend([32, 32, 4, 16]);
        let want = &dump[(tid * 64) as usize..(tid * 64) as usize + v.len()];
        if want != v.as_slice() {
            if mismatches < 5 {
                println!("tid {tid}: opus {want:?}\n        rust {v:?}");
            }
            mismatches += 1;
        }
    }
    println!("check-layouts {}: {} / {} threads mismatch", args[0], mismatches, BLOCK_SIZE);
    if mismatches != 0 {
        std::process::exit(2);
    }
}

fn main() {
    let o = parse_options();
    check!(hipSetDevice(0));
    let (m, n, k, batch) = (o.m as usize, o.n as usize, o.k as usize, o.batch as usize);
    let groups_k = k / 128;
    let groups_n = n / 128;
    let (a_batch, b_batch, c_batch) = (m * k, n * k, m * n);
    let (sfa_batch, sfb_batch) = (m * groups_k, groups_n * groups_k);
    for (what, v) in [("A", batch * a_batch), ("B", batch * b_batch), ("C bytes", batch * c_batch * 4)] {
        assert!(v <= i32::MAX as usize, "{what} exceeds the signed 32-bit indexing limit");
    }
    let c_elem = if o.bf16 { 2 } else { 4 };

    let seed = o.seed as u64;
    let mut a = vec![0u8; batch * a_batch];
    let mut b = vec![0u8; batch * b_batch];
    let mut sfa = vec![0u8; batch * sfa_batch];
    let mut sfb = vec![0u8; batch * sfb_batch];
    fill_fp8(&mut a, seed);
    fill_fp8(&mut b, seed ^ 0x3141_5926_5358_9793);
    fill_scales(&mut sfa, seed ^ 0x2718_2818_2845_9045);
    fill_scales(&mut sfb, seed ^ 0x6a09_e667_f3bc_c909);
    let b_packed = pack_weight_16x16(&b, batch, n, k);

    let dev = |bytes: usize| { let mut p = null_mut(); check!(hipMalloc(&mut p, bytes)); p };
    let (d_a, d_b, d_c, d_sfa, d_sfb) = (dev(a.len()), dev(b.len()), dev(batch * c_batch * c_elem), dev(sfa.len()), dev(sfb.len()));
    check!(hipMemcpy(d_a, a.as_ptr() as _, a.len(), H2D));
    check!(hipMemcpy(d_b, b_packed.as_ptr() as _, b.len(), H2D));
    check!(hipMemcpy(d_sfa, sfa.as_ptr() as _, sfa.len(), H2D));
    check!(hipMemcpy(d_sfb, sfb.as_ptr() as _, sfb.len(), H2D));
    drop(b_packed);

    let kargs = opus_gemm_scale_kargs {
        ptr_a: d_a, ptr_b: d_b, ptr_c: d_c, m: o.m, n: o.n, k: o.k, batch: o.batch,
        stride_a: o.k, stride_b: o.k, stride_c: o.n,
        stride_a_batch: a_batch as i32, stride_b_batch: b_batch as i32, stride_c_batch: c_batch as i32,
        ptr_sfa: d_sfa, ptr_sfb: d_sfb, stride_sfa: o.m, stride_sfb: groups_k as i32,
        stride_sfa_batch: sfa_batch as i32, stride_sfb_batch: sfb_batch as i32,
    };

    let (m_tiles, n_tiles) = (o.m / 256, o.n / 256);
    let tiles = if o.tiles != 0 { o.tiles } else {
        let (mut d, mut cus) = (0, 0);
        check!(hipGetDevice(&mut d));
        check!(hipDeviceGetAttribute(&mut cus, HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, d));
        if ((m_tiles + 3) / 4 * n_tiles * o.batch) >= cus { 4 } else { 1 }
    };
    let grid = (((m_tiles + tiles - 1) / tiles * n_tiles) as u32, o.batch as u32);
    println!("M={} N={} K={} batch={} dtype={} seed={} grid=({},1,{}) block={} output_tiles_per_wg={}{}",
             o.m, o.n, o.k, o.batch, if o.bf16 { "bf16" } else { "fp32" }, o.seed, grid.0, grid.1,
             layout::BLOCK_SIZE, tiles, if o.tiles == 0 { " (auto)" } else { " (forced)" });

    let kernels: Vec<kernel_module> = o.co.iter().map(|s| kernel_module::load(s)).collect();
    let mut all_valid = true;
    let mut out_bytes = vec![0u8; batch * c_batch * c_elem];
    for kn in &kernels {
        if o.verify == 0 {
            continue;
        }
        let (f, lds) = kn.function(tiles, o.bf16);
        // NaN poison makes a missing output store a deterministic failure.
        check!(hipMemset(d_c, 0xff, out_bytes.len()));
        launch(f, lds, grid, &kargs);
        check!(hipDeviceSynchronize());
        check!(hipMemcpy(out_bytes.as_mut_ptr() as _, d_c, out_bytes.len(), D2H));
        let got = |i: usize| if o.bf16 {
            bf16_to_f32(u16::from_le_bytes([out_bytes[2 * i], out_bytes[2 * i + 1]]))
        } else {
            f32::from_le_bytes(out_bytes[4 * i..4 * i + 4].try_into().unwrap())
        };
        for bi in 0..batch {
            let p = problem { a: &a[bi * a_batch..], b: &b[bi * b_batch..], sfa: &sfa[bi * sfa_batch..],
                              sfb: &sfb[bi * sfb_batch..], m, k };
            let points: Vec<(usize, usize)> = if o.verify == 1 {
                (0..m * n).map(|i| (i / n, i % n)).collect()
            } else {
                (0..o.samples).map(|i| { let h = mix_bits(0x5eed_0000 + i as u64); ((h % m as u64) as usize, ((h >> 32) % n as u64) as usize) }).collect()
            };
            let threads = std::thread::available_parallelism().map_or(16, |x| x.get());
            let chunk = points.len().div_ceil(threads).max(1);
            let stats: Vec<check_stats> = std::thread::scope(|s| {
                let hs: Vec<_> = points.chunks(chunk).map(|pts| {
                    let p = &p;
                    let got = &got;
                    s.spawn(move || {
                        let mut st = check_stats::default();
                        for &(r, c) in pts {
                            let (rf, mag) = p.reference(r, c);
                            check_one(&mut st, rf, mag, got(bi * c_batch + r * n + c), o.bf16, (r, c));
                        }
                        st
                    })
                }).collect();
                hs.into_iter().map(|h| h.join().unwrap()).collect()
            });
            let (checked, errors) = (stats.iter().map(|s| s.checked).sum::<usize>(), stats.iter().map(|s| s.errors).sum::<usize>());
            let max_ratio = stats.iter().map(|s| s.max_ratio).fold(0.0, f64::max);
            let max_diff = stats.iter().map(|s| s.max_diff).fold(0.0, f64::max);
            println!("[{} batch {}/{}] {}: errors={}/{} ({}), max_diff={:.3e}, max_ratio={:.3}",
                     kn.label, bi + 1, batch, if errors == 0 { "VALID" } else { "FAIL" }, errors, checked,
                     if o.verify == 1 { "full" } else { "sampled" }, max_diff, max_ratio);
            all_valid &= errors == 0;
        }
    }
    if !all_valid {
        println!("[Overall] SOME CHECKS FAILED");
        std::process::exit(2);
    }
    if o.iters == 0 {
        return;
    }

    let flop = 2.0 * o.m as f64 * o.n as f64 * o.k as f64 * o.batch as f64;
    let mut times: Vec<Vec<f64>> = vec![vec![]; kernels.len()];
    for r in 0..o.rounds {
        for (ki, kn) in kernels.iter().enumerate() {
            let (f, lds) = kn.function(tiles, o.bf16);
            let ms = time_ms(f, lds, grid, &kargs, o.warmup, o.iters);
            times[ki].push(ms);
            println!("round {r} {:<40} avg_time={:.4} ms, {:.2} TFlops", kn.label, ms, flop / 1e9 / ms);
        }
    }
    if o.rounds > 1 {
        for (ki, kn) in kernels.iter().enumerate() {
            let mut t = times[ki].clone();
            t.sort_by(f64::total_cmp);
            let med = t[t.len() / 2];
            println!("median {:<39} avg_time={:.4} ms, {:.2} TFlops", kn.label, med, flop / 1e9 / med);
        }
    }
    for p in [d_a, d_b, d_c, d_sfa, d_sfb] {
        check!(hipFree(p));
    }
}
