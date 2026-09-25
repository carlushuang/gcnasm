// Rust host for the Rust-written AMDGPU kernels in ../kernel.
// Mirrors gcnasm/bandwidth_memread/bandwidth_kernel.cu (persistent launch, float4, UNROLL=4, OCCUPANCY=1).
use std::ffi::{c_char, c_int, c_void, CStr};
use std::ptr::null_mut;

type HipError = c_int;
type HipModule = *mut c_void;
type HipFunction = *mut c_void;
type HipEvent = *mut c_void;
type HipStream = *mut c_void;

const HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 63;
const HIP_MEMCPY_HOST_TO_DEVICE: c_int = 1;

unsafe extern "C" {
    fn hipSetDevice(dev: c_int) -> HipError;
    fn hipGetDevice(dev: *mut c_int) -> HipError;
    fn hipDeviceGetAttribute(v: *mut c_int, attr: c_int, dev: c_int) -> HipError;
    fn hipGetErrorString(e: HipError) -> *const c_char;
    fn hipMalloc(p: *mut *mut c_void, size: usize) -> HipError;
    fn hipFree(p: *mut c_void) -> HipError;
    fn hipMemcpy(dst: *mut c_void, src: *const c_void, size: usize, kind: c_int) -> HipError;
    fn hipModuleLoadData(m: *mut HipModule, image: *const c_void) -> HipError;
    fn hipModuleGetFunction(f: *mut HipFunction, m: HipModule, name: *const c_char) -> HipError;
    fn hipModuleLaunchKernel(
        f: HipFunction,
        gx: u32, gy: u32, gz: u32,
        bx: u32, by: u32, bz: u32,
        shared: u32,
        stream: HipStream,
        params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> HipError;
    fn hipEventCreate(e: *mut HipEvent) -> HipError;
    fn hipEventDestroy(e: HipEvent) -> HipError;
    fn hipEventRecord(e: HipEvent, s: HipStream) -> HipError;
    fn hipEventSynchronize(e: HipEvent) -> HipError;
    fn hipEventElapsedTime(ms: *mut f32, a: HipEvent, b: HipEvent) -> HipError;
}

macro_rules! call {
    ($e:expr) => {{
        let err = unsafe { $e };
        if err != 0 {
            let s = unsafe { CStr::from_ptr(hipGetErrorString(err)) };
            eprintln!("'{}'({}) at {}:{}", s.to_string_lossy(), err, file!(), line!());
            std::process::exit(1);
        }
    }};
}

// code object produced by `cargo build --release` in ../kernel; 8-byte aligned copy for the loader
static KERNEL_ELF: &[u8] = include_bytes!("../../kernel/target/amdgcn-amd-amdhsa/release/bw_kernel.elf");

const BLOCK_SIZE: i64 = 1024;
const UNROLL: i64 = 4;
const OCCUPANCY: i64 = 1;
const WARMUP: usize = 25;
const LOOP: usize = 100;

struct Kernels {
    memread: HipFunction,
    memread_plain: HipFunction,
    memcpy: HipFunction,
}

fn load_kernels() -> Kernels {
    let image: Vec<u64> = KERNEL_ELF
        .chunks(8)
        .map(|c| {
            let mut b = [0u8; 8];
            b[..c.len()].copy_from_slice(c);
            u64::from_le_bytes(b)
        })
        .collect();
    let mut m: HipModule = null_mut();
    call!(hipModuleLoadData(&mut m, image.as_ptr() as *const c_void));
    let get = |name: &CStr| {
        let mut f: HipFunction = null_mut();
        call!(hipModuleGetFunction(&mut f, m, name.as_ptr()));
        f
    };
    Kernels {
        memread: get(c"memread_kernel"),
        memread_plain: get(c"memread_plain_kernel"),
        memcpy: get(c"memcpy_kernel"),
    }
}

fn num_cu() -> i64 {
    let mut dev = 0;
    let mut cu = 0;
    call!(hipGetDevice(&mut dev));
    call!(hipDeviceGetAttribute(&mut cu, HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
    cu as i64
}

fn bench_kernel(f: HipFunction, src: *mut c_void, dst: *mut c_void, dwords: i64) -> f32 {
    let pixels = dwords * 4 / 16;
    let gx = num_cu() * OCCUPANCY;
    let mut src = src;
    let mut dst = dst;
    let mut issues_per_block = (pixels / gx) as i32;
    let mut iters = (issues_per_block as i64 / BLOCK_SIZE / UNROLL) as i32;
    let mut args: [*mut c_void; 4] = [
        &mut src as *mut _ as *mut c_void,
        &mut dst as *mut _ as *mut c_void,
        &mut issues_per_block as *mut _ as *mut c_void,
        &mut iters as *mut _ as *mut c_void,
    ];
    let run = |count: usize, args: &mut [*mut c_void; 4]| {
        for _ in 0..count {
            call!(hipModuleLaunchKernel(
                f, gx as u32, 1, 1, BLOCK_SIZE as u32, 1, 1, 0, null_mut(),
                args.as_mut_ptr(), null_mut()
            ));
        }
    };
    run(WARMUP, &mut args);
    let (mut a, mut b): (HipEvent, HipEvent) = (null_mut(), null_mut());
    call!(hipEventCreate(&mut a));
    call!(hipEventCreate(&mut b));
    call!(hipEventRecord(a, null_mut()));
    run(LOOP, &mut args);
    call!(hipEventRecord(b, null_mut()));
    call!(hipEventSynchronize(b));
    let mut ms = 0f32;
    call!(hipEventElapsedTime(&mut ms, a, b));
    call!(hipEventDestroy(a));
    call!(hipEventDestroy(b));
    ms / LOOP as f32
}

fn b2s(bytes: u64) -> String {
    let b = bytes as f64;
    if bytes < 1024 {
        format!("{bytes}B")
    } else if bytes < 1024 * 1024 {
        format!("{:.2}KB", b / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.2}MB", b / (1024.0 * 1024.0))
    } else {
        format!("{:.2}GB", b / (1024.0 * 1024.0 * 1024.0))
    }
}

fn run(f: HipFunction, tag: &str, rw: bool, dwords: i64) {
    let bytes = dwords as usize * 4;
    let h_a: Vec<f32> = (0..dwords).map(|i| ((i * 7919) % 1000 + 1) as f32 / 1000.0).collect();
    let (mut a, mut b): (*mut c_void, *mut c_void) = (null_mut(), null_mut());
    call!(hipMalloc(&mut a, bytes));
    call!(hipMalloc(&mut b, bytes));
    call!(hipMemcpy(a, h_a.as_ptr() as *const c_void, bytes, HIP_MEMCPY_HOST_TO_DEVICE));
    // same argument order as the .cu: kernel(B, A, ...) reads from B, writes to A
    let ms = bench_kernel(f, b, a, dwords);
    let moved = bytes as f64 * if rw { 2.0 } else { 1.0 };
    println!("{:>9}({}) -> {:.4}ms, {:.3}(GB/s)", b2s(bytes as u64), tag, ms, moved / (ms as f64 / 1e3) / 1e9);
    call!(hipFree(a));
    call!(hipFree(b));
}

fn launch_once(f: HipFunction, src: *mut c_void, dst: *mut c_void, issues_per_block: i32, iters: i32, gx: u32) {
    let (mut src, mut dst, mut ipb, mut it) = (src, dst, issues_per_block, iters);
    let mut args: [*mut c_void; 4] = [
        &mut src as *mut _ as *mut c_void,
        &mut dst as *mut _ as *mut c_void,
        &mut ipb as *mut _ as *mut c_void,
        &mut it as *mut _ as *mut c_void,
    ];
    call!(hipModuleLaunchKernel(f, gx, 1, 1, BLOCK_SIZE as u32, 1, 1, 0, null_mut(), args.as_mut_ptr(), null_mut()));
}

// memcpy: dst == src. memread: every lane sums iters*UNROLL = 8 copies of 1250.0 == 10000.0,
// which trips the kernel's "magic" dead store, proving loads + accumulate + store all ran.
fn validate(k: &Kernels) {
    const HIP_MEMCPY_DEVICE_TO_HOST: c_int = 2;
    let gx = num_cu() * OCCUPANCY;
    let iters = 2i64;
    let issues = BLOCK_SIZE * UNROLL * iters;
    let dwords = (gx * issues * 4) as usize;
    let bytes = dwords * 4;
    let (mut a, mut b): (*mut c_void, *mut c_void) = (null_mut(), null_mut());
    call!(hipMalloc(&mut a, bytes));
    call!(hipMalloc(&mut b, bytes));

    let h: Vec<f32> = (0..dwords).map(|i| i as f32).collect();
    let zero = vec![0f32; dwords];
    let mut out = vec![0f32; dwords];
    call!(hipMemcpy(a, h.as_ptr() as *const c_void, bytes, HIP_MEMCPY_HOST_TO_DEVICE));
    call!(hipMemcpy(b, zero.as_ptr() as *const c_void, bytes, HIP_MEMCPY_HOST_TO_DEVICE));
    launch_once(k.memcpy, a, b, issues as i32, iters as i32, gx as u32);
    call!(hipMemcpy(out.as_mut_ptr() as *mut c_void, b, bytes, HIP_MEMCPY_DEVICE_TO_HOST));
    let bad = out.iter().zip(&h).filter(|(x, y)| x != y).count();
    println!("validate memcpy_kernel: {} ({bad} mismatches / {dwords})", if bad == 0 { "PASS" } else { "FAIL" });

    for (f, name) in [(k.memread, "memread_kernel"), (k.memread_plain, "memread_plain_kernel")] {
        let h: Vec<f32> = vec![1250.0; dwords];
        call!(hipMemcpy(a, h.as_ptr() as *const c_void, bytes, HIP_MEMCPY_HOST_TO_DEVICE));
        call!(hipMemcpy(b, zero.as_ptr() as *const c_void, bytes, HIP_MEMCPY_HOST_TO_DEVICE));
        launch_once(f, a, b, issues as i32, iters as i32, gx as u32);
        call!(hipMemcpy(out.as_mut_ptr() as *mut c_void, b, 16, HIP_MEMCPY_DEVICE_TO_HOST));
        let ok = out[..4].iter().all(|&x| x == 10000.0);
        println!("validate {name}: {} (dst[0] = {:?})", if ok { "PASS" } else { "FAIL" }, &out[..4]);
    }
    call!(hipFree(a));
    call!(hipFree(b));
}

fn env_int(name: &str, def: i64) -> i64 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(def)
}

fn main() {
    call!(hipSetDevice(0));
    let k = load_kernels();
    validate(&k);
    let cases: [(HipFunction, &str, bool); 3] = [
        (k.memread, "[ro]", false),
        (k.memread_plain, "[ro-plain]", false),
        (k.memcpy, "[rw]", true),
    ];

    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 {
        let dwords: i64 = args[1].parse().expect("dwords");
        for (f, tag, rw) in cases {
            run(f, tag, rw, dwords);
        }
        return;
    }

    let mut cu = num_cu();
    let btc = env_int("BANDWIDTH_TEST_CASE", 0);
    println!("cu:{cu}, rust kernel ({btc})");
    cu = if btc == 0 { cu } else if btc == -1 { 304 } else { btc };
    let list: Vec<i64> = [0i64, 0, 0, 116, 212, 476, 820, 1024, 1638, 3276, 5710]
        .iter()
        .enumerate()
        .map(|(i, &m)| match i {
            0 => 20000,
            1 => 400000,
            2 => 16711680,
            _ => m * cu * BLOCK_SIZE,
        })
        .collect();
    for (f, tag, rw) in cases {
        println!("---------------------------------------------");
        for &d in &list {
            run(f, tag, rw, d);
            std::thread::sleep(std::time::Duration::from_millis(200));
        }
    }
}
