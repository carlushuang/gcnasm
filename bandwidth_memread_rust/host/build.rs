fn main() {
    let rocm = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".into());
    println!("cargo:rustc-link-search=native={rocm}/lib");
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{rocm}/lib");
    println!("cargo:rerun-if-changed=../kernel/target/amdgcn-amd-amdhsa/release/bw_kernel.elf");
}
