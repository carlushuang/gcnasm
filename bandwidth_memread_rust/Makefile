# Rust port of ../bandwidth_memread: device kernels in kernel/ (amdgcn-amd-amdhsa), host in host/ (HIP runtime FFI)
ARCH ?= gfx950
CARGO ?= cargo

all:
	cd kernel && $(CARGO) +nightly build --release --config 'build.rustflags=["-Ctarget-cpu=$(ARCH)","-Cllvm-args=-unroll-runtime=false"]'
	cd host && $(CARGO) +nightly build --release
	cp host/target/release/bandwidth_kernel bandwidth_kernel.exe

asm: all
	/opt/rocm/llvm/bin/llvm-objdump -d --mcpu=$(ARCH) kernel/target/amdgcn-amd-amdhsa/release/bw_kernel.elf > bw_kernel.$(ARCH).s

clean:
	rm -rf kernel/target host/target bandwidth_kernel.exe bw_kernel.*.s
