# Rust port of ../mxfp8_gemm_16x16x128_blockscale_bpreshuffle: kernels and host in Rust.
ARCH             ?= gfx950
ROCM_PATH        ?= /opt/rocm
CARGO            ?= cargo
# Only for the C++/opus reference kernel (make opus) and the layout oracle (make check-layouts).
CLANG23_ROOT     ?= /root/workspace/llvm-src/build
OPUS_INCLUDE_DIR ?= /root/workspace/aiter/csrc/include

CPP_DIR   = ../mxfp8_gemm_16x16x128_blockscale_bpreshuffle
BUILD     = build
CO        = $(BUILD)/mxfp8_gemm_rust.co
EXE       = $(BUILD)/gemm_mxfp8_bpreshuffle
OPUS_CO   = $(BUILD)/opus_clang23.co
KATTRS    = --attr amdgpu-flat-work-group-size=1,512 --attr amdgpu-waves-per-eu=1 --attr uniform-work-group-size
LLVM_BIN ?= $(HOME)/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/bin
RUN       = $(if $(GPU),HIP_VISIBLE_DEVICES=$(GPU)) $(EXE)
SHAPE    ?= -m 8192 -n 8192 -k 8192 -b 1
# upstream clang 23.1.1 already types the v2f16 buffer atomic builtin as _Float16 (opus switches at clang 24)
OPUS_FLAGS = -x hip -std=c++17 -O3 -ffast-math --offload-arch=$(ARCH) -nogpuinc -nogpulib -D__HIPCC_RTC__ \
             -DOPUS_HAS_RAW_PTR_ATOMIC_FADD_V2F16_BUILTIN=0 -I$(CPP_DIR) -I$(OPUS_INCLUDE_DIR)

.PHONY: all kernel host opus check check-layouts verify benchmark compare clean

all: kernel host

# rustc emits optimized crate bitcode only (-Clinker-plugin-lto: no machine codegen), then
# tools/relink.py adds launch bounds + IR shims and compiles it with rustc's own opt/llc/lld.
kernel:
	cd kernel && RUSTFLAGS="-Ctarget-cpu=$(ARCH) -Clinker-plugin-lto" $(CARGO) +nightly rustc --release --crate-type lib -- --emit=llvm-bc
	mkdir -p $(BUILD)
	cd kernel && python3 ../tools/relink.py --target-dir target --crate mxfp8_gemm --mcpu $(ARCH) --out ../$(CO) $(KATTRS)

host:
	cd host && ROCM_PATH=$(ROCM_PATH) $(CARGO) build --release
	mkdir -p $(BUILD)
	cp host/target/release/gemm_mxfp8_bpreshuffle $(EXE)

# The C++/opus kernel as a device-only code object, launched by the same Rust host.
opus:
	mkdir -p $(BUILD)
	$(CLANG23_ROOT)/bin/clang++ $(OPUS_FLAGS) --cuda-device-only $(CPP_DIR)/gemm_a8w8_mxfp8_scale_kernel.cc -o $(BUILD)/opus.bundle
	$(CLANG23_ROOT)/bin/clang-offload-bundler --type=o --unbundle --input=$(BUILD)/opus.bundle \
	    --targets=hipv4-amdgcn-amd-amdhsa--$(ARCH) --output=$(OPUS_CO)

# Same bar as the C++ `make check`: 320 scaled MFMA per specialization, plus no spills.
check: kernel
	@n=$$($(LLVM_BIN)/llvm-objdump -d $(CO) | grep -c v_mfma_scale_f32_16x16x128); \
	s=$$($(LLVM_BIN)/llvm-readobj --notes $(CO) | awk '/vgpr_spill_count|sgpr_spill_count/ {s += $$2} END {print s + 0}'); \
	if [ "$$n" -eq 1280 ] && [ "$$s" -eq 0 ]; then echo "check: OK -- $$n MFMA, 0 spills"; \
	else echo "check: FAIL -- $$n MFMA (expected 1280), $$s spills"; exit 1; fi

# kernel/src/layout.rs against the opus layouts, all 512 threads, two stride sets.
check-layouts: host
	$(CLANG23_ROOT)/bin/clang++ $(subst -nogpulib,,$(OPUS_FLAGS)) -nogpulib oracle/layout_dump.cc \
	    -L$(ROCM_PATH)/lib -lamdhip64 -Wl,-rpath,$(ROCM_PATH)/lib -o $(BUILD)/layout_dump
	$(BUILD)/layout_dump 1152 1152 512 9 $(BUILD)/dump_1152.bin
	$(BUILD)/layout_dump 8192 8192 8192 64 $(BUILD)/dump_8192.bin
	$(EXE) --check-layouts $(BUILD)/dump_1152.bin 1152 1152 512 9
	$(EXE) --check-layouts $(BUILD)/dump_8192.bin 8192 8192 8192 64

# The README verification matrix, full CPU reference, both output types.
verify: all
	@for cfg in "256 256 128 1 1" "256 512 256 2 1" "256 512 1152 1 1" "1024 256 128 1 4" "1280 512 256 2 4" "1280 512 1152 1 4"; do \
	    set -- $$cfg; for dt in fp32 bf16; do \
	        $(RUN) -m $$1 -n $$2 -k $$3 -b $$4 -v 1 -i 0 --tiles $$5 --dtype $$dt --co rust:$(CO) | grep -E 'VALID|FAIL' || exit 1; \
	    done; done

benchmark: all
	$(RUN) $(SHAPE) -v 2 -w 200 -i 100 --co rust:$(CO)

# Rust vs C++/opus, interleaved rounds, same host/launch/timing path.
compare: all opus
	for dt in bf16 fp32; do $(RUN) $(SHAPE) -v 2 -w 200 -i 100 --rounds 5 --dtype $$dt --co opus:$(OPUS_CO) --co rust:$(CO); done

clean:
	rm -rf $(BUILD) kernel/target host/target
