OPUS_INCLUDE_DIR ?= /root/workspace/aiter/csrc/include
ARCH            ?= gfx950
ROCM_PATH       ?= /opt/rocm

# This kernel needs a compiler that honours the `#pragma unroll 8` on the main
# K loop.  clang 20/21/22 report "loop not unrolled" and emit 64 MFMA; clang 23
# unrolls this variant to 320 MFMA per specialization.  Point CLANG23_ROOT at
# an LLVM build of that vintage, or override HIPCC to use the ROCm compiler and
# accept the slower code.
CLANG23_ROOT    ?= /root/workspace/llvm-src/build
HIPCC           ?= $(CLANG23_ROOT)/bin/clang++

# The standalone LLVM build has no OpenMP of its own; borrow ROCm's.
OMP_INC         ?= $(ROCM_PATH)/lib/llvm/include
OMP_LIB         ?= $(ROCM_PATH)/lib/llvm/lib

COMMON_FLAGS = -I. -I$(OPUS_INCLUDE_DIR) -std=c++17 -O3 -Wall -ffast-math -fPIC \
               --offload-arch=$(ARCH) --rocm-path=$(ROCM_PATH)
LDFLAGS      = -fopenmp -L$(OMP_LIB) -lomp -Wl,-rpath,$(OMP_LIB) \
               -L$(ROCM_PATH)/lib -lamdhip64 -Wl,-rpath,$(ROCM_PATH)/lib

BUILD = build
OUT   = $(BUILD)/gemm_a8w8_blockscale_bpreshuffle.exe

KERNEL_SRC  = gemm_a8w8_mxfp8_scale_kernel.cc
HOST_SRC    = gemm_a8w8_mxfp8_scale_host.cc
KERNEL_OBJ  = $(BUILD)/gemm_a8w8_mxfp8_scale_kernel.o
HOST_OBJ    = $(BUILD)/gemm_a8w8_mxfp8_scale_host.o

HEADERS = gemm_a8w8_mxfp8_scale_common.h \
          gemm_a8w8_mxfp8_scale_kernel_template.hpp

# Optional card pin: `make benchmark GPU=2`.
ifdef GPU
RUN_ENV = HIP_VISIBLE_DEVICES=$(GPU)
endif
RUN = $(RUN_ENV) LD_LIBRARY_PATH=$(OMP_LIB):$$LD_LIBRARY_PATH $(OUT)

SHAPE ?= -m 8192 -n 8192 -k 8192 -b 1
VERIFY_SHAPE ?= -m 1280 -n 512 -k 1152 -b 1

.PHONY: all benchmark verify check regs clean

all: $(OUT) $(BUILD)/libblockscale_bpreshuffle.so

$(BUILD):
	mkdir -p $(BUILD)

$(KERNEL_OBJ): $(KERNEL_SRC) $(HEADERS) Makefile | $(BUILD)
	@test -x $(HIPCC) || { echo "compiler not found at $(HIPCC) -- set CLANG23_ROOT or HIPCC"; exit 1; }
	$(HIPCC) -x hip $< $(COMMON_FLAGS) -D__HIPCC_RTC__ -c -o $@

$(HOST_OBJ): $(HOST_SRC) gemm_a8w8_mxfp8_scale_common.h Makefile | $(BUILD)
	@test -x $(HIPCC) || { echo "compiler not found at $(HIPCC) -- set CLANG23_ROOT or HIPCC"; exit 1; }
	$(HIPCC) -x hip $< $(COMMON_FLAGS) -fno-finite-math-only -fopenmp -I$(OMP_INC) -c -o $@

$(OUT): $(KERNEL_OBJ) $(HOST_OBJ)
	$(HIPCC) $^ --offload-arch=$(ARCH) --rocm-path=$(ROCM_PATH) $(LDFLAGS) -o $@

$(BUILD)/launch.o: gemm_a8w8_blockscale_bpreshuffle_launch.cc gemm_a8w8_mxfp8_scale_common.h Makefile | $(BUILD)
	$(HIPCC) -x hip $< $(COMMON_FLAGS) -c -o $@

$(BUILD)/libblockscale_bpreshuffle.so: $(KERNEL_OBJ) $(BUILD)/launch.o
	$(HIPCC) $^ -shared --offload-arch=$(ARCH) --rocm-path=$(ROCM_PATH) \
	    -L$(ROCM_PATH)/lib -lamdhip64 -Wl,-rpath,$(ROCM_PATH)/lib -o $@

# Inspect the linked executable without rewriting it. Separate temporary
# directories also allow `make -j check regs` to run without shared files.
define UNBUNDLE
set -eu; \
inspect_dir=$$(mktemp -d "$(BUILD)/inspect.XXXXXX"); \
trap 'rm -rf "$$inspect_dir"' EXIT HUP INT TERM; \
$(ROCM_PATH)/llvm/bin/llvm-objcopy \
   --dump-section=.hip_fatbin="$$inspect_dir/fatbin" "$(OUT)" "$$inspect_dir/copy.exe"; \
$(ROCM_PATH)/llvm/bin/clang-offload-bundler -type=o \
   -targets=hipv4-amdgcn-amd-amdhsa--$(ARCH) -input="$$inspect_dir/fatbin" \
   -output="$$inspect_dir/device" -unbundle
endef

# Check that the K loop unrolled for all four instantiations: FP32/BF16,
# each with 1 or 4 output tiles per workgroup (320 MFMA per specialization).
MFMA_MIN ?= 1280

check: $(OUT)
	@$(UNBUNDLE); \
	$(ROCM_PATH)/llvm/bin/llvm-objdump -d "$$inspect_dir/device" > "$$inspect_dir/isa"; \
	n=$$(awk '/v_mfma_scale_f32_16x16x128/ {n++} END {print n+0}' "$$inspect_dir/isa"); \
	p=$$(awk '/s_setprio/ {n++} END {print n+0}' "$$inspect_dir/isa"); \
	if [ "$$n" -ge $(MFMA_MIN) ] && [ "$$p" -eq 0 ]; then \
	    echo "check: OK -- $$n MFMA, 0 s_setprio"; \
	else \
	    echo "check: FAIL -- $$n MFMA (expected >=$(MFMA_MIN)), $$p s_setprio (expected 0)"; exit 1; \
	fi

# Occupancy inputs, one metadata block per specialization.
regs: $(OUT)
	@$(UNBUNDLE); \
	$(ROCM_PATH)/llvm/bin/llvm-readelf --notes "$$inspect_dir/device" | \
	    grep -iE "vgpr_count|sgpr_count|spill_count|group_segment_fixed_size|private_segment_fixed_size"

benchmark: check
	$(RUN) $(SHAPE) -v 0 -w 200 -i 100

verify: check
	$(RUN) $(VERIFY_SHAPE) -v 1 -w 0 -i 1 --tiles 4

clean:
	rm -rf $(BUILD)
