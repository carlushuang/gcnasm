OPUS_INCLUDE_DIR ?= /path/to/aiter/csrc/include
ARCH            ?= gfx950
ROCM_PATH       ?= /opt/rocm

# This kernel needs a compiler that honours the `#pragma unroll 4` on the main
# K loop.  clang 20/21/22 report "loop not unrolled" and emit 64 MFMA; clang 23
# unrolls it to 192 MFMA, which is worth roughly +13%.  Point CLANG23_ROOT at
# an LLVM build of that vintage, or override HIPCC to use the ROCm compiler and
# accept the slower code.
CLANG23_ROOT    ?= /root/workspace/llvm-src/build
HIPCC           ?= $(CLANG23_ROOT)/bin/clang++

# The standalone LLVM build has no OpenMP of its own; borrow ROCm's.
OMP_INC         ?= $(ROCM_PATH)/lib/llvm/include
OMP_LIB         ?= $(ROCM_PATH)/lib/llvm/lib

COMMON_FLAGS = -I. -I$(OPUS_INCLUDE_DIR) -std=c++17 -O3 -Wall -ffast-math \
               --offload-arch=$(ARCH) --rocm-path=$(ROCM_PATH)
LDFLAGS      = -fopenmp -L$(OMP_LIB) -lomp -Wl,-rpath,$(OMP_LIB) \
               -L$(ROCM_PATH)/lib -lamdhip64 -Wl,-rpath,$(ROCM_PATH)/lib

BUILD = build
OUT   = $(BUILD)/gemm_a8w8_mxfp8_scale.exe

KERNEL_SRC  = gemm_a8w8_mxfp8_scale_kernel.cc
HOST_SRC    = gemm_a8w8_mxfp8_scale_host.cc
KERNEL_OBJ  = $(BUILD)/gemm_a8w8_mxfp8_scale_kernel.o
HOST_OBJ    = $(BUILD)/gemm_a8w8_mxfp8_scale_host.o

HEADERS = gemm_a8w8_mxfp8_scale_common.h \
          gemm_a8w8_mxfp8_scale_repack.hpp \
          gemm_a8w8_mxfp8_scale_kernel_template.hpp

# Optional card pin: `make benchmark GPU=2`.
ifdef GPU
RUN_ENV = HIP_VISIBLE_DEVICES=$(GPU)
endif
RUN = $(RUN_ENV) LD_LIBRARY_PATH=$(OMP_LIB):$$LD_LIBRARY_PATH $(OUT)

SHAPE ?= -m 8192 -n 8192 -k 8192 -b 1

.PHONY: all benchmark verify check regs clean

all: $(OUT)

$(BUILD):
	mkdir -p $(BUILD)

$(KERNEL_OBJ): $(KERNEL_SRC) $(HEADERS) | $(BUILD)
	@test -x $(HIPCC) || { echo "compiler not found at $(HIPCC) -- set CLANG23_ROOT or HIPCC"; exit 1; }
	$(HIPCC) -x hip $< $(COMMON_FLAGS) -D__HIPCC_RTC__ -c -o $@

$(HOST_OBJ): $(HOST_SRC) gemm_a8w8_mxfp8_scale_common.h gemm_a8w8_mxfp8_scale_repack.hpp | $(BUILD)
	@test -x $(HIPCC) || { echo "compiler not found at $(HIPCC) -- set CLANG23_ROOT or HIPCC"; exit 1; }
	$(HIPCC) -x hip $< $(COMMON_FLAGS) -fopenmp -I$(OMP_INC) -c -o $@

$(OUT): $(KERNEL_OBJ) $(HOST_OBJ)
	$(HIPCC) $^ --offload-arch=$(ARCH) --rocm-path=$(ROCM_PATH) $(LDFLAGS) -o $@

# Extract the device image from the *linked* executable, never from the .o --
# the linker is what decides the code that actually runs.
define UNBUNDLE
$(ROCM_PATH)/llvm/bin/llvm-objcopy --dump-section=.hip_fatbin=$(BUILD)/.fb $(OUT) 2>/dev/null; \
$(ROCM_PATH)/llvm/bin/clang-offload-bundler -type=o \
   -targets=hipv4-amdgcn-amd-amdhsa--$(ARCH) -input=$(BUILD)/.fb \
   -output=$(BUILD)/.dev -unbundle 2>/dev/null
endef

# The build is only correct if the K loop unrolled (192 MFMA per kernel, and
# the device TU carries two instantiations) and no s_setprio survived.  Both
# have silently regressed before.  Counting 384 rather than 192 matters: with
# the looser bound one un-unrolled instantiation would still pass.
MFMA_MIN ?= 384

check: $(OUT)
	@$(UNBUNDLE); \
	n=$$($(ROCM_PATH)/llvm/bin/llvm-objdump -d $(BUILD)/.dev 2>/dev/null | grep -c v_mfma_scale_f32_16x16x128); \
	p=$$($(ROCM_PATH)/llvm/bin/llvm-objdump -d $(BUILD)/.dev 2>/dev/null | grep -c s_setprio); \
	rm -f $(BUILD)/.fb $(BUILD)/.dev; \
	if [ "$$n" -ge $(MFMA_MIN) ] && [ "$$p" -eq 0 ]; then \
	    echo "check: OK -- $$n MFMA, 0 s_setprio"; \
	else \
	    echo "check: FAIL -- $$n MFMA (expected >=$(MFMA_MIN)), $$p s_setprio (expected 0)"; exit 1; \
	fi

# Occupancy inputs, one block per instantiation.  Expected: VGPR 242 / SGPR 101
# (4 tiles per WG) and VGPR 226 / SGPR 76 (1 tile), 0 spill, LDS 139264 both.
regs: $(OUT)
	@$(UNBUNDLE); \
	$(ROCM_PATH)/llvm/bin/llvm-readelf --notes $(BUILD)/.dev 2>/dev/null | \
	    grep -iE "vgpr_count|sgpr_count|spill_count|group_segment_fixed_size"; \
	rm -f $(BUILD)/.fb $(BUILD)/.dev

benchmark: check
	$(RUN) $(SHAPE) -v 0 -w 200 -i 100

verify: check
	$(RUN) $(SHAPE) -v 1 -w 20 -i 20

clean:
	rm -rf $(BUILD)
