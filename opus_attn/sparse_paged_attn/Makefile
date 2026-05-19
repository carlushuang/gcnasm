OPUS_INCLUDE_DIR ?= /home/kaiyang1/develop/aiter/csrc/include
HIPCC           ?= /opt/rocm/bin/hipcc
ARCH            ?= gfx950

COMMON_FLAGS = -I. -I$(OPUS_INCLUDE_DIR) -std=c++20 -fopenmp -O3 -Wall --offload-arch=$(ARCH) -ffast-math

BUILD = build
OUT   = $(BUILD)/pa_prefill.exe

KERNEL_SRCS = pa_prefill_bf16_kernel.cc pa_prefill_fp16_kernel.cc
HOST_SRC    = pa_host.cc
KERNEL_OBJS = $(patsubst %.cc,$(BUILD)/%.o,$(KERNEL_SRCS))
HOST_OBJ    = $(BUILD)/pa_host.o

.PHONY: all clean benchmark

all: $(OUT)

$(BUILD):
	mkdir -p $(BUILD)

# Kernel objects (heavy — device+host passes, parallelizable via make -j)
# -D__HIPCC_RTC__ skips implicit __clang_hip_runtime_wrapper.h on host pass (~250ms saved)
# Each kernel .cc only includes its own template hpp; depending on both keeps the rule simple.
$(BUILD)/%.o: %.cc pa_prefill_kernel_template.hpp pa_defs.h | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) -D__HIPCC_RTC__ -Rpass-analysis=kernel-resource-usage -save-temps=obj -c -o $@

# Host object (no -D__HIPCC_RTC__ — needs full HIP runtime for <<<>>> syntax)
$(HOST_OBJ): $(HOST_SRC) pa_defs.h | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) -c -o $@

# Link
$(OUT): $(KERNEL_OBJS) $(HOST_OBJ)
	$(HIPCC) $^ --offload-arch=$(ARCH) -fopenmp -o $@

benchmark: $(OUT)
	@echo "=== d=512 bf16 ===" && $(OUT) -dtype bf16 -d 512
	@echo "=== d=512 fp16 ===" && $(OUT) -dtype fp16 -d 512

clean:
	rm -rf $(BUILD)
