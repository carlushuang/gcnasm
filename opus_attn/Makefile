OPUS_INCLUDE_DIR ?= /home/carhuang/repo/aiter/csrc/include
HIPCC           ?= /opt/rocm/bin/hipcc
ARCH            ?= gfx950

COMMON_FLAGS = -I. -I$(OPUS_INCLUDE_DIR) -std=c++20 -fopenmp -O3 -Wall --offload-arch=$(ARCH) -ffast-math

BUILD = build
OUT   = $(BUILD)/gqa_attn.exe

KERNEL_SRCS = gqa_gfx950_kernel_causal.cc gqa_gfx950_kernel_noncausal.cc
HOST_SRC    = gqa_gfx950_host.cc
KERNEL_OBJS = $(patsubst %.cc,$(BUILD)/%.o,$(KERNEL_SRCS))
HOST_OBJ    = $(BUILD)/gqa_host.o

.PHONY: all clean benchmark

all: $(OUT)

$(BUILD):
	mkdir -p $(BUILD)

# Kernel objects (heavy — device+host passes, parallelizable via make -j)
$(BUILD)/%.o: %.cc gqa_gfx950_kernel_template.hpp gqa_common.h | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) -Rpass-analysis=kernel-resource-usage -c -o $@

# Host object
$(HOST_OBJ): $(HOST_SRC) gqa_common.h | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) -c -o $@

# Link
$(OUT): $(KERNEL_OBJS) $(HOST_OBJ)
	$(HIPCC) $^ --offload-arch=$(ARCH) -fopenmp -o $@

benchmark: $(OUT)
	@echo "=== Causal ===" && $(OUT)
	@echo "=== Non-causal ===" && $(OUT) --no-causal

clean:
	rm -rf $(BUILD)
