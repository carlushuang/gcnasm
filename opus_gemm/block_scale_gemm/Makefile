OPUS_INCLUDE_DIR ?= /path/to/aiter/csrc/include
HIPCC           ?= /opt/rocm/bin/hipcc
ARCH            ?= gfx950

COMMON_FLAGS = -I. -I$(OPUS_INCLUDE_DIR) -std=c++20 -fopenmp -O3 -Wall --offload-arch=$(ARCH) -ffast-math

BUILD = build
OUT   = $(BUILD)/gemm_a8w8_blockscale.exe

KERNEL_SRCS = gemm_a8w8_blockscale_kernel.cc
HOST_SRC    = gemm_a8w8_blockscale_host.cc
KERNEL_OBJS = $(patsubst %.cc,$(BUILD)/%.o,$(KERNEL_SRCS))
HOST_OBJ    = $(BUILD)/gemm_a8w8_blockscale_host.o

.PHONY: all clean benchmark

all: $(OUT)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/%.o: %.cc gemm_a8w8_blockscale_kernel_template.hpp gemm_a8w8_blockscale_common.h | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) -D__HIPCC_RTC__ -Rpass-analysis=kernel-resource-usage -c -o $@

$(HOST_OBJ): $(HOST_SRC) gemm_a8w8_blockscale_common.h | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) -c -o $@

$(OUT): $(KERNEL_OBJS) $(HOST_OBJ)
	$(HIPCC) $^ --offload-arch=$(ARCH) -fopenmp -o $@

benchmark: $(OUT)
	$(OUT)

clean:
	rm -rf $(BUILD)
