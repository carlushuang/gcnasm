ARCH ?= gfx1201
HIPCC ?= /opt/rocm/bin/hipcc
BUILD ?= build
EXE   := $(BUILD)/opus_attn_gfx1201.exe

CXXFLAGS  := -std=c++17 -O3 -Wall -fopenmp --offload-arch=$(ARCH)
DEVFLAGS  := $(CXXFLAGS)
HOSTFLAGS := $(CXXFLAGS)

KERNEL_OBJ := $(BUILD)/attn_gfx1201_kernel.o
HOST_OBJ   := $(BUILD)/attn_gfx1201_host.o

.PHONY: all clean run
all: $(EXE)

$(BUILD):
	mkdir -p $@

$(KERNEL_OBJ): attn_gfx1201_kernel.cc attn_gfx1201_kernel_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(DEVFLAGS) -c $< -o $@

$(HOST_OBJ): attn_gfx1201_host.cc attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(HOSTFLAGS) -c $< -o $@

$(EXE): $(KERNEL_OBJ) $(HOST_OBJ)
	$(HIPCC) $(CXXFLAGS) $^ -o $@

run: $(EXE)
	./$(EXE)

clean:
	rm -rf $(BUILD)
