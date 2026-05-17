ARCH ?= gfx1201
HIPCC ?= /opt/rocm/bin/hipcc
BUILD ?= build
EXE   := $(BUILD)/opus_attn_gfx1201.exe

CXXFLAGS := -std=c++17 -O3 -Wall -fopenmp --offload-arch=$(ARCH)

KERNELS  := attn_gfx1201_kernel_v0.cc \
            attn_gfx1201_kernel_v1.cc \
            attn_gfx1201_kernel_v2.cc \
            attn_gfx1201_kernel_v3.cc \
            attn_gfx1201_kernel_v4.cc \
            attn_gfx1201_kernel_v5.cc \
            attn_gfx1201_kernel_v6.cc \
            attn_gfx1201_kernel_v7.cc \
            attn_gfx1201_kernel_v8.cc \
            attn_gfx1201_kernel_v9.cc
KERN_OBJ := $(patsubst %.cc,$(BUILD)/%.o,$(KERNELS))
HOST_OBJ := $(BUILD)/attn_gfx1201_host.o

.PHONY: all clean run
all: $(EXE)

$(BUILD):
	mkdir -p $@

$(BUILD)/attn_gfx1201_kernel_v0.o: attn_gfx1201_kernel_v0.cc attn_gfx1201_kernel_v0_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(BUILD)/attn_gfx1201_kernel_v1.o: attn_gfx1201_kernel_v1.cc attn_gfx1201_kernel_v1_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(BUILD)/attn_gfx1201_kernel_v2.o: attn_gfx1201_kernel_v2.cc attn_gfx1201_kernel_v2_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(BUILD)/attn_gfx1201_kernel_v3.o: attn_gfx1201_kernel_v3.cc attn_gfx1201_kernel_v3_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(BUILD)/attn_gfx1201_kernel_v4.o: attn_gfx1201_kernel_v4.cc attn_gfx1201_kernel_v4_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(BUILD)/attn_gfx1201_kernel_v5.o: attn_gfx1201_kernel_v5.cc attn_gfx1201_kernel_v5_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(BUILD)/attn_gfx1201_kernel_v6.o: attn_gfx1201_kernel_v6.cc attn_gfx1201_kernel_v6_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(BUILD)/attn_gfx1201_kernel_v7.o: attn_gfx1201_kernel_v7.cc attn_gfx1201_kernel_v7_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(BUILD)/attn_gfx1201_kernel_v8.o: attn_gfx1201_kernel_v8.cc attn_gfx1201_kernel_v8_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(BUILD)/attn_gfx1201_kernel_v9.o: attn_gfx1201_kernel_v9.cc attn_gfx1201_kernel_v9_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(HOST_OBJ): attn_gfx1201_host.cc attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(EXE): $(KERN_OBJ) $(HOST_OBJ)
	$(HIPCC) $(CXXFLAGS) $^ -o $@

run: $(EXE)
	./$(EXE)

clean:
	rm -rf $(BUILD)
