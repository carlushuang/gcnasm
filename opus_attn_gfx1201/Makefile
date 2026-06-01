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
            attn_gfx1201_kernel_v9.cc \
            attn_gfx1201_kernel_v10.cc \
            attn_gfx1201_kernel_v11.cc \
            attn_gfx1201_kernel_v12.cc \
            attn_gfx1201_kernel_v13.cc \
            attn_gfx1201_kernel_v14.cc \
            attn_gfx1201_kernel_v15.cc \
            attn_gfx1201_kernel_v16.cc \
            attn_gfx1201_kernel_v17.cc \
            attn_gfx1201_kernel_v18.cc \
            attn_gfx1201_kernel_v19.cc \
            attn_gfx1201_kernel_v20.cc \
            attn_gfx1201_kernel_v21.cc \
            attn_gfx1201_kernel_v22.cc \
            attn_gfx1201_kernel_v23.cc \
            attn_gfx1201_kernel_v24.cc \
            attn_gfx1201_kernel_v25.cc \
            attn_gfx1201_kernel_v26.cc \
            attn_gfx1201_kernel_v27.cc \
            attn_gfx1201_kernel_v28.cc \
            attn_gfx1201_kernel_v29.cc \
            attn_gfx1201_kernel_v30.cc \
            attn_gfx1201_kernel_v31.cc \
            attn_gfx1201_kernel_v32.cc \
            attn_gfx1201_kernel_v33.cc \
            attn_gfx1201_kernel_v34.cc \
    attn_gfx1201_kernel_v35.cc \
    attn_gfx1201_kernel_v36.cc \
    attn_gfx1201_kernel_v37.cc \
    attn_gfx1201_kernel_v38.cc \
    attn_gfx1201_kernel_v39.cc \
    attn_gfx1201_kernel_v40.cc \
    attn_gfx1201_kernel_v41.cc \
    attn_gfx1201_kernel_v42.cc \
    attn_gfx1201_kernel_v43.cc \
    attn_gfx1201_kernel_v44.cc \
    attn_gfx1201_kernel_v45.cc \
    attn_gfx1201_kernel_v48.cc \
            attn_gfx1201_kernel_v139.cc
KERN_OBJ := $(patsubst %.cc,$(BUILD)/%.o,$(KERNELS))
HOST_OBJ := $(BUILD)/attn_gfx1201_host.o

.PHONY: all clean run ubench
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

$(BUILD)/attn_gfx1201_kernel_v10.o: attn_gfx1201_kernel_v10.cc attn_gfx1201_kernel_v10_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(BUILD)/attn_gfx1201_kernel_v11.o: attn_gfx1201_kernel_v11.cc attn_gfx1201_kernel_v11_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(HOST_OBJ): attn_gfx1201_host.cc attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(EXE): $(KERN_OBJ) $(HOST_OBJ)
	$(HIPCC) $(CXXFLAGS) $^ -o $@

run: $(EXE)
	./$(EXE)


$(BUILD)/ubench: ubench_gfx1201.cc | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) $< -o $@

ubench: $(BUILD)/ubench
	./$(BUILD)/ubench

clean:
	rm -rf $(BUILD)

$(BUILD)/attn_gfx1201_kernel_v12.o: attn_gfx1201_kernel_v12.cc attn_gfx1201_kernel_v12_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(BUILD)/attn_gfx1201_kernel_v13.o: attn_gfx1201_kernel_v13.cc attn_gfx1201_kernel_v13_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(BUILD)/attn_gfx1201_kernel_v14.o: attn_gfx1201_kernel_v14.cc attn_gfx1201_kernel_v14_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(BUILD)/attn_gfx1201_kernel_v15.o: attn_gfx1201_kernel_v15.cc attn_gfx1201_kernel_v15_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(BUILD)/attn_gfx1201_kernel_v16.o: attn_gfx1201_kernel_v16.cc attn_gfx1201_kernel_v16_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(BUILD)/attn_gfx1201_kernel_v17.o: attn_gfx1201_kernel_v17.cc attn_gfx1201_kernel_v17_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(BUILD)/attn_gfx1201_kernel_v18.o: attn_gfx1201_kernel_v18.cc attn_gfx1201_kernel_v18_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(BUILD)/attn_gfx1201_kernel_v19.o: attn_gfx1201_kernel_v19.cc attn_gfx1201_kernel_v19_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(BUILD)/attn_gfx1201_kernel_v20.o: attn_gfx1201_kernel_v20.cc attn_gfx1201_kernel_v20_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(BUILD)/attn_gfx1201_kernel_v21.o: attn_gfx1201_kernel_v21.cc attn_gfx1201_kernel_v21_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(BUILD)/attn_gfx1201_kernel_v22.o: attn_gfx1201_kernel_v22.cc attn_gfx1201_kernel_v22_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(BUILD)/attn_gfx1201_kernel_v23.o: attn_gfx1201_kernel_v23.cc attn_gfx1201_kernel_v23_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(BUILD)/attn_gfx1201_kernel_v24.o: attn_gfx1201_kernel_v24.cc attn_gfx1201_kernel_v24_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(BUILD)/attn_gfx1201_kernel_v25.o: attn_gfx1201_kernel_v25.cc attn_gfx1201_kernel_v25_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(BUILD)/attn_gfx1201_kernel_v26.o: attn_gfx1201_kernel_v26.cc attn_gfx1201_kernel_v26_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(BUILD)/attn_gfx1201_kernel_v27.o: attn_gfx1201_kernel_v27.cc attn_gfx1201_kernel_v27_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(BUILD)/attn_gfx1201_kernel_v28.o: attn_gfx1201_kernel_v28.cc attn_gfx1201_kernel_v28_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(BUILD)/attn_gfx1201_kernel_v29.o: attn_gfx1201_kernel_v29.cc attn_gfx1201_kernel_v29_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(BUILD)/attn_gfx1201_kernel_v30.o: attn_gfx1201_kernel_v30.cc attn_gfx1201_kernel_v30_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(BUILD)/attn_gfx1201_kernel_v31.o: attn_gfx1201_kernel_v31.cc attn_gfx1201_kernel_v31_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(BUILD)/attn_gfx1201_kernel_v34.o: attn_gfx1201_kernel_v34.cc \
    attn_gfx1201_kernel_v35.cc \
    attn_gfx1201_kernel_v36.cc \
    attn_gfx1201_kernel_v37.cc \
    attn_gfx1201_kernel_v38.cc \
    attn_gfx1201_kernel_v39.cc \
    attn_gfx1201_kernel_v40.cc \
    attn_gfx1201_kernel_v41.cc \
    attn_gfx1201_kernel_v42.cc \
    attn_gfx1201_kernel_v43.cc attn_gfx1201_kernel_v34_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(BUILD)/attn_gfx1201_kernel_v33.o: attn_gfx1201_kernel_v33.cc attn_gfx1201_kernel_v33_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(BUILD)/attn_gfx1201_kernel_v32.o: attn_gfx1201_kernel_v32.cc attn_gfx1201_kernel_v32_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(BUILD)/attn_gfx1201_kernel_v35.o: attn_gfx1201_kernel_v35.cc \
    attn_gfx1201_kernel_v36.cc \
    attn_gfx1201_kernel_v37.cc \
    attn_gfx1201_kernel_v38.cc \
    attn_gfx1201_kernel_v39.cc \
    attn_gfx1201_kernel_v40.cc \
    attn_gfx1201_kernel_v41.cc \
    attn_gfx1201_kernel_v42.cc \
    attn_gfx1201_kernel_v43.cc attn_gfx1201_kernel_v35_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(BUILD)/attn_gfx1201_kernel_v36.o: attn_gfx1201_kernel_v36.cc attn_gfx1201_kernel_v36_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(BUILD)/attn_gfx1201_kernel_v37.o: attn_gfx1201_kernel_v37.cc \
    attn_gfx1201_kernel_v38.cc \
    attn_gfx1201_kernel_v39.cc \
    attn_gfx1201_kernel_v40.cc \
    attn_gfx1201_kernel_v41.cc \
    attn_gfx1201_kernel_v42.cc \
    attn_gfx1201_kernel_v43.cc attn_gfx1201_kernel_v37_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(BUILD)/attn_gfx1201_kernel_v38.o: attn_gfx1201_kernel_v38.cc \
    attn_gfx1201_kernel_v39.cc \
    attn_gfx1201_kernel_v40.cc \
    attn_gfx1201_kernel_v41.cc \
    attn_gfx1201_kernel_v42.cc \
    attn_gfx1201_kernel_v43.cc attn_gfx1201_kernel_v38_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(BUILD)/attn_gfx1201_kernel_v39.o: attn_gfx1201_kernel_v39.cc \
    attn_gfx1201_kernel_v40.cc \
    attn_gfx1201_kernel_v41.cc \
    attn_gfx1201_kernel_v42.cc \
    attn_gfx1201_kernel_v43.cc attn_gfx1201_kernel_v39_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(BUILD)/attn_gfx1201_kernel_v40.o: attn_gfx1201_kernel_v40.cc \
    attn_gfx1201_kernel_v41.cc \
    attn_gfx1201_kernel_v42.cc \
    attn_gfx1201_kernel_v43.cc attn_gfx1201_kernel_v40_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(BUILD)/attn_gfx1201_kernel_v41.o: attn_gfx1201_kernel_v41.cc attn_gfx1201_kernel_v41_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(BUILD)/attn_gfx1201_kernel_v42.o: attn_gfx1201_kernel_v42.cc attn_gfx1201_kernel_v42_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(BUILD)/attn_gfx1201_kernel_v43.o: attn_gfx1201_kernel_v43.cc attn_gfx1201_kernel_v43_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(BUILD)/attn_gfx1201_kernel_v44.o: attn_gfx1201_kernel_v44.cc attn_gfx1201_kernel_v44_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(BUILD)/attn_gfx1201_kernel_v45.o: attn_gfx1201_kernel_v45.cc attn_gfx1201_kernel_v45_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

$(BUILD)/attn_gfx1201_kernel_v48.o: attn_gfx1201_kernel_v48.cc attn_gfx1201_kernel_v48_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@

# v49: hand-scheduled ASM kernel, loaded as HSACO at runtime
LLVM_BIN := /opt/rocm/lib/llvm/bin
$(BUILD)/attn_v49.hsaco: attn_gfx1201_kernel_v49.s | $(BUILD)
	$(LLVM_BIN)/clang-offload-bundler --type=s --input=$< --output=$(BUILD)/v49_device.s --output=/dev/null --targets=hip-amdgcn-amd-amdhsa--gfx1201,host-x86_64-unknown-linux-gnu- --unbundle
	$(LLVM_BIN)/clang++ -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 $(BUILD)/v49_device.s -o $@

# Make exe depend on HSACO so it's built before running
# [fix] v49.s source is absent in #24 — drop the forced HSACO dep so `make` builds the .cc kernels
#$(EXE): | $(BUILD)/attn_v49.hsaco

$(BUILD)/attn_gfx1201_kernel_v139.o: attn_gfx1201_kernel_v139.cc attn_gfx1201_kernel_v139_template.hpp attn_common.h | $(BUILD)
	$(HIPCC) -x hip $(CXXFLAGS) -c $< -o $@
