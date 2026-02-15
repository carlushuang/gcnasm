# Makefile for warp_bitonic_sort TVM FFI module
#
# Prerequisites:
#   - apache-tvm-ffi installed via pip (pip install apache-tvm-ffi)
#   - ROCm installed (set ROCM_PATH if not /opt/rocm)
#   - CK (composable_kernel) headers (set CK_DIR)
#
# Usage:
#   make                 # build the shared library
#   make clean           # remove build artifacts
#
# Environment variables:
#   ROCM_PATH   - path to ROCm installation (default: /opt/rocm)
#   CK_DIR      - path to composable_kernel repo (default: /raid0/carhuang/repo/composable_kernel)
#   GPU_ARCH    - target GPU architecture (default: native)

ROCM_PATH  ?= /opt/rocm
CK_DIR     ?= /raid0/carhuang/repo/composable_kernel
GPU_ARCH   ?= native

# Compilers
HIPCC      := $(ROCM_PATH)/bin/hipcc
CXX        := g++

# Directories
BUILD_DIR  := build
CSRC_DIR   := csrc

# Auto-detect tvm_ffi package paths from pip installation
TVM_FFI_PKG := $(shell python3 -c "import tvm_ffi, os; print(os.path.dirname(tvm_ffi.__file__))")
TVM_FFI_INC := -I$(TVM_FFI_PKG)/include -I$(TVM_FFI_PKG)/3rdparty/dlpack/include
TVM_FFI_LIB := -L$(TVM_FFI_PKG)/lib -ltvm_ffi -Wl,-rpath,$(TVM_FFI_PKG)/lib

# HIP flags
HIP_INC    := -I$(ROCM_PATH)/include -I$(CK_DIR)/include
HIP_FLAGS  := --offload-arch=$(GPU_ARCH) -O3 -fPIC $(HIP_INC) -I$(CSRC_DIR)

# CXX flags
CXX_FLAGS  := -std=c++17 -O2 -fPIC $(TVM_FFI_INC) -I$(CSRC_DIR)

# Linker flags
LD_FLAGS   := $(TVM_FFI_LIB) -L$(ROCM_PATH)/lib -lamdhip64

# Target
TARGET     := $(BUILD_DIR)/libwarp_bitonic_sort_tvm.so

# Source files
HIP_SRC    := $(CSRC_DIR)/warp_bitonic_sort.hip
TVM_SRC    := $(CSRC_DIR)/tvm_api.cc

# Object files
HIP_OBJ    := $(BUILD_DIR)/warp_bitonic_sort.o
TVM_OBJ    := $(BUILD_DIR)/tvm_api.o

.PHONY: all clean

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile HIP kernel
$(HIP_OBJ): $(HIP_SRC) $(CSRC_DIR)/warp_bitonic_sort.hpp | $(BUILD_DIR)
	$(HIPCC) $(HIP_FLAGS) -c $< -o $@

# Compile TVM FFI registration
$(TVM_OBJ): $(TVM_SRC) $(CSRC_DIR)/warp_bitonic_sort.hpp | $(BUILD_DIR)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

# Link shared library
$(TARGET): $(HIP_OBJ) $(TVM_OBJ)
	$(CXX) -shared -o $@ $^ $(LD_FLAGS)
	@echo ""
	@echo "==> Built: $(TARGET)"

clean:
	rm -rf $(BUILD_DIR)
