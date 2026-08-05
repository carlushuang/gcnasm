OPUS_INCLUDE_DIR ?= ../../mla_agentic_v0/third_party/aiter/csrc/include
MORI_INCLUDE_DIR ?= /workspace/mori/include
MORI_LIB_DIR     ?= /workspace/mori/build/src/cco
HIPCC            ?= /opt/rocm/bin/hipcc
ARCH             ?= gfx950
# Optional hardware-location records for ATT target-CU selection.
RECORD_WG_HW      ?= 0
# TRACE=1 keeps optimization while adding line tables for rocprofv3 ATT.
TRACE             ?= 0
ifeq ($(TRACE),1)
TRACE_FLAGS       := -g -gline-tables-only -fdebug-compilation-dir=/ \
	-ffile-prefix-map=.=$(CURDIR) -fdebug-prefix-map=.=$(CURDIR)
endif

MPI_INCLUDE_DIR ?= /usr/lib/x86_64-linux-gnu/openmpi/include
MPI_LIB         ?= -lmpi_cxx -lmpi

BUILD = build
OUT   = $(BUILD)/a2a_gemm_lsa.exe

COMMON_FLAGS = -I. -I$(OPUS_INCLUDE_DIR) -std=c++20 -fopenmp -O3 $(TRACE_FLAGS) -Wall --offload-arch=$(ARCH) -ffast-math -DBUILD_CCO_SDMA=1 -DA2A_GEMM_RECORD_WG_HW=$(RECORD_WG_HW)
HOST_INCLUDES = -I$(MORI_INCLUDE_DIR) -I$(MPI_INCLUDE_DIR)
HOST_LIBS = -L$(MORI_LIB_DIR) -lmori_cco -Wl,-rpath,$(MORI_LIB_DIR) -lrccl $(MPI_LIB)

OBJS = $(BUILD)/a2a_gemm_kernel.o $(BUILD)/a2a_gemm_sdma_fused_kernel.o $(BUILD)/a2a_sdma_kernel.o $(BUILD)/gemm_a16w16_quad_subtile_kernel.o $(BUILD)/a2a_gemm_host.o

.PHONY: all clean run

all: $(OUT)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/a2a_gemm_kernel.o: a2a_gemm_kernel.cc a2a_gemm_kernel_template.hpp gemm_defs.h | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) $(HOST_INCLUDES) -D__HIPCC_RTC__ -Rpass-analysis=kernel-resource-usage -save-temps=obj -c -o $@

$(BUILD)/a2a_sdma_kernel.o: a2a_sdma_kernel.cc | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) $(HOST_INCLUDES) -Rpass-analysis=kernel-resource-usage -c -o $@

$(BUILD)/a2a_gemm_sdma_fused_kernel.o: a2a_gemm_sdma_fused_kernel.cc a2a_gemm_kernel_template.hpp gemm_defs.h | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) $(HOST_INCLUDES) -D__HIPCC_RTC__ -Rpass-analysis=kernel-resource-usage -save-temps=obj -c -o $@

$(BUILD)/gemm_a16w16_quad_subtile_kernel.o: ../opus_gemm_a2a_lsa/gemm_a16w16_quad_subtile_kernel.cc ../opus_gemm_a2a_lsa/gemm_a16w16_quad_subtile_kernel_template.hpp ../opus_gemm_a2a_lsa/gemm_defs.h | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) $(HOST_INCLUDES) -DOPUS_PERSISTENT=0 -D__HIPCC_RTC__ -Rpass-analysis=kernel-resource-usage -save-temps=obj -c -o $@

$(BUILD)/a2a_gemm_host.o: a2a_gemm_host.cc gemm_defs.h | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) $(HOST_INCLUDES) -c -o $@

$(OUT): $(OBJS)
	$(HIPCC) $^ --offload-arch=$(ARCH) -fopenmp $(HOST_LIBS) -o $@

run: $(OUT)
	mpirun --allow-run-as-root -n 8 $(OUT)

clean:
	rm -rf $(BUILD)
