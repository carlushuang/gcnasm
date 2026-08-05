OPUS_INCLUDE_DIR ?= ../../mla_agentic_v0/third_party/aiter/csrc/include
MORI_INCLUDE_DIR ?= /workspace/mori/include
MORI_LIB_DIR     ?= /workspace/mori/build/src/cco
HIPCC            ?= /opt/rocm/bin/hipcc
ARCH             ?= gfx950
# The optimized peer/chunk/stripe tile order is fixed in the kernel.
# 1=global-counter persistent CTAs, 0=one output tile per CTA.
PERSISTENT       ?= 1
SDMA_FLAGS       := -DBUILD_CCO_SDMA=1
# TRACE=1 keeps -O3 but adds DWARF and maps relative source paths to this
# absolute directory so the ATT viewer can load source text.
TRACE            ?= 0
ifeq ($(TRACE),1)
TRACE_FLAGS      := -g -fdebug-compilation-dir=/ \
	-ffile-prefix-map=.=$(CURDIR) -fdebug-prefix-map=.=$(CURDIR)
endif

MPI_INCLUDE_DIR ?= /usr/lib/x86_64-linux-gnu/openmpi/include
MPI_LIB         ?= -lmpi_cxx -lmpi

BUILD = build
OUT   = $(BUILD)/quad_lsa_direct.exe
SDMA_OUT = $(BUILD)/sdma_a2a_bench.exe
SDMA_BENCH_OBJS = $(BUILD)/sdma_a2a_bench.o $(BUILD)/sdma_a2a_kernel.o

COMMON_FLAGS = -I. -I$(abspath $(OPUS_INCLUDE_DIR)) -std=c++20 -fopenmp -O3 $(TRACE_FLAGS) -Wall --offload-arch=$(ARCH) -ffast-math -DOPUS_PERSISTENT=$(PERSISTENT) $(SDMA_FLAGS)
HOST_INCLUDES = -I$(MORI_INCLUDE_DIR) -I$(MPI_INCLUDE_DIR)
HOST_LIBS = -L$(MORI_LIB_DIR) -lmori_cco -Wl,-rpath,$(MORI_LIB_DIR) $(MPI_LIB)

OBJS = $(BUILD)/gemm_a16w16_quad_subtile_kernel.o $(BUILD)/lsa_a2a_kernel.o $(BUILD)/sdma_a2a_kernel.o $(BUILD)/main.o

.PHONY: all clean run sdma_bench

all: $(OUT)

sdma_bench: $(SDMA_OUT)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/gemm_a16w16_quad_subtile_kernel.o: gemm_a16w16_quad_subtile_kernel.cc gemm_a16w16_quad_subtile_kernel_template.hpp gemm_defs.h | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) $(HOST_INCLUDES) -D__HIPCC_RTC__ -Rpass-analysis=kernel-resource-usage -save-temps=obj -c -o $@

$(BUILD)/sdma_a2a_kernel.o: sdma_a2a_kernel.cc | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) $(HOST_INCLUDES) -Rpass-analysis=kernel-resource-usage -c -o $@

$(BUILD)/lsa_a2a_kernel.o: lsa_a2a_kernel.cc gemm_defs.h | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) -Rpass-analysis=kernel-resource-usage -c -o $@

$(BUILD)/main.o: main.cc gemm_defs.h | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) $(HOST_INCLUDES) -c -o $@

$(OUT): $(OBJS)
	$(HIPCC) $^ --offload-arch=$(ARCH) -fopenmp $(HOST_LIBS) -o $@

$(BUILD)/sdma_a2a_bench.o: sdma_a2a_bench.cc | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) $(HOST_INCLUDES) -c -o $@

$(SDMA_OUT): $(SDMA_BENCH_OBJS)
	$(HIPCC) $^ --offload-arch=$(ARCH) -fopenmp $(HOST_LIBS) -o $@

run: $(OUT)
	mpirun --allow-run-as-root -n 4 $(OUT)

clean:
	rm -rf $(BUILD)
