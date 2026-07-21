OPUS_INCLUDE_DIR ?= /shared/amdgpu/home/jiahao_zhou_qle/blyu/aiter/csrc/include
MORI_INCLUDE_DIR ?= /shared/amdgpu/home/jiahao_zhou_qle/blyu/mori/include
MORI_LIB_DIR     ?= /opt/venv/lib/python3.12/site-packages/mori
HIPCC            ?= /opt/rocm/bin/hipcc
ARCH             ?= gfx950
# 0=baseline, 1=early partial C store, 2=lookahead next-tile loads, 3=drop tail barrier.
STORE_PIPELINE   ?= 3
# 0=baseline C store, 1=LDS staging coalesced store, 2=ds_bpermute pair-coalesced store.
C_STORE_MODE     ?= 2
# Default: round-robin scattered column tiles across peers and lightly stagger
# the final C stores. This reduced same-peer LSA store bursts in the large-shape
# sweep (M=8192,N=36864) by ~25% vs the previous p3+cstore2 baseline.
STORE_STAGGER_PHASES ?= 16
STORE_STAGGER_DELAY  ?= 4
TILE_ORDER       ?= 1

MPI_INCLUDE_DIR ?= /usr/lib/x86_64-linux-gnu/openmpi/include
MPI_LIB         ?= -lmpi_cxx -lmpi

BUILD = build
OUT   = $(BUILD)/quad_lsa_direct.exe

COMMON_FLAGS = -I. -I$(OPUS_INCLUDE_DIR) -std=c++20 -fopenmp -O3 -Wall --offload-arch=$(ARCH) -ffast-math -DOPUS_STORE_PIPELINE=$(STORE_PIPELINE) -DOPUS_C_STORE_MODE=$(C_STORE_MODE) -DOPUS_STORE_STAGGER_PHASES=$(STORE_STAGGER_PHASES) -DOPUS_STORE_STAGGER_DELAY=$(STORE_STAGGER_DELAY) -DOPUS_TILE_ORDER=$(TILE_ORDER)
HOST_INCLUDES = -I$(MORI_INCLUDE_DIR) -I$(MPI_INCLUDE_DIR)
HOST_LIBS = -L$(MORI_LIB_DIR) -lmori_cco -Wl,-rpath,$(MORI_LIB_DIR) $(MPI_LIB)

OBJS = $(BUILD)/gemm_a16w16_quad_subtile_kernel.o $(BUILD)/main.o

.PHONY: all clean run

all: $(OUT)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/gemm_a16w16_quad_subtile_kernel.o: gemm_a16w16_quad_subtile_kernel.cc gemm_a16w16_quad_subtile_kernel_template.hpp gemm_defs.h | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) -D__HIPCC_RTC__ -Rpass-analysis=kernel-resource-usage -save-temps=obj -c -o $@

$(BUILD)/main.o: main.cc gemm_defs.h | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) $(HOST_INCLUDES) -c -o $@

$(OUT): $(OBJS)
	$(HIPCC) $^ --offload-arch=$(ARCH) -fopenmp $(HOST_LIBS) -o $@

run: $(OUT)
	mpirun --allow-run-as-root -n 4 $(OUT)

clean:
	rm -rf $(BUILD)
