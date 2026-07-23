OPUS_INCLUDE_DIR ?= ../../mla_agentic_v0/third_party/aiter/csrc/include
MORI_INCLUDE_DIR ?= /workspace/mori/include
MORI_LIB_DIR     ?= /workspace/mori/build/src/cco
HIPCC            ?= /opt/rocm/bin/hipcc
ARCH             ?= gfx950
# 1=phase-shift comm WGs across peers, 0=all comm WGs finish one peer at a time.
COMM_PEER_ORDER  ?= 1
# 0=comm WGs at bx 0,8,... (same XCC); 1=contiguous bx 0..N-1 (spread XCCs).
COMM_WG_PLACEMENT ?= 1
RECORD_WG_HW      ?= 0
# 0=pointer copy + explicit wait, 1=compiler-managed wait, 2=buffer copy with cpol.
COMM_COPY_MODE    ?= 2
# 0=one ready counter per source shard, 1=one counter per source and M tile.
TILE_READY        ?= 0
READY_AWARE_K     ?= 0
A2A_C_STORE_MODE  ?= 2
PRESTORE_BARRIER  ?= 1

MPI_INCLUDE_DIR ?= /usr/lib/x86_64-linux-gnu/openmpi/include
MPI_LIB         ?= -lmpi_cxx -lmpi

BUILD = build
OUT   = $(BUILD)/a2a_gemm_lsa.exe

COMMON_FLAGS = -I. -I$(OPUS_INCLUDE_DIR) -std=c++20 -fopenmp -O3 -Wall --offload-arch=$(ARCH) -ffast-math -DA2A_GEMM_COMM_PEER_ORDER=$(COMM_PEER_ORDER) -DA2A_GEMM_COMM_WG_PLACEMENT=$(COMM_WG_PLACEMENT) -DA2A_GEMM_RECORD_WG_HW=$(RECORD_WG_HW) -DA2A_GEMM_COMM_COPY_MODE=$(COMM_COPY_MODE) -DA2A_GEMM_TILE_READY=$(TILE_READY) -DA2A_GEMM_READY_AWARE_K=$(READY_AWARE_K) -DA2A_GEMM_C_STORE_MODE=$(A2A_C_STORE_MODE) -DA2A_GEMM_PRESTORE_BARRIER=$(PRESTORE_BARRIER)
HOST_INCLUDES = -I$(MORI_INCLUDE_DIR) -I$(MPI_INCLUDE_DIR)
HOST_LIBS = -L$(MORI_LIB_DIR) -lmori_cco -Wl,-rpath,$(MORI_LIB_DIR) -lrccl $(MPI_LIB)

OBJS = $(BUILD)/a2a_gemm_kernel.o $(BUILD)/gemm_a16w16_quad_subtile_kernel.o $(BUILD)/a2a_gemm_host.o

.PHONY: all clean run

all: $(OUT)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/a2a_gemm_kernel.o: a2a_gemm_kernel.cc a2a_gemm_kernel_template.hpp gemm_defs.h | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) -D__HIPCC_RTC__ -Rpass-analysis=kernel-resource-usage -save-temps=obj -c -o $@

$(BUILD)/gemm_a16w16_quad_subtile_kernel.o: ../opus_gemm_a2a_lsa/gemm_a16w16_quad_subtile_kernel.cc ../opus_gemm_a2a_lsa/gemm_a16w16_quad_subtile_kernel_template.hpp ../opus_gemm_a2a_lsa/gemm_defs.h | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) -DOPUS_PERSISTENT=0 -D__HIPCC_RTC__ -Rpass-analysis=kernel-resource-usage -save-temps=obj -c -o $@

$(BUILD)/a2a_gemm_host.o: a2a_gemm_host.cc gemm_defs.h | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) $(HOST_INCLUDES) -c -o $@

$(OUT): $(OBJS)
	$(HIPCC) $^ --offload-arch=$(ARCH) -fopenmp $(HOST_LIBS) -o $@

run: $(OUT)
	mpirun --allow-run-as-root -n 8 $(OUT)

clean:
	rm -rf $(BUILD)
