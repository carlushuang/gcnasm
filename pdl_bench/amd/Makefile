HIPCC      ?= hipcc
HIP_ARCH   ?= --offload-arch=gfx950

CXXFLAGS   := -std=c++17 -O3
HIPCCFLAGS := $(HIP_ARCH) $(CXXFLAGS)

TARGET     := pdl_bench_amd.exe
SRC        := pdl_bench.hip

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(HIPCC) $(HIPCCFLAGS) $< -o $@

clean:
	rm -f $(TARGET)
