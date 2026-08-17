# The sibling aiter checkout supplies <opus/opus.hpp>; override for a different one.
OPUS_INCLUDE_DIR ?= ../../aiter/csrc/include
# Only the kernel TU needs the pin-enabled toolchain; the host side is ordinary
# HIP and builds with stock ROCm (which also supplies OpenMP for the reference).
HIPCC            ?= /opt/rocm/bin/hipcc
DEVICE_HIPCC     ?= $(HIPCC)
ARCH             ?= gfx1250

# Two builds of one kernel, differing only in whether the register plan is
# requested. Each lands in its own directory; dump_asm.sh reads both.
#   pin      __builtin_amdgcn_pin_vgpr on each written value (default)
#   nopin    control, same source with no placement request
VARIANT ?= pin
VARIANT_FLAGS_pin   =
VARIANT_FLAGS_nopin = -DOPUS_NO_PIN=1
VARIANT_FLAGS = $(VARIANT_FLAGS_$(VARIANT))

COMMON_FLAGS = -I. -I$(OPUS_INCLUDE_DIR) -std=c++20 -fopenmp -O3 -Wall --offload-arch=$(ARCH) $(VARIANT_FLAGS)

# Device-pass flags that matter for this pipeline:
#   -enable-post-misched          the opus pipeline relies on post-RA scheduling
#   -amdgpu-expert-scheduling-mode  honours the sched_group_barrier groups
# Both need the register-pinning LLVM (see README); stock ROCm rejects the
# second one outright and does not know the pin builtin at all.
DEVICE_FLAGS = -mllvm -enable-post-misched=1 -mllvm -amdgpu-expert-scheduling-mode

ifeq ($(filter $(VARIANT),pin nopin),)
  $(error VARIANT must be one of: pin nopin)
endif

BUILD = build_$(VARIANT)
OUT   = $(BUILD)/gemm_a16w16_4wave_compute.exe

KERNEL_SRC  = gemm_a16w16_4wave_compute_kernel.cc
HOST_SRC    = gemm_host.cc
KERNEL_OBJ  = $(BUILD)/gemm_a16w16_4wave_compute_kernel.o
HOST_OBJ    = $(BUILD)/gemm_host.o

.PHONY: all clean run variants

all: $(OUT)

# Both builds, for the asm comparison.
variants:
	@for v in pin nopin; do $(MAKE) --no-print-directory VARIANT=$$v || exit 1; done

$(BUILD):
	mkdir -p $(BUILD)

# Kernel object: heavy device-pass template compilation. -D__HIPCC_RTC__ skips
# the implicit __clang_hip_runtime_wrapper.h on the host pass. -save-temps=obj
# keeps the .s next to the object, which is what the scheduling work reads.
$(KERNEL_OBJ): $(KERNEL_SRC) gemm_a16w16_4wave_compute_kernel_template.hpp gemm_defs.h | $(BUILD)
	$(DEVICE_HIPCC) $< $(COMMON_FLAGS) $(DEVICE_FLAGS) -D__HIPCC_RTC__ \
	    -Rpass-analysis=kernel-resource-usage -save-temps=obj -c -o $@

# Host object: needs the full HIP runtime for <<<>>> launch syntax.
$(HOST_OBJ): $(HOST_SRC) gemm_defs.h | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) -c -o $@

$(OUT): $(KERNEL_OBJ) $(HOST_OBJ)
	$(HIPCC) $^ --offload-arch=$(ARCH) -fopenmp -o $@

run: $(OUT)
	@$(OUT)

clean:
	rm -rf build_pin build_nopin asm
