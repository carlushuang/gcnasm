#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

typedef int int32x4_t __attribute__((ext_vector_type(4)));
typedef int int32x2_t __attribute__((ext_vector_type(2)));
typedef _Float16 half_t;
typedef _Float16_2 half2_t;

using index_t = int32_t;

#include <stdint.h>
__device__ half2_t llvm_amdgcn_raw_buffer_atomic_add_fp16x2(
    half2_t vdata,
    int32x4_t rsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.atomic.fadd.v2f16");



template <typename T>
union BufferResource
{
    __device__ constexpr BufferResource() : content{} {}

    // 128 bit SGPRs to supply buffer resource in buffer instructions
    // https://rocm-documentation.readthedocs.io/en/latest/GCN_ISA_Manuals/testdocbook.html#vector-memory-buffer-instructions
    int32x4_t content;
    struct{
        T * address;
        uint32_t range;
        uint32_t config;
    };
};

#define BUFFER_RESOURCE_3RD_DWORD 0x00020000U    // gfx90a

template <typename T>
__device__ int32x4_t make_wave_buffer_resource(T* p_wave)
{
    BufferResource<T> wave_buffer_resource;

    // wavewise base address (64 bit)
    wave_buffer_resource.address = const_cast<std::remove_cv_t<T>*>(p_wave);
    // wavewise range (32 bit)
    wave_buffer_resource.range = 0xffffffffU;
    // wavewise setting (32 bit)
    wave_buffer_resource.config = BUFFER_RESOURCE_3RD_DWORD;

    return wave_buffer_resource.content;
}