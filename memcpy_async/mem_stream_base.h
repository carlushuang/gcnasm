#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <random>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <numeric>
#include <utility>


#define WARMUP 200
#define LOOP 200

#define RAND_INT 0

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define HIP_CALL(call) do{  \
    hipError_t err = call;  \
    if(err != hipSuccess){  \
        printf("[hiperror](%d) fail to call %s",(int)err,#call);    \
        exit(0);            \
    }                       \
} while(0)

#define ABS(x) ((x) > 0 ? (x) : -(x))

using fp32_t = float;

using int32x4_t  = int32_t __attribute__((ext_vector_type(4)));

using fp32x1_t = fp32_t __attribute__((ext_vector_type(1)));
using fp32x2_t = fp32_t __attribute__((ext_vector_type(2)));
using fp32x4_t = fp32_t __attribute__((ext_vector_type(4)));
using fp32x16_t = fp32_t __attribute__((ext_vector_type(16)));
using index_t  = int;

template<int bytes>
struct bytes_to_vector;

template<> struct bytes_to_vector<16> { using type = fp32x4_t; };
template<> struct bytes_to_vector<8> { using type = fp32x2_t; };
template<> struct bytes_to_vector<4> { using type = fp32_t; };

template<int bytes>
using bytes_to_vector_t = typename bytes_to_vector<bytes>::type;



template<std::size_t N>
struct num { static const constexpr auto value = N; };

template <class F, std::size_t... Is>
__host__ __device__ void for_(F func, std::index_sequence<Is...>)
{
    (func(num<Is>{}), ...);
}

template <typename Kernel, typename... Args>
__global__ void kentry(Args... args)
{
    Kernel{}(args...);
}

#define BUFFER_LOAD_DWORD3 0x00020000
struct buffer_resource {
    const void * ptr;
    uint32_t range;
    uint32_t config;
};
__device__ int32x4_t make_buffer_resource(const void * ptr, uint32_t range = 0xffffffff)
{
    buffer_resource res {ptr, range, BUFFER_LOAD_DWORD3};
    return __builtin_bit_cast(int32x4_t, res);
}

__device__ void
llvm_amdgcn_raw_buffer_load_lds(int32x4_t rsrc,
                                __attribute__((address_space(3))) uint32_t* lds_ptr,
                                index_t size,
                                index_t voffset,
                                index_t soffset,
                                index_t offset,
                                index_t aux) __asm("llvm.amdgcn.raw.buffer.load.lds");


__device__ fp32x4_t llvm_amdgcn_raw_buffer_load_fp32x4(int32x4_t srsrc, index_t voffset, index_t soffset, index_t glc_slc)
                            __asm("llvm.amdgcn.raw.buffer.load.v4f32");

__device__ void llvm_amdgcn_raw_buffer_store_fp32x4(fp32x4_t vdata, int32x4_t rsrc, index_t voffset, index_t soffset, index_t glc_slc)
                            __asm("llvm.amdgcn.raw.buffer.store.v4f32");


template<typename T>
__device__ void buffer_load_dwordx4_raw(T & value, int32x4_t res/*buffer resource*/, index_t v_offset, index_t s_offset, index_t i_offset/*max 0xFFF*/, index_t /*flag*/ = 0){
    static_assert(sizeof(T) == 16);
    using v_type = float __attribute__((ext_vector_type(4)));
    asm volatile("buffer_load_dwordx4 %0, %1, %2, %3 offen offset:%4"
        : "+v"(reinterpret_cast<v_type&>(value)) : "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset) : "memory");
}

__device__ void async_buffer_load_dword_v(void* smem,
                                              int32x4_t rsrc,
                                              index_t voffset,
                                              index_t /*soffset*/,
                                              index_t ioffset /*max 0xFFF*/,
                                              index_t /*flag*/       = 0)
{
    asm volatile("buffer_load_dword %1, %2, 0 offen offset:%3 lds"
                    : "=r"(smem) /*dummy dependency for smem*/
                    : "v"(voffset), "s"(rsrc), "n"(ioffset)
                    : "memory");
}

__device__ void m0_set_with_memory(index_t v)
{
    asm volatile("s_mov_b32 m0, %0" : : "s"(v) : "memory");
}

__device__ void m0_inc_with_memory(index_t v)
{
    asm volatile("s_add_u32 m0, %0, m0" : : "n"(v) : "memory");
}

template<typename T>
__device__ void buffer_store_dwordx4_raw(const T & value, int32x4_t res/*buffer resource*/, index_t v_offset, index_t s_offset, index_t i_offset/*max 0xFFF*/, index_t /*flag*/ = 0){
    static_assert(sizeof(T) == 16);
    using v_type = float __attribute__((ext_vector_type(4)));
    asm volatile("buffer_store_dwordx4 %0, %1, %2, %3 offen offset:%4"
        : : "v"(reinterpret_cast<const v_type&>(value)), "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset) : "memory");
}

__device__ void buffer_fence(index_t cnt)
{
    asm volatile("s_waitcnt vmcnt(%0)" : : "n" (cnt) : "memory");
}

template<typename T>
__device__ __forceinline__ T nt_load(const T& ref)
{
    return __builtin_nontemporal_load(&ref);
}

template<typename T>
__device__ __forceinline__ void nt_store(const T& value, T& ref) {
    __builtin_nontemporal_store(value, &ref);
}

template <typename Derived, int BLOCK_SIZE, int GRID_SIZE>
struct mem_stream_base {
    struct karg {
        void * src;
        void * dst;
        int64_t bytes;
        uint32_t iters;
        uint32_t issue_per_block;
    };

    using harg = karg;
    karg consts;
    
    __host__ mem_stream_base(harg harg_, int bytes_per_issue, int unroll, int inner = 1) {
        consts = harg_;
        auto elems = consts.bytes / bytes_per_issue;
        auto issue_per_block = elems / GRID_SIZE;
        consts.issue_per_block = issue_per_block;
        consts.iters = issue_per_block / BLOCK_SIZE / unroll / inner;
    }

    __host__ virtual ~mem_stream_base() {} 
    
    __host__ void operator()() const {
        kentry<typename Derived::kernel><<<GRID_SIZE, BLOCK_SIZE>>>(consts);
    }
};

