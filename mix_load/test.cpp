#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <random>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <numeric>
#define HALF
#ifdef HALF
#include "half.hpp"
#endif

using index_t = int;

#define MAX(x, y) ((x) > (y) ? (x) : (y))


using fp32 = float;
using fp16 = _Float16;

using fp16x2 = fp16 __attribute__((ext_vector_type(2)));
using fp16x4 = fp16 __attribute__((ext_vector_type(4)));
using fp16x8 = fp16 __attribute__((ext_vector_type(8)));
using fp16x16 = fp16 __attribute__((ext_vector_type(16)));
using fp32x16 = fp32 __attribute__((ext_vector_type(16)));

using dword_t = index_t;
using dwordx4_t = dword_t __attribute__((ext_vector_type(4)));

#define BLOCK_SIZE 256

#define BUFFER_LOAD_DWORD3 0x00020000
struct buffer_resource {
    const void * ptr;
    uint32_t range;
    uint32_t config;
};
__device__ dwordx4_t make_buffer_resource(const void * ptr)
{
    buffer_resource res {ptr, 0xffffffff, BUFFER_LOAD_DWORD3};
    return __builtin_bit_cast(dwordx4_t, res);
}

__device__ void
llvm_amdgcn_raw_buffer_load_lds(dwordx4_t rsrc,
                                __attribute__((address_space(3))) uint32_t* lds_ptr,
                                index_t size,
                                index_t voffset,
                                index_t soffset,
                                index_t offset,
                                index_t aux) __asm("llvm.amdgcn.raw.buffer.load.lds");

template<typename T, index_t N>
struct static_buffer {
    T data[N];
    __host__ __device__ auto & get(index_t i) {return data[i]; }
    __host__ __device__ const auto & get(index_t i) const {return data[i]; }
};

template<typename T, index_t N>
struct vector_type {
    using type = T __attribute__((ext_vector_type(N)));
    type data;

    template<typename Tx>
    __device__ __host__ auto & as(){
        static_assert(sizeof(T) * N % sizeof(Tx) == 0);
        constexpr int vx = sizeof(T) * N / sizeof(Tx);
        return reinterpret_cast<static_buffer<Tx, vx>&>(data);
    }
    template<typename Tx>
    __device__ __host__ const auto & as() const {
        static_assert(sizeof(T) * N % sizeof(Tx) == 0);
        constexpr int vx = sizeof(T) * N / sizeof(Tx);
        return reinterpret_cast<const static_buffer<Tx, vx>&>(data);
    }

    __device__ __host__ auto & to(){
        return reinterpret_cast<static_buffer<T, N>&>(data);
    }

    __device__ __host__ const auto & to() const {
        return reinterpret_cast<const static_buffer<T, N>&>(data);
    }

    template<typename Tx>
    __device__ __host__ auto & at(index_t i){
        return as<Tx>().get(i);
    }

    template<typename Tx>
    __device__ __host__ const auto & at(index_t i) const {
        return as<Tx>().get(i);
    }

    __device__ __host__ auto &at(index_t i) {
        return to().get(i);
    }
    __device__ __host__ const auto &at(index_t i) const {
        return to().get(i);
    }

    __device__ __host__ auto & get() {
        return data;
    }
    __device__ __host__ const auto & get() const {
        return data;
    }
};

//  128x32 tile
__global__ void /* __launch_bounds__(THREADS) */
compute_gemm_gemm(const void* __restrict__ ptr_q,
                  const void* __restrict__ ptr_k,
                  const void* __restrict__ ptr_v,
                  void* __restrict__ ptr_o,
                  uint32_t loops)
{
    if(blockIdx.x > 0)
        return;

    __shared__  uint32_t smem_0[8192] __attribute__((address_space(3)));
    __shared__  uint32_t smem_1[8192] __attribute__((address_space(3)));
    vector_type<fp16, 16> q;

    q.template at<fp16x8>(0) = reinterpret_cast<const fp16x8*>(ptr_q)[threadIdx.x];
    q.template at<fp16x8>(1) = reinterpret_cast<const fp16x8*>(ptr_q)[threadIdx.x + 256];

    vector_type<fp32, 16> acc;  // no clear zero
    for(auto i = 0; i < 16; i++) {
        acc.at(i) = .0f;
    }

    // load K
    for(auto i = 0; i < 8; i++)
        llvm_amdgcn_raw_buffer_load_lds(make_buffer_resource(ptr_k),
            reinterpret_cast<__attribute__((address_space(3))) uint32_t*>(smem_0 + (threadIdx.x + i * 256) * sizeof(dword_t)),
            sizeof(dword_t), (threadIdx.x + i * 256) * sizeof(dword_t), 0, 0, 0);

    while (loops > 0) {
        vector_type<fp32, 16> acc_0;  // no clear zero

        // load V
        __builtin_amdgcn_s_barrier();
        for(auto i = 0; i < 8; i++)
            llvm_amdgcn_raw_buffer_load_lds(make_buffer_resource(ptr_v),
                reinterpret_cast<__attribute__((address_space(3))) uint32_t*>(smem_1 + (threadIdx.x + i * 256) * sizeof(dword_t)),
                sizeof(dword_t), (threadIdx.x + i * 256) * sizeof(dword_t), 0, 0, 0);

        // compute gemm_0
        {
            vector_type<fp16, 16> k;
            k.template at<fp16x8>(0) = reinterpret_cast<__attribute__((address_space(3))) fp16x8*>(smem_0)[threadIdx.x];
            k.template at<fp16x8>(1) = reinterpret_cast<__attribute__((address_space(3))) fp16x8*>(smem_0)[threadIdx.x + 256];

            for(auto i = 0; i < 4; i++)
                acc_0.get() = __builtin_amdgcn_mfma_f32_32x32x8f16(q.template at<fp16x4>(i),
                                                            k.template at<fp16x4>(i),
                                                            acc_0.get(), 0, 0, 0);
        }

        __builtin_amdgcn_s_barrier();
        loops--;
        // load K
        if(loops > 0) {
            for(auto i = 0; i < 8; i++)
                llvm_amdgcn_raw_buffer_load_lds(make_buffer_resource(ptr_k),
                    reinterpret_cast<__attribute__((address_space(3))) uint32_t*>(smem_0 + (threadIdx.x + i * 256) * sizeof(dword_t)),
                    sizeof(dword_t), (threadIdx.x + i * 256) * sizeof(dword_t), 0, 0, 0);
        }

        // compute gemm 1
        {
            vector_type<fp16, 16> v;
            v.template at<fp16x8>(0) = reinterpret_cast<__attribute__((address_space(3))) fp16x8*>(smem_1)[threadIdx.x];
            v.template at<fp16x8>(1) = reinterpret_cast<__attribute__((address_space(3))) fp16x8*>(smem_1)[threadIdx.x + 256];
            for(auto i = 0; i < 4; i++) {
                vector_type<fp16, 16> tmp;
                for(auto j = 0; j < 16; j++) {
                    tmp.at(j) = static_cast<fp16>(acc_0.template at<fp32>(j));
                }
                acc.get() = __builtin_amdgcn_mfma_f32_32x32x8f16(tmp.template at<fp16x4>(i),
                                                v.template at<fp16x4>(i),
                                                acc.get(), 0, 0, 0);
            }
        }

        // __builtin_amdgcn_s_barrier();
    }

    vector_type<fp16, 16> o;
    for(auto j = 0; j < 16; j++) {
        o.at(j) = static_cast<fp16>(acc.at(j));
    }
    reinterpret_cast<fp16x8*>(ptr_o)[threadIdx.x] = o.template at<fp16x8>(0);
    reinterpret_cast<fp16x8*>(ptr_o)[threadIdx.x + 256] = o.template at<fp16x8>(1);
}


int main()
{
    
}