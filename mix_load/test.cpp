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

#define TO_SBUF(type_, n_, v_) reinterpret_cast<static_buffer<type_, n_>&>(v_)

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
    fp16x16 q;

    TO_SBUF(fp16x8, 2, q).get(0) = reinterpret_cast<const fp16x8*>(ptr_q)[threadIdx.x];
    TO_SBUF(fp16x8, 2, q).get(1) = reinterpret_cast<const fp16x8*>(ptr_q)[threadIdx.x + 256];

    fp32x16 acc;  // no clear zero

    // load K
    for(auto i = 0; i < 8; i++)
        llvm_amdgcn_raw_buffer_load_lds(make_buffer_resource(ptr_k),
            reinterpret_cast<__attribute__((address_space(3))) uint32_t*>(smem_0 + (threadIdx.x + i * 256) * sizeof(dword_t)),
            sizeof(dword_t), (threadIdx.x + i * 256) * sizeof(dword_t), 0, 0, 0);

    while (loops > 0) {
        fp32x16 acc_0;  // no clear zero

        // load V
        __builtin_amdgcn_s_barrier();
        for(auto i = 0; i < 8; i++)
            llvm_amdgcn_raw_buffer_load_lds(make_buffer_resource(ptr_v),
                reinterpret_cast<__attribute__((address_space(3))) uint32_t*>(smem_1 + (threadIdx.x + i * 256) * sizeof(dword_t)),
                sizeof(dword_t), (threadIdx.x + i * 256) * sizeof(dword_t), 0, 0, 0);

        // compute gemm_0
        {
            fp16x16 k;
            TO_SBUF(fp16x8, 2, k).get(0) = reinterpret_cast<__attribute__((address_space(3))) fp16x8*>(smem_0)[threadIdx.x];
            TO_SBUF(fp16x8, 2, k).get(1) = reinterpret_cast<__attribute__((address_space(3))) fp16x8*>(smem_0)[threadIdx.x + 256];

            for(auto i = 0; i < 4; i++)
                acc_0 = __builtin_amdgcn_mfma_f32_32x32x8f16(TO_SBUF(fp16x4, 4, q).get(i),
                                                            TO_SBUF(fp16x4, 4, k).get(i),
                                                            acc_0, 0, 0, 0);
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
            fp16x16 v;
            TO_SBUF(fp16x8, 2, v).get(0) = reinterpret_cast<__attribute__((address_space(3))) fp16x8*>(smem_1)[threadIdx.x];
            TO_SBUF(fp16x8, 2, v).get(1) = reinterpret_cast<__attribute__((address_space(3))) fp16x8*>(smem_1)[threadIdx.x + 256];
            for(auto i = 0; i < 4; i++) {
                fp16x16 tmp;
                for(auto j = 0; j < 16; j++) {
                    TO_SBUF(fp16, 16, tmp).get(j) = static_cast<fp16>(TO_SBUF(fp32, 16, acc_0).get(j));
                }
                acc = __builtin_amdgcn_mfma_f32_32x32x8f16(TO_SBUF(fp16x4, 4, tmp).get(i),
                                                TO_SBUF(fp16x4, 4, v).get(i),
                                                acc, 0, 0, 0);
            }
        }

        // __builtin_amdgcn_s_barrier();
    }

    fp16x16 o;
    for(auto j = 0; j < 16; j++) {
        TO_SBUF(fp16, 16, o).get(j) = static_cast<fp16>(TO_SBUF(fp32, 16, acc).get(j));
    }
    reinterpret_cast<fp16x8*>(ptr_o)[threadIdx.x] = TO_SBUF(fp16x8, 2, o).get(0);
    reinterpret_cast<fp16x8*>(ptr_o)[threadIdx.x + 256] = TO_SBUF(fp16x8, 2, o).get(1);
}


int main()
{
    
}