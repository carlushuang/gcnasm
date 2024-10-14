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

#define LOCAL_SCRATCH 1
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
using fp16_t = _Float16;
using float16 = half_float::half; // cpu type

using fp16x2_t = fp16_t __attribute__((ext_vector_type(2)));
using fp16x4_t = fp16_t __attribute__((ext_vector_type(4)));
using fp16x8_t = fp16_t __attribute__((ext_vector_type(8)));
using fp16x16_t = fp16_t __attribute__((ext_vector_type(16)));
using fp32x16_t = fp32_t __attribute__((ext_vector_type(16)));

using int32x4_t = int32_t __attribute__((ext_vector_type(4)));
#define BUFFER_LOAD_DWORD3 0x00020000   // This is valid for 
struct buffer_resource {
    const void * ptr;
    uint32_t range;
    uint32_t config;
};
__device__ int32x4_t make_buffer_resource(const void * ptr, uint32_t size = 0xffffffff)
{
    buffer_resource res {ptr, size, BUFFER_LOAD_DWORD3};
    return __builtin_bit_cast(int32x4_t, res);
}
// A: M*K, B: N*K, C:M*N, use 32x32x8 fp16
/*
* V0/V1/   is 32bit register holding A/B matrix data, each register contains 2 fp16 pixel along gemm-k
* a0/a1... is 32bit register holding C matrix data in fp32 (this instruction use fp32 as acc)
* L0, L1.. is lane id with in a single wave, here we only have lane 0~63 (wave64)
* each thread need 2 registers for A, 2 regs for B, 16 regs for C

                                 L0 L1 L2 L3 L4 L5 L6 L7 L8 L9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
                       Matrix B   __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
                                 |v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0| k0  L0~31
                                 |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__| k1
                                 |v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1| k2
                                _|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|_k3
                                 |v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0| k4  L32~63
                                 |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__| k5
                                 |v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1| k6
                                _|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|_k7
     Matrix A
     L0~31       L32~63           Matrix C
     k0 k1 k2 k3 k4 k5 k6 k7      L0 L1 L2 L3 L4 L5 L6 L7 L8 L9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
     _____ _____|_____ _____      __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
L0  |v0   |v1   |v0   |v1   |    |a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0| L0~31
L1  |v0   |v1   |v0   |v1   |    |a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|
L2  |v0   |v1   |v0   |v1   |    |a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|
L3  |v0   |v1   |v0   |v1   |   _|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|_
L4  |v0   |v1   |v0   |v1   |    |a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0| L32~63
L5  |v0   |v1   |v0   |v1   |    |a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|
L6  |v0   |v1   |v0   |v1   |    |a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|
L7  |v0   |v1   |v0   |v1   |   _|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|_
L8  |v0   |v1   |v0   |v1   |    |a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4| L0~31
L9  |v0   |v1   |v0   |v1   |    |a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|
L10 |v0   |v1   |v0   |v1   |    |a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|
L11 |v0   |v1   |v0   |v1   |   _|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|_
L12 |v0   |v1   |v0   |v1   |    |a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4| L32~63
L13 |v0   |v1   |v0   |v1   |    |a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|
L14 |v0   |v1   |v0   |v1   |    |a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|
L15 |v0   |v1   |v0   |v1   |   _|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|_
L16 |v0   |v1   |v0   |v1   |    |a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8| L0~31
L17 |v0   |v1   |v0   |v1   |    |a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|
L18 |v0   |v1   |v0   |v1   |    |10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|
L19 |v0   |v1   |v0   |v1   |   _|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|_
L20 |v0   |v1   |v0   |v1   |    |a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8| L32~63
L21 |v0   |v1   |v0   |v1   |    |a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|
L22 |v0   |v1   |v0   |v1   |    |10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|
L23 |v0   |v1   |v0   |v1   |   _|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|_
L24 |v0   |v1   |v0   |v1   |    |12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12| L0~31
L25 |v0   |v1   |v0   |v1   |    |13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|
L26 |v0   |v1   |v0   |v1   |    |14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|
L27 |v0   |v1   |v0   |v1   |   _|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|_
L28 |v0   |v1   |v0   |v1   |    |12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12| L32~63
L29 |v0   |v1   |v0   |v1   |    |13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|
L30 |v0   |v1   |v0   |v1   |    |14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|
L31 |v0___|v1___|v0___|v1___|   _|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|_
                |
*/
__global__ void 
matrix_core_kernel_standard(const void* __restrict__ ptr_a,
                   const void* __restrict__ ptr_b,
                   void* __restrict__ ptr_c,
                   int stride_a, // stride in unit of pixel
                   int stride_b,
                   int stride_c)
{
    // 32x32x8 gemm, assume only launced 1 wave
    int offset_a = (threadIdx.x / 32 * 4) + (threadIdx.x % 32 * stride_a);
    int offset_b = (threadIdx.x / 32 * 4) + (threadIdx.x % 32 * stride_b);

    fp16x4_t v_a = *reinterpret_cast<const fp16x4_t*>(reinterpret_cast<const fp16_t*>(ptr_a) + offset_a);
    fp16x4_t v_b = *reinterpret_cast<const fp16x4_t*>(reinterpret_cast<const fp16_t*>(ptr_b) + offset_b);
    fp32x16_t v_c = {.0f};  // clear

    v_c = __builtin_amdgcn_mfma_f32_32x32x8f16(v_a, v_b, v_c, 0, 0, 0);

    fp16x16_t v_c_f16;
    for(auto i = 0; i < 16; i++) {
        v_c_f16[i] = static_cast<fp16_t>(v_c[i]);
    }

    int col_id_c = threadIdx.x % 32;
    int row_id_c = threadIdx.x / 32 * 4;
    int offset_c = row_id_c * stride_c + col_id_c;

    for(auto i = 0; i < 16; i++) {
        int row_offset = (i % 4) + (i / 4 * 8);
        *(reinterpret_cast<fp16_t*>(ptr_c) + offset_c + row_offset * stride_c) = v_c_f16[i];
    }
}

__global__ void 
matrix_core_kernel_standard_agpr(const void* __restrict__ ptr_a,
                   const void* __restrict__ ptr_b,
                   void* __restrict__ ptr_c,
                   int stride_a, // stride in unit of pixel
                   int stride_b,
                   int stride_c)
{
    // 32x32x8 gemm, assume only launced 1 wave
    int offset_a = (threadIdx.x / 32 * 4) + (threadIdx.x % 32 * stride_a);
    int offset_b = (threadIdx.x / 32 * 4) + (threadIdx.x % 32 * stride_b);

    auto res_a = make_buffer_resource(ptr_a);
    auto res_b = make_buffer_resource(ptr_b);
    fp16x4_t v_a, v_b;

    asm volatile("buffer_load_dwordx2 %0, %1, %2, 0 offen offset:%3"
            :"+a"(v_a) :  "v"(static_cast<int>(offset_a * sizeof(fp16_t))), "s"(res_a), "n"(0) : "memory");

    asm volatile("buffer_load_dwordx2 %0, %1, %2, 0 offen offset:%3"
            :"+a"(v_b) :  "v"(static_cast<int>(offset_b * sizeof(fp16_t))), "s"(res_b), "n"(0) : "memory");

    fp32x16_t v_c = {.0f};  // clear

#if LOCAL_SCRATCH
    // create 2 local scratch, note this is x8, not x4(purposely)
    fp16x8_t v_aa;
    fp16x8_t v_bb;
    for(auto i = 0; i < 4; i++)
        v_aa[i] = v_a[i];
    for(auto i = 0; i < 4; i++)
        v_bb[i] = v_b[i];

    // Note the local scratch re-assignment is before this waitcnt
    // but this is fine, since finally compiler will remove all such
    // (redundant) movement for us
    asm volatile("s_waitcnt vmcnt(0)"  : : : "memory");
    
    // this is local scratch used for mfma
    fp16x4_t v_ar, v_br;
    for(auto i = 0; i < 4; i++)
        v_ar[i] = v_aa[i];
    
    for(auto i = 0; i < 4; i++)
        v_br[i] = v_bb[i];
    asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %3\n"
                 "s_nop 16"         // TODO: better resolve data dependency
                 : "+v"(v_c)
                 :  "a"(v_ar), "a"(v_br),  "v"(v_c) : );
#else
    asm volatile("s_waitcnt vmcnt(0)"  : : : "memory");
    asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %3\n"
                 "s_nop 16"         // TODO: better resolve data dependency
                 : "+v"(v_c)
                 :  "a"(v_a), "a"(v_b),  "v"(v_c) : );
#endif

    fp16x16_t v_c_f16;
    for(auto i = 0; i < 16; i++) {
        v_c_f16[i] = static_cast<fp16_t>(v_c[i]);
    }

    int col_id_c = threadIdx.x % 32;
    int row_id_c = threadIdx.x / 32 * 4;
    int offset_c = row_id_c * stride_c + col_id_c;

    for(auto i = 0; i < 16; i++) {
        int row_offset = (i % 4) + (i / 4 * 8);
        *(reinterpret_cast<fp16_t*>(ptr_c) + offset_c + row_offset * stride_c) = v_c_f16[i];
    }
}

// kernel-2, swap A/B pointer to transpose C matrix, now we can do vector store
/*
* Note: C matrix now is transposed, we can do vectore store out(assum C fast changing dim is N)

                                 L0 L1 L2 L3 L4 L5 L6 L7 L8 L9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
             (swapped) Matrix A   __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
                                 |v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0| k0  L0~31
                                 |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__| k1
                                 |v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1| k2
                                _|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|_k3
                                 |v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0| k4  L32~63
                                 |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__| k5
                                 |v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1| k6
                                _|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|_k7
     Matrix B (swapped)
     L0~31       L32~63           Matrix C (transposed)
     k0 k1 k2 k3 k4 k5 k6 k7      L0~31       L32~63      L0~31       L32~63      L0~31       L32~63      L0~31       L32~63
     _____ _____|_____ _____      __ __ __ __|__ __ __ __|__ __ __ __|__ __ __ __|__ __ __ __|__ __ __ __|__ __ __ __|__ __ __ __
L0  |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L0 
L1  |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L1 
L2  |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L2 
L3  |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L3 
L4  |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L4 
L5  |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L5 
L6  |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L6 
L7  |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L7 
L8  |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L8 
L9  |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L9 
L10 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L10
L11 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L11
L12 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L12
L13 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L13
L14 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L14
L15 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L15
L16 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L16
L17 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L17
L18 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L18
L19 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L19
L20 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L20
L21 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L21
L22 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L22
L23 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L23
L24 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L24
L25 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L25
L26 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L26
L27 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L27
L28 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L28
L29 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L29
L30 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L30
L31 |v0___|v1___|v0___|v1___|    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L31
                |                            |           |           |           |           |           |           |

*/
__global__ void 
matrix_core_kernel_swap_a_b(const void* __restrict__ ptr_a,
                   const void* __restrict__ ptr_b,
                   void* __restrict__ ptr_c,
                   int stride_a, // stride in unit of pixel
                   int stride_b,
                   int stride_c)
{
    // 32x32x8 gemm, assume only launced 1 wave
    int offset_a = (threadIdx.x / 32 * 4) + (threadIdx.x % 32 * stride_a);
    int offset_b = (threadIdx.x / 32 * 4) + (threadIdx.x % 32 * stride_b);

    fp16x4_t v_a = *reinterpret_cast<const fp16x4_t*>(reinterpret_cast<const fp16_t*>(ptr_a) + offset_a);
    fp16x4_t v_b = *reinterpret_cast<const fp16x4_t*>(reinterpret_cast<const fp16_t*>(ptr_b) + offset_b);
    fp32x16_t v_c = {.0f};  // clear

    v_c = __builtin_amdgcn_mfma_f32_32x32x8f16(v_b, v_a, v_c, 0, 0, 0);

    fp16x16_t v_c_f16;
    for(auto i = 0; i < 16; i++) {
        v_c_f16[i] = static_cast<fp16_t>(v_c[i]);
    }

    int col_id_c = threadIdx.x / 32 * 4; 
    int row_id_c = threadIdx.x % 32;
    int offset_c = row_id_c * stride_c + col_id_c;

    for(auto i = 0; i < (16 / 4); i++) {
        int col_offset = i * 8;
        fp16x4_t tmp;
        tmp.x = v_c_f16[4 * i + 0]; tmp.y = v_c_f16[4 * i + 1];
        tmp.z = v_c_f16[4 * i + 2]; tmp.w = v_c_f16[4 * i + 3];
        *reinterpret_cast<fp16x4_t*>(reinterpret_cast<fp16_t*>(ptr_c) + offset_c + col_offset) = tmp;
    }
}

// kernel-3, swap A/B pointer to transpose C matrix, and swizzle b(vector size is larger)
/*
* Note: C matrix now is transposed, we can do vectore store out(assum C fast changing dim is N), and vector size is larger

                                 L0 L1 L2 L3 L4 L5 L6 L7 L8 L9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
             (swapped) Matrix A   __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
                                 |v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0| k0  L0~31
                                 |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__| k1
                                 |v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1| k2
                                _|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|_k3
                                 |v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0| k4  L32~63
                                 |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__| k5
                                 |v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1| k6
                                _|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|_k7
Matrix B (swapped+swizzled)
     L0~31       L32~63           Matrix C (transposed + increased vector size)
     k0 k1 k2 k3 k4 k5 k6 k7      L0~31                   L32~63                  L0~31                   L32~63 
     _____ _____|_____ _____      __ __ __ __ __ __ __ __|__ __ __ __ __ __ __ __|__ __ __ __ __ __ __ __|__ __ __ __ __ __ __ __
L0  |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L0 
L1  |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L1 
L2  |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L2 
L3  |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L3 
L8 *|v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L4 
L9 *|v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L5 
L10*|v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L6 
L11*|v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L7 
L4 *|v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L8 
L5 *|v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L9 
L6 *|v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L10
L7 *|v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L11
L12 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L12
L13 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L13
L14 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L14
L15 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L15
L16 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L16
L17 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L17
L18 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L18
L19 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L19
L24*|v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L20
L25*|v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L21
L26*|v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L22
L27*|v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L23
L20*|v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L24
L21*|v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L25
L22*|v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L26
L23*|v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L27
L28 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L28
L29 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L29
L30 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L30
L31 |v0___|v1___|v0___|v1___|    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L31
                |                                        |                       |                       |            
*/
__global__ void 
matrix_core_kernel_swap_swb(const void* __restrict__ ptr_a,
                   const void* __restrict__ ptr_b,
                   void* __restrict__ ptr_c,
                   int stride_a, // stride in unit of pixel
                   int stride_b,
                   int stride_c)
{
    // 32x32x8 gemm, assume only launced 1 wave
    int row_group_id_b = threadIdx.x % 32 / 4;
    int row_id_b = threadIdx.x % 4 + row_group_id_b % 2 * 8 + row_group_id_b % 4 / 2 * 4 + row_group_id_b / 4 * 16;
    int offset_a = (threadIdx.x / 32 * 4) + (threadIdx.x % 32 * stride_a);
    int offset_b = (threadIdx.x / 32 * 4) + (row_id_b * stride_b);

    // printf("tid:%d, rid:%d,%d, %d\n", static_cast<int>(threadIdx.x), row_id_b,row_group_id_b,  row_group_id_b % 4 / 2 * 4);

    fp16x4_t v_a = *reinterpret_cast<const fp16x4_t*>(reinterpret_cast<const fp16_t*>(ptr_a) + offset_a);
    fp16x4_t v_b = *reinterpret_cast<const fp16x4_t*>(reinterpret_cast<const fp16_t*>(ptr_b) + offset_b);
    fp32x16_t v_c = {.0f};  // clear

    v_c = __builtin_amdgcn_mfma_f32_32x32x8f16(v_b, v_a, v_c, 0, 0, 0);

    fp16x16_t v_c_f16;
    for(auto i = 0; i < 16; i++) {
        v_c_f16[i] = static_cast<fp16_t>(v_c[i]);
    }

    int col_id_c = threadIdx.x / 32 * 8;
    int row_id_c = threadIdx.x % 32;
    int offset_c = row_id_c * stride_c + col_id_c;

    for(auto i = 0; i < (16 / 8); i++) {
        int col_offset = i * 16;
        fp16x8_t tmp;
        for(auto j = 0; j < 8; j++) tmp[j] = v_c_f16[i * 8 + j];
        *reinterpret_cast<fp16x8_t*>(reinterpret_cast<fp16_t*>(ptr_c) + offset_c + col_offset) = tmp;
    }
}

#ifdef RAND_INT
#define PER_PIXEL_CHECK
#endif

static inline bool valid_vector( const float* ref, const float16* pred, int n, double nrms = 1e-3 )
{    
    double s0=0.0;
    double s1=0.0;
#ifdef PER_PIXEL_CHECK
    int pp_err = 0;
#endif
    int i_start = 0, i_end=n;
    
    for( int i=i_start; i<i_end; ++i ){
        double ri=(double)ref[i];
        double pi=(double)pred[i];
        double d=ri-pi;
        double dd=d*d;
        double rr=2.0*ri*ri;
        s0+=dd;
        s1+=rr;
        
#ifdef PER_PIXEL_CHECK
        double delta = ABS(ri-pi)/ri;
        if(delta>1e-3){
            if(pp_err<100)
                printf("diff at %4d, ref:%lf, pred:%lf(0x%04x), d:%lf\n",i,ri,pi,((uint16_t*)pred)[i],delta);
            pp_err++;
        }
#endif
    }
    // int i_num = i_end - i_start;
    // printf("pp_crr:%d, pp_err:%d, crr_ratio:%.3f, nrms:%lf, s0:%lf, s1:%lf\n",i_num-pp_err, pp_err, (float)(i_num-pp_err)/(float)i_num, sqrt(s0/s1),s0,s1);

    return (sqrt(s0/s1)<nrms)
#ifdef PER_PIXEL_CHECK
        && (pp_err==0)
#endif
    ;
}

void rand_vector_2d(float* v, int row, int col, int ld, float min_v = 0, float max_v = 1){
    int r,c;
    static int flag = 0;
    if(!flag){ srand(time(NULL)); flag = 1; }
    for(r=0;r<row;r++){
        for(c=0;c<col;c++){
            float tmp = float(std::rand()) / float(RAND_MAX);
            v[r*ld+c] = static_cast<float>(min_v + tmp * (max_v - min_v));
            // v[r*ld+c] =   ((float)(r*ld+c)) / (row/2 * col/2) - 5;
        }
    }
}

void rand_vector_2d_int(float* v, int row, int col, int ld){
    int r,c;
    static int flag = 0;
    if(!flag){ srand(time(NULL)); flag = 1; }
    for(r=0;r<row;r++){
        for(c=0;c<col;c++){
            v[r*ld+c] = ((float)(rand() % 10)) - 5;
        }
    }
}

void gemm_rcr(
    const float*  __restrict__ ptr_a,
    const float*  __restrict__ ptr_b,
    float*  ptr_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc)
{
    for(auto i_m = 0 ; i_m < m; i_m++) {
        for(auto i_n = 0; i_n < n; i_n++) {
            float acc = 0;
            for(auto i_k = 0; i_k < k; i_k++) {
                acc += ptr_a[i_m * lda + i_k] * ptr_b[i_n * ldb + i_k];
            }
            ptr_c[i_m * ldc + i_n] = acc;
        }
    }
}

int main(int argc, char ** argv)
{
    int m = 32;
    int n = 32;
    int k = 8;

    int lda = k;
    int ldb = k;
    int ldc = n;

    float *host_a, *host_b, *host_c;
    float16 *fp16_a, *fp16_b, *fp16_c, *dev_a, *dev_b, *dev_c;

    //fp32 on host
    host_a = (float*)malloc(lda*m*sizeof(float));
    host_b = (float*)malloc(ldb*n*sizeof(float));
    host_c = (float*)malloc(ldc*m*sizeof(float));

#ifdef RAND_INT
    rand_vector_2d_int(host_a, m, k, lda);
    rand_vector_2d_int(host_b, n, k, ldb);
#else
    rand_vector_2d(host_a, m, k, lda, 0.0, 1.0);
    rand_vector_2d(host_b, n, k, ldb, -0.5, 0.5);
#endif

    //fp16 on host
    fp16_a = (float16*)malloc(lda*m*sizeof(float16));
    fp16_b = (float16*)malloc(ldb*n*sizeof(float16));
    fp16_c = (float16*)malloc(ldc*m*sizeof(float16));
    //convert fp32 a and b into fp16 on host
    for(int i=0; i<lda*m; i++)fp16_a[i]=__float2half_rn(host_a[i]);
    for(int i=0; i<ldb*n; i++)fp16_b[i]=__float2half_rn(host_b[i]);

    HIP_CALL(hipMalloc(&dev_a, lda*m*sizeof(float16)));
    HIP_CALL(hipMalloc(&dev_b, ldb*n*sizeof(float16)));
    HIP_CALL(hipMalloc(&dev_c, ldc*m*sizeof(float16)));
    //fp16 cpy to device
    HIP_CALL(hipMemcpy(dev_a, fp16_a, lda*m*sizeof(float16), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(dev_b, fp16_b, ldb*n*sizeof(float16), hipMemcpyHostToDevice));

    printf("m:%d,n:%d,k:%d,lda:%d,ldb:%d,ldc:%d\n",  m, n, k, lda, ldb, ldc); fflush(stdout);
    gemm_rcr(host_a, host_b, host_c, m,n,k,lda,ldb,ldc);


    {
        matrix_core_kernel_standard<<<1, 64>>>(dev_a, dev_b, dev_c, lda, ldb, ldc);

        HIP_CALL(hipMemcpy(fp16_c, dev_c, ldc*m*sizeof(float16), hipMemcpyDeviceToHost));
        bool res = valid_vector( host_c, fp16_c, m*n, 1e-3);
        printf("[32x32x8, standard], %s",res?"valid":"fail");fflush(stdout);
        printf("\n"); fflush(stdout);
    }
    {
        matrix_core_kernel_standard_agpr<<<1, 64>>>(dev_a, dev_b, dev_c, lda, ldb, ldc);

        HIP_CALL(hipMemcpy(fp16_c, dev_c, ldc*m*sizeof(float16), hipMemcpyDeviceToHost));
        bool res = valid_vector( host_c, fp16_c, m*n, 1e-3);
        printf("[32x32x8, std_agpr], %s",res?"valid":"fail");fflush(stdout);
        printf("\n"); fflush(stdout);
    }
    {
        matrix_core_kernel_swap_a_b<<<1, 64>>>(dev_a, dev_b, dev_c, lda, ldb, ldc);

        HIP_CALL(hipMemcpy(fp16_c, dev_c, ldc*m*sizeof(float16), hipMemcpyDeviceToHost));
        bool res = valid_vector( host_c, fp16_c, m*n, 1e-3);
        printf("[32x32x8, swap_a_b], %s",res?"valid":"fail");fflush(stdout);
        printf("\n"); fflush(stdout);
    }
    {
        matrix_core_kernel_swap_swb<<<1, 64>>>(dev_a, dev_b, dev_c, lda, ldb, ldc);
        HIP_CALL(hipMemcpy(fp16_c, dev_c, ldc*m*sizeof(float16), hipMemcpyDeviceToHost));
        bool res = valid_vector( host_c, fp16_c, m*n, 1e-3);
        printf("[32x32x8, swap_swb], %s",res?"valid":"fail");fflush(stdout);
        printf("\n"); fflush(stdout);
    }
    
    free(host_a);
    free(host_b);
    free(host_c);
    free(fp16_a);
    free(fp16_b);
    free(fp16_c);
    
    HIP_CALL(hipFree(dev_a));
    HIP_CALL(hipFree(dev_b));
    HIP_CALL(hipFree(dev_c));
}
