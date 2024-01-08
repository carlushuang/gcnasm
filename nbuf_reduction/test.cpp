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

template<index_t bytes>
struct gld;

template<> struct gld<4>{
    template<typename T>
    __device__ void operator()(T & value, dwordx4_t res/*buffer resource*/, index_t v_offset, index_t s_offset, index_t i_offset/*max 0xFFF*/, index_t /*flag*/ = 0){
        static_assert(sizeof(T) == 4);
        asm volatile("buffer_load_dword %0, %1, %2, %3 offen offset:%4"
            : "+v"(value) : "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset) : "memory");
    }
};

__device__ void gld_fence(index_t cnt)
{
    asm volatile("s_waitcnt vmcnt(%0)" : : "n" (cnt) : "memory");
}

template<typename T, index_t N>
__device__ void gld_fence(vector_type<T, N> & /*buf*/, index_t cnt)
{
    asm volatile("s_waitcnt vmcnt(%0)" : : "n" (cnt) : "memory");
    // constexpr index_t total = sizeof(T) * N / sizeof(float);
    // for(auto i = 0; i < total; i++) {
    //     asm volatile("" :"+v"(buf.template as<float>().get(i)) : :);
    // }
}

template<typename T, typename W>
__device__ void v_acc(T & x, const W & w)
{
    asm volatile("v_add_f32 %0, %1, %0" : "+v"(x) : "v"(w) :);
}

__global__ void
__launch_bounds__(256, 2)
reduce(const void* ptr_src,
            void* ptr_dst,
            uint32_t rows)
{
    if(blockIdx.x > 0)
        return;

    float acc = .0f;

    int col_offset = threadIdx.x;

    using buf_type = vector_type<float, 1>;

    static_buffer<buf_type, 2> gbuf;
    bool odd = __builtin_amdgcn_readfirstlane(rows & 1);

    const float * p_src = reinterpret_cast<const float*>(ptr_src);
    float * p_dst = reinterpret_cast<float*>(ptr_dst);

    int ir = __builtin_amdgcn_readfirstlane(0);

    gld<4>{}(gbuf.get(0), make_buffer_resource(p_src), col_offset * sizeof(float), ir*256*sizeof(float), 0);
    ir++;

    while(ir < rows) {
        gld<4>{}(gbuf.get(1), make_buffer_resource(p_src), col_offset * sizeof(float), ir*256*sizeof(float), 0);
        ir++;
        gld_fence(gbuf.get(0), 1);
        //acc += gbuf.get(0).template at<float>(0);
        v_acc(acc, gbuf.get(0));
        
        if(ir >= rows)
            break;

        gld<4>{}(gbuf.get(0), make_buffer_resource(p_src), col_offset * sizeof(float), ir*256*sizeof(float), 0);
        ir++;
        gld_fence(gbuf.get(1), 1);
        // acc += gbuf.get(1).template at<float>(0);
        v_acc(acc, gbuf.get(1));
    }

    if(odd) {
        gld_fence(gbuf.get(0), 0);
        // acc += gbuf.get(0).template at<float>(0);
        v_acc(acc, gbuf.get(0));
    }
    else {
        gld_fence(gbuf.get(1), 0);
        // acc += gbuf.get(1).template at<float>(0);
        v_acc(acc, gbuf.get(1));
    }

    p_dst[col_offset] = acc;
}


int main()
{
    
}