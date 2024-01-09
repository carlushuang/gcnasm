#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <random>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <numeric>

#define USE_INLINE_ASM 0
#define DYNAMIC_BUF 1

using index_t = int;

#define MAX(x, y) ((x) > (y) ? (x) : (y))


using fp32 = float;
using fp32x16 = fp32 __attribute__((ext_vector_type(16)));
using fp32x8 = fp32 __attribute__((ext_vector_type(8)));
using fp32x4 = fp32 __attribute__((ext_vector_type(4)));
using fp32x2 = fp32 __attribute__((ext_vector_type(2)));
using fp32x1 = fp32 __attribute__((ext_vector_type(1)));

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
    static constexpr index_t size = N;
    using data_type = T;

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

__device__ fp32 llvm_amdgcn_raw_buffer_load_fp32(dwordx4_t srsrc, index_t voffset, index_t soffset, index_t glc_slc)
                            __asm("llvm.amdgcn.raw.buffer.load.f32");
__device__ fp32x2 llvm_amdgcn_raw_buffer_load_fp32x2(dwordx4_t srsrc, index_t voffset, index_t soffset, index_t glc_slc)
                            __asm("llvm.amdgcn.raw.buffer.load.v2f32");                            
__device__ fp32x4 llvm_amdgcn_raw_buffer_load_fp32x4(dwordx4_t srsrc, index_t voffset, index_t soffset, index_t glc_slc)
                            __asm("llvm.amdgcn.raw.buffer.load.v4f32");

template<> struct gld<16>{
    template<typename T>
    __device__ void operator()(T & value, dwordx4_t res/*buffer resource*/, index_t v_offset, index_t s_offset, index_t i_offset/*max 0xFFF*/, index_t /*flag*/ = 0){
        static_assert(sizeof(T) == 16);
#if USE_INLINE_ASM
        asm volatile("buffer_load_dwordx4 %0, %1, %2, %3 offen offset:%4"
            : "+v"(value.get()) : "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset) : "memory");
#else
        auto tmp = llvm_amdgcn_raw_buffer_load_fp32x4(res, v_offset, s_offset + i_offset, 0);
        value =  __builtin_bit_cast(T, tmp);
#endif
    }
};

template<> struct gld<8>{
    template<typename T>
    __device__ void operator()(T & value, dwordx4_t res/*buffer resource*/, index_t v_offset, index_t s_offset, index_t i_offset/*max 0xFFF*/, index_t /*flag*/ = 0){
        static_assert(sizeof(T) == 8);
#if USE_INLINE_ASM
        asm volatile("buffer_load_dwordx2 %0, %1, %2, %3 offen offset:%4"
            : "+v"(value) : "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset) : "memory");
#else
        auto tmp = llvm_amdgcn_raw_buffer_load_fp32x2(res, v_offset, s_offset + i_offset, 0);
        value =  __builtin_bit_cast(T, tmp);
#endif
    }
};

template<> struct gld<4>{
    template<typename T>
    __device__ void operator()(T & value, dwordx4_t res/*buffer resource*/, index_t v_offset, index_t s_offset, index_t i_offset/*max 0xFFF*/, index_t /*flag*/ = 0){
        static_assert(sizeof(T) == 4);
#if USE_INLINE_ASM
        asm volatile("buffer_load_dword %0, %1, %2, %3 offen offset:%4"
            : "+v"(value) : "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset) : "memory");
#else
        auto tmp = llvm_amdgcn_raw_buffer_load_fp32(res, v_offset, s_offset + i_offset, 0);
        value =  __builtin_bit_cast(T, tmp);
#endif
    }
};

__device__ void gld_fence(index_t cnt)
{
#if USE_INLINE_ASM
    asm volatile("s_waitcnt vmcnt(%0)" : : "n" (cnt) : "memory");
#else
    (void) cnt;
#endif
}

template<typename T, index_t N>
__device__ void gld_fence(vector_type<T, N> & /*buf*/, index_t cnt)
{
#if USE_INLINE_ASM
    asm volatile("s_waitcnt vmcnt(%0)" : : "n" (cnt) : "memory");
    // constexpr index_t total = sizeof(T) * N / sizeof(float);
    // for(auto i = 0; i < total; i++) {
    //     asm volatile("" :"+v"(buf.template as<float>().get(i)) : :);
    // }
#else
    (void) cnt;
#endif
}

template<typename T, index_t N>
__device__ void clear_buf(vector_type<T, N> & buf)
{
#if USE_INLINE_ASM
    for(auto i = 0; i < N; i++)
        asm volatile("v_mov_b32 %0, 0" : "+v"(buf.at(i)) :  : "memory");
#else
    for(auto i = 0; i < N; i++)
        buf.at(i) = .0f;
#endif
}

template<typename T, typename W>
__device__ void v_acc(T & x, const W & a, const W & b)
{
    // TODO: T/W must be vector type
    static_assert(T::size == W::size);
#if USE_INLINE_ASM
    // TODO: force to fp32
    for(auto i = 0; i < T::size; i++)
        asm volatile("v_fmac_f32 %0, %1, %2" : "+v"(x.at(i)) : "v"(a.at(i)), "v"(b.at(i)):);
#else
    for(auto i = 0; i < T::size; i++)
        x.at(i) += a.at(i) * b.at(i);
#endif
}

#if 1
__global__ void
reduce_n2(const void* ptr_a,
        const void* ptr_b,
            void* ptr_dst,
            uint32_t rows)
{
    if(blockIdx.x > 0)
        return;

    int col_offset = threadIdx.x;

    using buf_type = vector_type<float, 4>;

    //buf_type acc {.0f};
    buf_type acc;
    clear_buf(acc);

    static_buffer<buf_type, 2> g_a;
    static_buffer<buf_type, 2> g_b;
    int odd = __builtin_amdgcn_readfirstlane(rows & 1);

    const buf_type * p_a = reinterpret_cast<const buf_type*>(ptr_a);
    const buf_type * p_b = reinterpret_cast<const buf_type*>(ptr_b);
    buf_type * p_dst = reinterpret_cast<buf_type*>(ptr_dst);

    int ir = __builtin_amdgcn_readfirstlane(0);

    gld<sizeof(buf_type)>{}(g_a.get(0), make_buffer_resource(p_a), col_offset * sizeof(buf_type), ir*256*sizeof(buf_type), 0);
    gld<sizeof(buf_type)>{}(g_b.get(0), make_buffer_resource(p_b), col_offset * sizeof(buf_type), ir*256*sizeof(buf_type), 0);
    ir++;
#if DYNAMIC_BUF
    while(ir < rows) {
        gld<sizeof(buf_type)>{}(g_a.get(1), make_buffer_resource(p_a), col_offset * sizeof(buf_type), ir*256*sizeof(buf_type), 0);
        gld<sizeof(buf_type)>{}(g_b.get(1), make_buffer_resource(p_b), col_offset * sizeof(buf_type), ir*256*sizeof(buf_type), 0);
        ir++;
        gld_fence(2);
        v_acc(acc, g_a.get(0), g_b.get(0));
        
        if(ir >= rows)
            break;

        gld<sizeof(buf_type)>{}(g_a.get(0), make_buffer_resource(p_a), col_offset * sizeof(buf_type), ir*256*sizeof(buf_type), 0);
        gld<sizeof(buf_type)>{}(g_b.get(0), make_buffer_resource(p_b), col_offset * sizeof(buf_type), ir*256*sizeof(buf_type), 0);
        ir++;
        gld_fence(2);
        v_acc(acc, g_a.get(1), g_b.get(1));
    }

    if(odd) {
        gld_fence(0);
        v_acc(acc, g_a.get(0), g_b.get(0));
    }
    else {
        gld_fence(0);
        v_acc(acc, g_a.get(1), g_b.get(1));
    }
#else
    if(odd) {
        while(ir < rows) {
            gld<sizeof(buf_type)>{}(g_a.get(1), make_buffer_resource(p_a), col_offset * sizeof(buf_type), ir*256*sizeof(buf_type), 0);
            gld<sizeof(buf_type)>{}(g_b.get(1), make_buffer_resource(p_b), col_offset * sizeof(buf_type), ir*256*sizeof(buf_type), 0);
            ir++;
            gld_fence(2);
            v_acc(acc, g_a.get(0), g_b.get(0));

            gld<sizeof(buf_type)>{}(g_a.get(0), make_buffer_resource(p_a), col_offset * sizeof(buf_type), ir*256*sizeof(buf_type), 0);
            gld<sizeof(buf_type)>{}(g_b.get(0), make_buffer_resource(p_b), col_offset * sizeof(buf_type), ir*256*sizeof(buf_type), 0);
            ir++;
            gld_fence(2);
            v_acc(acc, g_a.get(1), g_b.get(1));
        }
        gld_fence(0);
        v_acc(acc, g_a.get(0), g_b.get(0));
    }
    else {
        gld<sizeof(buf_type)>{}(g_a.get(1), make_buffer_resource(p_a), col_offset * sizeof(buf_type), ir*256*sizeof(buf_type), 0);
        gld<sizeof(buf_type)>{}(g_b.get(1), make_buffer_resource(p_b), col_offset * sizeof(buf_type), ir*256*sizeof(buf_type), 0);
        ir++;
        while(ir < rows) {
            gld_fence(2);
            v_acc(acc, g_a.get(0), g_b.get(0));

            gld<sizeof(buf_type)>{}(g_a.get(0), make_buffer_resource(p_a), col_offset * sizeof(buf_type), ir*256*sizeof(buf_type), 0);
            gld<sizeof(buf_type)>{}(g_b.get(0), make_buffer_resource(p_b), col_offset * sizeof(buf_type), ir*256*sizeof(buf_type), 0);
            ir++;
            gld_fence(2);
            v_acc(acc, g_a.get(1), g_b.get(1));

            gld<sizeof(buf_type)>{}(g_a.get(1), make_buffer_resource(p_a), col_offset * sizeof(buf_type), ir*256*sizeof(buf_type), 0);
            gld<sizeof(buf_type)>{}(g_b.get(1), make_buffer_resource(p_b), col_offset * sizeof(buf_type), ir*256*sizeof(buf_type), 0);
            ir++;
        }
        gld_fence(2);
        v_acc(acc, g_a.get(0), g_b.get(0));
        gld_fence(0);
        v_acc(acc, g_a.get(1), g_b.get(1));
    }
#endif
    p_dst[col_offset] = acc;
}
#endif

// __device__ int amdgcn_if(int v) __asm("llvm.amdgcn.ixxxf");

#if 1
__global__ void
reduce_n3(const void* __restrict__ ptr_a,
            const void* __restrict__ ptr_b,
            void* __restrict__ ptr_dst,
            uint32_t rows)
{
    if(blockIdx.x > 0)
        return;

    int col_offset = threadIdx.x;

    using buf_type = vector_type<float, 4>;

    //buf_type acc {.0f};
    buf_type acc;
    clear_buf(acc);

    static_buffer<buf_type, 3> g_a;
    static_buffer<buf_type, 3> g_b;
    int mod = __builtin_amdgcn_readfirstlane(rows & 2);

    const buf_type * __restrict__ p_a = reinterpret_cast<const buf_type* __restrict__>(ptr_a);
    const buf_type * __restrict__ p_b = reinterpret_cast<const buf_type* __restrict__>(ptr_b);
    buf_type * __restrict__ p_dst = reinterpret_cast<buf_type* __restrict__>(ptr_dst);

    int ir = __builtin_amdgcn_readfirstlane(0);

#define GLD_A(i_buf_) gld<sizeof(buf_type)>{}(g_a.get(i_buf_), make_buffer_resource(p_a), col_offset * sizeof(buf_type), ir*256*sizeof(buf_type), 0)
#define GLD_B(i_buf_) gld<sizeof(buf_type)>{}(g_b.get(i_buf_), make_buffer_resource(p_b), col_offset * sizeof(buf_type), ir*256*sizeof(buf_type), 0);
#define ACC_C(i_buf_) v_acc(acc, g_a.get(i_buf_), g_b.get(i_buf_))

    // assume loop larget than 3
    GLD_A(0); GLD_B(0); ir++;
    GLD_A(1); GLD_B(1); ir++;

#if DYNAMIC_BUF
    while(ir < rows) {
        GLD_A(2); GLD_B(2); ir++;
        gld_fence(4);
        ACC_C(0);
        
        if(ir >= rows) break;

        GLD_A(0); GLD_B(0); ir++;
        gld_fence(4);
        ACC_C(1);

        if(ir >= rows) break;

        GLD_A(1); GLD_B(1); ir++;
        gld_fence(4);
        ACC_C(2);
    }

    if(mod == 0) {
        gld_fence(2);
        ACC_C(1);
        gld_fence(0);
        ACC_C(2);
    }
    else if(mod == 1) {
        gld_fence(2);
        ACC_C(2);
        gld_fence(0);
        ACC_C(0);
    }
    else {
        gld_fence(2);
        ACC_C(0);
        gld_fence(0);
        ACC_C(1);
    }
#else
    if(mod == 0) {
        GLD_A(2); GLD_B(2); ir++;
        while(ir < rows) {
            gld_fence(4);
            ACC_C(0);
            GLD_A(0); GLD_B(0); ir++;

            gld_fence(4);
            ACC_C(1);
            GLD_A(1); GLD_B(1); ir++;

            gld_fence(4);
            ACC_C(2);
            GLD_A(2); GLD_B(2); ir++;
        }
        gld_fence(4);
        ACC_C(0);
        gld_fence(2);
        ACC_C(1);
        gld_fence(0);
        ACC_C(2);
    }
    else if(mod == 1) {
        GLD_A(2); GLD_B(2); ir++;
        gld_fence(4);
        ACC_C(0);
        GLD_A(0); GLD_B(0); ir++;
        while(ir < rows) {
            gld_fence(4);
            ACC_C(1);
            GLD_A(1); GLD_B(1); ir++;

            gld_fence(4);
            ACC_C(2);
            GLD_A(2); GLD_B(2); ir++;

            gld_fence(4);
            ACC_C(0);
            GLD_A(0); GLD_B(0); ir++;
        }
        gld_fence(4);
        ACC_C(1);
        gld_fence(2);
        ACC_C(2);
        gld_fence(0);
        ACC_C(0);
    }
    else {
        GLD_A(2); GLD_B(2); ir++;
        gld_fence(4);
        ACC_C(0);
        GLD_A(0); GLD_B(0); ir++;

        gld_fence(4);
        ACC_C(1);
        GLD_A(1); GLD_B(1); ir++;

        while(ir < rows) {
            gld_fence(4);
            ACC_C(2);
            GLD_A(2); GLD_B(2); ir++;

            gld_fence(4);   
            ACC_C(0);
            GLD_A(0); GLD_B(0); ir++;

            gld_fence(4);
            ACC_C(1);
            GLD_A(1); GLD_B(1); ir++;
        }
        gld_fence(4);
        ACC_C(2);
        gld_fence(2);
        ACC_C(0);
        gld_fence(0);
        ACC_C(1);
    }
#endif
    p_dst[col_offset] = acc;
}
#endif

int main()
{
    
}