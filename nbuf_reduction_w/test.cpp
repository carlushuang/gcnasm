#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <random>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <numeric>

#define CALL(cmd) \
do {\
    hipError_t cuda_error  = cmd;\
    if (cuda_error != hipSuccess) { \
        std::cout<<"'"<<hipGetErrorString(cuda_error)<<"'("<<cuda_error<<")"<<" at "<<__FILE__<<":"<<__LINE__<<std::endl;\
        exit(EXIT_FAILURE);\
    }\
} while(0)

#define UNROLL_BUF 1

using index_t = int;

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


template<typename T_, index_t N_>
struct static_buffer {
    using type = T_;
    static constexpr index_t size = N_;
    type data[size];
    __host__ __device__ auto & get() {return data; }
    __host__ __device__ const auto & get() const {return data; }
    __host__ __device__ auto & get(index_t i) {return data[i]; }
    __host__ __device__ const auto & get(index_t i) const {return data[i]; }

#define SB_COMMON_AS() \
            static_assert(sizeof(type) * size % sizeof(Tx) == 0); \
            constexpr int vx = sizeof(type) * size / sizeof(Tx)

    template<typename Tx>
    __host__ __device__ auto & get_as() {SB_COMMON_AS();
            return reinterpret_cast<static_buffer<Tx, vx>&>(data);}
    template<typename Tx>
    __host__ __device__ const auto & get_as() const {SB_COMMON_AS();
            return reinterpret_cast<const static_buffer<Tx, vx>&>(data);}
    template<typename Tx>
    __host__ __device__ auto & get_as(index_t i) {SB_COMMON_AS();
            return reinterpret_cast<static_buffer<Tx, vx>&>(data).get(i);}
    template<typename Tx>
    __host__ __device__ const auto & get_as(index_t i) const {SB_COMMON_AS();
            return reinterpret_cast<const static_buffer<Tx, vx>&>(data).get(i);}
#undef SB_COMMON_AS
};

template<index_t bytes>
struct gld;

template<> struct gld<16>{
    template<typename T>
    __device__ void operator()(T & value, dwordx4_t res/*buffer resource*/, index_t v_offset, index_t s_offset, index_t i_offset/*max 0xFFF*/, index_t /*flag*/ = 0){
        static_assert(sizeof(T) == 16);
        using v_type = float __attribute__((ext_vector_type(4)));
        asm volatile("buffer_load_dwordx4 %0, %1, %2, %3 offen offset:%4"
            : "+v"(reinterpret_cast<v_type&>(value)) : "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset) : "memory");
    }
};

template<> struct gld<8>{
    template<typename T>
    __device__ void operator()(T & value, dwordx4_t res/*buffer resource*/, index_t v_offset, index_t s_offset, index_t i_offset/*max 0xFFF*/, index_t /*flag*/ = 0){
        static_assert(sizeof(T) == 8);
        using v_type = float __attribute__((ext_vector_type(2)));
        asm volatile("buffer_load_dwordx2 %0, %1, %2, %3 offen offset:%4"
            : "+v"(reinterpret_cast<v_type&>(value)) : "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset) : "memory");
    }
};

template<> struct gld<4>{
    template<typename T>
    __device__ void operator()(T & value, dwordx4_t res/*buffer resource*/, index_t v_offset, index_t s_offset, index_t i_offset/*max 0xFFF*/, index_t /*flag*/ = 0){
        static_assert(sizeof(T) == 4);
        using v_type = float;
        asm volatile("buffer_load_dword %0, %1, %2, %3 offen offset:%4"
            : "+v"(reinterpret_cast<v_type&>(value)) : "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset) : "memory");
    }
};

// need prepare data in unit of dword for this subdword case
template<> struct gld<2>{
    template<typename T>
    __device__ void operator()(T & value, dwordx4_t res/*buffer resource*/, index_t v_offset, index_t s_offset, index_t i_offset/*max 0xFFF*/, index_t /*flag*/ = 0){
        using v_type = float;
        static_assert(sizeof(T) == sizeof(v_type));
        asm volatile("buffer_load_ushort %0, %1, %2, %3 offen offset:%4"
            : "+v"(reinterpret_cast<v_type&>(value)) : "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset) : "memory");
    }
};

// need prepare data in unit of dword for this subdword case
template<> struct gld<1>{
    template<typename T>
    __device__ void operator()(T & value, dwordx4_t res/*buffer resource*/, index_t v_offset, index_t s_offset, index_t i_offset/*max 0xFFF*/, index_t /*flag*/ = 0){
        using v_type = float;
        static_assert(sizeof(T) == sizeof(v_type));
        asm volatile("buffer_load_sbyte %0, %1, %2, %3 offen offset:%4"
            : "+v"(reinterpret_cast<v_type&>(value)) : "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset) : "memory");
    }
};

template<index_t bytes>
struct gst;

template<> struct gst<16>{
    template<typename T>
    __device__ void operator()(const T & value, dwordx4_t res/*buffer resource*/, index_t v_offset, index_t s_offset, index_t i_offset/*max 0xFFF*/, index_t /*flag*/ = 0){
        static_assert(sizeof(T) == 16);
        using v_type = float __attribute__((ext_vector_type(4)));
        asm volatile("buffer_store_dwordx4 %0, %1, %2, %3 offen offset:%4"
            : : "v"(reinterpret_cast<const v_type&>(value)), "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset) : "memory");
    }
};

template<> struct gst<8>{
    template<typename T>
    __device__ void operator()(const T & value, dwordx4_t res/*buffer resource*/, index_t v_offset, index_t s_offset, index_t i_offset/*max 0xFFF*/, index_t /*flag*/ = 0){
        static_assert(sizeof(T) == 8);
        using v_type = float __attribute__((ext_vector_type(2)));
        asm volatile("buffer_store_dwordx2 %0, %1, %2, %3 offen offset:%4"
            : : "v"(reinterpret_cast<const v_type&>(value)), "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset) : "memory");
    }
};

template<> struct gst<4>{
    template<typename T>
    __device__ void operator()(const T & value, dwordx4_t res/*buffer resource*/, index_t v_offset, index_t s_offset, index_t i_offset/*max 0xFFF*/, index_t /*flag*/ = 0){
        static_assert(sizeof(T) == 4);
        using v_type = float;
        asm volatile("buffer_store_dword %0, %1, %2, %3 offen offset:%4"
            : : "v"(reinterpret_cast<const v_type&>(value)), "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset) : "memory");
    }
};

// need prepare data in unit of dword for this subdword case
template<> struct gst<2>{
    template<typename T>
    __device__ void operator()(const T & value, dwordx4_t res/*buffer resource*/, index_t v_offset, index_t s_offset, index_t i_offset/*max 0xFFF*/, index_t /*flag*/ = 0){
        using v_type = float;
        static_assert(sizeof(T) == sizeof(v_type));
        asm volatile("buffer_store_short %0, %1, %2, %3 offen offset:%4"
            : : "v"(reinterpret_cast<const v_type&>(value)), "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset) : "memory");
    }
};

// need prepare data in unit of dword for this subdword case
template<> struct gst<1>{
    template<typename T>
    __device__ void operator()(const T & value, dwordx4_t res/*buffer resource*/, index_t v_offset, index_t s_offset, index_t i_offset/*max 0xFFF*/, index_t /*flag*/ = 0){
        using v_type = float;
        static_assert(sizeof(T) == sizeof(v_type));
        asm volatile("buffer_store_byte %0, %1, %2, %3 offen offset:%4"
            : : "v"(reinterpret_cast<const v_type&>(value)), "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset) : "memory");
    }
};

__device__ void gld_fence(index_t cnt)
{
    asm volatile("s_waitcnt vmcnt(%0)" : : "n" (cnt) : "memory");
}

namespace impl{

template<index_t N>
__device__ void insert_dummy_dep_per_dword(static_buffer<float, N>& buf)
{
    for (auto i = 0; i < buf.size; i++) asm volatile(" " : "+v"(buf.get(i)) : : "memory");
}

#if 0
template<>
__device__ void insert_dummy_dep_per_dword<2>(static_buffer<float, 2>& buf)
{
    asm volatile(" " : "+v"(buf.get(0)), "+v"(buf.get(1)) : : "memory");
}

template<>
__device__ void insert_dummy_dep_per_dword<3>(static_buffer<float, 3>& buf)
{
    asm volatile(" " : "+v"(buf.get(0)), "+v"(buf.get(1)), "+v"(buf.get(2)) : : "memory");
}

template<>
__device__ void insert_dummy_dep_per_dword<4>(static_buffer<float, 4>& buf)
{
    asm volatile(" " : "+v"(buf.get(0)), "+v"(buf.get(1)), "+v"(buf.get(2)), "+v"(buf.get(3)) : : "memory");
}

template<>
__device__ void insert_dummy_dep_per_dword<8>(static_buffer<float, 8>& buf)
{
    asm volatile(" " : "+v"(buf.get(0)), "+v"(buf.get(1)), "+v"(buf.get(2)), "+v"(buf.get(3)),
                       "+v"(buf.get(4)), "+v"(buf.get(5)), "+v"(buf.get(6)), "+v"(buf.get(7)) : : "memory");
}
#endif

__device__ void insert_dummy_dep() {}

template<typename T>
__device__ void insert_dummy_dep(T & buffer)
{
    using da_type = static_buffer<float, (sizeof(T) + 3) / 4>;
    auto & dummy = reinterpret_cast<da_type&>(buffer);
    insert_dummy_dep_per_dword(dummy);
}

template<typename Tx, typename... Ty>
__device__ void insert_dummy_dep(Tx& bx, Ty&... by)
{
    insert_dummy_dep(bx);
    insert_dummy_dep(by...);
}
}

template<typename... T>
__device__ void gld_fence(index_t cnt, T&... o)
{
    asm volatile("s_waitcnt vmcnt(%0)" : : "n" (cnt) : "memory");
    impl::insert_dummy_dep(o...);
}
template<typename T, index_t N>
__device__ void clear_buf(static_buffer<T, N> & buf)
{
#if 0
    for(auto i = 0; i < N; i++)
        asm volatile("v_mov_b32 %0, 0" : "+v"(buf.at(i)) :  : "memory");
#else
    for(auto i = 0; i < N; i++)
        buf.get(i) = static_cast<T>(0);
#endif
}

template<typename T, typename W>
__device__ void v_acc(T & x, const W & a, const W & b)
{
    // TODO: T/W must be vector type
    static_assert(T::size == W::size);
#if 0
    // TODO: force to fp32
    for(auto i = 0; i < T::size; i++)
        asm volatile("v_fmac_f32 %0, %1, %2" : "+v"(x.get(i)) : "v"(a.get(i)), "v"(b.get(i)):);
#else
    for(auto i = 0; i < T::size; i++)
        x.get(i) += a.get(i) * b.get(i);
#endif
}

#ifndef ABS
#define ABS(x) ((x)>0?(x):-1*(x))
#endif
template<typename T>
int valid_vector(const T* lhs, const T * rhs, size_t len, T delta = (T)1e-3){
    size_t i;
    int err_cnt = 0;
    for(i = 0;i < len; i++){
        T d = lhs[i]- rhs[i];
        d = ABS(d);
        if(d > delta){
            printf(" diff at %d, lhs:%f, rhs:%f\n", (int)i, lhs[i], rhs[i]);
            err_cnt++;
        }
    }
    return err_cnt;
}

#if 1
template<typename T, index_t alignment>
__global__ void
reduce_n2(const void* ptr_a,
            const void* ptr_b,
            void* ptr_dst,
            uint32_t rows)
{
    if(blockIdx.x > 0)
        return;

    int col_offset = threadIdx.x;

    using buf_type = static_buffer<T, alignment>;

    buf_type acc;
    clear_buf(acc);

    static_buffer<buf_type, 2> g_a;
    static_buffer<buf_type, 2> g_b;
    int odd = __builtin_amdgcn_readfirstlane(rows & 1);

    const buf_type * p_a = reinterpret_cast<const buf_type*>(ptr_a);
    const buf_type * p_b = reinterpret_cast<const buf_type*>(ptr_b);
    buf_type * p_dst = reinterpret_cast<buf_type*>(ptr_dst);

    int ir = __builtin_amdgcn_readfirstlane(0);

#define GLD_A(i_buf_) gld<sizeof(buf_type)>{}(g_a.get(i_buf_), make_buffer_resource(p_a), col_offset * sizeof(buf_type), ir*256*sizeof(buf_type), 0)
#define GLD_B(i_buf_) gld<sizeof(buf_type)>{}(g_b.get(i_buf_), make_buffer_resource(p_b), col_offset * sizeof(buf_type), ir*256*sizeof(buf_type), 0);
#define ACC_C(i_buf_) v_acc(acc, g_a.get(i_buf_), g_b.get(i_buf_))

#if !UNROLL_BUF
    GLD_A(0); GLD_B(0); ir++;
    while(ir < rows) {
        GLD_A(1); GLD_B(1); ir++;
        gld_fence(2, g_a.get(0), g_b.get(0));
        ACC_C(0);
        
        if(ir >= rows)
            break;

        GLD_A(0); GLD_B(0); ir++;
        gld_fence(2, g_a.get(1), g_b.get(1));
        ACC_C(1);
    }

    if(odd) {
        gld_fence(0, g_a.get(0), g_b.get(0));
        ACC_C(0);
    }
    else {
        gld_fence(0, g_a.get(1), g_b.get(1));
        ACC_C(1);
    }
#else
    if(odd) {
        GLD_A(0); GLD_B(0); ir++;   // if put this to before if...else... will compute error
        while(ir < rows) {
            GLD_A(1); GLD_B(1); ir++;
            gld_fence(2, g_a.get(0), g_b.get(0));
            ACC_C(0);

            GLD_A(0); GLD_B(0); ir++;
            gld_fence(2, g_a.get(1), g_b.get(1));
            ACC_C(1);
        }
        gld_fence(0, g_a.get(0), g_b.get(0));
        ACC_C(0);
    }
    else {
        GLD_A(0); GLD_B(0); ir++;
        GLD_A(1); GLD_B(1); ir++;
        while(ir < rows) {
            gld_fence(2, g_a.get(0), g_b.get(0));
            ACC_C(0);

            GLD_A(0); GLD_B(0); ir++;
            gld_fence(2, g_a.get(1), g_b.get(1));
            ACC_C(1);

            GLD_A(1); GLD_B(1); ir++;
        }
        gld_fence(2, g_a.get(0), g_b.get(0));
        ACC_C(0);
        gld_fence(0, g_a.get(1), g_b.get(1));
        ACC_C(1);
    }
#endif
    gst<sizeof(buf_type)>{}(acc.get(), make_buffer_resource(p_dst), col_offset * sizeof(buf_type), 0, 0);


#undef GLD_A
#undef GLD_B
#undef ACC_C

}
#endif


#if 1
template<typename T, index_t alignment>
__global__ void
reduce_n3(const void* __restrict__ ptr_a,
            const void* __restrict__ ptr_b,
            void* __restrict__ ptr_dst,
            uint32_t rows)
{
    if(blockIdx.x > 0)
        return;

    int col_offset = threadIdx.x;

    using buf_type = static_buffer<T, alignment>;

    buf_type acc;
    clear_buf(acc);

    static_buffer<buf_type, 3> g_a;
    static_buffer<buf_type, 3> g_b;
    int mod = __builtin_amdgcn_readfirstlane(rows % 3);

    const buf_type * __restrict__ p_a = reinterpret_cast<const buf_type* __restrict__>(ptr_a);
    const buf_type * __restrict__ p_b = reinterpret_cast<const buf_type* __restrict__>(ptr_b);
    buf_type * __restrict__ p_dst = reinterpret_cast<buf_type* __restrict__>(ptr_dst);

    int ir = __builtin_amdgcn_readfirstlane(0);

#define GLD_A(i_buf_) gld<sizeof(buf_type)>{}(g_a.get(i_buf_), make_buffer_resource(p_a), col_offset * sizeof(buf_type), ir*256*sizeof(buf_type), 0)
#define GLD_B(i_buf_) gld<sizeof(buf_type)>{}(g_b.get(i_buf_), make_buffer_resource(p_b), col_offset * sizeof(buf_type), ir*256*sizeof(buf_type), 0);
#define ACC_C(i_buf_) v_acc(acc, g_a.get(i_buf_), g_b.get(i_buf_))

    // assume loop larget than 3

#if !UNROLL_BUF
    GLD_A(0); GLD_B(0); ir++;
    GLD_A(1); GLD_B(1); ir++;
    while(ir < rows) {
        GLD_A(2); GLD_B(2); ir++;
        gld_fence(4, g_a.get(0), g_b.get(0));
        ACC_C(0);
        
        if(ir >= rows) break;

        GLD_A(0); GLD_B(0); ir++;
        gld_fence(4, g_a.get(1), g_b.get(1));
        ACC_C(1);

        if(ir >= rows) break;

        GLD_A(1); GLD_B(1); ir++;
        gld_fence(4, g_a.get(2), g_b.get(2));
        ACC_C(2);
    }

    if(mod == 0) {
        gld_fence(2, g_a.get(1), g_b.get(1));
        ACC_C(1);
        gld_fence(0, g_a.get(2), g_b.get(2));
        ACC_C(2);
    }
    else if(mod == 1) {
        gld_fence(2, g_a.get(2), g_b.get(2));
        ACC_C(2);
        gld_fence(0, g_a.get(0), g_b.get(0));
        ACC_C(0);
    }
    else {
        gld_fence(2, g_a.get(0), g_b.get(0));
        ACC_C(0);
        gld_fence(0, g_a.get(1), g_b.get(1));
        ACC_C(1);
    }
#else
    if(mod == 0) {
        GLD_A(0); GLD_B(0); ir++;
        GLD_A(1); GLD_B(1); ir++;
        GLD_A(2); GLD_B(2); ir++;
        while(ir < rows) {
            gld_fence(4, g_a.get(0), g_b.get(0));
            ACC_C(0);
            GLD_A(0); GLD_B(0); ir++;

            gld_fence(4, g_a.get(1), g_b.get(1));
            ACC_C(1);
            GLD_A(1); GLD_B(1); ir++;

            gld_fence(4, g_a.get(2), g_b.get(2));
            ACC_C(2);
            GLD_A(2); GLD_B(2); ir++;
        }
        gld_fence(4, g_a.get(0), g_b.get(0));
        ACC_C(0);
        gld_fence(2, g_a.get(1), g_b.get(1));
        ACC_C(1);
        gld_fence(0, g_a.get(2), g_b.get(2));
        ACC_C(2);
    }
    else if(mod == 1) {
        GLD_A(0); GLD_B(0); ir++;
        GLD_A(1); GLD_B(1); ir++;
        GLD_A(2); GLD_B(2); ir++;
        gld_fence(4, g_a.get(0), g_b.get(0));
        ACC_C(0);
        GLD_A(0); GLD_B(0); ir++;
        while(ir < rows) {
            gld_fence(4, g_a.get(1), g_b.get(1));
            ACC_C(1);
            GLD_A(1); GLD_B(1); ir++;

            gld_fence(4, g_a.get(2), g_b.get(2));
            ACC_C(2);
            GLD_A(2); GLD_B(2); ir++;

            gld_fence(4, g_a.get(0), g_b.get(0));
            ACC_C(0);
            GLD_A(0); GLD_B(0); ir++;
        }
        gld_fence(4, g_a.get(1), g_b.get(1));
        ACC_C(1);
        gld_fence(2, g_a.get(2), g_b.get(2));
        ACC_C(2);
        gld_fence(0, g_a.get(0), g_b.get(0));
        ACC_C(0);
    }
    else {
        GLD_A(0); GLD_B(0); ir++;
        GLD_A(1); GLD_B(1); ir++;
        GLD_A(2); GLD_B(2); ir++;
        gld_fence(4, g_a.get(0), g_b.get(0));
        ACC_C(0);
        GLD_A(0); GLD_B(0); ir++;

        gld_fence(4, g_a.get(1), g_b.get(1));
        ACC_C(1);
        GLD_A(1); GLD_B(1); ir++;

        while(ir < rows) {
            gld_fence(4, g_a.get(2), g_b.get(2));
            ACC_C(2);
            GLD_A(2); GLD_B(2); ir++;

            gld_fence(4, g_a.get(0), g_b.get(0));   
            ACC_C(0);
            GLD_A(0); GLD_B(0); ir++;

            gld_fence(4, g_a.get(1), g_b.get(1));
            ACC_C(1);
            GLD_A(1); GLD_B(1); ir++;
        }
        gld_fence(4, g_a.get(2), g_b.get(2));
        ACC_C(2);
        gld_fence(2, g_a.get(0), g_b.get(0));
        ACC_C(0);
        gld_fence(0, g_a.get(1), g_b.get(1));
        ACC_C(1);
    }
#endif
    gst<sizeof(buf_type)>{}(acc.get(), make_buffer_resource(p_dst), col_offset * sizeof(buf_type), 0, 0);

#undef GLD_A
#undef GLD_B
#undef ACC_C

}
#endif

template<typename T, int num_col>
void host_reduce(const T* a, const T* b, T* dst, uint32_t rows)
{
    for(auto ic = 0; ic < num_col; ic++) {
        T acc = 0;
        for(auto ir = 0; ir < rows; ir++) {
            index_t idx = ir * num_col + ic;
            acc += a[idx] * b[idx];
        }
        dst[ic] = acc;
    }
}

template<typename T>
void rand_vector(T* v, int num){
    static int flag = 0;
    if(!flag){ srand(time(NULL)); flag = 1; }

    for(int i = 0; i < num; i++){
        float value = (((float)(rand() % 20)) / 10.0f) - 10.f;
        v[i] = static_cast<T>(value);
    }
}

template<typename T, int alignment, int block_size, int nprefetch /* 2 or 3*/>
struct test_reduce{
    void operator()(uint32_t rows){
        using vec_t = static_buffer<T, alignment>;
        vec_t *A, *B, *Dst;

        vec_t * h_A = (vec_t*)malloc(rows * block_size*sizeof(vec_t));
        vec_t * h_B = (vec_t*)malloc(rows * block_size*sizeof(vec_t));
        vec_t * h_Dst = (vec_t*)malloc(block_size*sizeof(vec_t));
        vec_t * h_Dst_host = (vec_t*)malloc(block_size*sizeof(vec_t));

        // for(auto i = 0; i < rows * block_size; i++) { h_A[i] = vec_t{static_cast<T>(i + 1)}; }
        // for(auto i = 0; i < rows * block_size; i++) { h_B[i] = vec_t{static_cast<T>(i + 1)}; }
        rand_vector(reinterpret_cast<T*>(h_A), rows * block_size * alignment);
        rand_vector(reinterpret_cast<T*>(h_B), rows * block_size * alignment);

        CALL(hipMalloc(&A, rows * block_size*sizeof(vec_t)));
        CALL(hipMalloc(&B, rows * block_size*sizeof(vec_t)));
        CALL(hipMalloc(&Dst, block_size*sizeof(vec_t)));
        CALL(hipMemcpy(A, h_A, rows * block_size*sizeof(vec_t), hipMemcpyHostToDevice));
        CALL(hipMemcpy(B, h_B, rows * block_size*sizeof(vec_t), hipMemcpyHostToDevice));

        if constexpr (nprefetch == 2)
            reduce_n2<T, alignment><<<1, block_size>>>(A, B, Dst, rows);
        else if constexpr (nprefetch == 3)
            reduce_n3<T, alignment><<<1, block_size>>>(A, B, Dst, rows); 
        CALL(hipMemcpy(h_Dst, Dst, block_size * sizeof(vec_t), hipMemcpyDeviceToHost));

        host_reduce<T, alignment * block_size>(reinterpret_cast<const T*>(h_A), reinterpret_cast<const T*>(h_B),
                                                reinterpret_cast<T*>(h_Dst_host), rows);

        int err = valid_vector(reinterpret_cast<T*>(h_Dst), reinterpret_cast<T*>(h_Dst_host),
                                    alignment * block_size);
        printf("row:%d, col:%d, prefetch:%d, %s\n", rows, alignment * block_size, nprefetch, err == 0 ? "valid":"error");
        fflush(stdout);

        free(h_A); free(h_B); free(h_Dst); free(h_Dst_host);
        CALL(hipFree(A)); CALL(hipFree(B)); CALL(hipFree(Dst));
    }
};

int main(int argc, char ** argv)
{
    if (argc > 1) {
        int rows = atoi(argv[1]);
        if (rows <= 3) {
            printf("not support rows smaller than 3(for 3 buffer)");
            return -1;
        }
        test_reduce<float, 4, 256, 2>{}(rows);
        test_reduce<float, 4, 256, 3>{}(rows);
    }
    else {
        for(auto i = 0; i < 7; i++) {
            int rows = rand() % 55 + 3;
            test_reduce<float, 4, 256, 2>{}(rows);
            test_reduce<float, 4, 256, 3>{}(rows);
        }
    }
}