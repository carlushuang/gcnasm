#include <hip/hip_runtime.h>
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>

using f32 = float;
// using f16 = _Float16;

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;

using index_t = u32;

typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
typedef uint32_t u32x1 __attribute__((ext_vector_type(1)));

typedef f32 f32x8 __attribute__((ext_vector_type(8)));
typedef f32 f32x4 __attribute__((ext_vector_type(4)));
typedef f32 f32x2 __attribute__((ext_vector_type(2)));
typedef f32 f32x1 __attribute__((ext_vector_type(1)));

typedef u32x4 dwordx4_t;
typedef u32x2 dwordx2_t;
typedef u32   dword_t;

typedef uint8_t u8x16 __attribute__((ext_vector_type(16)));
typedef uint8_t u8x8  __attribute__((ext_vector_type(8)));
typedef uint8_t u8x4  __attribute__((ext_vector_type(4)));
typedef uint8_t u8x2  __attribute__((ext_vector_type(2)));
typedef uint8_t u8x1  __attribute__((ext_vector_type(1)));


#define BUFFER_LOAD_DWORD3 0x00020000   // This is valid for 
struct buffer_resource {
    const void * ptr;
    dword_t range;
    dword_t config;
};
__device__ dwordx4_t make_buffer_resource(const void * ptr, uint32_t size = 0xffffffff)
{
    buffer_resource res {ptr, size, BUFFER_LOAD_DWORD3};
    return __builtin_bit_cast(dwordx4_t, res);
}

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


__device__ void gld_fence(index_t cnt)
{
    asm volatile("s_waitcnt vmcnt(%0)" : : "n" (cnt) : "memory");
}


template<typename T, index_t force_buffer_size = sizeof(T)>
__global__
void oob_test_kernel(T* __restrict__ dst, const T* __restrict__ src){
    if(blockIdx.x == 0 && threadIdx.x == 0) {
        T data;
        gld<sizeof(T)>{}(data, make_buffer_resource(src, force_buffer_size), 0, 0, 0);
        gld_fence(0);
        gst<sizeof(T)>{}(data, make_buffer_resource(dst), 0, 0, 0);
    }
}

#define CALL(cmd) \
do {\
    hipError_t cuda_error  = cmd;\
    if (cuda_error != hipSuccess) { \
        std::cout<<"'"<<hipGetErrorString(cuda_error)<<"'("<<cuda_error<<")"<<" at "<<__FILE__<<":"<<__LINE__<<std::endl;\
        exit(EXIT_FAILURE);\
    }\
} while(0)

namespace impl {
    template<typename T, int N>
    struct to_vec { using type = T __attribute__((ext_vector_type(N))); };
}
template<typename T, int N>
using to_vec_t = typename impl::to_vec<T, N>::type;

template<int total_bytes, int force_bytes>
struct test_oob{
    void operator()(){
        using vec_type = to_vec_t<uint8_t, total_bytes>;
        vec_type *A, *B;

        uint8_t * h_A = (uint8_t*)malloc(total_bytes*sizeof(uint8_t));
        uint8_t * h_B = (uint8_t*)malloc(total_bytes*sizeof(uint8_t));

        for(auto i = 0; i < total_bytes; i++) { h_A[i] = i + 1; }

        CALL(hipMalloc(&A, total_bytes * sizeof(uint8_t)));
        CALL(hipMalloc(&B, total_bytes * sizeof(uint8_t)));
        CALL(hipMemcpy(A, h_A, total_bytes * sizeof(uint8_t), hipMemcpyHostToDevice));

        oob_test_kernel<vec_type, force_bytes><<<1, 64>>>(B, A); 
        CALL(hipMemcpy(h_B, B, total_bytes * sizeof(uint8_t), hipMemcpyDeviceToHost));
        printf("[%2d/%2d] ", force_bytes, total_bytes);
        for(auto i = 0; i < total_bytes; i++) printf("%02x ", h_A[i]);
        printf(" -> ");
        for(auto i = 0; i < total_bytes; i++) printf("%02x ", h_B[i]);
        printf("\n"); fflush(stdout);

        free(h_A); free(h_B);
        CALL(hipFree(A)); CALL(hipFree(B));
    }
};


int main(int argc, char ** argv) {
    printf("buffer_load_dwordx4 ------\n");
    test_oob<16, 16>{}();
    test_oob<16, 14>{}();
    test_oob<16, 11>{}();
    test_oob<16, 8>{}();
    test_oob<16, 6>{}();
    test_oob<16, 4>{}();
    test_oob<16, 3>{}();
    test_oob<16, 2>{}();
    test_oob<16, 1>{}();

    printf("buffer_load_dwordx2 ------\n");
    test_oob<8, 8>{}();
    test_oob<8, 6>{}();
    test_oob<8, 4>{}();
    test_oob<8, 3>{}();
    test_oob<8, 2>{}();
    test_oob<8, 1>{}();

    printf("buffer_load_dword ------\n");
    test_oob<4, 4>{}();
    test_oob<4, 3>{}();
    test_oob<4, 2>{}();
    test_oob<4, 1>{}();
}
