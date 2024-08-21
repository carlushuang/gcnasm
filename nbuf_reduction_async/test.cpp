#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <random>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <numeric>

#ifndef USE_BUF_RSRC_BUILTIN
#define USE_BUF_RSRC_BUILTIN 1
#endif

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


#define BUFFER_LOAD_DWORD3 0x00020000
struct buffer_resource {
    const void * ptr;
    uint32_t range;
    uint32_t config;
};


#if USE_BUF_RSRC_BUILTIN
__device__ __attribute__((address_space(8))) void* llvm_amdgcn_make_buffer_rsrc(void * ptr, int16_t stride_16bit, int32_t num_records, int32_t flags)
                            __asm("llvm.amdgcn.make.buffer.rsrc");
#endif

__device__ auto make_buffer_resource(const void * ptr)
{
#if USE_BUF_RSRC_BUILTIN
    // TODO: here we skip the const qualifier
    return llvm_amdgcn_make_buffer_rsrc(const_cast<void*>(ptr), 0, 0xffffffff, BUFFER_LOAD_DWORD3);
#else
    buffer_resource res {ptr, 0xffffffff, BUFFER_LOAD_DWORD3};
    return __builtin_bit_cast(dwordx4_t, res);
#endif
}

__device__ void
llvm_amdgcn_raw_buffer_load_lds(
#if USE_BUF_RSRC_BUILTIN
    __attribute__((address_space(8))) void*,
#else
    dwordx4_t rsrc,
#endif
    __attribute__((address_space(3))) uint32_t* lds_ptr,
    index_t size,
    index_t voffset,
    index_t soffset,
    index_t offset,
    index_t aux) __asm("llvm.amdgcn.raw.buffer.load.lds");

__device__ float llvm_amdgcn_raw_buffer_load_fp32(
#if USE_BUF_RSRC_BUILTIN
__attribute__((address_space(8))) void*,
#else
dwordx4_t srsrc,
#endif
index_t voffset, index_t soffset, index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.f32");

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
template <typename T>
using remove_cv_t = typename std::remove_cv<T>::type;

template <typename T>
using remove_cvref_t = remove_cv_t<std::remove_reference_t<T>>;


// #define SPTR(_ptr_) reinterpret_cast<__attribute__((address_space(3))) uint32_t*>(reinterpret_cast<uintptr_t>(_ptr_))
#define SPTR(_ptr_) reinterpret_cast<__attribute__((address_space(3))) uint32_t*>(_ptr_)
#define FPTR(_ptr_) reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(_ptr_))


// TODO: for simplicity assume rows is 2x
template<int block_size=256>
__global__ void reduce_fp32_n2(const void* ptr_a,
            const void* ptr_b,
            void* ptr_dst,
            uint32_t rows,
            uint32_t row_stride)
{
    if(blockIdx.x > 0)
        return;

    float a[2];
    float b[2];

    float acc = .0f;

    auto a_res = make_buffer_resource(ptr_a);
    auto b_res = make_buffer_resource(ptr_b);
    int wave_id = __builtin_amdgcn_readfirstlane(threadIdx.x / 64);
    int lane_id = threadIdx.x % 64;
    int col_id = threadIdx.x;
    int col_id_swi = (3 - wave_id) * 64 + lane_id;   // swizzle col id inorder to test s_barrier
    // int col_id_swi = col_id;
    constexpr index_t WAVE_LEN = 65;    // Note: with padding!!
    int col_id_swi_dr = (3 - wave_id) * WAVE_LEN + lane_id;

    // __attribute__((address_space(3))) is needed otherwise compiler will not properly figure out the dependency
    // ... and result in 2 smem merged into a unified smem
    __shared__ __attribute__((address_space(3))) float smem_a0[4 * WAVE_LEN];
    __shared__ __attribute__((address_space(3))) float smem_a1[4 * WAVE_LEN];


    b[0] = llvm_amdgcn_raw_buffer_load_fp32(b_res, col_id_swi * sizeof(float), 0, 0);
    llvm_amdgcn_raw_buffer_load_lds(a_res, SPTR(smem_a0 + wave_id * WAVE_LEN), sizeof(uint32_t), col_id * sizeof(float), 0, 0, 0);

    uint32_t i_r = 1;
    while(i_r < (rows - 1)) {
        b[1] = llvm_amdgcn_raw_buffer_load_fp32(b_res, (i_r * row_stride + col_id_swi) * sizeof(float), 0, 0);
        llvm_amdgcn_raw_buffer_load_lds(a_res, SPTR(smem_a1 + wave_id * WAVE_LEN), sizeof(uint32_t), (i_r * row_stride + col_id) * sizeof(float), 0, 0, 0);
        
        i_r++;
        __builtin_amdgcn_s_waitcnt(0x0f72); // vmcnt(2), must add, otherwise compiler will generate vmcnt(*) after s_barrier
        __builtin_amdgcn_s_barrier();
        a[0] = smem_a0[col_id_swi_dr];
        acc += a[0] * b[0];

        b[0] = llvm_amdgcn_raw_buffer_load_fp32(b_res, (i_r * row_stride + col_id_swi) * sizeof(float), 0, 0);
        llvm_amdgcn_raw_buffer_load_lds(a_res, SPTR(smem_a0 + wave_id * WAVE_LEN), sizeof(uint32_t), (i_r * row_stride + col_id) * sizeof(float), 0, 0, 0);

        i_r++;
        __builtin_amdgcn_s_waitcnt(0x0f72); // vmcnt(2)
        __builtin_amdgcn_s_barrier();
        a[1] = smem_a1[col_id_swi_dr];
        acc += a[1] * b[1];
    }

    b[1] = llvm_amdgcn_raw_buffer_load_fp32(b_res, (i_r * row_stride + col_id_swi) * sizeof(float), 0, 0);
    llvm_amdgcn_raw_buffer_load_lds(a_res, SPTR(smem_a1 + wave_id * WAVE_LEN), sizeof(uint32_t), (i_r * row_stride + col_id) * sizeof(float), 0, 0, 0);

    __builtin_amdgcn_s_waitcnt(0x0f72);  // vmcnt(2)
    __builtin_amdgcn_s_barrier();
    a[0] = smem_a0[col_id_swi_dr];
    acc += a[0] * b[0];

    __builtin_amdgcn_s_waitcnt(0x0f70);  // vmcnt(0)
    __builtin_amdgcn_s_barrier();
    a[1] = smem_a1[col_id_swi_dr];
    acc += a[1] * b[1];

    reinterpret_cast<float*>(ptr_dst)[col_id_swi] = acc;
}

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
        using vec_t = float;
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
            reduce_fp32_n2<block_size><<<1, block_size>>>(A, B, Dst, rows, block_size);
        // else if constexpr (nprefetch == 3)
        //     reduce_n3<T, alignment><<<1, block_size>>>(A, B, Dst, rows); 
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
        if (rows <= 3 || rows % 2 != 0) {
            printf("not support rows smaller than 3(for 3 buffer)");
            return -1;
        }
        test_reduce<float, 1, 256, 2>{}(rows);
        // test_reduce<float, 4, 256, 3>{}(rows);
    }
    else {
        for(auto i = 0; i < 7; i++) {
            int rows = (rand() % 13 + 3) * 2;
            test_reduce<float, 1, 256, 2>{}(rows);
            // test_reduce<float, 4, 256, 3>{}(rows);
        }
    }
}