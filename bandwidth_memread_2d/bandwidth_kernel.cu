#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>

#ifdef __NVCC__
using fp32 =  float;
using fp32x2 = float2;
using fp32x4 = float4;
#else
using fp32 = __attribute__((__ext_vector_type__(1))) float;
using fp32x2 = __attribute__((__ext_vector_type__(2))) float;
using fp32x4 = __attribute__((__ext_vector_type__(4))) float;
#endif

#define BLOCK_SIZE 1024

#ifndef USE_NT_LOAD
#define USE_NT_LOAD 1
#endif

#if USE_NT_LOAD
template<typename T>
__device__ __forceinline__ T nt_load(const T& ref)
{
#ifdef __HIPCC__
    return __builtin_nontemporal_load(&ref);
#else
    return ref;
#endif
}
#endif

template<typename T>
__device__  __forceinline__ void acc(T& v, const T& other)
{
    constexpr int vec = sizeof(T) / sizeof(float);
    if constexpr(vec == 1) {
        v += other;
    }
    if constexpr(vec == 4) {
        v.x += other.x;
        v.y += other.y;
        v.z += other.z;
        v.w += other.w;
    }
}

static inline __device__ uint32_t floatAsSortableUint(float x) {
  uint32_t bits = __float_as_uint(x);
  bits = (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;
  return bits;
}

template <int step>
static inline __device__ uint32_t extractBinIdx(float x) {
  uint32_t bits = floatAsSortableUint(x);

  if constexpr (step == 0) {
    return bits >> 26;
  } else if constexpr (step == 1) {
    return (bits >> 10) & 0x7ff;
  } else {
    return bits & 0x3ff;
  }
}

// 1 WG 1 row
template<typename DType, int vec = 1, int need_bin = 0>
__global__
void memread_kernel_2d(DType* p_src, DType* p_dst, int rows, int cols, int stride)
{
    int i_row = blockIdx.x;
    p_src += i_row * stride;
    DType v {};

    __shared__ DType smemHistogram[64];

    int iters = cols / vec / BLOCK_SIZE;

    for(auto i = 0; i < iters; i++) {
        auto offs = threadIdx.x + i * BLOCK_SIZE;
#if USE_NT_LOAD
        auto f = nt_load(p_src[offs]);
#else
        auto f = p_src[offs];
#endif
        acc(v, f);

        if constexpr (need_bin) {
            uint32_t binIdx = extractBinIdx<0>(f);
            // if(threadIdx.x == 0)
            atomicAdd(&smemHistogram[binIdx], 1);
            // smemHistogram[binIdx] = smemHistogram[binIdx] + 1;
        }
    }

    // constexpr int vec = sizeof(T) / sizeof(float);
    if constexpr(vec == 1) {
        if(v == 10000) {
            *p_dst  = v;
            *(p_dst + threadIdx.x + blockIdx.x * BLOCK_SIZE) = smemHistogram[threadIdx.x];
        }

    }
    if constexpr(vec == 4) {
        if(v.x == 10000 && v.y == 10000 && v.z == 10000 && v.w == 10000)
            *p_dst  = v;
    }
}

#define CALL(cmd) \
do {\
    cudaError_t cuda_error  = cmd;\
    if (cuda_error != cudaSuccess) { \
        std::cout<<"'"<<cudaGetErrorString(cuda_error)<<"'("<<cuda_error<<")"<<" at "<<__FILE__<<":"<<__LINE__<<std::endl;\
        exit(EXIT_FAILURE);\
    }\
} while(0)

#define WARMUP 25
#define LOOP 100

static inline void b2s(size_t bytes, char * str){
	if(bytes<1024){
		sprintf(str, "%luB", bytes);
	}else if(bytes<(1024*1024)){
		double b= (double)bytes/1024.0;
		sprintf(str, "%.2fKB", b);
	}else if(bytes<(1024*1024*1024)){
		double b= (double)bytes/(1024.0*1024);
		sprintf(str, "%.2fMB", b);
	}else{
		double b= (double)bytes/(1024.0*1024*1024);
		sprintf(str, "%.2fGB", b);
	}
}

static inline int env_get_int(const char * var_name, int def_v)
{
    char * v = getenv(var_name);
    int r = def_v;
    if(v)
        r = atoi(v);
    return r;
}
static inline float get_rand(){
    static int inited = 0;
    float v;
    if(!inited){ srand(time(NULL)); inited = 1; }
    v = rand() % 1000 + 1;
    return v / 1000.0f;
}

static inline int valid_vec(const float * vec_a, const float * vec_b, int num)
{
    int err_cnt = 0;
    for(int i=0;i<num;i++){
        if(vec_a[i] != vec_b[i])
            err_cnt++;
    }
    return err_cnt;
}

template<typename DType, int vec = 1, int need_bin = 0>
float bench_memread_kernel(void * B, void * A, int rows, int cols, int stride)
{
    // benchmark kernel
    // TODO: ignore check can even divide or not
    int bx = BLOCK_SIZE;
    int gx = rows;

    cudaEvent_t start_ev, stop_ev;
    CALL(cudaEventCreate(&start_ev));
    CALL(cudaEventCreate(&stop_ev));

    for(int i=0;i<WARMUP;i++)
        memread_kernel_2d<DType, vec, need_bin><<<gx, bx>>>(reinterpret_cast<DType*>(B), reinterpret_cast<DType*>(A), rows, cols, stride);

    CALL(cudaEventRecord(start_ev, 0));
    for(int i=0;i<LOOP;i++)
        memread_kernel_2d<DType, vec, need_bin><<<gx, bx>>>(reinterpret_cast<DType*>(B), reinterpret_cast<DType*>(A), rows, cols, stride);
    CALL(cudaEventRecord( stop_ev, 0 ));
    CALL(cudaEventSynchronize(stop_ev));

    float ms;
    CALL(cudaEventElapsedTime(&ms,start_ev, stop_ev));
    ms/=LOOP;
    return ms;
}

template<int need_bin = 0>
int run(int rows, int cols, int stride)
{
    int64_t dwords = static_cast<int64_t>(rows) * stride;
    unsigned char *A, *B;
    
    float * h_A = (float*)malloc(dwords*sizeof(float));
    float * h_B = (float*)malloc(dwords*sizeof(float));
	for (int i = 0; i < dwords; ++i)
        h_A[i] = get_rand();

    CALL(cudaMalloc(&A, dwords * sizeof(float)));
    CALL(cudaMalloc(&B, dwords * sizeof(float)));
    CALL(cudaMemcpy(A, h_A, dwords * sizeof(float), cudaMemcpyHostToDevice));

    float ms = bench_memread_kernel<float, 1, need_bin>(B, A, rows, cols, stride);

    auto get_gbps = [](int rows_, int cols_, float ms_){
        int64_t dwords_ = static_cast<int64_t>(rows_) * cols_;
        return  ((double)dwords_*sizeof(float))/((double)ms_/1000)/1000000000.0;
    };
    char str[64];
    b2s(dwords*sizeof(float), str);
    printf("[%4dx%4d]%9s -> %.4fms, %.3f(GB/s), %s\n",rows, cols, str, ms, get_gbps(rows, cols, ms),
        need_bin ? "atomic_smem" : "no_smem"
    );

    free(h_A);
    free(h_B);
    CALL(cudaFree(A));
    CALL(cudaFree(B));
    return 0;
}

struct bench_data {
    int rows;
    int cols;
};

int main(int argc, char ** argv) {
    CALL(cudaSetDevice(0));
    if(argc > 2){
        int rows = atoi(argv[1]);
        int cols = atoi(argv[2]);
        run<0>(rows, cols, cols);
        run<1>(rows, cols, cols);
    } else {
#if 1
        bench_data dwords_list[] = {
            {320, 16384},
            {3200, 16384},
            {16384, 16384}};

        int iters = sizeof(dwords_list) / sizeof(dwords_list[0]);

        for(auto i = 0; i < iters; i++) {
            int rows = dwords_list[i].rows;
            int cols = dwords_list[i].cols;
            run<0>(rows, cols, cols);
            usleep(100000);
            run<1>(rows, cols, cols);
            usleep(100000);
        }
#endif
    }
}
