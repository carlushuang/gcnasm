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
    if constexpr(vec == 4) {
        v.x += other.x;
        v.y += other.y;
        v.z += other.z;
        v.w += other.w;
    }
}

// simple memread kernel implementation, launch based on CU number
template<typename T, int UNROLL = 8>
__global__
void memread_kernel(T* p_src, T* p_dst, int issues_per_block, int iters)
{
    auto current = blockIdx.x * issues_per_block;
    T v {};
    for(auto i = 0; i < iters; i++) {
        auto offs = UNROLL * BLOCK_SIZE * i + threadIdx.x;
        #pragma unroll
        for(auto j = 0; j < UNROLL; j++) {
#if USE_NT_LOAD
            acc(v, nt_load(p_src[current + offs]));
#else
            acc(v, p_src[current + offs]);
#endif
            offs += BLOCK_SIZE;
        }
    }

    constexpr int vec = sizeof(T) / sizeof(float);
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

template<typename VEC, int OCCUPANCY, int UNROLL>
float bench_memread_kernel(void * B, void * A, int64_t dwords)
{
    // benchmark kernel
    int num_cu = [&](){
        cudaDeviceProp dev_prop;
        int dev;
        CALL(cudaGetDevice(&dev));
        CALL(cudaGetDeviceProperties(&dev_prop,dev ));
        return dev_prop.multiProcessorCount;
    }();

    // TODO: ignore check can even divide or not
    int64_t pixels = dwords * 4 / sizeof(VEC);
    int bx = BLOCK_SIZE;
    int gx = num_cu * OCCUPANCY;
    int issues_per_block = pixels / gx;
    int iters = issues_per_block / bx / UNROLL;

    cudaEvent_t start_ev, stop_ev;
    CALL(cudaEventCreate(&start_ev));
    CALL(cudaEventCreate(&stop_ev));

    for(int i=0;i<WARMUP;i++)
        memread_kernel<VEC, UNROLL><<<gx, bx>>>(reinterpret_cast<VEC*>(B), reinterpret_cast<VEC*>(A), issues_per_block, iters);

    CALL(cudaEventRecord(start_ev, 0));
    for(int i=0;i<LOOP;i++)
        memread_kernel<VEC, UNROLL><<<gx, bx>>>(reinterpret_cast<VEC*>(B), reinterpret_cast<VEC*>(A), issues_per_block, iters);
    CALL(cudaEventRecord( stop_ev, 0 ));
    CALL(cudaEventSynchronize(stop_ev));

    float ms;
    CALL(cudaEventElapsedTime(&ms,start_ev, stop_ev));
    ms/=LOOP;
    return ms;
}

int run(int64_t dwords)
{
    unsigned char *A, *B;
    
    float * h_A = (float*)malloc(dwords*sizeof(float));
    float * h_B = (float*)malloc(dwords*sizeof(float));
	for (int i = 0; i < dwords; ++i)
        h_A[i] = get_rand();

    CALL(cudaMalloc(&A, dwords * sizeof(float)));
    CALL(cudaMalloc(&B, dwords * sizeof(float)));
    CALL(cudaMemcpy(A, h_A, dwords * sizeof(float), cudaMemcpyHostToDevice));

    float ms = bench_memread_kernel<fp32x4, 1, 4>(B, A, dwords);


    auto get_gbps = [](int64_t dwords_, float ms_){
        return  ((double)dwords_*sizeof(float))/((double)ms_/1000)/1000000000.0;
    };
    char str[64];
    b2s(dwords*sizeof(float), str);
    printf("%9s -> %.4fms, %.3f(GB/s)\n", str, ms, get_gbps(dwords, ms));

    free(h_A);
    free(h_B);
    CALL(cudaFree(A));
    CALL(cudaFree(B));
    return 0;
}

int main(int argc, char ** argv) {
	CALL(cudaSetDevice(0));
    if(argc > 1){
        int64_t dwords = atoll(argv[1]);
        run(dwords);
    } else {
        int num_cu = [&](){
            cudaDeviceProp dev_prop;
            int dev;
            CALL(cudaGetDevice(&dev));
            CALL(cudaGetDeviceProperties(&dev_prop,dev ));
            return dev_prop.multiProcessorCount;
        }();
        printf("cu:%d\n", num_cu);

        int64_t dwords_list[] = {
            20000,
            400000,
            116 * num_cu * BLOCK_SIZE,
            212 * num_cu * BLOCK_SIZE,
            476 * num_cu * BLOCK_SIZE,
            820 * num_cu * BLOCK_SIZE,
            1638 * num_cu * BLOCK_SIZE,
            3276 * num_cu * BLOCK_SIZE,
            12032 * num_cu * BLOCK_SIZE};

        int iters = sizeof(dwords_list) / sizeof(dwords_list[0]);

        for(auto i = 0; i < iters; i++) {
            int64_t dwords = dwords_list[i];
            run(dwords);
            usleep(100000);
        }
    }
}
