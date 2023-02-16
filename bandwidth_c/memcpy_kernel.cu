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

#define BLOCK_SIZE 256
template<typename T>
__global__
void memcpy_kernel(T* __restrict__ dst, const T* __restrict__ src, uint32_t n){
    int idx = (blockIdx.x * BLOCK_SIZE + threadIdx.x);
    if(idx < n)
        dst[idx] = src[idx];
}

#define CALL(cmd) \
do {\
    cudaError_t cuda_error  = cmd;\
    if (cuda_error != cudaSuccess) { \
        std::cout<<"'"<<cudaGetErrorString(cuda_error)<<"'("<<cuda_error<<")"<<" at "<<__FILE__<<":"<<__LINE__<<std::endl;\
        exit(EXIT_FAILURE);\
    }\
} while(0)

#define WARMUP 3
#define LOOP 10

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

template<typename VEC>
float bench_memcpy_kernel(void * B, void * A, int dwords)
{
    // benchmark kernel
    int bx = BLOCK_SIZE;
    int pixels = dwords * 4 / sizeof(VEC);
    int gx = (pixels + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // printf("%d, %d\n", pixels, sizeof(VEC));fflush(stdout);

    cudaEvent_t start_ev, stop_ev;
    CALL(cudaEventCreate(&start_ev));
    CALL(cudaEventCreate(&stop_ev));

    for(int i=0;i<WARMUP;i++)
        memcpy_kernel<VEC><<<gx, bx>>>(reinterpret_cast<VEC*>(B), reinterpret_cast<const VEC*>(A), pixels);

    CALL(cudaEventRecord(start_ev, 0));
    for(int i=0;i<LOOP;i++)
        memcpy_kernel<VEC><<<gx, bx>>>(reinterpret_cast<VEC*>(B), reinterpret_cast<const VEC*>(A), pixels);
    CALL(cudaEventRecord( stop_ev, 0 ));
    CALL(cudaEventSynchronize(stop_ev));

    float ms;
    CALL(cudaEventElapsedTime(&ms,start_ev, stop_ev));
    ms/=LOOP;
    return ms;
}

float bench_memcpy_api(void * B, void * A, int dwords)
{
    // benchmark memcpy api
    cudaEvent_t start_ev, stop_ev;
    CALL(cudaEventCreate(&start_ev));
    CALL(cudaEventCreate(&stop_ev));
    for(int i=0;i<WARMUP;i++)
        CALL(cudaMemcpy(B, A, dwords * sizeof(float), cudaMemcpyDeviceToDevice));
    CALL(cudaEventRecord( start_ev, 0));
    for(int i=0;i<LOOP;i++)
        CALL(cudaMemcpy(B, A, dwords * sizeof(float), cudaMemcpyDeviceToDevice));
    CALL(cudaEventRecord( stop_ev, 0 ));
    CALL(cudaEventSynchronize(stop_ev));

    float ms_api;
    CALL(cudaEventElapsedTime(&ms_api,start_ev, stop_ev));
    ms_api/=LOOP;
    return ms_api;
}


int main(int argc, char ** argv) {
	cudaSetDevice(0);
    unsigned char *A, *B;
    int dwords = 64*3*224*224;
    if(argc > 1){
        dwords = atoi(argv[1]);
    }
    float * h_A = (float*)malloc(dwords*sizeof(float));
    float * h_B = (float*)malloc(dwords*sizeof(float));
	for (int i = 0; i < dwords; ++i)
        h_A[i] = get_rand();

    CALL(cudaMalloc(&A, dwords * sizeof(float)));
    CALL(cudaMalloc(&B, dwords * sizeof(float)));
    CALL(cudaMemcpy(A, h_A, dwords * sizeof(float), cudaMemcpyHostToDevice));

    float msx1 = bench_memcpy_kernel<fp32>(B, A, dwords);
    sleep(1);
    float msx2 = bench_memcpy_kernel<fp32x2>(B, A, dwords);
    sleep(1);
    float msx4 = bench_memcpy_kernel<fp32x4>(B, A, dwords);

    CALL(cudaMemcpy(h_B, B, dwords * sizeof(float), cudaMemcpyDeviceToHost));
    sleep(1);

    // benchmark memcpy api
    float ms_api = bench_memcpy_api(B, A, dwords);
    
    auto get_gbps = [](int dwords_, float ms_){
        return  ((double)dwords_*sizeof(float)*2)/((double)ms_/1000)/1000000000.0;
    };
    char str[64];
    b2s(dwords*sizeof(float), str);
    printf("%s, kernel_x1:%.3f(GB/s), kernel_x2:%.3f(GB/s), kernel_x4:%.3f(GB/s), api:%.3f(GB/s)\n",
        str, get_gbps(dwords, msx1), get_gbps(dwords, msx2), get_gbps(dwords, msx4),
       get_gbps(dwords, ms_api) );

    free(h_A);
    free(h_B);
    CALL(cudaFree(A));
    CALL(cudaFree(B));
}
