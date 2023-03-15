#include <hip/hip_runtime.h>
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include "common.h"

#define CALL(cmd) \
do {\
    hipError_t cuda_error  = cmd;\
    if (cuda_error != hipSuccess) { \
        std::cout<<"'"<<hipGetErrorString(cuda_error)<<"'("<<cuda_error<<")"<<" at "<<__FILE__<<":"<<__LINE__<<std::endl;\
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

float bench_memcpy_kernel(void * B, void * A, int dwords, memtpy_kernel_t f_memcpy)
{
    // benchmark kernel
    hipEvent_t start_ev, stop_ev;
    CALL(hipEventCreate(&start_ev));
    CALL(hipEventCreate(&stop_ev));

    for(int i=0;i<WARMUP;i++)
        f_memcpy(B, A, dwords);

    CALL(hipEventRecord(start_ev, 0));
    for(int i=0;i<LOOP;i++)
        f_memcpy(B, A, dwords);

    CALL(hipEventRecord( stop_ev, 0 ));
    CALL(hipEventSynchronize(stop_ev));

    float ms;
    CALL(hipEventElapsedTime(&ms,start_ev, stop_ev));
    ms/=LOOP;
    return ms;
}

float bench_memcpy_api(void * B, void * A, int dwords)
{
    // benchmark memcpy api
    hipEvent_t start_ev, stop_ev;
    CALL(hipEventCreate(&start_ev));
    CALL(hipEventCreate(&stop_ev));
    for(int i=0;i<WARMUP;i++)
        CALL(hipMemcpy(B, A, dwords * sizeof(float), hipMemcpyDeviceToDevice));
    CALL(hipEventRecord( start_ev, 0));
    for(int i=0;i<LOOP;i++)
        CALL(hipMemcpy(B, A, dwords * sizeof(float), hipMemcpyDeviceToDevice));
    CALL(hipEventRecord( stop_ev, 0 ));
    CALL(hipEventSynchronize(stop_ev));

    float ms_api;
    CALL(hipEventElapsedTime(&ms_api,start_ev, stop_ev));
    ms_api/=LOOP;
    return ms_api;
}


int main(int argc, char ** argv) {
	hipSetDevice(0);
    unsigned char *A, *B;
    int dwords = 64*3*224*224;
    if(argc > 1){
        dwords = atoi(argv[1]);
    }
    float * h_A = (float*)malloc(dwords*sizeof(float));
    float * h_B = (float*)malloc(dwords*sizeof(float));
	for (int i = 0; i < dwords; ++i)
        h_A[i] = get_rand();

    CALL(hipMalloc(&A, dwords * sizeof(float)));
    CALL(hipMalloc(&B, dwords * sizeof(float)));
    CALL(hipMemcpy(A, h_A, dwords * sizeof(float), hipMemcpyHostToDevice));

    memcpy_module_t mm;

    float msx1 = bench_memcpy_kernel(B, A, dwords, mm.memcpy_fp32);
    sleep(1);
    float msx2 = bench_memcpy_kernel(B, A, dwords, mm.memcpy_fp32x2);
    sleep(1);
    float msx4 = bench_memcpy_kernel(B, A, dwords, mm.memcpy_fp32x4);

    CALL(hipMemcpy(h_B, B, dwords * sizeof(float), hipMemcpyDeviceToHost));
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
    CALL(hipFree(A));
    CALL(hipFree(B));
}
