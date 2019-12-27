#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <stdlib.h>

extern "C" __global__
void memcpy_kernel(unsigned char* __restrict__ output, const unsigned char* __restrict__ input){
    output += (blockIdx.x<<13)|(threadIdx.x<<2); 
    *((float* __restrict__ )&output[0])       = *((float* __restrict__ )&input[0]);
    *((float* __restrict__ )&output[0x400])   = *((float* __restrict__ )&input[0x400]);
    *((float* __restrict__ )&output[0x800])   = *((float* __restrict__ )&input[0x800]);
    *((float* __restrict__ )&output[0xc00])   = *((float* __restrict__ )&input[0xc00]);
    *((float* __restrict__ )&output[0x1000])  = *((float* __restrict__ )&input[0x1000]);
    *((float* __restrict__ )&output[0x1400])  = *((float* __restrict__ )&input[0x1400]);
    *((float* __restrict__ )&output[0x1800])  = *((float* __restrict__ )&input[0x1800]);
    *((float* __restrict__ )&output[0x1c00])  = *((float* __restrict__ )&input[0x1c00]);
}

#define CALL(cmd) \
do {\
    cudaError_t cuda_error  = cmd;\
    if (cuda_error != cudaSuccess) { \
        std::cout<<"'"<<cudaGetErrorString(cuda_error)<<"'("<<cuda_error<<")"<<" at "<<__FILE__<<":"<<__LINE__<<std::endl;\
        exit(EXIT_FAILURE);\
    }\
} while(0)

#define WARMUP 2
#define LOOP 10

int main() {
	cudaSetDevice(0);
    unsigned char *A, *B;
    const int total_float =64*3*224*224;
	static float h_A[total_float], h_B[total_float];
	for (int i = 0; i < total_float; ++i)
		h_A[i] = i % 71;

    CALL(cudaMalloc(&A, total_float * sizeof(float)));
    CALL(cudaMalloc(&B, total_float * sizeof(float)));
    CALL(cudaMemcpy(A, h_A, total_float * sizeof(float), cudaMemcpyHostToDevice));

    int bx = 256;
    int gx = (total_float+255)>>8;
    assert(total_float/bx);

    cudaEvent_t start_ev, stop_ev;
    CALL(cudaEventCreate(&start_ev));
    CALL(cudaEventCreate(&stop_ev));

    for(int i=0;i<WARMUP;i++)
        memcpy_kernel<<<gx, bx>>>(B, A);

    CALL(cudaDeviceSynchronize());
    CALL(cudaEventRecord( start_ev, 0));
    for(int i=0;i<LOOP;i++)
        memcpy_kernel<<<gx, bx>>>(B, A);
    CALL(cudaEventRecord( stop_ev, 0 ));

    float ms;
    CALL(cudaEventElapsedTime(&ms,start_ev, stop_ev));
    ms/=LOOP;

    printf("total %dB, gflops:%f\n", total_float, ((double)total_float*sizeof(float)*2)/((double)ms/1000)/1000000000.0 );
}
