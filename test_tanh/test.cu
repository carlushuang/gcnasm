#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>

#define BLOCK_SIZE 256
#define DEFAULT_FLOAT (1e-9)

#define CALL(cmd) \
do {\
    cudaError_t cuda_error  = cmd;\
    if (cuda_error != cudaSuccess) { \
        std::cout<<"'"<<cudaGetErrorString(cuda_error)<<"'("<<cuda_error<<")"<<" at "<<__FILE__<<":"<<__LINE__<<std::endl;\
        exit(EXIT_FAILURE);\
    }\
} while(0)

namespace impl {
#ifdef __NVCC__
__forceinline__ __device__ float tanh(float x) {
  float y;
  asm volatile("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
#else
__forceinline__ __device__ float tanh(float x) {
  return ::tanhf(x);
}
#endif
}

// only 1 workgroup
__global__ void
kernel(float* __restrict__ src, float* __restrict__ dst)
{
    float x = src[threadIdx.x];
    dst[threadIdx.x] = impl::tanh(x); 
}

int main(int argc, char ** argv)
{
    uint32_t v = 0x78;
    float test_float = *reinterpret_cast<float*>(&v);
    if(argc > 1){
        test_float = atof(argv[1]);
    }

    float * h_A = (float*)malloc(BLOCK_SIZE*sizeof(float));
    float * h_B = (float*)malloc(BLOCK_SIZE*sizeof(float));
	for (int i = 0; i < BLOCK_SIZE; ++i) {
        h_A[i] = test_float;
    }

    float *A, *B;
    CALL(cudaMalloc(&A, BLOCK_SIZE * sizeof(float)));
    CALL(cudaMalloc(&B, BLOCK_SIZE * sizeof(float)));
    CALL(cudaMemcpy(A, h_A, BLOCK_SIZE * sizeof(float), cudaMemcpyHostToDevice));


    kernel<<<1, BLOCK_SIZE >>>(A, B);

    CALL(cudaMemcpy(h_B, B, BLOCK_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    // only need to verify first element within A/B
    printf("src:%f(0x%08x) -> dst:%f(0x%08x)\n", h_A[0], *reinterpret_cast<uint32_t*>(h_A), h_B[0], *reinterpret_cast<uint32_t*>(h_B));


    free(h_A);
    free(h_B);
    CALL(cudaFree(A));
    CALL(cudaFree(B));
}