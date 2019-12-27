#include <cuda_runtime.h>
#include <stdio.h>

extern "C" __global__
void memcpy_kernel(unsigned char* __restrict__ output,  unsigned char* __restrict__ input){
    output += (blockIdx.x<<13)|(threadIdx.x<<2); 
    *((float*)&output[0])       = *((float*)&input[0]);
    *((float*)&output[0x400])   = *((float*)&input[0x400]);
    *((float*)&output[0x800])   = *((float*)&input[0x800]);
    *((float*)&output[0xc00])   = *((float*)&input[0xc00]);
    *((float*)&output[0x1000])  = *((float*)&input[0x1000]);
    *((float*)&output[0x1400])  = *((float*)&input[0x1400]);
    *((float*)&output[0x1800])  = *((float*)&input[0x1800]);
    *((float*)&output[0x1c00])  = *((float*)&input[0x1c00]);
}
#define WARMUP 2
#define LOOP 10

int main() {
	cudaSetDevice(0);
    float *A, *B, *output;
    const int total_float =64*3*224*224;
	static float h_A[total_float], h_B[total_float];
	for (int i = 0; i < total_float; ++i)
		h_A[i] = i % 71;

	cudaMalloc(&A, total_float * sizeof(float));
	cudaMalloc(&B, total_float * sizeof(float));
	cudaMemcpy(A, h_A, sizeof(h_A), cudaMemcpyHostToDevice);
	//cudaMemcpy(B, h_B, sizeof(h_B), cudaMemcpyHostToDevice);
    int bx = 256;
    int gx = (total_float+255)>>8;

    cudaEvent_t start_ev, stop_ev;
    cudaEventCreate(&start_ev);
    cudaEventCreate(&stop_ev);

    for(int i=0;i<WARMUP;i++)
        memcpy_kernel<<<gx, bx, 0, 0>>>(B, A);

    cudaDeviceSynchronize();
    cudaEventRecord( start_ev, NULL);
    for(int i=0;i<LOOP;i++)
        memcpy_kernel<<<gx, bx, 0, 0>>>(B, A);
    cudaEventRecord( stop_ev, NULL );

    float ms;
    cudaEventElapsedTime(&ms,start_ev, stop_ev);
    ms/=LOOP;

    printf("total %dB, gflops:%f\n", (total_float*sizeof(float)*2)/((double)ms/1000)/1000000000.0 );

}
