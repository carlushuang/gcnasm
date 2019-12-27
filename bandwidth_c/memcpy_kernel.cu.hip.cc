#include <hip/hip_runtime.h>
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
	hipSetDevice(0);
    float *A, *B, *output;
    const int total_float =64*3*224*224;
	static float h_A[total_float], h_B[total_float];
	for (int i = 0; i < total_float; ++i)
		h_A[i] = i % 71;

	hipMalloc(&A, total_float * sizeof(float));
	hipMalloc(&B, total_float * sizeof(float));
	hipMemcpy(A, h_A, sizeof(h_A), hipMemcpyHostToDevice);
	//hipMemcpy(B, h_B, sizeof(h_B), hipMemcpyHostToDevice);
    int bx = 256;
    int gx = (total_float+255)>>8;

    hipEvent_t start_ev, stop_ev;
    hipEventCreate(&start_ev);
    hipEventCreate(&stop_ev);

    for(int i=0;i<WARMUP;i++)
        hipLaunchKernelGGL((memcpy_kernel), dim3(gx), dim3(bx), 0, 0, B, A);

    hipDeviceSynchronize();
    hipEventRecord( start_ev, NULL);
    for(int i=0;i<LOOP;i++)
        hipLaunchKernelGGL((memcpy_kernel), dim3(gx), dim3(bx), 0, 0, B, A);
    hipEventRecord( stop_ev, NULL );

    float ms;
    hipEventElapsedTime(&ms,start_ev, stop_ev);
    ms/=LOOP;

    printf("total %dB, gflops:%f\n", (total_float*sizeof(float)*2)/((double)ms/1000)/1000000000.0 );

}
