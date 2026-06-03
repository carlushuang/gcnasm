#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <cstdio>
#include <cstdlib>

__global__ void busy(float* out, int n, int iters, int tag) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = (float)idx + tag;
    #pragma unroll 1
    for (int i = 0; i < iters; ++i) acc = fmaf(acc, 1.0001f, float(i & 0xFF));
    if (acc == 1.2345e30f && idx < n) out[idx] = acc;
}

#define CK(x) do{ hipError_t e=(x); if(e!=hipSuccess){printf("ERR %s\n",hipGetErrorString(e)); return 1;} }while(0)

int main(int argc, char** argv){
    int mode = argc>1 ? atoi(argv[1]) : 1;   // 0=normal in-order, 1=anyOrder
    int nk   = argc>2 ? atoi(argv[2]) : 4;   // number of kernels on the stream
    int grid = argc>3 ? atoi(argv[3]) : 8;   // small grid -> spare occupancy
    int iters= argc>4 ? atoi(argv[4]) : 50000;
    const int BS=256; int N=grid*BS;
    float* d; CK(hipMalloc(&d, N*sizeof(float)));
    hipStream_t s; CK(hipStreamCreate(&s));
    dim3 g(grid), b(BS);

    for (int k=0;k<nk;k++){
        if (mode==0) {
            busy<<<g,b,0,s>>>(d,N,iters,k);
        } else {
            void* a[]={&d,&N,&iters,&k};
            CK(hipExtLaunchKernel((void*)busy,g,b,a,0,s,nullptr,nullptr,hipExtAnyOrderLaunch));
        }
    }
    CK(hipStreamSynchronize(s));
    printf("mode=%s nk=%d grid=%d iters=%d : done\n", mode?"anyOrder":"normal", nk, grid, iters);
    hipFree(d); hipStreamDestroy(s);
    return 0;
}
