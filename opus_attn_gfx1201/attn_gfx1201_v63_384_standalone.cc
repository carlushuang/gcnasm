// Minimal v63 standalone for ISA analysis, BLOCK_M=384 only
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include "attn_common.h"
#define HIP_CALL(c) do { hipError_t e = (c); if (e != hipSuccess) { fprintf(stderr, "HIP %s @ %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); std::exit(1); } } while(0)
#include "attn_gfx1201_kernel_v63_template.hpp"
__global__ void v_transpose_kernel(const bf16_t* V, bf16_t* VT, int B, int H, int N, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * N * D;
    if (idx >= total) return;
    int d = idx % D; int n = (idx / D) % N; int h = (idx / (D * N)) % H; int b = idx / (D * N * H);
    VT[b*(H*D*N) + h*(D*N) + d*N + n] = V[b*(H*N*D) + h*(N*D) + n*D + d];
}
int main() {
    int B=4,H=8,N=7680,D=128;
    fp32_t scale=1.0f/std::sqrt((fp32_t)D); size_t sz=(size_t)B*H*N*D;
    std::vector<bf16_t> hQ(sz),hK(sz),hV(sz);
    std::mt19937 rng(42); std::uniform_real_distribution<float> u(-0.5f,0.5f);
    for(auto&x:hQ)x=bf16_from_f32(u(rng)); for(auto&x:hK)x=bf16_from_f32(u(rng)); for(auto&x:hV)x=bf16_from_f32(u(rng));
    bf16_t*dQ,*dK,*dV,*dO,*dVT;
    HIP_CALL(hipMalloc(&dQ,sz*sizeof(bf16_t))); HIP_CALL(hipMalloc(&dK,sz*sizeof(bf16_t)));
    HIP_CALL(hipMalloc(&dV,sz*sizeof(bf16_t))); HIP_CALL(hipMalloc(&dVT,sz*sizeof(bf16_t)));
    HIP_CALL(hipMalloc(&dO,sz*sizeof(bf16_t)));
    HIP_CALL(hipMemcpy(dQ,hQ.data(),sz*sizeof(bf16_t),hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(dK,hK.data(),sz*sizeof(bf16_t),hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(dV,hV.data(),sz*sizeof(bf16_t),hipMemcpyHostToDevice));
    {int t=256,bl=(sz+t-1)/t; v_transpose_kernel<<<bl,t>>>(dV,dVT,B,H,N,D); HIP_CALL(hipDeviceSynchronize());}
    opus_attn_kargs kargs{}; kargs.ptr_q=dQ;kargs.ptr_k=dK;kargs.ptr_v=dVT;kargs.ptr_o=dO;
    kargs.B=B;kargs.H=H;kargs.N=N;kargs.D=D;kargs.scale=scale;
    using Traits = opus_attn_traits<384,32,128>;
    auto kern = opus_attn_gfx1201_kernel_v63<Traits>;
    int nb=N/Traits::BLOCK_M; dim3 grid(nb,H,B),block(Traits::BLOCK_SIZE);
    for(int i=0;i<5;++i)kern<<<grid,block,0,0>>>(kargs); HIP_CALL(hipDeviceSynchronize());
    hipEvent_t ev0,ev1; HIP_CALL(hipEventCreate(&ev0)); HIP_CALL(hipEventCreate(&ev1));
    HIP_CALL(hipEventRecord(ev0)); for(int i=0;i<100;++i)kern<<<grid,block,0,0>>>(kargs);
    HIP_CALL(hipEventRecord(ev1)); HIP_CALL(hipEventSynchronize(ev1));
    float ms=0; HIP_CALL(hipEventElapsedTime(&ms,ev0,ev1)); ms/=100;
    double tflops=4.0*B*H*(double)N*N*D/(ms*1e9);
    printf("v63 BLOCK_M=384 avg=%.3f ms  %.2f TFLOPS\n",ms,tflops);
    return 0;
}
