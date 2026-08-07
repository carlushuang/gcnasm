// v82: unconditional rescale + hoisted V addresses + fused rescale-PV
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>
#include "attn_common.h"
#define HIP_CALL(c) do { hipError_t e = (c); if (e != hipSuccess) { fprintf(stderr, "HIP %s @ %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); std::exit(1); } } while(0)
#include "attn_gfx1201_kernel_v82_template.hpp"
__global__ void v_transpose_kernel(const bf16_t* V, bf16_t* VT, int B, int H, int N, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * N * D;
    if (idx >= total) return;
    int d = idx % D; int n = (idx / D) % N; int h = (idx / (D * N)) % H; int b = idx / (D * N * H);
    VT[b*(H*D*N) + h*(D*N) + d*N + n] = V[b*(H*N*D) + h*(N*D) + n*D + d];
}
static void cpu_reference(int B, int H, int N, int D, const bf16_t* Q, const bf16_t* K, const bf16_t* V, bf16_t* O, fp32_t scale) {
    std::vector<fp32_t> S(N), P(N);
    for (int b = 0; b < B; ++b) for (int h = 0; h < H; ++h) {
        auto *Qbh=Q+(b*H+h)*N*D, *Kbh=K+(b*H+h)*N*D, *Vbh=V+(b*H+h)*N*D; auto *Obh=O+(b*H+h)*N*D;
        for (int m = 0; m < N; ++m) {
            fp32_t rm=-3.4e38f;
            for (int n = 0; n < N; ++n) { fp32_t s=0; for (int d=0;d<D;++d) s+=bf16_to_f32(Qbh[m*D+d])*bf16_to_f32(Kbh[n*D+d]); s*=scale; S[n]=s; if(s>rm)rm=s; }
            fp32_t rs=0; for (int n=0;n<N;++n) { P[n]=std::exp(S[n]-rm); rs+=P[n]; }
            fp32_t inv=rs>0?1.0f/rs:0;
            for (int d=0;d<D;++d) { fp32_t o=0; for (int n=0;n<N;++n) o+=P[n]*bf16_to_f32(Vbh[n*D+d]); Obh[m*D+d]=bf16_from_f32(o*inv); }
        }
    }
}
int main(int argc, char** argv) {
    int B=1,H=1,N=256,D=128,verify=1,warmups=5,iters=100;
    int block_m=384;
    for (int i=1;i<argc;++i) { const char*a=argv[i];
        auto eq=[&](const char*k,int&dst){auto kn=std::strlen(k);if(std::strncmp(a,k,kn)==0){dst=std::atoi(a[kn]=='='?a+kn+1:argv[++i]);return true;}return false;};
        if(eq("-b",B)||eq("--batch",B))continue; if(eq("-h",H)||eq("--heads",H))continue;
        if(eq("-n",N)||eq("--seq",N))continue; if(eq("-d",D)||eq("--dim",D))continue;
        if(eq("--verify",verify))continue; if(eq("--iters",iters))continue;
        if(eq("--block_m",block_m))continue;
    }

    fp32_t scale=1.0f/std::sqrt((fp32_t)D); size_t sz=(size_t)B*H*N*D;
    std::vector<bf16_t> hQ(sz),hK(sz),hV(sz),hO(sz),hRef(sz);
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

    auto run = [&](auto traits_tag) {
        using Traits = decltype(traits_tag);
        printf("running version v82-blockM%d  B=%d H=%d N=%d D=%d\n",Traits::BLOCK_M,B,H,N,D);
        auto kern = opus_attn_gfx1201_kernel_v82<Traits>;
        int nb=N/Traits::BLOCK_M; dim3 grid(nb,H,B),block(Traits::BLOCK_SIZE);
        for(int i=0;i<warmups;++i)kern<<<grid,block,0,0>>>(kargs); HIP_CALL(hipDeviceSynchronize());
        if(verify){
            HIP_CALL(hipMemcpy(hO.data(),dO,sz*sizeof(bf16_t),hipMemcpyDeviceToHost));
            cpu_reference(B,H,N,D,hQ.data(),hK.data(),hV.data(),hRef.data(),scale);
            double ma=0,me=0,mr=0;int nb2=0;
            for(size_t i=0;i<sz;++i){double a=(double)bf16_to_f32(hO[i]),r=(double)bf16_to_f32(hRef[i]),d=std::abs(a-r);
                ma=std::max(ma,d);me+=d;double rel=std::abs(r)>1e-3?d/std::abs(r):0;mr=std::max(mr,rel);if(d>0.05)++nb2;}
            me/=sz;printf("VERIFY B=%d H=%d N=%d D=%d  max_abs=%.4f  mean_abs=%.5f  max_rel=%.4f  n_bad(>0.05)=%d/%zu\n",B,H,N,D,ma,me,mr,nb2,sz);
        }
        hipEvent_t ev0,ev1; HIP_CALL(hipEventCreate(&ev0)); HIP_CALL(hipEventCreate(&ev1));
        HIP_CALL(hipEventRecord(ev0)); for(int i=0;i<iters;++i)kern<<<grid,block,0,0>>>(kargs);
        HIP_CALL(hipEventRecord(ev1)); HIP_CALL(hipEventSynchronize(ev1));
        float ms=0; HIP_CALL(hipEventElapsedTime(&ms,ev0,ev1)); ms/=iters;
        double tflops=4.0*B*H*(double)N*N*D/(ms*1e9);
        printf("BENCH  iters=%d  avg=%.3f ms  %.2f TFLOPS\n",iters,ms,tflops);
    };

    if (block_m == 128) run(opus_attn_traits<128,32,128>{});
    else if (block_m == 192) run(opus_attn_traits<192,32,128>{});
    else if (block_m == 256) run(opus_attn_traits<256,32,128>{});
    else if (block_m == 320) run(opus_attn_traits<320,32,128>{});
    else if (block_m == 384) run(opus_attn_traits<384,32,128>{});
    else if (block_m == 448) run(opus_attn_traits<448,32,128>{});
    else if (block_m == 512) run(opus_attn_traits<512,32,128>{});
    else if (block_m == 640) run(opus_attn_traits<640,32,128>{});
    else if (block_m == 768) run(opus_attn_traits<768,32,128>{});
    else { printf("unsupported block_m=%d\n", block_m); return 1; }

    HIP_CALL(hipFree(dQ));HIP_CALL(hipFree(dK));HIP_CALL(hipFree(dV));HIP_CALL(hipFree(dVT));HIP_CALL(hipFree(dO));
    return 0;
}
