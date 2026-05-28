// v88: BLOCK_N=32 hand-ASM kernel, loaded as HSACO
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>

using bf16_t = unsigned short;
using fp32_t = float;

static inline fp32_t bf16_to_f32(bf16_t v) {
    unsigned int bits = (unsigned int)v << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static inline bf16_t bf16_from_f32(fp32_t f) {
    unsigned int x;
    memcpy(&x, &f, 4);
    x += ((x >> 16) & 1) + 0x7FFF;
    return (bf16_t)(x >> 16);
}

struct opus_attn_kargs {
    void* ptr_q;
    void* ptr_k;
    void* ptr_v;
    void* ptr_o;
    int B, H, N, D;
    fp32_t scale;
};

#define HIP_CALL(c) do { hipError_t e = (c); if (e != hipSuccess) { fprintf(stderr, "HIP %s @ %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); std::exit(1); } } while(0)

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
            for (int n = 0; n < N; ++n) { fp32_t s=0; for (int d2=0;d2<D;++d2) s+=bf16_to_f32(Qbh[m*D+d2])*bf16_to_f32(Kbh[n*D+d2]); s*=scale; S[n]=s; if(s>rm)rm=s; }
            fp32_t rs=0; for (int n=0;n<N;++n) { P[n]=std::exp(S[n]-rm); rs+=P[n]; }
            fp32_t inv=rs>0?1.0f/rs:0;
            for (int d2=0;d2<D;++d2) { fp32_t o=0; for (int n=0;n<N;++n) o+=P[n]*bf16_to_f32(Vbh[n*D+d2]); Obh[m*D+d2]=bf16_from_f32(o*inv); }
        }
    }
}

int main(int argc, char** argv) {
    int B=1,H=1,N=384,D=128,verify=1,warmups=50,iters=100;
    const char* hsaco_path = "v88.hsaco";
    const char* func_name = "v88_kernel";
    for (int i=1;i<argc;++i) { const char*a=argv[i];
        auto eq=[&](const char*k,int&dst){auto kn=std::strlen(k);if(std::strncmp(a,k,kn)==0){dst=std::atoi(a[kn]=='='?a+kn+1:argv[++i]);return true;}return false;};
        if(eq("-b",B))continue; if(eq("-h",H))continue;
        if(eq("-n",N))continue; if(eq("-d",D))continue;
        if(eq("--verify",verify))continue; if(eq("--iters",iters))continue;
        if(std::strncmp(a,"--hsaco",7)==0){hsaco_path=a[7]=='='?a+8:argv[++i];continue;}
        if(std::strncmp(a,"--func",6)==0){func_name=a[6]=='='?a+7:argv[++i];continue;}
    }

    constexpr int BLOCK_M = 384;
    constexpr int BLOCK_N = 32;
    constexpr int BLOCK_SIZE = (BLOCK_M / 16) * 32;

    if (N % BLOCK_M != 0) { printf("ERROR: N=%d not divisible by BLOCK_M=%d\n",N,BLOCK_M); return 1; }
    if (N % BLOCK_N != 0) { printf("ERROR: N=%d not divisible by BLOCK_N=%d\n",N,BLOCK_N); return 1; }

    hipModule_t mod;
    hipFunction_t func;
    HIP_CALL(hipModuleLoad(&mod, hsaco_path));
    hipError_t ferr = hipModuleGetFunction(&func, mod, func_name);
    if (ferr != hipSuccess) {
        fprintf(stderr, "Failed to find function '%s' in %s: %s\n", func_name, hsaco_path, hipGetErrorString(ferr));
        return 1;
    }
    printf("Loaded HSACO from %s, function '%s'\n", hsaco_path, func_name);

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
    {int t=256,bl=(int)((sz+t-1)/t); v_transpose_kernel<<<bl,t>>>(dV,dVT,B,H,N,D); HIP_CALL(hipDeviceSynchronize());}

    opus_attn_kargs kargs{}; kargs.ptr_q=dQ;kargs.ptr_k=dK;kargs.ptr_v=dVT;kargs.ptr_o=dO;
    kargs.B=B;kargs.H=H;kargs.N=N;kargs.D=D;kargs.scale=scale;

    int nb=N/BLOCK_M;
    size_t karg_size = sizeof(kargs);
    void* extra[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, &kargs,
        HIP_LAUNCH_PARAM_BUFFER_SIZE, &karg_size,
        HIP_LAUNCH_PARAM_END
    };

    printf("v88 (BLOCK_N=32 ASM) B=%d H=%d N=%d D=%d BLOCK_M=%d BLOCK_N=%d grid=(%d,%d,%d)\n",B,H,N,D,BLOCK_M,BLOCK_N,nb,H,B);
    printf("kargs: ptr_q=%p ptr_k=%p ptr_v=%p ptr_o=%p B=%d H=%d N=%d D=%d scale=%.6f\n",
        kargs.ptr_q, kargs.ptr_k, kargs.ptr_v, kargs.ptr_o, kargs.B, kargs.H, kargs.N, kargs.D, kargs.scale);
    printf("sizeof(kargs)=%zu\n", sizeof(kargs));

    // Try a single launch first
    printf("Attempting single launch...\n"); fflush(stdout);
    hipError_t lerr = hipModuleLaunchKernel(func, nb, H, B, BLOCK_SIZE, 1, 1, 0, 0, NULL, extra);
    printf("Launch returned: %s\n", hipGetErrorString(lerr)); fflush(stdout);
    hipError_t serr = hipDeviceSynchronize();
    printf("Sync returned: %s\n", hipGetErrorString(serr)); fflush(stdout);
    if (serr != hipSuccess) { printf("Kernel crashed, aborting.\n"); return 1; }

    for(int i=1;i<warmups;++i)
        HIP_CALL(hipModuleLaunchKernel(func, nb, H, B, BLOCK_SIZE, 1, 1, 0, 0, NULL, extra));
    HIP_CALL(hipDeviceSynchronize());

    if(verify){
        HIP_CALL(hipMemset(dO, 0xAA, sz*sizeof(bf16_t)));
        HIP_CALL(hipModuleLaunchKernel(func, nb, H, B, BLOCK_SIZE, 1, 1, 0, 0, NULL, extra));
        HIP_CALL(hipDeviceSynchronize());
        HIP_CALL(hipMemcpy(hO.data(),dO,sz*sizeof(bf16_t),hipMemcpyDeviceToHost));
        cpu_reference(B,H,N,D,hQ.data(),hK.data(),hV.data(),hRef.data(),scale);
        double ma=0,me=0;int nb2=0;
        for(size_t i=0;i<sz;++i){double a=(double)bf16_to_f32(hO[i]),r=(double)bf16_to_f32(hRef[i]),d2=std::abs(a-r);
            ma=std::max(ma,d2);me+=d2;if(d2>0.05)++nb2;}
        me/=sz;
        printf("VERIFY max_abs=%.4f  mean_abs=%.5f  n_bad(>0.05)=%d/%zu\n",ma,me,nb2,sz);
        printf("First 8 GPU:");for(int i=0;i<8;++i)printf(" %.4f",(double)bf16_to_f32(hO[i]));printf("\n");
        printf("First 8 REF:");for(int i=0;i<8;++i)printf(" %.4f",(double)bf16_to_f32(hRef[i]));printf("\n");
        int nz=0,naa=0;for(size_t i=0;i<sz;++i){if(bf16_to_f32(hO[i])==0.0f)++nz;if(hO[i]==0xAAAA)++naa;}
        printf("GPU zeros=%d  fill(0xAA)=%d/%zu\n",nz,naa,sz);
        if(ma>0.01) printf(">>> FAILED <<<\n"); else printf(">>> PASSED <<<\n");
    }

    hipEvent_t ev0,ev1; HIP_CALL(hipEventCreate(&ev0)); HIP_CALL(hipEventCreate(&ev1));
    HIP_CALL(hipEventRecord(ev0));
    for(int i=0;i<iters;++i)
        HIP_CALL(hipModuleLaunchKernel(func, nb, H, B, BLOCK_SIZE, 1, 1, 0, 0, NULL, extra));
    HIP_CALL(hipEventRecord(ev1)); HIP_CALL(hipEventSynchronize(ev1));
    float ms=0; HIP_CALL(hipEventElapsedTime(&ms,ev0,ev1)); ms/=iters;
    double tflops=4.0*B*H*(double)N*N*D/(ms*1e9);
    printf("BENCH  iters=%d  avg=%.3f ms  %.2f TFLOPS\n",iters,ms,tflops);

    HIP_CALL(hipFree(dQ));HIP_CALL(hipFree(dK));HIP_CALL(hipFree(dV));HIP_CALL(hipFree(dVT));HIP_CALL(hipFree(dO));
    HIP_CALL(hipModuleUnload(mod));
    return 0;
}
