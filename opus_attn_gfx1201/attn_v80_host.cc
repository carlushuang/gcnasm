// v80: BLOCK_N=64 hand-ASM kernel, loaded as HSACO
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>

// bf16 as raw 16-bit — same format as attn_common.h
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
    // RNE rounding
    unsigned int rounding = ((x >> 16) & 1) + 0x7FFF;
    x += rounding;
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
            for (int n = 0; n < N; ++n) { fp32_t s=0; for (int d=0;d<D;++d) s+=bf16_to_f32(Qbh[m*D+d])*bf16_to_f32(Kbh[n*D+d]); s*=scale; S[n]=s; if(s>rm)rm=s; }
            fp32_t rs=0; for (int n=0;n<N;++n) { P[n]=std::exp(S[n]-rm); rs+=P[n]; }
            fp32_t inv=rs>0?1.0f/rs:0;
            for (int d=0;d<D;++d) { fp32_t o=0; for (int n=0;n<N;++n) o+=P[n]*bf16_to_f32(Vbh[n*D+d]); Obh[m*D+d]=bf16_from_f32(o*inv); }
        }
    }
}

int main(int argc, char** argv) {
    int B=4,H=8,N=7680,D=128,verify=0,warmups=5,iters=100;
    const char* hsaco_path = "v80.hsaco";
    for (int i=1;i<argc;++i) { const char*a=argv[i];
        auto eq=[&](const char*k,int&dst){auto kn=std::strlen(k);if(std::strncmp(a,k,kn)==0){dst=std::atoi(a[kn]=='='?a+kn+1:argv[++i]);return true;}return false;};
        if(eq("-b",B)||eq("--batch",B))continue; if(eq("-h",H)||eq("--heads",H))continue;
        if(eq("-n",N)||eq("--seq",N))continue; if(eq("-d",D)||eq("--dim",D))continue;
        if(eq("--verify",verify))continue; if(eq("--iters",iters))continue;
        if(std::strncmp(a,"--hsaco",7)==0){hsaco_path=a[7]=='='?a+8:argv[++i];continue;}
    }

    constexpr int BLOCK_M = 384;
    constexpr int BLOCK_N = 64;
    constexpr int BLOCK_SIZE = (BLOCK_M / 16) * 32; // 24 waves * 32 = 768

    if (N % BLOCK_M != 0) { printf("ERROR: N=%d not divisible by BLOCK_M=%d\n",N,BLOCK_M); return 1; }
    if (N % BLOCK_N != 0) { printf("ERROR: N=%d not divisible by BLOCK_N=%d\n",N,BLOCK_N); return 1; }

    // Load HSACO
    hipModule_t mod;
    hipFunction_t func;
    HIP_CALL(hipModuleLoad(&mod, hsaco_path));
    const char* func_name = "opus_attn_gfx1201_kernel_v80";
    hipError_t ferr = hipModuleGetFunction(&func, mod, func_name);
    if (ferr != hipSuccess) {
        fprintf(stderr, "Failed to find function '%s' in %s: %s\n", func_name, hsaco_path, hipGetErrorString(ferr));
        return 1;
    }
    printf("Loaded HSACO from %s\n", hsaco_path);

    fp32_t scale=1.0f/std::sqrt((fp32_t)D); size_t sz=(size_t)B*H*N*D;
    std::vector<bf16_t> hQ(sz),hK(sz),hV(sz),hO(sz),hRef(sz);
    std::mt19937 rng(42); std::uniform_real_distribution<float> u(-0.5f,0.5f);
    int debug_mode = 0;
    for(auto&x:hQ)x=bf16_from_f32(u(rng)); for(auto&x:hK)x=bf16_from_f32(u(rng)); for(auto&x:hV)x=bf16_from_f32(u(rng));
    if (const char* dm = std::getenv("DEBUG_MODE")) debug_mode = std::atoi(dm);
    if (debug_mode == 1) {
        for(auto&x:hQ)x=bf16_from_f32(0.0f); for(auto&x:hK)x=bf16_from_f32(0.0f);
        for(size_t i=0;i<sz;++i)hV[i]=bf16_from_f32(((float)(i%D))/D);
        printf("DEBUG_MODE=1: Q=0, K=0, V=d/D\n");
    } else if (debug_mode == 2) {
        for(auto&x:hQ)x=bf16_from_f32(0.1f);
        printf("DEBUG_MODE=2: Q=0.1, K=random, V=random\n");
    } else if (debug_mode == 3) {
        for(auto&x:hQ)x=bf16_from_f32(0.0f);
        for(auto&x:hK)x=bf16_from_f32(0.0f);
        for(auto&x:hV)x=bf16_from_f32(0.0f);
        for(int i=0;i<D;++i)hK[i]=bf16_from_f32(1.0f);
        for(int i=0;i<D;++i)hV[i]=bf16_from_f32(1.0f);
        printf("DEBUG_MODE=3: Q=0, K[0,:]=1, V[0,:]=1, rest=0\n");
    } else if (debug_mode == 4) {
        for(auto&x:hK)x=bf16_from_f32(0.0f);
        printf("DEBUG_MODE=4: Q=random, K=0, V=random (uniform attn)\n");
    }
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
    int total_blocks = nb * H * B;
    void* kptr = &kargs;
    size_t karg_size = sizeof(kargs);

    printf("v80 (BLOCK_N=64 ASM) B=%d H=%d N=%d D=%d BLOCK_M=%d BLOCK_N=%d total_blocks=%d\n",B,H,N,D,BLOCK_M,BLOCK_N,total_blocks);

    for(int i=0;i<warmups;++i) {
        HIP_CALL(hipModuleLaunchKernel(func, total_blocks, 1, 1, BLOCK_SIZE, 1, 1, 0, 0, &kptr, NULL));
    }
    HIP_CALL(hipDeviceSynchronize());

    if(verify){
        // Clear output and run one more time
        HIP_CALL(hipMemset(dO, 0xAA, sz*sizeof(bf16_t)));
        HIP_CALL(hipModuleLaunchKernel(func, total_blocks, 1, 1, BLOCK_SIZE, 1, 1, 0, 0, &kptr, NULL));
        HIP_CALL(hipDeviceSynchronize());
        HIP_CALL(hipMemcpy(hO.data(),dO,sz*sizeof(bf16_t),hipMemcpyDeviceToHost));
        cpu_reference(B,H,N,D,hQ.data(),hK.data(),hV.data(),hRef.data(),scale);
        // Single-tile reference (first 64 KV only)
        int N_single = 64;
        std::vector<bf16_t> hRef64(sz);
        {
            std::vector<fp32_t> S64(N_single), P64(N_single);
            for (int b=0;b<B;++b) for (int h=0;h<H;++h) {
                auto *Qbh=hQ.data()+(b*H+h)*N*D, *Kbh=hK.data()+(b*H+h)*N*D, *Vbh=hV.data()+(b*H+h)*N*D;
                auto *Obh=hRef64.data()+(b*H+h)*N*D;
                for (int m=0;m<N;++m) {
                    fp32_t rm=-3.4e38f;
                    for (int n=0;n<N_single;++n) { fp32_t s=0; for(int d2=0;d2<D;++d2) s+=bf16_to_f32(Qbh[m*D+d2])*bf16_to_f32(Kbh[n*D+d2]); s*=scale; S64[n]=s; if(s>rm)rm=s; }
                    fp32_t rs=0; for(int n=0;n<N_single;++n) { P64[n]=std::exp(S64[n]-rm); rs+=P64[n]; }
                    fp32_t inv=rs>0?1.0f/rs:0;
                    for(int d2=0;d2<D;++d2) { fp32_t o=0; for(int n=0;n<N_single;++n) o+=P64[n]*bf16_to_f32(Vbh[n*D+d2]); Obh[m*D+d2]=bf16_from_f32(o*inv); }
                }
            }
            double ma64=0; for(size_t i=0;i<sz;++i){double a=(double)bf16_to_f32(hO[i]),r=(double)bf16_to_f32(hRef64[i]),d2=std::abs(a-r);ma64=std::max(ma64,d2);}
            printf("SINGLE_TILE_REF (first 64 KV): max_abs=%.4f\n",ma64);
            printf("First 8 REF64:");for(int i=0;i<8;++i)printf(" %.4f",(double)bf16_to_f32(hRef64[i]));printf("\n");
        }
        double ma=0,me=0,mr=0;int nb2=0;
        for(size_t i=0;i<sz;++i){double a=(double)bf16_to_f32(hO[i]),r=(double)bf16_to_f32(hRef[i]),d2=std::abs(a-r);
            ma=std::max(ma,d2);me+=d2;double rel=std::abs(r)>1e-3?d2/std::abs(r):0;mr=std::max(mr,rel);if(d2>0.05)++nb2;}
        me/=sz;printf("VERIFY B=%d H=%d N=%d D=%d  max_abs=%.4f  mean_abs=%.5f  max_rel=%.4f  n_bad(>0.05)=%d/%zu\n",B,H,N,D,ma,me,mr,nb2,sz);
        printf("First 8 GPU bf16:");for(int i=0;i<8;++i)printf(" %.4f",(double)bf16_to_f32(hO[i]));printf("\n");
        printf("First 8 REF values:");for(int i=0;i<8;++i)printf(" %.4f",(double)bf16_to_f32(hRef[i]));printf("\n");
        printf("First 8 GPU raw16:");
        for(int i=0;i<8;++i) printf(" %04x(%.4g)", (unsigned)hO[i], bf16_to_f32(hO[i]));
        printf("\n");
        int nz=0;for(size_t i=0;i<sz;++i)if(bf16_to_f32(hO[i])==0.0f)++nz;
        printf("GPU zeros: %d/%zu\n",nz,sz);
        // Per-head diagnostics
        for(int b=0;b<B;++b) for(int h=0;h<H;++h) {
            int hz=0, haa=0; auto* base=hO.data()+(b*H+h)*N*D;
            for(int i=0;i<N*D;++i) { if(bf16_to_f32(base[i])==0.0f)++hz; if(base[i]==0xAAAA)++haa; }
            printf("  b=%d h=%d: zeros=%d fill(0xAA)=%d/%d first16:", b, h, hz, haa, N*D);
            for(int i=0;i<16;++i) printf(" %04x", (unsigned)base[i]);
            printf("\n");
        }
        // Also dump raw bytes from start of output buffer
        printf("Raw output dwords at ptr_o:\n");
        const unsigned* raw_dw = reinterpret_cast<const unsigned*>(hO.data());
        for(int i=0;i<24;++i) printf(" [%d]=%08x", i, raw_dw[i]);
        printf("\n");
        // Also check for canary 0xDEAD0001 in first 1024 dwords
        for(int i=0;i<1024;++i) {
            if(raw_dw[i]==0xDEAD0001) printf("Found canary at dword[%d] (byte %d): %08x %08x %08x %08x\n", i, i*4, raw_dw[i], raw_dw[i+1], raw_dw[i+2], raw_dw[i+3]);
        }
    }

    hipEvent_t ev0,ev1; HIP_CALL(hipEventCreate(&ev0)); HIP_CALL(hipEventCreate(&ev1));
    HIP_CALL(hipEventRecord(ev0));
    for(int i=0;i<iters;++i)
        HIP_CALL(hipModuleLaunchKernel(func, total_blocks, 1, 1, BLOCK_SIZE, 1, 1, 0, 0, &kptr, NULL));
    HIP_CALL(hipEventRecord(ev1)); HIP_CALL(hipEventSynchronize(ev1));
    float ms=0; HIP_CALL(hipEventElapsedTime(&ms,ev0,ev1)); ms/=iters;
    double tflops=4.0*B*H*(double)N*N*D/(ms*1e9);
    printf("BENCH  iters=%d  avg=%.3f ms  %.2f TFLOPS\n",iters,ms,tflops);

    HIP_CALL(hipFree(dQ));HIP_CALL(hipFree(dK));HIP_CALL(hipFree(dV));HIP_CALL(hipFree(dVT));HIP_CALL(hipFree(dO));
    HIP_CALL(hipModuleUnload(mod));
    return 0;
}
