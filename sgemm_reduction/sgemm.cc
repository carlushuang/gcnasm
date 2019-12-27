#include <stdio.h>
#include <hip/hip_runtime.h>
#include <random>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#ifdef USE_MKL
#include <mkl.h>
#endif

#define PER_PIXEL_CHECK
#define ASSERT_ON_FAIL

#ifndef ABS
#define ABS(x) ((x)>0?(x):-1*(x))
#endif

static inline bool valid_vector( const float* ref, const float* pred, int n, double nrms = 1e-6 )
{
    double s0=0.0;
    double s1=0.0;
#ifdef PER_PIXEL_CHECK
    int pp_err = 0;
#endif
    for( int i=0; i<n; ++i ){
        double ri=(double)ref[i];
        double pi=(double)pred[i];
        double d=ri-pi;
        double dd=d*d;
        double rr=2.0*ri*ri;
        s0+=dd;
        s1+=rr;
#ifdef PER_PIXEL_CHECK
        double delta = ABS(ri-pi)/ri;
        if(delta>3e-5){
#ifdef ASSERT_ON_FAIL
            //if(pp_err<100)
                printf("diff at %4d, ref:%lf, pred:%lf(0x%08x), d:%lf\n",i,ri,pi,((uint32_t*)pred)[i],delta);
#endif
            pp_err++;
        }
#endif
    }
    //printf("nrms:%lf, s0:%lf, s1:%lf\n",sqrt(s0/s1),s0,s1);
    return (sqrt(s0/s1)<nrms)
#ifdef PER_PIXEL_CHECK
        && (pp_err==0)
#endif
    ;
}

void sgemm_cr(
    float*  ptr_c,
    const float*  __restrict__ ptr_a,
    const float*  __restrict__ ptr_b,
    float alpha,
    unsigned int m,
    unsigned int n,
    unsigned int k,
    unsigned int lda,
    unsigned int ldb,
    unsigned int ldc)
{
#ifdef USE_MKL
    cblas_sgemm(CblasColMajor,CblasNoTrans,CblasTrans,
        m,n,k,alpha,ptr_a,lda,ptr_b,ldb,0,ptr_c,ldc);
#else
    unsigned int im,in,ik;
    for(in=0;in<n;in++){
        for(im=0;im<m;im++){
            float c = .0;
            for(ik=0;ik<k;ik++)
                c += ptr_a[ik*lda+im]*ptr_b[ik*ldb+in];
            ptr_c[in*ldc+im] = alpha*c;
        }
    }
#endif
}

#define HIP_CALL(call) do{  \
    hipError_t err = call;  \
    if(err != hipSuccess){  \
        printf("[hiperror](%d) fail to call %s",(int)err,#call);    \
        exit(0);            \
    }                       \
} while(0)

static inline int get_int(const char* env_name, int def_value)
{
    char * v = getenv(env_name);
    if(v)
        return atoi(v);
    return def_value;
}

void rand_vector_2d(float* v, int row, int col, int ld){
    int r,c;
    static int flag = 0;
    if(!flag){ srand(time(NULL)); flag = 1; }

    for(r=0;r<row;r++){
        for(c=0;c<col;c++){
            v[r*ld+c] = ((float)(rand() % 100)) / 100.0f;
            //v[r*ld+c] = ((float)(r % 100)+1) / 100.0f + ((float)(c % 100)+1) / 1000.0f;
            //v[r*ld+c] = 0.01;
        }
    }
}

#define HSACO "kernel.co"
#define HSA_KERNEL "sgemm_64x64_wr"

#define SGEMM_M 768
#define SGEMM_N 768
#define SGEMM_K 8192

int main(int argc, char ** argv){
    hipModule_t module;
    hipFunction_t kernel_func;
    hipEvent_t evt_00, evt_11;
    HIP_CALL(hipSetDevice(0));

    HIP_CALL(hipModuleLoad(&module, HSACO));
    HIP_CALL(hipModuleGetFunction(&kernel_func, module, HSA_KERNEL));

    int validate = get_int("VALIDATE", 0);
    int m = get_int("M", SGEMM_M);
    int n = get_int("N", SGEMM_N);
    int k = get_int("K", SGEMM_K);
    int lda = m*sizeof(float);
    int ldb = n*sizeof(float);
    int ldc = m*sizeof(float);
    float alpha = 1.0f;
    float *host_a, *host_b, *host_c, *dev_a, *dev_b, *dev_c, *host_ch;
    int bdx = 256;
    int gdx = ((m+63)>>6)*((n+63)>>6);

    host_a = (float*)malloc(lda*k);
    host_b = (float*)malloc(ldb*k);
    host_c = (float*)malloc(ldc*n);
    rand_vector_2d(host_a, k, m, lda/sizeof(float));
    rand_vector_2d(host_b, k, n, ldb/sizeof(float));

    HIP_CALL(hipSetDevice(0));
    HIP_CALL(hipMalloc(&dev_a, lda*k));
    HIP_CALL(hipMalloc(&dev_b, ldb*k));
    HIP_CALL(hipMalloc(&dev_c, ldc*n));
    HIP_CALL(hipMemcpy(dev_a, host_a, lda*k, hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(dev_b, host_b, ldb*k, hipMemcpyHostToDevice));

    int total_loop=100;
    int warm_ups = 5;
    int i;

    struct __attribute__((packed)) {
        void*  ptr_c;
        void*  ptr_a;
        void*  ptr_b;
        float alpha;
        unsigned int m;
        unsigned int n;
        unsigned int k;
        unsigned int lda;
        unsigned int ldb;
        unsigned int ldc;
    } args;
    size_t arg_size = sizeof(args);
    args.ptr_c  = (void*)dev_c;
    args.ptr_a  = (void*)dev_a;
    args.ptr_b  = (void*)dev_b;
    args.alpha  = alpha;
    args.m      = m;
    args.n      = n;
    args.k      = k;
    args.lda    = lda;
    args.ldb    = ldb;
    args.ldc    = ldc;
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                    &arg_size, HIP_LAUNCH_PARAM_END};

    for(i=0;i<warm_ups;i++)
        HIP_CALL(hipModuleLaunchKernel(kernel_func, gdx,1,1, bdx,1,1,  0, 0, NULL, (void**)&config ));

    hipEventCreate(&evt_00);
    hipEventCreate(&evt_11);

    hipDeviceSynchronize();
    hipEventRecord(evt_00, NULL);
    for(i=0;i<total_loop;i++)
        HIP_CALL(hipModuleLaunchKernel(kernel_func, gdx,1,1, bdx,1,1,  0, 0, NULL, (void**)&config ));

    float elapsed_ms;
    hipEventRecord(evt_11, NULL);
    hipEventSynchronize(evt_11);
    hipDeviceSynchronize();
    hipEventElapsedTime(&elapsed_ms, evt_00, evt_11);
    hipEventDestroy(evt_00);
    hipEventDestroy(evt_11);

    float time_per_loop = elapsed_ms/total_loop;
    float gflops = (float)2*m*n*k/time_per_loop/(1e6);
    printf("m:%d,n:%d,k:%d,gflops:%.3f",m,n,k,gflops);
    if(validate){
        sgemm_cr(host_c, host_a, host_b, alpha, m,n,k,lda/sizeof(float),ldb/sizeof(float),ldc/sizeof(float));
        host_ch = (float*)malloc(ldc*n);
        HIP_CALL(hipMemcpy(host_ch, dev_c, ldc*n, hipMemcpyDeviceToHost));
        bool res = valid_vector( host_c, host_ch, m*n );
        printf(",%s",res?"valid":"fail");
    }
    printf("\n");
}
