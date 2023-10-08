#include "hgemm_mfma_invoker.hpp"

#include <stdio.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <random>
#include <iostream>
#include <stdlib.h>
#include <math.h>


using float16=half_float::half;

static inline bool valid_vector( const float* ref, const float16* pred, int n, double nrms = 1e-3 )
{    
    double s0=0.0;
    double s1=0.0;
#ifdef PER_PIXEL_CHECK
    int pp_err = 0;
#endif
    int i_start = 0, i_end=n;
    
    for( int i=i_start; i<i_end; ++i ){
        double ri=(double)ref[i];
        double pi=(double)pred[i];
        double d=ri-pi;
        double dd=d*d;
        double rr=2.0*ri*ri;
        s0+=dd;
        s1+=rr;
        
#ifdef PER_PIXEL_CHECK
        double delta = ABS(ri-pi)/ri;
        if(delta>1e-3){
#ifdef ASSERT_ON_FAIL
            if(pp_err<100)
            printf("diff at %4d, ref:%lf, pred:%lf(0x%04x), d:%lf\n",i,ri,pi,((uint16_t*)pred)[i],delta);
#endif
            pp_err++;
        }
#endif
    }
    // int i_num = i_end - i_start;
    // printf("pp_crr:%d, pp_err:%d, crr_ratio:%.3f, nrms:%lf, s0:%lf, s1:%lf\n",i_num-pp_err, pp_err, (float)(i_num-pp_err)/(float)i_num, sqrt(s0/s1),s0,s1);

    return (sqrt(s0/s1)<nrms)
#ifdef PER_PIXEL_CHECK
        && (pp_err==0)
#endif
    ;
}

void rand_vector_2d(float* v, int row, int col, int ld){
    int r,c;
    static int flag = 0;
    if(!flag){ srand(time(NULL)); flag = 1; }
    for(r=0;r<row;r++){
        for(c=0;c<col;c++){
            v[r*ld+c] = ((float)(rand() % 100)) / 100.0f;
            // v[r*ld+c] = ((float)(rand() % 6)) - 3;
            // v[r*ld+c] = 1;
        }
    }
}

void rand_vector_2d_int(float* v, int row, int col, int ld){
    int r,c;
    static int flag = 0;
    if(!flag){ srand(time(NULL)); flag = 1; }
    for(r=0;r<row;r++){
        for(c=0;c<col;c++){
            v[r*ld+c] = ((float)(rand() % 10)) - 5;
        }
    }
}

void gemm_rcr(
    float*  ptr_c,
    const float*  __restrict__ ptr_a,
    const float*  __restrict__ ptr_b,
    index_t m,
    index_t n,
    index_t k,
    index_t lda,
    index_t ldb,
    index_t ldc)
{
    for(auto i_m = 0 ; i_m < m; i_m++) {
        for(auto i_n = 0; i_n < n; i_n++) {
            float acc = 0;
            for(auto i_k = 0; i_k < k; i_k++) {
                acc += ptr_a[i_m * lda + i_k] * ptr_b[i_n * ldb + i_k];
            }
            ptr_c[i_m * ldc + i_n] = acc;
        }
    }
}


int main(int argc, char ** argv)
{
    int validation = 0;
    int m = 3840;
    int n = 4096;
    int k = 4096;
    if(argc >= 2) {
        validation = atoi(argv[1]);
    }
    if(argc >= 5) {
        m = atoi(argv[2]);
        n = atoi(argv[3]);
        k = atoi(argv[4]);
    }
    int lda = k;
    int ldb = k;
    int ldc = n;

    if(argc >= 8) {
        lda = atoi(argv[5]);
        ldb = atoi(argv[6]);
        ldc = atoi(argv[7]);
    }

    float *host_a, *host_b, *host_c;
    float16 *fp16_a, *fp16_b, *fp16_c, *dev_a, *dev_b, *dev_c;

    //fp32 on host
    host_a = (float*)malloc(lda*m*sizeof(float));
    host_b = (float*)malloc(ldb*n*sizeof(float));
    host_c = (float*)malloc(ldc*m*sizeof(float));
    rand_vector_2d(host_a, m, k, lda);
    rand_vector_2d(host_b, n, k, ldb);
    //fp16 on host
    fp16_a = (float16*)malloc(lda*m*sizeof(float16));
    fp16_b = (float16*)malloc(ldb*n*sizeof(float16));
    fp16_c = (float16*)malloc(ldc*m*sizeof(float16));
    //convert fp32 a and b into fp16 on host
    for(int i=0; i<lda*m; i++)fp16_a[i]=__float2half_rn(host_a[i]);
    for(int i=0; i<ldb*n; i++)fp16_b[i]=__float2half_rn(host_b[i]);

    HIP_CALL(hipMalloc(&dev_a, lda*m*sizeof(float16)));
    HIP_CALL(hipMalloc(&dev_b, ldb*n*sizeof(float16)));
    HIP_CALL(hipMalloc(&dev_c, ldc*m*sizeof(float16)));
    //fp16 cpy to device
    HIP_CALL(hipMemcpy(dev_a, fp16_a, lda*m*sizeof(float16), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(dev_b, fp16_b, ldb*n*sizeof(float16), hipMemcpyHostToDevice));

    printf("m:%d,n:%d,k:%d,lda:%d,ldb:%d,ldc:%d",  m, n, k, lda, ldb, ldc); fflush(stdout);

    auto invoker = gemm_invoker<kernel_t>{};
    auto karg = invoker.make_karg(dev_a, dev_b, dev_c, m, n, k, lda, ldb, ldc);
    bool applicable = invoker.is_applicable(karg);
    float ms = 0;
    if(applicable) ms = invoker.bench(karg);

    float gflops = applicable ? (float)2*m*n*k/ms/(1e9) : 0;
    printf(" ms:%f, tflops:%.3f",ms,gflops); fflush(stdout);
    if(!applicable) {
        printf(", not applicable"); fflush(stdout);
    }

    if(applicable && validation){
        gemm_rcr(host_c, host_a, host_b, m,n,k,lda,ldb,ldc);
        HIP_CALL(hipMemcpy(fp16_c, dev_c, ldc*m*sizeof(float16), hipMemcpyDeviceToHost));
        bool res = valid_vector( host_c, fp16_c, m*n );
        printf(", %s",res?"valid":"fail");fflush(stdout);
    }

    printf("\n"); fflush(stdout);
    
    free(host_a);
    free(host_b);
    free(host_c);
    free(fp16_a);
    free(fp16_b);
    free(fp16_c);
    
    HIP_CALL(hipFree(dev_a));
    HIP_CALL(hipFree(dev_b));
    HIP_CALL(hipFree(dev_c));
}