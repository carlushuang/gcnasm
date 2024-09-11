#include <stdio.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <random>
#include <map>
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#ifdef USE_MKL
#include <mkl.h>
#endif
#define HALF
#ifdef HALF
#include "half.hpp"
#endif
#include "fmha_asm.hpp"
// #define PER_PIXEL_CHECK
#define ASSERT_ON_FAIL
#define MFMA
// #define ASM_PRINT

#ifndef ABS
#define ABS(x) ((x) > 0 ? (x) : -1 * (x))
#endif

using float16 = half_float::half;
static inline bool valid_vector(const float *ref, const float16 *pred, int n, double nrms = 1e-3)
{
    double s0 = 0.0;
    double s1 = 0.0;
#ifdef PER_PIXEL_CHECK
    int pp_err = 0;
#endif
    int i_start = 0, i_end = n;
    int i_num = i_end - i_start;
    for (int i = i_start; i < i_end; ++i)
    {
        double ri = (double)ref[i];
        double pi = (double)pred[i];
        double d = ri - pi;
        double dd = d * d;
        double rr = 2.0 * ri * ri;
        s0 += dd;
        s1 += rr;

#ifdef PER_PIXEL_CHECK
        double delta = ABS(ri - pi) / ri;
        if (delta > 1e-3)
        {
#ifdef ASSERT_ON_FAIL
            if (pp_err < 100)
                printf("diff at %4d, ref:%lf, pred:%lf(0x%04x), d:%lf\n", i, ri, pi, ((uint16_t *)pred)[i], delta);
#endif
            pp_err++;
        }
#endif
    }
    //    printf("pp_crr:%d, pp_err:%d, crr_ratio:%.3f, nrms:%lf, s0:%lf, s1:%lf\n",i_num-pp_err, pp_err, (float)(i_num-pp_err)/(float)i_num, sqrt(s0/s1),s0,s1);

    return (sqrt(s0 / s1) < nrms)
#ifdef PER_PIXEL_CHECK
           && (pp_err == 0)
#endif
        ;
}

void hgemm_cr_kpack2(
    float *ptr_c,
    const float *__restrict__ ptr_a,
    const float *__restrict__ ptr_b,
    float alpha,
    unsigned int m,
    unsigned int n,
    unsigned int k,
    unsigned int lda,
    unsigned int ldb,
    unsigned int ldc)
{
#ifdef USE_MKL
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                m, n, k, alpha, ptr_a, lda, ptr_b, ldb, 0, ptr_c, ldc);
#else
    // change the layout
    unsigned int im, in, ik;
    for (in = 0; in < n; in++)
    {
        for (im = 0; im < m; im++)
        {
#ifndef MFMA
            float c = .0;
            for (ik = 0; ik < (k >> 1); ik++)
            {
                c += ptr_a[ik * lda * 2 + im * 2] * ptr_b[ik * ldb * 2 + in * 2];
                c += ptr_a[ik * lda * 2 + im * 2 + 1] * ptr_b[ik * ldb * 2 + in * 2 + 1];
            }
            ptr_c[in * ldc + im] = alpha * c;
#endif

#ifdef MFMA
            float c = .0;
            for (ik = 0; ik < (k >> 2); ik++)
            {
                c += ptr_a[ik * 4 * lda + im * 4 + 0] * ptr_b[ik * 4 * ldb + in * 4 + 0] + ptr_a[ik * 4 * lda + im * 4 + 1] * ptr_b[ik * 4 * ldb + in * 4 + 1] + ptr_a[ik * 4 * lda + im * 4 + 2] * ptr_b[ik * 4 * ldb + in * 4 + 2] + ptr_a[ik * 4 * lda + im * 4 + 3] * ptr_b[ik * 4 * ldb + in * 4 + 3];
            }
            ptr_c[in * ldc + im] = alpha * c;
#endif
        }
    }
#endif
}

#define HIP_CALL(call)                                                 \
    do                                                                 \
    {                                                                  \
        hipError_t err = call;                                         \
        if (err != hipSuccess)                                         \
        {                                                              \
            printf("[hiperror](%d) fail to call %s", (int)err, #call); \
            exit(0);                                                   \
        }                                                              \
    } while (0)

static inline int get_int(const char *env_name, int def_value)
{
    char *v = getenv(env_name);
    if (v)
        return atoi(v);
    return def_value;
}

void rand_vector_2d(float *v, int row, int col, int ld)
{
    int r, c;
    static int flag = 0;
    if (!flag)
    {
        srand(time(NULL));
        flag = 1;
    }

    for (r = 0; r < row; r++)
    {
        for (c = 0; c < col; c++)
        {
            v[r * ld + c] = ((float)(rand() % 100)) / 100.0f;
            // v[r*ld+c] = ((float)(r % 100)+1) / 100.0f + ((float)(c % 100)+1) / 1000.0f;
            // v[r*ld+c] = 1.0;
        }
    }
}

// #define HSACO "hgemm128x128.hsaco"
#define HSACO "kernel.co"
// #define HSACO "hgemm_128x128_kpack2"
#define HSA_KERNEL "kernel_func"

// #define Batch    1
// #define Head_num 1
// #define seq_len  128
// #define head_dim  128
// #define seq_len  128

std::map<std::string, int> parse_options(const std::vector<std::string> &optionList)
{
    std::map<std::string, int> options;
    for (const std::string &option : optionList)
    {
        size_t equalPos = option.find('=');
        if (equalPos != std::string::npos)
        {
            std::string key = option.substr(0, equalPos);
            std::string value = option.substr(equalPos + 1);
            options[key] = std::stoull(value);
        }
        else
        {
            // Handle error or ignore options with no equal sign
            std::cerr << "Error: Invalid option format: " << option << std::endl;
        }
    }
    return options;
}

void get_param(std::map<std::string, int> parsedOptions, std::string key, int &value)
{
    auto it = parsedOptions.find(key);
    if (it != parsedOptions.end())
    {
        value = it->second;
    }
}

int main(int argc, char **argv)
{
    hipModule_t module;
    hipFunction_t kernel_func;
    hipEvent_t evt_00, evt_11;
    HIP_CALL(hipSetDevice(0));

    HIP_CALL(hipModuleLoad(&module, HSACO));
    HIP_CALL(hipModuleGetFunction(&kernel_func, module, HSA_KERNEL));

    int validate = get_int("VALIDATE", 0);
    // int m = get_int("M", HGEMM_M);
    // int n = get_int("N", HGEMM_N);
    // int k = get_int("K", HGEMM_K);

    // int lda = m*sizeof(float);
    // int ldb = n*sizeof(float);
    // int ldc = m*sizeof(float);
    int b = 1;     // get_int("M", HGEMM_M);
    int h = 16;    // get_int("N", HGEMM_N);
    int s = 16384; // get_int("K", HGEMM_K);
    int d = 128;
    int atm_f32 = 1;
    int skip_dq_rd = 1;
    int mask = 0;
    int mask_kb = 1;
    int ioperm = 1; //0: bshd, 1:bhsd -----lse,D,dQ always keep in bhsd layout

    int ts_qo = 32;
    int ts_kv = 128;
    int dump_result = 0;
    std::vector<std::string> options;
    for (int i = 1; i < argc; i++)
    {
        options.push_back(argv[i]);
    }
    std::map<std::string, int> parsedOptions = parse_options(options);
    get_param(parsedOptions, "b", b);
    get_param(parsedOptions, "h", h);
    get_param(parsedOptions, "s", s);
    get_param(parsedOptions, "d", d);
    get_param(parsedOptions, "dump_result", dump_result);
    get_param(parsedOptions, "atm_f32", atm_f32);
    get_param(parsedOptions, "mask", mask);
    get_param(parsedOptions, "mask_kb", mask_kb);
    get_param(parsedOptions, "ioperm", ioperm);

    std::cout << "b:" << b << std::endl;
    std::cout << "h:" << h << std::endl;
    std::cout << "s:" << s << std::endl;
    std::cout << "d:" << d << std::endl;
    std::cout << "dump_result:" << dump_result << std::endl;
    std::cout << "atm_f32:" << atm_f32 << std::endl;
    std::cout << "skip_dq_rd:" << skip_dq_rd << std::endl;
    std::cout << "mask:" << mask << std::endl;
    std::cout << "mask_kb:" << mask_kb << std::endl;
    std::cout << "ioperm:" << ioperm << std::endl;

    int stride_tg = ts_kv * d * 2;
    int stride_head = s * d * 2;
    int stride_batch = h * s * d * 2;
    int stride_seqlen = d * 2;

    if (ioperm == 0)//bshd
    {
        stride_seqlen = h * d * 2;
        stride_tg = ts_kv * stride_seqlen;
        stride_head = d * 2;
        stride_batch = h * s * d * 2;
    }

    // int lda = m*sizeof(float);
    // int ldb = n*sizeof(float);
    // int ldc = m*sizeof(float);
    float k_log2e = log2f(expf(1));
    float k_scalar = sqrt(d);

    k_scalar = (float)(1.0 / (double)k_scalar);
    // float alpha = 1.0f;
    // float16 *host_dq, *host_dk, *host_dv;
    float *host_q, *host_k, *host_v, *host_do;
    float *host_lse, *host_odo;

    long sz_mx = b * h * s * d;
    int sz_lsd = b * h * s;
    int sz_mx_pad = 0;//ts_qo * 4 * d;
    int sz_lsd_pad = 0;//ts_qo * 4;
    long sz_mx_dq = 0;

    if (atm_f32 == 2)
       sz_mx_dq = sz_mx * (s/ts_kv);
    else
       sz_mx_dq = sz_mx;

    // float16 *fp16_a, *fp16_b, *fp16_c, *dev_a, *dev_b, *dev_c;
    float16 *host_fp16_q, *host_fp16_k, *host_fp16_v, *host_fp16_do;
    float16 *host_fp16_dq, *host_fp16_dk, *host_fp16_dv;
    float16 *dev_q, *dev_k, *dev_v, *dev_do, *dev_dk, *dev_dv;
    float *dev_lse, *dev_odo, *host_fp32_dq, *dev_dq;

    // int bdx = 256;
    // int gdx = ((m+127)>>7)*((n+127)>>7);

    // fp32 on host
    // host_a = (float*)malloc(lda*k);
    // host_b = (float*)malloc(ldb*k);
    // host_c = (float*)malloc(ldc*n);
    host_q = (float *)malloc(sz_mx * sizeof(float));
    host_k = (float *)malloc(sz_mx * sizeof(float));
    host_v = (float *)malloc(sz_mx * sizeof(float));
    host_do = (float *)malloc(sz_mx * sizeof(float));

    host_lse = (float *)malloc(sz_lsd * sizeof(float));
    host_odo = (float *)malloc(sz_lsd * sizeof(float));

    // rand_vector_2d(host_a, k, m, lda/sizeof(float));
    // rand_vector_2d(host_b, k, n, ldb/sizeof(float));
    int off_mx = 0;
    int off_lsd = 0;
    int init_pattern = 0;
    get_param(parsedOptions, "init_pattern", init_pattern);

    // for (int i = 0; i < b; i++)
    //{
    //     for (int j = 0; j < h; j++)
    //     {
    //         rand_vector_2d(host_q+off_mx, d, s, s);
    //         rand_vector_2d(host_k+off_mx, d, s, s);
    //         rand_vector_2d(host_v+off_mx, d, s, s);
    //         rand_vector_2d(host_do+off_mx, d, s, s);
    //         off_mx += s*d;
    //         rand_vector_2d(host_lse+off_lsd, s, 1, 1);
    //         rand_vector_2d(host_odo+off_lsd, s, 1, 1);
    //         off_lsd += s;
    //     }
    // }

    // fp16 on host
    // fp16_a = (float16*)malloc(lda*(k>>1));
    // fp16_b = (float16*)malloc(ldb*(k>>1));
    // fp16_c = (float16*)malloc(ldc*(n>>1));

    host_fp16_q = (float16 *)malloc(sz_mx * sizeof(float) / 2);
    host_fp16_k = (float16 *)malloc(sz_mx * sizeof(float) / 2);
    host_fp16_v = (float16 *)malloc(sz_mx * sizeof(float) / 2);
    host_fp16_do = (float16 *)malloc(sz_mx * sizeof(float) / 2);

    srand(1);
    fmha_batch_init(host_fp16_q, b, h, s, d, FP16, init_pattern);
    srand(2);
    fmha_batch_init(host_fp16_k, b, h, s, d, FP16, init_pattern);
    srand(3);
    fmha_batch_init(host_fp16_v, b, h, s, d, FP16, init_pattern);
    srand(4);
    fmha_batch_init(host_fp16_do, b, h, s, d, FP16, init_pattern);
    srand(5);
    fmha_batch_init(host_lse, b, h, s, 1, FP32, init_pattern);
    srand(6);
    fmha_batch_init(host_odo, b, h, s, 1, FP32, init_pattern);
    // convert fp32 a and b into fp16 on host
    // for(int i=0; i<m*k; i++)fp16_a[i]=__float2half_rn(host_a[i]);
    // for(int i=0; i<n*k; i++)fp16_b[i]=__float2half_rn(host_b[i]);
    // for(int i=0; i<sz_mx; i++) host_fp16_q[i] =__float2half_rn(host_q[i]);
    // for(int i=0; i<sz_mx; i++) host_fp16_k[i] =__float2half_rn(host_k[i]);
    // for(int i=0; i<sz_mx; i++) host_fp16_v[i] =__float2half_rn(host_v[i]);
    // for(int i=0; i<sz_mx; i++) host_fp16_do[i]=__float2half_rn(host_do[i]);

    host_fp32_dq = (float *)malloc(sz_mx_dq * sizeof(float));
    host_fp16_dq = (float16 *)malloc(sz_mx * sizeof(float) / 2);
    host_fp16_dk = (float16 *)malloc(sz_mx * sizeof(float) / 2);
    host_fp16_dv = (float16 *)malloc(sz_mx * sizeof(float) / 2);
    memset(host_fp32_dq, 0, sz_mx * sizeof(float));

    if (dump_result)
    {
        fmha_dump_batch_inHex(host_fp16_q, "Q.hex", b, h, s, d, FP16);
        fmha_dump_batch_inHex(host_fp16_k, "K.hex", b, h, s, d, FP16);
        fmha_dump_batch_inHex(host_fp16_v, "V.hex", b, h, s, d, FP16);
        fmha_dump_batch_inHex(host_fp16_do, "dO.hex", b, h, s, d, FP16);
        fmha_dump_batch_inHex(host_lse, "L.hex", b, h, s, 1, FP32);
        fmha_dump_batch_inHex(host_odo, "D.hex", b, h, s, 1, FP32);
    }

    printf("dqsize: %zuM\n", sz_mx_dq*4/1024/1024);

    HIP_CALL(hipSetDevice(0));
    // fp16 on device
    // HIP_CALL(hipMalloc(&dev_a, lda*(k>>1)));
    // HIP_CALL(hipMalloc(&dev_b, ldb*(k>>1)));
    // HIP_CALL(hipMalloc(&dev_c, ldc*(n>>1)));
    HIP_CALL(hipMalloc(&dev_dq, sz_mx_dq * sizeof(float)));
    HIP_CALL(hipMalloc(&dev_dk, sz_mx * sizeof(float) / 2));
    HIP_CALL(hipMalloc(&dev_dv, sz_mx * sizeof(float) / 2));

    HIP_CALL(hipMalloc(&dev_q, (sz_mx + sz_mx_pad) * sizeof(float) / 2));
    HIP_CALL(hipMalloc(&dev_k, (sz_mx + sz_mx_pad) * sizeof(float) / 2));
    HIP_CALL(hipMalloc(&dev_v, (sz_mx + sz_mx_pad) * sizeof(float) / 2));
    HIP_CALL(hipMalloc(&dev_do, (sz_mx + sz_mx_pad) * sizeof(float) / 2));

    HIP_CALL(hipMalloc(&dev_lse, (sz_lsd + sz_lsd_pad) * sizeof(float)));
    HIP_CALL(hipMalloc(&dev_odo, (sz_lsd + sz_lsd_pad) * sizeof(float)));

    printf("dev_dq: %p\n", dev_dq);
    printf("dev_dk: %p\n", dev_dk);
    printf("dev_dv: %p\n", dev_dv);
    printf("dev_q: %p\n", dev_q);
    printf("dev_k: %p\n", dev_k);
    printf("dev_v: %p\n", dev_v);
    printf("dev_do: %p\n", dev_do);
    printf("dev_lse: %p\n", dev_lse);
    printf("dev_odo: %p\n", dev_odo);

    // fp16 cpy to device
    // HIP_CALL(hipMemcpy(dev_a, fp16_a, lda*(k>>1), hipMemcpyHostToDevice));
    // HIP_CALL(hipMemcpy(dev_b, fp16_b, ldb*(k>>1), hipMemcpyHostToDevice));
    if (ioperm == 1)
    {
        HIP_CALL(hipMemcpy(dev_q, host_fp16_q, sz_mx * sizeof(float) / 2, hipMemcpyHostToDevice));
        HIP_CALL(hipMemcpy(dev_k, host_fp16_k, sz_mx * sizeof(float) / 2, hipMemcpyHostToDevice));
        HIP_CALL(hipMemcpy(dev_v, host_fp16_v, sz_mx * sizeof(float) / 2, hipMemcpyHostToDevice));
        HIP_CALL(hipMemcpy(dev_do, host_fp16_do, sz_mx * sizeof(float) / 2, hipMemcpyHostToDevice));
    }
    else 
    {
        float16 *host_fp16_perm_q = (float16 *)malloc(sz_mx * sizeof(float) / 2);
        float16 *host_fp16_perm_k = (float16 *)malloc(sz_mx * sizeof(float) / 2);
        float16 *host_fp16_perm_v = (float16 *)malloc(sz_mx * sizeof(float) / 2);
        float16 *host_fp16_perm_do = (float16 *)malloc(sz_mx * sizeof(float) / 2);

        fmha_batch_reshape(host_fp16_perm_q,  host_fp16_q,  b, h, s, d, FP16, 1, ioperm);
        fmha_batch_reshape(host_fp16_perm_k,  host_fp16_k,  b, h, s, d, FP16, 1, ioperm);
        fmha_batch_reshape(host_fp16_perm_v,  host_fp16_v,  b, h, s, d, FP16, 1, ioperm);
        fmha_batch_reshape(host_fp16_perm_do,  host_fp16_do,  b, h, s, d, FP16, 1, ioperm);

        HIP_CALL(hipMemcpy(dev_q, host_fp16_perm_q, sz_mx * sizeof(float) / 2, hipMemcpyHostToDevice));
        HIP_CALL(hipMemcpy(dev_k, host_fp16_perm_k, sz_mx * sizeof(float) / 2, hipMemcpyHostToDevice));
        HIP_CALL(hipMemcpy(dev_v, host_fp16_perm_v, sz_mx * sizeof(float) / 2, hipMemcpyHostToDevice));
        HIP_CALL(hipMemcpy(dev_do, host_fp16_perm_do, sz_mx * sizeof(float) / 2, hipMemcpyHostToDevice));

        free(host_fp16_perm_q);
        free(host_fp16_perm_k);
        free(host_fp16_perm_v);
        free(host_fp16_perm_do);
    }
    HIP_CALL(hipMemcpy(dev_dq, host_fp32_dq, sz_mx * sizeof(float), hipMemcpyHostToDevice));

    HIP_CALL(hipMemcpy(dev_lse, host_lse, sz_lsd * sizeof(float), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(dev_odo, host_odo, sz_lsd * sizeof(float), hipMemcpyHostToDevice));

#ifdef ASM_PRINT
    // debug pointer
    float *host_print, *print;
    host_print = (float *)malloc(bdx * 8);
    HIP_CALL(hipMalloc(&print, bdx * 8));
#endif
    struct p3
    {
        unsigned int _p0;
        unsigned int _p1;
        unsigned int _p2;
    };
    struct p2
    {
        unsigned int _p0;

        unsigned int _p1;
    };
    struct __attribute__((packed))
    {
        void *ptr_dq;
        p2 _p0;
        void *ptr_dk;
        p2 _p1;
        void *ptr_dv;
        p2 _p2;
        void *ptr_q;
        p2 _p3;
        void *ptr_k;
        p2 _p4;
        void *ptr_v;
        p2 _p5;
        void *ptr_do;
        p2 _p6;
        void *ptr_lse;
        p2 _p7;
        void *ptr_odo;
        p2 _p8;
        float scalar;
        p3 _p9;
        float log2e;
        p3 _p10;
        unsigned int seq_len;
        p3 _p11;
        unsigned int Ts;
        p3 _p12;
        unsigned int Hs;
        p3 _p13;
        unsigned int BAs;
        p3 _p14;
        unsigned int Seqs;
        p3 _p15;
#ifdef ASM_PRINT
        void *print;
#endif
    } args;
    size_t arg_size = sizeof(args);
    // args.ptr_c  = (void*)dev_c;
    // args.ptr_a  = (void*)dev_a;
    // args.ptr_b  = (void*)dev_b;
    // args.alpha  = alpha;
    // args.m      = m;
    // args.n      = n;
    // args.k      = k;
    // args.lda    = lda;
    // args.ldb    = ldb;
    // args.ldc    = ldc;
    args.ptr_dq = (void *)dev_dq;
    args.ptr_dk = (void *)dev_dk;
    args.ptr_dv = (void *)dev_dv;
    args.ptr_q = (void *)dev_q;
    args.ptr_k = (void *)dev_k;
    args.ptr_v = (void *)dev_v;
    args.ptr_do = (void *)dev_do;
    args.ptr_lse = (void *)dev_lse;
    args.ptr_odo = (void *)dev_odo;
    args.scalar = k_scalar;
    args.log2e = k_log2e;
    args.seq_len = s;
    args.Ts = stride_tg;
    args.Hs = stride_head;
    args.BAs = stride_batch;
    args.Seqs = stride_seqlen;
#ifdef ASM_PRINT
    args.print = (void *)print;
#endif

    printf("argsize: %zu\n", arg_size);

    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &arg_size, HIP_LAUNCH_PARAM_END};

    int total_loop = 1;
    int warm_ups = 0;
    int i;

    int bdx = 256;
    int gdx = s / ts_kv;
    int gdy = h;
    int gdz = b;

    if (mask && mask_kb)
    {
        int num_tg = s / ts_kv;
        gdx = (num_tg%2) ? (num_tg/2+1) : (num_tg/2);
    }

    for (i = 0; i < warm_ups; i++)
    {
        HIP_CALL(hipModuleLaunchKernel(kernel_func, gdx, gdy, gdz, bdx, 1, 1, 0, 0, NULL, (void **)&config));
        std::cout << "safe here" << std::endl;
    }

#ifdef ASM_PRINT
    int max_i = 256;
    HIP_CALL(hipMemcpy(host_print, print, 8 * max_i, hipMemcpyDeviceToHost));
    for (int i = 0; i < max_i; i++)
    {
        if (((uint32_t *)host_print)[2 * i + 1] != 0x5c005c00)
            printf("Thread%d, PrintVal:0x%x\n", ((int *)host_print)[2 * i], ((uint32_t *)host_print)[2 * i + 1]);
        // std::cout<<"Thread"<<((int*) host_print)[2*i]<<", PrintVal1:"<<(((float16*)host_print)[4*i+2])<<
        //", PrintVal2:"<<( ( (float16*)host_print )[4*i+3] )<<std::endl;
    }
#endif

    hipEventCreate(&evt_00);
    hipEventCreate(&evt_11);
    hipDeviceSynchronize();
    hipEventRecord(evt_00, NULL);
    for (i = 0; i < total_loop; i++)
        HIP_CALL(hipModuleLaunchKernel(kernel_func, gdx, gdy, gdz, bdx, 1, 1, 0, 0, NULL, (void **)&config));

    std::cout << "we are done" << std::endl;
    float elapsed_ms;
    hipEventRecord(evt_11, NULL);
    hipEventSynchronize(evt_11);
    hipDeviceSynchronize();
    hipEventElapsedTime(&elapsed_ms, evt_00, evt_11);
    hipEventDestroy(evt_00);
    hipEventDestroy(evt_11);

    float time_per_loop = elapsed_ms / total_loop;
    float gflops = (float)2 * 5 * b * h * d * s * s / time_per_loop / (1e6);
    if(mask)
       gflops = gflops/2;
    printf("b:%d,h:%d,s:%d,d:%d, time: %.3f, gflops:%.3f\n", b, h, s, d, time_per_loop, gflops);
    // if(validate){
    //     hgemm_cr_kpack2(host_c, host_a, host_b, alpha, m,n,k,lda/sizeof(float),ldb/sizeof(float),ldc/sizeof(float));
    //     HIP_CALL(hipMemcpy(fp16_c, dev_c, ldc*(n>>1), hipMemcpyDeviceToHost));
    //     bool res = valid_vector( host_c, fp16_c, m*n );
    //     printf(",%s",res?"valid":"fail");
    // }
    // printf("\n");
    
    if ((atm_f32 == 1) || ((!skip_dq_rd)&&(atm_f32 == 2)))
       HIP_CALL(hipMemcpy(host_fp32_dq, dev_dq, sz_mx_dq * sizeof(float), hipMemcpyDeviceToHost));
    else
    {
       if (ioperm == 1)
            HIP_CALL(hipMemcpy((void*)host_fp16_dq, dev_dq, sz_mx * sizeof(float) / 2, hipMemcpyDeviceToHost));
       else
       {
            float16 *host_fp16_perm_dq = (float16 *)malloc(sz_mx * sizeof(float) / 2);
            HIP_CALL(hipMemcpy(host_fp16_perm_dq, dev_dq, sz_mx * sizeof(float) / 2, hipMemcpyDeviceToHost));
            fmha_batch_reshape(host_fp16_dq, host_fp16_perm_dq, b, h, s, d, FP16, ioperm, 1);
            free(host_fp16_perm_dq);    
       } 
    }
    if (ioperm == 1)
    {
       HIP_CALL(hipMemcpy(host_fp16_dk, dev_dk, sz_mx * sizeof(float) / 2, hipMemcpyDeviceToHost));
       HIP_CALL(hipMemcpy(host_fp16_dv, dev_dv, sz_mx * sizeof(float) / 2, hipMemcpyDeviceToHost));
    }
    else 
    {
        float16 *host_fp16_perm_dk = (float16 *)malloc(sz_mx * sizeof(float) / 2);
        float16 *host_fp16_perm_dv = (float16 *)malloc(sz_mx * sizeof(float) / 2);

        HIP_CALL(hipMemcpy(host_fp16_perm_dk, dev_dk, sz_mx * sizeof(float) / 2, hipMemcpyDeviceToHost));
        HIP_CALL(hipMemcpy(host_fp16_perm_dv, dev_dv, sz_mx * sizeof(float) / 2, hipMemcpyDeviceToHost));

        fmha_batch_reshape(host_fp16_dk, host_fp16_perm_dk, b, h, s, d, FP16, ioperm, 1);
        fmha_batch_reshape(host_fp16_dv, host_fp16_perm_dv, b, h, s, d, FP16, ioperm, 1);

        free(host_fp16_perm_dk);
        free(host_fp16_perm_dv);
    }
    if (atm_f32 == 1)
       fmha_batch_cvt(host_fp16_dq, host_fp32_dq, b, h, s, d, FP16);
    else if ((atm_f32 == 2)&&(!skip_dq_rd))
    {
        fmha_bwd_dQ_redc(host_fp32_dq, b, h, s, d, s/ts_kv);
        fmha_batch_cvt(host_fp16_dq, host_fp32_dq, b, h, s, d, FP16);
    }

    if (dump_result)
    { 
        fmha_dump_batch_inHex(host_fp16_dq, "gpu_dQ.hex", b, h, s, d, FP16);
        //fmha_dump_batch_inHex(host_fp32_dq, "gpu_dQ32.hex", b, h, s, d, FP32);
        fmha_dump_batch_inHex(host_fp16_dk, "gpu_dK.hex", b, h, s, d, FP16);
        fmha_dump_batch_inHex(host_fp16_dv, "gpu_dV.hex", b, h, s, d, FP16);
    }
    // free(host_a);
    // free(host_b);
    // free(host_c);
    // free(fp16_a);
    // free(fp16_b);
    // free(fp16_c);

    // hipFree(dev_a);
    // hipFree(dev_b);
    // hipFree(dev_c);

    free(host_q);
    free(host_k);
    free(host_v);
    free(host_do);
    free(host_lse);
    free(host_odo);
    free(host_fp16_q);
    free(host_fp16_k);
    free(host_fp16_v);
    free(host_fp16_do);
    free(host_fp16_dq);
    free(host_fp32_dq);
    free(host_fp16_dk);
    free(host_fp16_dv);

    hipFree(dev_q);
    hipFree(dev_k);
    hipFree(dev_v);
    hipFree(dev_do);
    hipFree(dev_dq);
    hipFree(dev_dk);
    hipFree(dev_dv);
    hipFree(dev_lse);
    hipFree(dev_odo);

#ifdef ASM_PRINT
    free(host_print);
    hipFree(print);
#endif
    // printf("CU:%d, TIPS:%.3f(2x:%.3f, 4x:%.3f), cost:%fms per loop\n", num_cu, tips, 2*tips, 4*tips, time_per_loop);
}
