#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <random>

#define HIP_CALL(call) do{  \
    hipError_t err = call;  \
    if(err != hipSuccess){  \
        printf("[hiperror](%d) fail to call %s",(int)err,#call);    \
        exit(0);            \
    }                       \
} while(0)

void host_transpose_32x32(const float * in, float *out){
    size_t i,j;
    for(j=0;j<32;j++){
        for(i=0;i<32;i++){
            out[i*32+j] = in[j*32+i];
        }
    }
}
template<typename T>
void rand_vec(T * seq, size_t len){
    static std::random_device rd;   // seed
    static std::mt19937 mt(rd());
    static std::uniform_real_distribution<T> dist(-10.0, 10.0);

    for(size_t i=0;i<len;i++) seq[i] =  dist(mt);
}
#ifndef ABS
#define ABS(x) ((x)>0?(x):-1*(x))
#endif

template<typename T>
int valid_vector(const T* lhs, const T * rhs, size_t len, T delta = (T)0.0001){
    size_t i;
    int err_cnt = 0;
    for(i = 0;i < len; i++){
        T d = lhs[i]- rhs[i];
        d = ABS(d);
        if(d > delta){
            printf(" diff at %d, lhs:%f, rhs:%f\n", (int)i, lhs[i], rhs[i]);
            err_cnt++;
        }
    }
    return err_cnt;
}
template<typename T>
void copy_vec(T * src, T* dst, size_t len){
    for(size_t i=0;i<len;i++) dst[i] = src[i];
}

#define HSACO "kernel.co"
#define HSA_KERNEL "transpose_32x32"
#define LEN 32

int main(){
    hipModule_t module;
    hipFunction_t kernel_func;
    float * host_in, * host_out;
    float * dev_in, * dev_out;

    host_in = new float [LEN*LEN];
    host_out = new float [LEN*LEN];

    rand_vec(host_in, LEN*LEN);

    HIP_CALL(hipSetDevice(0));
    HIP_CALL(hipMalloc(&dev_in, sizeof(float)*LEN*LEN));
    HIP_CALL(hipMalloc(&dev_out, sizeof(float)*LEN*LEN));
    HIP_CALL(hipMemcpy(dev_in, host_in, sizeof(float)*LEN*LEN, hipMemcpyHostToDevice));

    struct {
        float * in;
        float * out;
    } args;
    args.in = dev_in;
    args.out = dev_out;
    size_t arg_size = sizeof(args);
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &arg_size, HIP_LAUNCH_PARAM_END};

    HIP_CALL(hipModuleLoad( &module, HSACO ));
    HIP_CALL(hipModuleGetFunction(&kernel_func, module, HSA_KERNEL));
    HIP_CALL(hipModuleLaunchKernel(kernel_func, 1,1,1, 32,1,1,  0, 0, NULL, (void**)&config ));
    HIP_CALL(hipMemcpy(host_out, dev_out, sizeof(float)*LEN*LEN, hipMemcpyDeviceToHost));
    {
        float * host_out_2 = new float [LEN*LEN];
        host_transpose_32x32(host_in, host_out_2);
        valid_vector(host_out, host_out_2, LEN*LEN);
        delete [] host_out_2;
    }
    delete [] host_in;
    delete [] host_out;
}
