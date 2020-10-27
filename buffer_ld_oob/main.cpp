#include <stdio.h>
#include <hip/hip_runtime.h>
#include <random>
#include <iostream>

#define HIP_CALL(call) do{  \
    hipError_t err = call;  \
    if(err != hipSuccess){  \
        printf("[hiperror](%d) fail to call %s",(int)err,#call);    \
        exit(0);            \
    }                      \
} while(0)

#define HSACO "buffer_ld_oob.hsaco"
#define HSA_KERNEL "kernel_func"


#define PER_PIXEL_CHECK
#define ASSERT_ON_FAIL

#ifndef ABS
#define ABS(x) ((x)>0?(x):-1*(x))
#endif

template<typename T>
void rand_vec(T * seq, size_t len){
    static std::random_device rd;   // seed
    static std::mt19937 mt(rd());
    static std::uniform_real_distribution<T> dist(-10.0, 10.0);

    for(size_t i=0;i<len;i++) seq[i] =  dist(mt);
}

template<typename T>
void set_vec(T * seq, size_t len, T value){
    for(size_t i=0;i<len;i++) seq[i] =  value;
}

static inline bool valid_vector( const float* ref, const float* pred, int n, double nrms = 1e-6 )
{
    int pp_err = 0;
    for( int i=0; i<n; ++i ){
        if(ref[i] != pred[i]){
#ifdef ASSERT_ON_FAIL
            if(pp_err<100)
                printf("diff at %4d, ref:%lf, pred:%lf(0x%08x), d:%lf\n",i,ref[i],pred[i],((uint32_t*)pred)[i],ABS(ref[i] - pred[i]));
#endif
            pp_err++;
        }
    }
    return pp_err == 0;
}

int main(int argc, char ** argv){
    hipModule_t module;
    hipFunction_t kernel_func;
    HIP_CALL(hipSetDevice(0));

    HIP_CALL(hipModuleLoad(&module, HSACO));
    HIP_CALL(hipModuleGetFunction(&kernel_func, module, HSA_KERNEL));

    int num_cu;
    {
        hipDeviceProp_t dev_prop;
        hipDevice_t dev;
        HIP_CALL(hipGetDevice( &dev ));
        HIP_CALL(hipGetDeviceProperties( &dev_prop, dev ));
        num_cu = dev_prop.multiProcessorCount;
    }

    int bdx = 256;
    int gdx = num_cu;
    float * host_in, * host_out, *dev_in, *dev_out;

    // int total_floats = 1073741760;
    // int total_floats = 256*2;
    int total_floats = 256*2;

    host_in   = new float[total_floats];
    host_out  = new float[total_floats];
    HIP_CALL(hipMalloc(&dev_in,  sizeof(float) * total_floats));
    HIP_CALL(hipMalloc(&dev_out, sizeof(float) * total_floats));

    rand_vec(host_in, total_floats);
    set_vec(host_out, total_floats, (float)0);

    HIP_CALL(hipMemcpy(dev_in, host_in, sizeof(float)*total_floats, hipMemcpyHostToDevice));

    printf("memcpy, input:%p, output:%p, floats:%d\n",dev_in,dev_out, total_floats);

    struct __attribute__((packed)){
        float * input;
        float * output;
        int     total_number;
        int     _pack;
    } args;

    size_t arg_size = sizeof(args);
    args.input = dev_in;
    args.output = dev_out;
    args.total_number = total_floats;

    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                    &arg_size, HIP_LAUNCH_PARAM_END};

    HIP_CALL(hipModuleLaunchKernel(kernel_func, gdx,1,1, bdx,1,1,  0, 0, NULL, (void**)&config ));

    HIP_CALL(hipMemcpy(host_out, dev_out, sizeof(float)*total_floats, hipMemcpyDeviceToHost));

    bool is_valid = valid_vector(host_in, host_out, total_floats);
    if(!is_valid){
        printf("not valid, please check\n");
    }

    delete [] host_in;
    delete [] host_out;
    hipFree(dev_in);
    hipFree(dev_out);
}
