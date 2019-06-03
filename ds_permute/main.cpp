#include <stdio.h>
#include <hip/hip_runtime.h>
#define HIP_CALL(call) do{  \
    hipError_t err = call;  \
    if(err != hipSuccess){  \
        printf("[hiperror](%d) fail to call %s",(int)err,#call);    \
        exit(0);            \
    }                       \
} while(0)

int get_int(const char* name, int default_val){
    char * v = getenv(name);
    if(v){
        return atoi(v);
    }
    return default_val;
}

#define HSACO "kernel.co"
#define HSA_KERNEL "shfl_xor_test"
int main(int argc, char ** argv){
    hipModule_t module;
    hipFunction_t kernel_func;
    int len = 32;
    int * host_in, * host_out, *dev_in, *dev_out;
    int mask = get_int("mask",1);
    int width = get_int("width",32);
    printf("mask:%d, width:%d\n",mask, width);

    host_in   = new int[len];
    host_out  = new int[len];
    for(int i=0;i<len;i++){
        host_in[i] = i;
    }
    HIP_CALL(hipSetDevice(0));
    HIP_CALL(hipMalloc(&dev_in, sizeof(int)*len));
    HIP_CALL(hipMalloc(&dev_out, sizeof(int)*len));
    HIP_CALL(hipMemcpy(dev_in, host_in, sizeof(int)*len, hipMemcpyHostToDevice));
    struct {
        int * in;
        int * out;
        int mask;
        int width;
    } args;
    args.in = dev_in;
    args.out = dev_out;
    args.mask = mask;
    args.width = width;
    size_t arg_size = sizeof(args);
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &arg_size, HIP_LAUNCH_PARAM_END};

    HIP_CALL(hipModuleLoad( &module, HSACO ));
    HIP_CALL(hipModuleGetFunction(&kernel_func, module, HSA_KERNEL));
    HIP_CALL(hipModuleLaunchKernel(kernel_func, 1,1,1, 32,1,1,  0, 0, NULL, (void**)&config ));
    HIP_CALL(hipMemcpy(host_out, dev_out, sizeof(int)*len, hipMemcpyDeviceToHost));
    for(int i=0;i<len;i++){
        printf("%2d - %2d\n", host_in[i], host_out[i]);
    }
    delete [] host_in;
    delete [] host_out;
}
