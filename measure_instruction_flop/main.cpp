#include <stdio.h>
#include <hip/hip_runtime.h>
#include <random>
#include <iostream>

#define HIP_CALL(call) do{  \
    hipError_t err = call;  \
    if(err != hipSuccess){  \
        printf("[hiperror](%d) fail to call %s",(int)err,#call);    \
        exit(0);            \
    }                       \
} while(0)


#define HSACO "kernel.co"
#define HSA_KERNEL "kernel_func"
#define NUM_CU 64

int main(int argc, char ** argv){
    hipModule_t module;
    hipFunction_t kernel_func;
    hipEvent_t evt_00, evt_11;
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


    int total_loop=200;
    int warm_ups = 3;
    int i;
    int inst_blocks = 1024*8;
    int inst_loop = 256;
    int fma_factor = 2;
    int bdx = 256;
    int gdx = num_cu;   //

    struct {
        int * dummy_ptr;
        int inst_blocks;
    } args;
    size_t arg_size = sizeof(args);
    args.inst_blocks = inst_blocks;
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                    &arg_size, HIP_LAUNCH_PARAM_END};

    for(i=0;i<warm_ups;i++)
        HIP_CALL(hipModuleLaunchKernel(kernel_func, gdx,1,1, bdx,1,1,  0, 0, NULL, (void**)&config ));

    hipEventCreate(&evt_00);
    hipEventCreate(&evt_11);

    hipCtxSynchronize();
    hipEventRecord(evt_00, NULL);
    for(i=0;i<total_loop;i++)
        HIP_CALL(hipModuleLaunchKernel(kernel_func, gdx,1,1, bdx,1,1,  0, 0, NULL, (void**)&config ));

    float elapsed_ms;
    hipEventRecord(evt_11, NULL);
    hipEventSynchronize(evt_11);
    hipCtxSynchronize();
    hipEventElapsedTime(&elapsed_ms, evt_00, evt_11);
    hipEventDestroy(evt_00);
    hipEventDestroy(evt_11);

    float time_per_loop = elapsed_ms/total_loop;

    printf("%d CU, %fTflops, cost:%fms per loop\n",num_cu,  (double)inst_loop*inst_blocks*num_cu*bdx*fma_factor/time_per_loop/1e9,   time_per_loop);

}
