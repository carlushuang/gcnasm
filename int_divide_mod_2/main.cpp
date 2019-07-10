#include <stdio.h>
#include <hip/hip_runtime.h>
#include <random>

#define HIP_CALL(call) do{  \
    hipError_t err = call;  \
    if(err != hipSuccess){  \
        printf("[hiperror](%d) fail to call %s",(int)err,#call);    \
        exit(0);            \
    }                       \
} while(0)


int host_func(int * in_vec, int * out_vec, int num, int divider){
    for(int i=0;i<num;i++){
        out_vec[i] = in_vec[i] / divider;
    }
    return 0;
}
template<typename T>
void rand_vec(T * seq, size_t len){
    static std::random_device rd;   // seed
    static std::mt19937 mt(rd());
    static std::uniform_real_distribution<float> dist(0.0,2147483648); // 2**20-1

    for(size_t i=0;i<len;i++) seq[i] =  (T)dist(mt);
}
template<typename T>
void dump_vector(const T*vec, size_t len){
    for(size_t i=0;i<len;i++){
        std::cout<<vec[i]<<", ";
    }
    std::cout<<"\n";
}
template<typename T>
int valid_vector_integer(const T* lhs, const T * rhs, size_t len){
    int err_cnt = 0;
    for(size_t i = 0;i < len; i++){
        if(lhs[i] == rhs[i])
            ;
        else{
            printf(" diff at %d, lhs:%d, rhs:%d\n", (int)i, lhs[i], rhs[i]);
            err_cnt++;
        }
    }
    return err_cnt;
}
#define HSACO "kernel.co"
#define HSA_KERNEL "kernel_func"
int main(int argc, char ** argv){
    hipModule_t module;
    hipFunction_t kernel_func;
    int len = 128;
    int * host_in, * host_out, *dev_in, *dev_out;

    host_in   = new int[len];
    host_out  = new int[len];

    HIP_CALL(hipSetDevice(0));
    HIP_CALL(hipMalloc(&dev_in, sizeof(int)*len));
    HIP_CALL(hipMalloc(&dev_out, sizeof(int)*len));

    HIP_CALL(hipModuleLoad(&module, HSACO));
    HIP_CALL(hipModuleGetFunction(&kernel_func, module, HSA_KERNEL));

    int max_divider = 1000000;
    int divider;
    for(divider=1;divider<=max_divider;divider++){
        printf("-- start of divider:%d\n",divider);
        int total_loop=200;

        while(total_loop--){
            rand_vec(host_in, len);
            HIP_CALL(hipMemcpy(dev_in, host_in, sizeof(int)*len, hipMemcpyHostToDevice));
            struct {
                int * in;
                int * out;
                int divider;
            } args;
            args.in = dev_in;
            args.out = dev_out;
            args.divider = divider;
            size_t arg_size = sizeof(args);
            void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                            &arg_size, HIP_LAUNCH_PARAM_END};

            HIP_CALL(hipModuleLaunchKernel(kernel_func, len,1,1, 64,1,1,  0, 0, NULL, (void**)&config ));
            HIP_CALL(hipMemcpy(host_out, dev_out, sizeof(int)*len, hipMemcpyDeviceToHost));
            {
                int * host_out_2 = new int[len];
        #if 0
                dump_vector(host_in, len);
                {
                    for(int k=0;k<len;k++){
                        printf("0x%08x, ", ((unsigned int*)host_out)[k]);
                    }
                    printf("\n");
                }
        #endif

                host_func(host_in, host_out_2, len, divider);
                int err_cnt = valid_vector_integer(host_out, host_out_2, len);
                delete [] host_out_2;
                if(err_cnt!=0) goto out;
            }
        } // while()
    } // for divider++

out:
    delete [] host_in;
    delete [] host_out;
}
