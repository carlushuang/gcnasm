#include <stdio.h>
#include <hip/hip_runtime.h>
#include <random>
#include <stdint.h>

#define HIP_CALL(call) do{  \
    hipError_t err = call;  \
    if(err != hipSuccess){  \
        printf("[hiperror](%d) fail to call %s",(int)err,#call);    \
        exit(0);            \
    }                       \
} while(0)


int host_func(int * in_vec, int * out_vec, int num, int divider){
    for(int i=0;i<num;i++){
        out_vec[i] = in_vec[i] % divider;
    }
    return 0;
}
template<typename T>
void rand_vec(T * seq, size_t len){
    static std::random_device rd;   // seed
    static std::mt19937 mt(rd());
    static std::uniform_real_distribution<float> dist(0.0, 100.1); // 2**20-1

    for(size_t i=0;i<len;i++) seq[i] =  (T)dist(mt);
}
template<typename T>
void dump_vector(const T*vec, size_t len){
    for(size_t i=0;i<len;i++){
        std::cout<<vec[i]<<", ";
    }
    std::cout<<"\n";
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

#define HSACO "kernel.co"
#define HSA_KERNEL "kernel_func"

#define DWORD_PER_UNIT 1ull
#define BLOCK_DIM_X 256ull
#define GRID_DIM_X 64*2*16*8ull
#define UNIT_PER_THRD 16ull
#define P_LOOP 1

void host_func(float * in, float * out){
    for(size_t i=0;i<DWORD_PER_UNIT*BLOCK_DIM_X*GRID_DIM_X*UNIT_PER_THRD;i++){
        out[i] = in[i];
    }
}

#define WARMUP 2
#define LOOP 5

int main(int argc, char ** argv){
    hipModule_t module;
    hipFunction_t kernel_func;
    float * host_in, * host_out, *dev_in, *dev_out;

    size_t dword_size = DWORD_PER_UNIT*BLOCK_DIM_X*GRID_DIM_X*UNIT_PER_THRD;

    host_in   = new float[dword_size];
    host_out  = new float[dword_size];

    HIP_CALL(hipSetDevice(0));
    HIP_CALL(hipMalloc(&dev_in, sizeof(float)*dword_size ));
    HIP_CALL(hipMalloc(&dev_out, sizeof(float)*dword_size ));

    HIP_CALL(hipModuleLoad(&module, HSACO));
    HIP_CALL(hipModuleGetFunction(&kernel_func, module, HSA_KERNEL));


    rand_vec(host_in, dword_size);
    HIP_CALL(hipMemcpy(dev_in, host_in, sizeof(float)*dword_size, hipMemcpyHostToDevice));
    struct {
        float * in;
        float * out;
    } args;
    args.in = dev_in;
    args.out = dev_out;
    size_t arg_size = sizeof(args);
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &arg_size, HIP_LAUNCH_PARAM_END};
    hipEvent_t evt_0, evt_1;
    hipEventCreate(&evt_0);
    hipEventCreate(&evt_1);

    hipCtxSynchronize();
    for(int i=0;i<WARMUP;i++){
        HIP_CALL(hipModuleLaunchKernel(kernel_func, GRID_DIM_X,1,1, BLOCK_DIM_X,1,1,  0, 0, NULL, (void**)&config ));
    }
    hipCtxSynchronize();
    hipEventRecord(evt_0, NULL);
    for(int i=0;i<LOOP;i++){
        HIP_CALL(hipModuleLaunchKernel(kernel_func, GRID_DIM_X,1,1, BLOCK_DIM_X,1,1,  0, 0, NULL, (void**)&config ));
    }
    hipEventRecord(evt_1, NULL);
    hipEventSynchronize(evt_1);
    hipCtxSynchronize();
    float elapsed_ms;
    hipEventElapsedTime(&elapsed_ms, evt_0, evt_1);
    double t = elapsed_ms/LOOP;
    double tp = (double)dword_size*sizeof(float)*2 / ((double)t/1000) / 1000000000.0 * P_LOOP;
    double per = tp/484.0 * 100;
    printf("cost:%f ms, throughput:%f GB/s(%.4f%%)\n",t,tp,per );

    {
        float * host_out_2 = new float[dword_size];
        host_func(host_in, host_out_2);
        HIP_CALL(hipMemcpy(host_out, dev_out, sizeof(float)*dword_size, hipMemcpyDeviceToHost));
        valid_vector(host_out_2, host_out, dword_size);
        delete [] host_out_2;
    }

out:
    delete [] host_in;
    delete [] host_out;
}
