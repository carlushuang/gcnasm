#include <stdio.h>
#include <hip/hip_runtime.h>
#include <random>
#include <iostream>
#include <assert.h>
#include <thread>

#define HIP_CALL(call) do{  \
    hipError_t err = call;  \
    if(err != hipSuccess){  \
        printf("[hiperror](%d) fail to call %s",(int)err,#call);    \
        exit(0);            \
    }                      \
} while(0)

#define HSACO "magic_div.hsaco"
#define HSA_KERNEL "kernel_func"


struct magic_div_u32_t {
    uint32_t magic;
    uint32_t shift;
};

static inline struct magic_div_u32_t magic_div_u32_gen(uint32_t d) {
    assert(d >= 1 && d <= INT32_MAX);
    uint32_t shift;
    for (shift = 0; shift < 32; shift++)
        if ((1U << shift) >= d)
            break;

    uint64_t one = 1;
    uint64_t magic = ((one << 32) * ((one << shift) - d)) / d + 1;
    assert(magic <= 0xffffffffUL);

    magic_div_u32_t result;
    result.magic = magic;
    result.shift = shift;
    return result;
}

// host side version
static inline uint32_t magic_div_mulhi_u32(uint32_t x, uint32_t y) {
    uint64_t xl = x, yl = y;
    uint64_t rl = xl * yl;
    return (uint32_t)(rl >> 32);
}
uint32_t magic_div_u32_do(uint32_t numer, const struct magic_div_u32_t *denom) {
    uint32_t tmp = magic_div_mulhi_u32(denom->magic, numer);
    return (tmp + numer) >> denom->shift;
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

    int inst_loop = 512;
    int bdx = 256;
    int gdx = 256;

    struct {
        uint32_t * numerater_ptr;
        uint32_t * quot_ptr;   
        uint32_t * rem_ptr;        
        uint32_t denom;
        uint32_t magic;
        uint32_t shift;
        uint32_t total_size;
    } args;

    uint32_t * numerater_ptr    = new uint32_t[inst_loop * bdx * gdx];
    uint32_t * quot_ptr         = new uint32_t[inst_loop * bdx * gdx];
    uint32_t * rem_ptr          = new uint32_t[inst_loop * bdx * gdx];

    uint32_t * dev_numerater_ptr;
    uint32_t * dev_quot_ptr;
    uint32_t * dev_rem_ptr;

    HIP_CALL(hipMalloc(&dev_numerater_ptr,  inst_loop * bdx * gdx * sizeof(uint32_t)));
    HIP_CALL(hipMalloc(&dev_quot_ptr,       inst_loop * bdx * gdx * sizeof(uint32_t)));
    HIP_CALL(hipMalloc(&dev_rem_ptr,        inst_loop * bdx * gdx * sizeof(uint32_t)));

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads < 4)
        num_threads = 4;

    for(uint32_t denom = 1; denom <= INT32_MAX; denom ++){
        for(uint32_t numer = 0; numer <= INT32_MAX; numer += inst_loop * bdx * gdx){
            for(int i=0; i< inst_loop * bdx * gdx; i++){
                numerater_ptr[i] = numer + i;       // ignore outside check
            }
            HIP_CALL(hipMemcpy(dev_numerater_ptr, numerater_ptr, inst_loop * bdx * gdx * sizeof(uint32_t), hipMemcpyHostToDevice));

            magic_div_u32_t mdiv = magic_div_u32_gen(denom);

            args.numerater_ptr  = dev_numerater_ptr;
            args.quot_ptr       = dev_quot_ptr;
            args.rem_ptr        = dev_rem_ptr;
            args.denom          = denom;
            args.magic          = mdiv.magic;
            args.shift          = mdiv.shift;
            args.total_size     = inst_loop * bdx * gdx;

            size_t arg_size = sizeof(args);
            void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                        &arg_size, HIP_LAUNCH_PARAM_END};

            HIP_CALL(hipModuleLaunchKernel(kernel_func, gdx,1,1, bdx,1,1,  0, 0, NULL, (void**)&config ));

            HIP_CALL(hipMemcpy(quot_ptr, dev_quot_ptr, inst_loop * bdx * gdx * sizeof(uint32_t), hipMemcpyDeviceToHost));
            HIP_CALL(hipMemcpy(rem_ptr,  dev_rem_ptr,  inst_loop * bdx * gdx * sizeof(uint32_t), hipMemcpyDeviceToHost));

#if 0
            for(int i=0; i< inst_loop * bdx * gdx; i++){
                if((numer + i) > INT32_MAX)
                    break;
                uint32_t n = numer + i;
                uint32_t q = n / denom;
                uint32_t r = n % denom;
                if((q != quot_ptr[i]) || (r != rem_ptr[i])){
                    printf("WRONG! %u/%u, q:%u, r:%u, but from gpu  q:%u, r:%u\n", n, denom, q, r, quot_ptr[i], rem_ptr[i]);
                    assert(0);
                }
            }
#else
            std::vector<std::thread> threads;
            for (uint32_t t = 0; t < num_threads; t++){
                threads.push_back( std::thread(
                    [numer, denom, quot_ptr, rem_ptr](uint32_t thread_id, uint32_t block_size, uint32_t total_len){
                        for(uint32_t idx = thread_id; idx < total_len; idx += block_size){
                            if((numer + idx) > INT32_MAX)
                                break;
                            uint32_t n = numer + idx;
                            uint32_t q = n / denom;
                            uint32_t r = n % denom;
                            if((q != quot_ptr[idx]) || (r != rem_ptr[idx])){
                                printf("WRONG! %u/%u, q:%u, r:%u, but from gpu  q:%u, r:%u\n", n, denom, q, r, quot_ptr[idx], rem_ptr[idx]);
                                assert(0);
                            }
                        }
                    }, t, num_threads, inst_loop * bdx * gdx)
                );
            }

            for (auto &th : threads)
                th.join();
#endif

            uint32_t numer_end =  (numer+inst_loop * bdx * gdx) > INT32_MAX ? INT32_MAX : (numer+inst_loop * bdx * gdx);
            printf("ok %u ... %u / %u\n", numer, numer_end, denom);
        }
    }

    free(numerater_ptr   );
    free(quot_ptr        );
    free(rem_ptr         );
}
