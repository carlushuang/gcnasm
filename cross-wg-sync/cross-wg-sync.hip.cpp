#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdio.h>

#define HIP_CALL(call) do{              \
    hipError_t err = call;              \
    if(err != hipSuccess){              \
        printf("[hiperror](%d) fail to call %s",(int)err,#call);    \
        exit(0);                        \
    }                                   \
} while(0)

#ifndef ABS
#define ABS(x) ((x)>0?(x):-1*(x))
#endif

#define WARP_SIZE 64  // need runtime detecting for correct value
#define BLOCK_SIZE 256
#define GRID_SIZE 3000

#define PER_PIXEL_CHECK
static inline bool valid_vector( const float* ref, const float * pred, int n, double nrms = 1e-3 )
{    
    double s0=0.0;
    double s1=0.0;
#ifdef PER_PIXEL_CHECK
    int pp_err = 0;
#endif
    int i_start = 0, i_end=n;
    // int i_num = i_end - i_start;
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
//#ifdef ASSERT_ON_FAIL
            if(pp_err<100)
            printf("diff at %4d, ref:%lf, pred:%lf(0x%04x), d:%lf\n",i,ri,pi,((uint16_t*)pred)[i],delta);
//#endif
            pp_err++;
        }
#endif
    }
    // printf("pp_crr:%d, pp_err:%d, crr_ratio:%.3f, nrms:%lf, s0:%lf, s1:%lf\n",i_num-pp_err, pp_err, (float)(i_num-pp_err)/(float)i_num, sqrt(s0/s1),s0,s1);

    return (sqrt(s0/s1)<nrms)
#ifdef PER_PIXEL_CHECK
        && (pp_err==0)
#endif
    ;
}

// offset is per-thread offset(threadIdx.x/y/z dependent), not per-wave offset(threadIdx.x/y/z independent)
__device__ float atomic_load_fp32(float * addr, uint32_t offset = 0)
{
    return __builtin_bit_cast(float, __atomic_load_n(reinterpret_cast<uint32_t*>(addr + offset), __ATOMIC_RELAXED));
}

// offset is per-thread offset(threadIdx.x/y/z dependent), not per-wave offset(threadIdx.x/y/z independent)
__device__ void atomic_store_fp32(float * addr, float value, uint32_t offset = 0)
{
    // __hip_atomic_store() does not work
    // currently intrinsic doesn't produce sc1 bit for gfx94*
#if (defined(__gfx908__) || defined(__gfx90a__))
    asm volatile("global_store_dword %0, %1, %2 glc\n"
                "s_waitcnt vmcnt(0)"
                :
                : "v"(static_cast<uint32_t>(offset * sizeof(uint32_t))), "v"(value), "s"(addr)
                : "memory");
#elif (defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__))
    asm volatile("global_store_dword %0, %1, %2 sc0 sc1\n"
                "s_waitcnt vmcnt(0)"
                :
                : "v"(static_cast<uint32_t>(offset * sizeof(uint32_t))), "v"(value), "s"(addr)
                : "memory");
#endif
}

// always use tid = 0 to wait
// assume initial value is zero
struct workgroup_barrier {
    __device__ workgroup_barrier(uint32_t * ptr) :
        base_ptr(ptr)
    {}

    __device__ uint32_t ld(uint32_t offset  = 0)
    {
        return __atomic_load_n(base_ptr + offset, __ATOMIC_RELAXED);
    }

    __device__ void wait_eq(uint32_t value, uint32_t offset = 0)
    {
        if(threadIdx.x == 0){
            while(ld(offset) != value){}
        }
        __builtin_amdgcn_s_barrier();
    }

    __device__ void wait_lt(uint32_t value, uint32_t offset = 0)
    {
        if(threadIdx.x == 0){
            while(ld(offset) < value){}
        }
        __builtin_amdgcn_s_barrier();
    }

    __device__ void wait_set(uint32_t compare, uint32_t value, uint32_t offset = 0)
    {
        if(threadIdx.x == 0){
            while(atomicCAS_system(base_ptr + offset, compare, value) != compare){}
        }
        __builtin_amdgcn_s_barrier();
    }

    // enter critical zoon, assume buffer is zero when launch kernel
    __device__ void aquire(uint32_t offset = 0)
    {
        wait_set(0, 1, offset);
    }

    // exit critical zoon, assume buffer is zero when launch kernel
    __device__ void release(uint32_t offset = 0)
    {
        wait_set(1, 0, offset);
    }

    __device__ void inc(uint32_t offset = 0)
    {
        __builtin_amdgcn_s_barrier();
        if(threadIdx.x == 0){
            atomicAdd(base_ptr + offset, 1);
        }
    }

    uint32_t * base_ptr;
};

/*
* simple example to reduce element between workgroups.
* input(2d) GRID_SIZE * BLOCK_SIZE reduce to output(1d) BLOCK_SIZE
* 
* each thread within a WG load a pixel from input row(where the row-id = blockIdx.x)
* , and atomically load a pixel from output.
* then reduce(add) the result, atomically store the result to output row.
*
* we can just use atomicAdd() to achieve this reduction, but here we use this
* load->modify->store flow deliberately, which introduce the necessity of workgroup_barrier
* data structure implemented for the cross-wg sync
* 
* this reduction stage can be serialized between WGs, or out-of-order. We test both.
* (they should be the same except some numeric difference)
*
* besides, we use one extra wg to copy the final reduced value from p_out to p_out_aux
* if everything is OK, p_out and p_out_aux should be identical
* for simplicity, only test it if serialized_reduce==true, since p_cnt now have enough counter
* to indicate the finish of first period (if ooo reduce, need one extra buffer to serve as counter)
* 
*/
template<bool serialized_reduce = true>
__global__ void simple_workgroup_reduce(uint32_t * p_cnt, float* p_in, float * p_out, float * p_out_aux)
{
    workgroup_barrier barrier(p_cnt);
    if constexpr (serialized_reduce) {
        if (blockIdx.x == (gridDim.x - 1)) {
            barrier.wait_eq(gridDim.x - 1);
            float o_data = atomic_load_fp32(p_out, threadIdx.x);  // atomic load
            *(p_out_aux + threadIdx.x) = o_data;  // not need to use atomic store
            return;
        }
    }
    if constexpr (serialized_reduce)
        barrier.wait_eq(blockIdx.x);
    else
        barrier.aquire();

    // critical area start
    float o_data = atomic_load_fp32(p_out, threadIdx.x);  // atomic load
    float i_data = *(p_in + blockIdx.x * blockDim.x + threadIdx.x); // no need to use atomic
    float result = i_data + o_data;
    atomic_store_fp32(p_out, result, threadIdx.x);
    // critical area end

    if constexpr (serialized_reduce)
        barrier.inc();
    else
        barrier.release();
}


void host_workgroup_reduce(float* p_in, float * p_out, int groups, int length)
{
    for(int l = 0; l < length; l++){
        float sum = .0f;
        for(int g = 0; g < groups; g++){
            sum += p_in[g * length + l];
        }
        p_out[l] = sum;
    }
}

void rand_vector(float* v, int num){
    static int flag = 0;
    if(!flag){ srand(time(NULL)); flag = 1; }

    for(int i = 0; i < num; i++){
        v[i] = ((float)(rand() % 100)) / 100.0f;
    }
}

template<bool serialized_reduce = true>
void invoke(int argc, char ** argv)
{
    // use 1 can also be correct, here we just hope this buffer to be cacheline aligned
    constexpr int cacheline_size_in_dword = 32;
    int reduce_groups = GRID_SIZE;
    if(argc >= 2) {
        reduce_groups = std::atoi(argv[1]);
    }
    uint32_t * dev_cnt;
    float * dev_in;
    float * dev_out;
    float * dev_out_aux;

    int i_sz = BLOCK_SIZE * reduce_groups;
    int o_sz = BLOCK_SIZE;

    float * host_in = new float[i_sz];
    float * host_out = new float[o_sz];
    float * host_out_dev = new float[o_sz];
    float * host_out_aux_dev = new float[o_sz];

    HIP_CALL(hipMalloc(&dev_cnt,  cacheline_size_in_dword * sizeof(uint32_t)));
    HIP_CALL(hipMalloc(&dev_in,  i_sz * sizeof(float)));
    HIP_CALL(hipMalloc(&dev_out,  o_sz * sizeof(float)));
    HIP_CALL(hipMalloc(&dev_out_aux,  o_sz * sizeof(float)));

    hipMemset(dev_cnt, 0, cacheline_size_in_dword * sizeof(uint32_t));
    hipMemset(dev_out, 0, o_sz * sizeof(float));

    rand_vector(host_in, i_sz);
    HIP_CALL(hipMemcpy(dev_in, host_in, i_sz* sizeof(float), hipMemcpyHostToDevice));

    host_workgroup_reduce(host_in, host_out, reduce_groups, BLOCK_SIZE);

    constexpr int aux_wg = serialized_reduce ? 1 : 0;
    simple_workgroup_reduce<serialized_reduce><<<dim3(reduce_groups + aux_wg), dim3(BLOCK_SIZE), 0, 0>>>
        (dev_cnt, dev_in, dev_out, dev_out_aux);
    HIP_CALL(hipMemcpy(host_out_dev, dev_out,  o_sz * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(host_out_aux_dev, dev_out_aux,  o_sz * sizeof(float), hipMemcpyDeviceToHost));

    bool valid = valid_vector(host_out, host_out_dev, o_sz);
    printf("[%s] %d groups, valid:%s", serialized_reduce? "serialized_reduce" : "outoforder_reduce",
                              reduce_groups,  valid?"y":"n");
    if constexpr(serialized_reduce) {
        bool valid_aux = true;
        for(auto i =0; i < o_sz; i++) {
            uint32_t oo = __builtin_bit_cast(uint32_t, host_out_dev[i]);
            uint32_t oa = __builtin_bit_cast(uint32_t, host_out_aux_dev[i]);
            if(oo != oa) {
                valid_aux = false;
            }
        }
        printf(", aux:%s", valid_aux?"y":"n");
    }
    printf("\n");
    free(host_in);
    free(host_out);
    free(host_out_dev);
    free(host_out_aux_dev);
    hipFree(dev_cnt);
    hipFree(dev_in);
    hipFree(dev_out);
    hipFree(dev_out_aux);
}

int main(int argc, char ** argv)
{
    invoke<true>(argc, argv);
    invoke<false>(argc, argv);
}
