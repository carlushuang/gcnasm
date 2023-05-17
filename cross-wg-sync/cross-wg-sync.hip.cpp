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
#define BLOCK_SIZE 64 
#define GRID_SIZE (104 * 2)

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



typedef int32_t int32x4_t __attribute__((ext_vector_type(4)));
#define AMDGCN_BUFFER_RES_3 0x00027000 // for gfx90a

template <typename T>
union amdgcn_buffer_resource
{
    // https://rocm-documentation.readthedocs.io/en/latest/GCN_ISA_Manuals/testdocbook.html#vector-memory-buffer-instructions
    int32x4_t content;
    struct
    {
        T* address;
        int32_t range;
        int32_t config;
    };
};

template <typename T>
__device__ int32x4_t amdgcn_make_buffer_resource(const T* addr)
{
    amdgcn_buffer_resource<T> buffer_resource;
    buffer_resource.address = const_cast<T*>(addr);
    buffer_resource.range   = 0xffffffff;
    buffer_resource.config  = AMDGCN_BUFFER_RES_3; // for gfx9

    return buffer_resource.content;
}

#define AMDGCN_BUFFER_DEFAULT   0
#define AMDGCN_BUFFER_GLC       1
#define AMDGCN_BUFFER_SLC       2
#define AMDGCN_BUFFER_GLC_SLC   3

__device__ float
llvm_amdgcn_raw_buffer_load_fp32(int32x4_t srsrc,
                                 int32_t voffset,
                                 int32_t soffset,
                                 int32_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.f32");

__device__ uint32_t
llvm_amdgcn_raw_buffer_load_u32(int32x4_t srsrc,
                                 int32_t voffset,
                                 int32_t soffset,
                                 int32_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.u32");

__device__ void
llvm_amdgcn_raw_buffer_store_fp32(float vdata,
                                  int32x4_t rsrc,
                                  int32_t voffset,
                                  int32_t soffset,
                                  int32_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.f32");

// always use tid = 0 to wait
// assume initial value is zero
struct workgroup_barrier {
    __device__ workgroup_barrier(uint32_t * ptr) :
        base_ptr(ptr)
    {}

    __device__ uint32_t aquire(int32_t offset)
    {
#if 0
        float d = llvm_amdgcn_raw_buffer_load_fp32(
                        get_res(),
                        0,
                        offset,
                        AMDGCN_BUFFER_GLC);
        union cvt {
            float f32;
            uint32_t u32;
        };
        cvt x;
        x.f32 = d;
        return x.u32;
#endif

        return __atomic_load_n(base_ptr + offset, __ATOMIC_RELAXED);
    }

    __device__ void wait_eq(int32_t offset, uint32_t value)
    {
        if(threadIdx.x == 0){
            //#pragma unroll 1
            while(aquire(offset) != value){}
        }
        __syncthreads();
    }

    __device__ void wait_lt(int32_t offset, uint32_t value)
    {
        if(threadIdx.x == 0){
            //#pragma unroll 1
            while(aquire(offset) < value){}
        }
        __syncthreads();
    }

    __device__ void inc(int32_t offset)
    {
        __syncthreads();
        if(threadIdx.x == 0){
            atomicAdd(base_ptr + offset, 1);
        }
    }

    __device__ int32x4_t get_res() const
    {
        return amdgcn_make_buffer_resource(base_ptr);
    }

    uint32_t * base_ptr;
};

/*
* simple example to reduce element between workgroups.
* number of groups equal to GRID_SIZE
* input GRID_SIZE * 256, output 256
* 
*/
__global__ void simple_workgroup_reduce(uint32_t * p_cnt, float* p_in, float * p_out)
{
    workgroup_barrier barrier(p_cnt);
    barrier.wait_eq(0, blockIdx.x);

    int32x4_t i_res = amdgcn_make_buffer_resource<float>(p_in + blockIdx.x * BLOCK_SIZE);
    int32x4_t o_res = amdgcn_make_buffer_resource<float>(p_out);

    // slc_glc: 0-no, 1-glc, 2-slc, 3-glc+slc
    float o_data = llvm_amdgcn_raw_buffer_load_fp32(o_res, threadIdx.x * sizeof(float), 0, AMDGCN_BUFFER_GLC);
    float i_data = llvm_amdgcn_raw_buffer_load_fp32(i_res, threadIdx.x * sizeof(float), 0, AMDGCN_BUFFER_DEFAULT);
    float result = i_data + o_data;
    llvm_amdgcn_raw_buffer_store_fp32(result, o_res,  threadIdx.x * sizeof(float), 0, AMDGCN_BUFFER_GLC);

    barrier.inc(0);
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

int main(int argc, char ** argv)
{
    int reduce_groups = GRID_SIZE;
    if(argc >= 2) {
        reduce_groups = std::atoi(argv[1]);
    }
    uint32_t * dev_cnt;
    float * dev_in;
    float * dev_out;

    int i_sz = BLOCK_SIZE * reduce_groups;
    int o_sz = BLOCK_SIZE;

    float * host_in = new float[i_sz];
    float * host_out = new float[o_sz];
    float * host_out_dev = new float[o_sz];

    HIP_CALL(hipMalloc(&dev_cnt,  1 * sizeof(uint32_t)));
    HIP_CALL(hipMalloc(&dev_in,  i_sz * sizeof(float)));
    HIP_CALL(hipMalloc(&dev_out,  o_sz * sizeof(float)));

    hipMemset(dev_cnt, 0, 1 * sizeof(uint32_t));
    hipMemset(dev_out, 0, o_sz * sizeof(float));

    rand_vector(host_in, i_sz);
    HIP_CALL(hipMemcpy(dev_in, host_in, i_sz* sizeof(float), hipMemcpyHostToDevice));

    host_workgroup_reduce(host_in, host_out, reduce_groups, BLOCK_SIZE);

    simple_workgroup_reduce<<<dim3(reduce_groups), dim3(BLOCK_SIZE), 0, 0>>>(dev_cnt, dev_in, dev_out);
    HIP_CALL(hipMemcpy(host_out_dev, dev_out,  o_sz * sizeof(float), hipMemcpyDeviceToHost));

    bool valid = valid_vector(host_out, host_out_dev, o_sz);
    printf("valid:%s\n", valid?"y":"n");
}
