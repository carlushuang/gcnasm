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

#define BLOCK_SIZE 256
#define GRID_SIZE 8

#define PER_PIXEL_CHECK
static inline bool valid_vector( const float* ref, const float * pred, int n, double nrms = 1e-3 )
{    
    double s0=0.0;
    double s1=0.0;
#ifdef PER_PIXEL_CHECK
    int pp_err = 0;
#endif
    int i_start = 0, i_end=n;
    int i_num = i_end - i_start;
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

__device__ float
llvm_amdgcn_raw_buffer_load_fp32(int32x4_t srsrc,
                                 int32_t voffset,
                                 int32_t soffset,
                                 int32_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.f32");

__device__ void
llvm_amdgcn_raw_buffer_store_fp32(float vdata,
                                  int32x4_t rsrc,
                                  int32_t voffset,
                                  int32_t soffset,
                                  int32_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.f32");
/*
* simple example to reduce element between workgroups.
* number of groups equal to GRID_SIZE
* input GRID_SIZE * 256, output 256
* 
*/
__global__ void simple_workgroup_reduce(int * p_cnt, float* p_in, float * p_out)
{
    while(atomicCAS(p_cnt, blockIdx.x, blockIdx.x) != blockIdx.x) ;

    int32x4_t i_res = amdgcn_make_buffer_resource<float>(p_in + blockIdx.x * BLOCK_SIZE);
    int32x4_t o_res = amdgcn_make_buffer_resource<float>(p_out);

    // slc_glc: 0-no, 1-glc, 2-slc, 3-glc+slc
    float o_data = llvm_amdgcn_raw_buffer_load_fp32(o_res, threadIdx.x * sizeof(float), 0, 2);
    float i_data = llvm_amdgcn_raw_buffer_load_fp32(i_res, threadIdx.x * sizeof(float), 0, 0);
    llvm_amdgcn_raw_buffer_store_fp32(i_data + o_data, o_res,  threadIdx.x * sizeof(float), 0, 2);

    // atomicAdd(p_cnt, (int)1); // atomic add seems fail... will stuck forever
    atomicCAS(p_cnt, blockIdx.x, blockIdx.x+1);
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

int main()
{
    int * dev_cnt;
    float * dev_in;
    float * dev_out;
    
    int i_sz = BLOCK_SIZE * GRID_SIZE;
    int o_sz = BLOCK_SIZE;

    float * host_in = new float[i_sz];
    float * host_out = new float[o_sz];
    float * host_out_dev = new float[o_sz];

    HIP_CALL(hipMalloc(&dev_cnt,  1 * sizeof(int)));
    HIP_CALL(hipMalloc(&dev_in,  i_sz * sizeof(float)));
    HIP_CALL(hipMalloc(&dev_out,  o_sz * sizeof(float)));

    hipMemset(dev_cnt, 0, 1 * sizeof(int));
    hipMemset(dev_out, 0, o_sz * sizeof(float));

    rand_vector(host_in, i_sz);
    HIP_CALL(hipMemcpy(dev_in, host_in, i_sz* sizeof(float), hipMemcpyHostToDevice));

    host_workgroup_reduce(host_in, host_out, GRID_SIZE, BLOCK_SIZE);

    simple_workgroup_reduce<<<dim3(GRID_SIZE), dim3(BLOCK_SIZE), 0, 0>>>(dev_cnt, dev_in, dev_out);
    HIP_CALL(hipMemcpy(host_out_dev, dev_out,  o_sz * sizeof(float), hipMemcpyDeviceToHost));

    bool valid = valid_vector(host_out, host_out_dev, o_sz);
    printf("valid:%s\n", valid?"y":"n");
}
