#include <hip/hip_runtime.h>
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>



#define CALL(cmd) \
do {\
    hipError_t cuda_error  = cmd;\
    if (cuda_error != hipSuccess) { \
        std::cout<<"'"<<hipGetErrorString(cuda_error)<<"'("<<cuda_error<<")"<<" at "<<__FILE__<<":"<<__LINE__<<std::endl;\
        exit(EXIT_FAILURE);\
    }\
} while(0)

static inline float get_rand_float(){
    static int inited = 0;
    float v;
    if(!inited){ srand(time(NULL)); inited = 1; }
    v = rand() % 100 + 1;
    return v / 100.0f;
}

static inline float get_rand_int(int max){
    static int inited = 0;
    float v;
    if(!inited){ srand(time(NULL)); inited = 1; }
    v = rand() % max;
    return v;
}

// always single wave
template<int lanegroup_size>
__global__ void test_ds_permute_kernel(float * p_dst, float* p_src, int * p_idx)
{
    if(threadIdx.x < lanegroup_size) {
        int idx = p_idx[threadIdx.x];
        float src = p_src[threadIdx.x];
        float dst = -0.3;
        dst = __builtin_bit_cast(float,
                            __builtin_amdgcn_ds_permute(idx << 2, __builtin_bit_cast(int, src)));
        dst = dst == 0.f ? 1.f : dst;
        p_dst[threadIdx.x] = dst;
    }
}

template<int lanegroup_size>
__global__ void test_ds_permute_kernel_v2(float * p_dst, float* p_src, int * p_idx)
{
    if(threadIdx.x < lanegroup_size) {
        int idx = p_idx[threadIdx.x];
        float src = p_src[threadIdx.x];
        float src_1 = p_src[threadIdx.x + 256];
        float dst = -0.3;
        dst = __builtin_bit_cast(float,
                            __builtin_amdgcn_ds_permute(idx << 2, __builtin_bit_cast(int, src)));
        if(dst ==  99999999.f) {
            p_dst[blockIdx.x * 256 + threadIdx.x * 2] = dst  + src;
        }
        dst = __builtin_bit_cast(float,
                            __builtin_amdgcn_ds_permute((idx ^ 1) << 2, __builtin_bit_cast(int, src_1)));
        p_dst[threadIdx.x] = dst;
    }
}


int main(int argc, char ** argv) {
    constexpr int wave_size = 64;
    constexpr int lanegroup_size = 8;
    {
        int * h_idx;
        float * h_src, * h_dst;
        int * d_idx;
        float * d_src, * d_dst;

        h_idx = (int*)malloc(lanegroup_size * sizeof(int));
        h_src = (float*)malloc(lanegroup_size * sizeof(float));
        h_dst = (float*)malloc(lanegroup_size * sizeof(float));

        for(int i = 0; i < lanegroup_size ; i++) {
            h_src[i] = get_rand_float();
            h_idx[i] = get_rand_int(lanegroup_size);
        }

        CALL(hipMalloc(&d_idx, lanegroup_size * sizeof(int)));
        CALL(hipMalloc(&d_src, lanegroup_size * sizeof(float)));
        CALL(hipMalloc(&d_dst, lanegroup_size * sizeof(float)));

        CALL(hipMemcpy(d_src, h_src, lanegroup_size * sizeof(float), hipMemcpyHostToDevice));
        CALL(hipMemcpy(d_idx, h_idx, lanegroup_size * sizeof(int), hipMemcpyHostToDevice));

        test_ds_permute_kernel<lanegroup_size><<<1, wave_size>>>(d_dst, d_src, d_idx);

        CALL(hipMemcpy(h_dst, d_dst, lanegroup_size * sizeof(float), hipMemcpyDeviceToHost));

        for(int i = 0; i < lanegroup_size; i++) {
            printf("%.2f ", h_src[i]);
        }
        printf("\n");
        
        for(int i = 0; i < lanegroup_size; i++) {
            printf("%4d ", h_idx[i]);
        }
        printf("\n");

        for(int i = 0; i < lanegroup_size; i++) {
            printf("%.2f ", h_dst[i]);
        }
        printf("\n");

    }
}
