#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include "common.h"

// block size is 256
// reduce size 128, so will reduce within & accross workgroup
// ... in unit of bf16x2

#define NUM_BLOCKS 1000
#define BLOCK_SIZE 256
#define REDUCE_SIZE 128

#define CALL(cmd) \
do {\
    hipError_t cuda_error  = cmd;\
    if (cuda_error != hipSuccess) { \
        std::cout<<"'"<<hipGetErrorString(cuda_error)<<"'("<<cuda_error<<")"<<" at "<<__FILE__<<":"<<__LINE__<<std::endl;\
        exit(EXIT_FAILURE);\
    }\
} while(0)

__device__ void atomicAddNoRet(bf16x2_t * addr, bf16x2_t value)
{
    uint32_t * dword_addr = reinterpret_cast<uint32_t*>(addr);
    uint32_t cur_v = *dword_addr;
    uint32_t old_v, new_v;

    do {
        old_v = cur_v;
        bf16x2_t new_ = add_bf16x2_t(*reinterpret_cast<bf16x2_t*>(&cur_v), value);
        new_v = *reinterpret_cast<uint32_t*>(&new_);
        cur_v = atomicCAS(dword_addr, old_v, new_v);
    }while(cur_v != old_v);
}

extern "C" __global__
void reduction_kernel(void * __restrict__ output, void* __restrict__ input){
    bf16x2_t * p_in = reinterpret_cast<bf16x2_t*>(input);
    bf16x2_t * p_out = reinterpret_cast<bf16x2_t*>(output);

    p_in += blockIdx.x * BLOCK_SIZE + threadIdx.x;
    p_out += threadIdx.x % REDUCE_SIZE;

    atomicAddNoRet(p_out, *p_in);
}


void simple_reduction(void * __restrict__ output, void* __restrict__ input)
{
    int reduce_size = BLOCK_SIZE / REDUCE_SIZE * NUM_BLOCKS;
    bf16x2_t * p_in = reinterpret_cast<bf16x2_t*>(input);
    bf16x2_t * p_out = reinterpret_cast<bf16x2_t*>(output);

    for(int i = 0 ; i < REDUCE_SIZE; i++){
        bf16x2_t acc = {0, 0};
        for(int j = 0; j < reduce_size; j++){
            acc = add_bf16x2_t(acc, p_in[j * REDUCE_SIZE + i]);
        }
        p_out[i] = acc;
    }
}

#define EPISILON 1e-3

bool validate(void * ref, void * pred, int num)
{
    bf16x2_t * p_r = reinterpret_cast<bf16x2_t*>(ref);
    bf16x2_t * p_d = reinterpret_cast<bf16x2_t*>(pred);
    int err = 0;

    for(int i = 0; i < num; i++){
#if BF16ASU16
        if(p_r[i].x != p_d[i].x || p_r[i].y != p_d[i].y)
            err++;
#else
        float r_x = bf16_2_float(p_r[i].x);
        float r_y = bf16_2_float(p_r[i].y);
        float d_x = bf16_2_float(p_d[i].x);
        float d_y = bf16_2_float(p_d[i].y);
        double e_x = std::abs(d_x - r_x) / r_x;
        double e_y = std::abs(d_y - r_y) / r_y;
        if(e_x > EPISILON || e_y > EPISILON){
            printf("mismatch at %d, r:%f,%f, p:%f,%f\n", i, r_x, r_y, d_x, d_y);
            err++;
        }
#endif
    }
    return err == 0;
}

static inline float get_rand_float(){
    static int inited = 0;
    float v;
    if(!inited){ srand(time(NULL)); inited = 1; }
    v = rand() % 2000 + 1;
    return v / 1000.0f - 1.0; // -1.0 ~ 1.0
}

static inline int get_rand_int(){
    static int inited = 0;
    if(!inited){ srand(time(NULL)); inited = 1; }
    return (rand() % 20) - 10;  // -10 ~ 10
}

int main() {
	hipSetDevice(0);
    void *A, *B;
    int num_a = BLOCK_SIZE * NUM_BLOCKS;
    int num_b = REDUCE_SIZE;
    bf16x2_t * h_A = (bf16x2_t*)malloc(num_a * sizeof(bf16x2_t));
    bf16x2_t * h_B = (bf16x2_t*)malloc(num_b * sizeof(bf16x2_t));
	for (int i = 0; i < num_a; ++i){
#if BF16ASU16
        h_A[i].x = get_rand_int();
        h_A[i].y = get_rand_int();
#else
        // h_A[i].x = float_2_bf16(get_rand_float());
        // h_A[i].y = float_2_bf16(get_rand_float());
        h_A[i].x = float_2_bf16(get_rand_int());
        h_A[i].y = float_2_bf16(get_rand_int());
#endif
    }

    CALL(hipMalloc(&A, num_a * sizeof(bf16x2_t)));
    CALL(hipMalloc(&B, num_b * sizeof(bf16x2_t)));
    CALL(hipMemcpy(A, h_A, num_a * sizeof(bf16x2_t), hipMemcpyHostToDevice));
    CALL(hipMemset(B, 0, num_b * sizeof(bf16x2_t)));

    // benchmark kernel
    int bx = BLOCK_SIZE;
    int gx = NUM_BLOCKS;

    reduction_kernel<<<gx, bx>>>(B, A);

    CALL(hipMemcpy(h_B, B, num_b * sizeof(bf16x2_t), hipMemcpyDeviceToHost));

    for(int i = 0; i < num_b; i++){
#if BF16ASU16
        printf("{%u, %u} ", h_B[i].x, h_B[i].y );
#else
        printf("{%.2f, %.2f} ", bf16_2_float(h_B[i].x), bf16_2_float(h_B[i].y));
#endif
        if((i + 1) % 8 == 0) printf("\n");
    }

    {
        // validation
        bf16x2_t * h_B_2 = (bf16x2_t*)malloc(num_b * sizeof(bf16x2_t));
        simple_reduction(h_B_2, h_A);
        bool valid = validate(h_B_2, h_B, REDUCE_SIZE);
        free(h_B_2);
        printf("valid:%s\n", valid ? "y" : "n");
    }

    free(h_A);
    free(h_B);
    CALL(hipFree(A));
    CALL(hipFree(B));
}