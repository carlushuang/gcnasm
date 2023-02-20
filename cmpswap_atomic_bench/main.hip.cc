#include <hip/hip_runtime.h>
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include "common.h"
#include "half.hpp"

// block size is 256
// reduce size 128, so will reduce within & accross workgroup
// ... in unit of bf16x2

#define BLOCK_SIZE 256

#define DEFAULT_OUT_DWORDS (64 * 1024 * 1024)
#define DEFAULT_INP_MULTIP 4

#define WARMUP 2
#define LOOP 10

#define CALL(cmd) \
do {\
    hipError_t cuda_error  = cmd;\
    if (cuda_error != hipSuccess) { \
        std::cout<<"'"<<hipGetErrorString(cuda_error)<<"'("<<cuda_error<<")"<<" at "<<__FILE__<<":"<<__LINE__<<std::endl;\
        exit(EXIT_FAILURE);\
    }\
} while(0)

static inline int get_int(const char* env_name, int def_value)
{
    char * v = getenv(env_name);
    if(v)
        return atoi(v);
    return def_value;
}


__device__ void _atomicAddNoRet(half2_t * addr, half2_t value)
{
    uint32_t * dword_addr = reinterpret_cast<uint32_t*>(addr);
    uint32_t cur_v = *dword_addr;
    uint32_t old_v, new_v;

    do {
        old_v = cur_v;
        //half2_t new_ = add_bf16x2_t(*reinterpret_cast<half2_t*>(&cur_v), value);
        half2_t new_ = value + *reinterpret_cast<half2_t*>(&cur_v);
        new_v = *reinterpret_cast<uint32_t*>(&new_);
        cur_v = atomicCAS(dword_addr, old_v, new_v);
    }while(cur_v != old_v);
}

template<int BLOCK_MAPPING_MODE>
__global__
void block_reduce_kernel_fp16x2_cmpswap(void * __restrict__ output, void* __restrict__ input, int32_t inp_mul, int32_t output_dwords){
    half2_t * p_in = reinterpret_cast<half2_t*>(input);
    half2_t * p_out = reinterpret_cast<half2_t*>(output);

    int32_t i_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int32_t o_idx;
    if constexpr(BLOCK_MAPPING_MODE == 0){
        o_idx = (blockIdx.x / inp_mul) * BLOCK_SIZE + threadIdx.x;
    }
    else if constexpr(BLOCK_MAPPING_MODE == 1){
        o_idx = (blockIdx.x * BLOCK_SIZE + threadIdx.x) % output_dwords;
    }
    if(o_idx < output_dwords)
        _atomicAddNoRet(p_out + o_idx, p_in[i_idx]);   // atomic swap hack
}

template<int BLOCK_MAPPING_MODE>
__global__
void block_reduce_kernel_fp16x2_atomic(void * __restrict__ output, void* __restrict__ input, int32_t inp_mul, int32_t output_dwords){
    half2_t * p_in = reinterpret_cast<half2_t*>(input);
    half2_t * p_out = reinterpret_cast<half2_t*>(output);

    int32_t i_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int32_t o_idx;
    if constexpr(BLOCK_MAPPING_MODE == 0){
        o_idx = (blockIdx.x / inp_mul) * BLOCK_SIZE + threadIdx.x;
    }
    else if constexpr(BLOCK_MAPPING_MODE == 1){
        o_idx = (blockIdx.x * BLOCK_SIZE + threadIdx.x) % output_dwords;
    }

    if(o_idx < output_dwords){
        int32x4_t res = make_wave_buffer_resource(p_out + o_idx);
        llvm_amdgcn_raw_buffer_atomic_add_fp16x2(p_in[i_idx], res, 0, 0, 0);
    }

    // atomicAdd(reinterpret_cast<float*>(p_out + o_idx), float(p_in[i_idx].x));
}

struct fp16x2_cpu_t {
    half_float::half x;
    half_float::half y;

    fp16x2_cpu_t() : x(0), y(0) {}
};

template<int BLOCK_MAPPING_MODE>
void simple_block_reduce(void * __restrict__ output, void* __restrict__ input, int32_t inp_mul, int32_t output_dwords)
{
    fp16x2_cpu_t * p_in = reinterpret_cast<fp16x2_cpu_t*>(input);
    fp16x2_cpu_t * p_out = reinterpret_cast<fp16x2_cpu_t*>(output);
    
    auto get_input_idx = [&] (int32_t i, int32_t j){
        if constexpr(BLOCK_MAPPING_MODE == 0){
            return i / BLOCK_SIZE * inp_mul * BLOCK_SIZE + j * BLOCK_SIZE + (i % BLOCK_SIZE);
        }
        else if constexpr(BLOCK_MAPPING_MODE == 1){
            return j * output_dwords + i;
        }
        return 0;
    };
    for(int32_t i = 0; i < output_dwords; i++){
        fp16x2_cpu_t acc;
        for(int32_t j = 0; j < inp_mul; j++){
            int32_t input_idx = get_input_idx(i, j);
            fp16x2_cpu_t in_data = p_in[input_idx];
            acc.x += in_data.x;
            acc.y += in_data.y;
        }
        p_out[i] = acc;
    }
}

#define EPISILON 1e-3

bool validate(void * ref, void * pred, int num)
{
    fp16x2_cpu_t * p_r = reinterpret_cast<fp16x2_cpu_t*>(ref);
    fp16x2_cpu_t * p_d = reinterpret_cast<fp16x2_cpu_t*>(pred);
    int err = 0;

    for(int i = 0; i < num; i++){
        float r_x = float(p_r[i].x);
        float r_y = float(p_r[i].y);
        float d_x = float(p_d[i].x);
        float d_y = float(p_d[i].y);
        double e_x = std::abs(d_x - r_x) / r_x;
        double e_y = std::abs(d_y - r_y) / r_y;
        if(e_x > EPISILON || e_y > EPISILON){
            printf("mismatch at %d, r:%f,%f, p:%f,%f\n", i, r_x, r_y, d_x, d_y);
            err++;
        }
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

static inline void b2s(size_t bytes, char * str){
	if(bytes<1024){
		sprintf(str, "%luB", bytes);
	}else if(bytes<(1024*1024)){
		double b= (double)bytes/1024.0;
		sprintf(str, "%.2fKB", b);
	}else if(bytes<(1024*1024*1024)){
		double b= (double)bytes/(1024.0*1024);
		sprintf(str, "%.2fMB", b);
	}else{
		double b= (double)bytes/(1024.0*1024*1024);
		sprintf(str, "%.2fGB", b);
	}
}

/* BLOCK_MAPPING_MODE
* 0 : every contiguous "input_multiply" blocks will reduce into single output block.
* 1 : strided reduction, blocks of 0 * tltal_out_blocks, 0 * tltal_out_blocks... will
*     reduce into single output block.
*
*
*/
template<int BLOCK_MAPPING_MODE>
struct _bench{
    static void run(half2_t * h_A, half2_t * h_B, half2_t * h_B_2, void * A, void * B, int32_t input_multiply, int32_t output_dwords){
        int num_a = output_dwords * input_multiply;
        int num_b = output_dwords;
        int need_valid = get_int("VALIDATE", 1);

        {
            // validation
            if(need_valid)
                simple_block_reduce<BLOCK_MAPPING_MODE>(h_B_2, h_A, input_multiply, output_dwords);
        }

        // benchmark kernel
        int bx = BLOCK_SIZE;
        int gx = (num_a + BLOCK_SIZE - 1) / BLOCK_SIZE;
        float ms = 0;
        bool valid = false;
        double total_bytes = (num_a + num_b) * sizeof(half2_t);


        {
            ms = 0;
            for(int i = 0; i < WARMUP; i++){
                block_reduce_kernel_fp16x2_atomic<BLOCK_MAPPING_MODE><<<gx, bx>>>(B, A, input_multiply, output_dwords);
            }
            hipEvent_t start_ev, stop_ev;
            CALL(hipEventCreate(&start_ev));
            CALL(hipEventCreate(&stop_ev));

            CALL(hipEventRecord(start_ev, 0));
            for(int i = 0; i < LOOP; i++){
                block_reduce_kernel_fp16x2_atomic<BLOCK_MAPPING_MODE><<<gx, bx>>>(B, A, input_multiply, output_dwords);
            }
            CALL(hipEventRecord( stop_ev, 0 ));
            CALL(hipEventSynchronize(stop_ev));
            
            CALL(hipEventElapsedTime(&ms,start_ev, stop_ev));
            ms/=LOOP;

            if(need_valid){
                CALL(hipMemset(B, 0, num_b * sizeof(half2_t)));
                block_reduce_kernel_fp16x2_atomic<BLOCK_MAPPING_MODE><<<gx, bx>>>(B, A, input_multiply, output_dwords);
                CALL(hipMemcpy(h_B, B, num_b * sizeof(half2_t), hipMemcpyDeviceToHost));
                valid = validate(h_B_2, h_B, output_dwords);
            }
        }
        printf("atomic<%d> %.3fms %.2fGB/s(%s)", BLOCK_MAPPING_MODE, ms, (total_bytes / (ms * 1e6)), need_valid?(valid?"y":"n"):"");fflush(stdout);


        printf(", ");fflush(stdout);


        {
            ms = 0;
            for(int i = 0; i < WARMUP; i++){
                block_reduce_kernel_fp16x2_cmpswap<BLOCK_MAPPING_MODE><<<gx, bx>>>(B, A, input_multiply, output_dwords);
            }
            hipEvent_t start_ev, stop_ev;
            CALL(hipEventCreate(&start_ev));
            CALL(hipEventCreate(&stop_ev));

            CALL(hipEventRecord(start_ev, 0));
            for(int i = 0; i < LOOP; i++){
                block_reduce_kernel_fp16x2_cmpswap<BLOCK_MAPPING_MODE><<<gx, bx>>>(B, A, input_multiply, output_dwords);
            }
            CALL(hipEventRecord( stop_ev, 0 ));
            CALL(hipEventSynchronize(stop_ev));
            
            CALL(hipEventElapsedTime(&ms,start_ev, stop_ev));
            ms/=LOOP;

            if(need_valid){
                CALL(hipMemset(B, 0, num_b * sizeof(half2_t)));
                block_reduce_kernel_fp16x2_cmpswap<BLOCK_MAPPING_MODE><<<gx, bx>>>(B, A, input_multiply, output_dwords);
                CALL(hipMemcpy(h_B, B, num_b * sizeof(half2_t), hipMemcpyDeviceToHost));
                valid = validate(h_B_2, h_B, output_dwords);
            }
        }
        printf("cmpswap<%d> %.3fms %.2fGB/s(%s)", BLOCK_MAPPING_MODE, ms, (total_bytes / (ms * 1e6)), need_valid?(valid?"y":"n"):"");fflush(stdout);


        printf("\n");fflush(stdout);
    }
};


int main(int argc, char ** argv) {
	hipSetDevice(0);
    int32_t input_multiply = DEFAULT_INP_MULTIP;
    int32_t output_dwords = DEFAULT_OUT_DWORDS;

    if(argc >= 2){
        input_multiply  = atoi(argv[1]);
    }
    if(argc >= 3){
        output_dwords = atoi(argv[2]);
    }

    printf("%d, mul:%d", output_dwords, input_multiply);fflush(stdout);
    assert(output_dwords % BLOCK_SIZE == 0);
   

    void *A, *B;
    int num_a = output_dwords * input_multiply;
    int num_b = output_dwords;
    half2_t * h_A = (half2_t*)malloc(num_a * sizeof(half2_t));
    half2_t * h_B = (half2_t*)malloc(num_b * sizeof(half2_t));
    half2_t * h_B_2 = (half2_t*)malloc(num_b * sizeof(half2_t));

    char dstr[128];
    b2s((num_a + num_b) * sizeof(half2_t), dstr);
    printf("(%s)", dstr);
    printf("\n");fflush(stdout);

	for (int i = 0; i < num_a; ++i){
#if 1
        h_A[i].x = half_float::half(get_rand_int());
        h_A[i].y = half_float::half(get_rand_int());
#else
        h_A[i].x = half_float::half(get_rand_float());
        h_A[i].y = half_float::half(get_rand_float());
#endif
    }

    CALL(hipMalloc(&A, num_a * sizeof(half2_t)));
    CALL(hipMalloc(&B, num_b * sizeof(half2_t)));
    CALL(hipMemcpy(A, h_A, num_a * sizeof(half2_t), hipMemcpyHostToDevice));
    CALL(hipMemset(B, 0, num_b * sizeof(half2_t)));


    _bench<0>::run(h_A, h_B, h_B_2, A, B, input_multiply, output_dwords);
    _bench<1>::run(h_A, h_B, h_B_2, A, B, input_multiply, output_dwords);
    //printf("\n");fflush(stdout);


    free(h_A);
    free(h_B);
    free(h_B_2);
    CALL(hipFree(A));
    CALL(hipFree(B));
}