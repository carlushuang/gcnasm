#include <hip/hip_runtime.h>
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>

/*
    for a single wave64 to construct a 8x8 matrix, printout below matrix

    0 1 2 3 4 5 6 7
    1 0 1 2 3 4 5 6
    2 1 0 1 2 3 4 5
    3 2 1 0 1 2 3 4
    4 3 2 1 0 1 2 3
    5 4 3 2 1 0 1 2
    6 5 4 3 2 1 0 1 
    7 6 5 4 3 2 1 0

*/
__device__ uint32_t sad(uint32_t x, uint32_t y, uint32_t acc){
    // return x > y ? (x - y) : (y - x);
    // return __sad(x, y, acc);
    return __builtin_amdgcn_sad_u16(x, y, acc);
}
__global__ void absdiff_8x8(uint32_t * dst){
    // TODO: single WG, 64 local size
    uint32_t row_id = threadIdx.x / 8;
    uint32_t col_id = threadIdx.x % 8;
    uint32_t zero_position = row_id;

    //uint32_t v = __builtin_amdgcn_sad_u16(col_id, zero_position, 0);
    // uint32_t v = col_id > zero_position ? (col_id - zero_position) : (zero_position - col_id);
    uint32_t v = sad(col_id, zero_position, 0);
    dst[row_id * 8 + col_id] = v;
}

#define CALL(cmd) \
do {\
    hipError_t cuda_error  = cmd;\
    if (cuda_error != hipSuccess) { \
        std::cout<<"'"<<hipGetErrorString(cuda_error)<<"'("<<cuda_error<<")"<<" at "<<__FILE__<<":"<<__LINE__<<std::endl;\
        exit(EXIT_FAILURE);\
    }\
} while(0)


int main(int argc, char ** argv) {
    uint32_t * d;
    uint32_t * h = (uint32_t*)malloc(64*sizeof(uint32_t));


    CALL(hipMalloc(&d, 64 * sizeof(uint32_t)));

    absdiff_8x8<<<1, 64>>>(d);
    CALL(hipMemcpy(h, d, 64 * sizeof(uint32_t), hipMemcpyDeviceToHost));

    for (uint32_t r = 0 ; r < 8; r++) {
        for (uint32_t c = 0 ; c < 8; c++) {
            printf("%u ", h[r * 8 + c]);
        }
        printf("\n");
    }


    free(h);
    CALL(hipFree(d));
    return 0;
}
