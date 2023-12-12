#include <hip/hip_runtime.h>
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>

using f32 = float;
// using f16 = _Float16;

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;

using index_t = u32;

typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x3 __attribute__((ext_vector_type(2)));
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
typedef uint32_t u32x1 __attribute__((ext_vector_type(1)));

typedef f32 f32x8 __attribute__((ext_vector_type(8)));
typedef f32 f32x4 __attribute__((ext_vector_type(4)));
typedef f32 f32x2 __attribute__((ext_vector_type(2)));
typedef f32 f32x1 __attribute__((ext_vector_type(1)));

typedef u32x4 dwordx4_t;
typedef u32x3 dwordx3_t;
typedef u32x2 dwordx2_t;
typedef u32   dword_t;


#define C_XCC_ID                   20
#define C_XCC_ID_XCC_ID_SIZE       4
#define C_XCC_ID_XCC_ID_OFFSET     0

#define C_HW_ID                    4
#define C_HW_ID_CU_ID_SIZE         4
#define C_HW_ID_CU_ID_OFFSET       8
#define C_HW_ID_SE_ID_SIZE         2
#define C_HW_ID_SE_ID_OFFSET       13
 

template<index_t LDS_SIZE, index_t BLOCK_SIZE>
__global__
void smid_kernel(float * A, float * B){
    __shared__ char smem[LDS_SIZE];
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)

    unsigned xcc_id = __builtin_amdgcn_s_getreg(GETREG_IMMED(C_XCC_ID_XCC_ID_SIZE - 1, C_XCC_ID_XCC_ID_OFFSET, C_XCC_ID));
    unsigned cu_id = __builtin_amdgcn_s_getreg(GETREG_IMMED(C_HW_ID_CU_ID_SIZE - 1, C_HW_ID_CU_ID_OFFSET, C_HW_ID));
    unsigned se_id = __builtin_amdgcn_s_getreg(GETREG_IMMED(C_HW_ID_SE_ID_SIZE - 1, C_HW_ID_SE_ID_OFFSET, C_HW_ID));
    if(threadIdx.x == 0 ) {
        printf("%-3d, %-3d, %-3d, %u, %u, %u\n",
                static_cast<int>(blockIdx.x),
                static_cast<int>(blockIdx.y),
                static_cast<int>(blockIdx.z),
                xcc_id, se_id, cu_id
        );
    }
#endif
    // dummy read/write
    smem[threadIdx.x] = A[threadIdx.x + blockIdx.x * BLOCK_SIZE];
    B[threadIdx.x + blockIdx.x * BLOCK_SIZE] = smem[threadIdx.x];
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
    uint32_t bx = 304;
    uint32_t by = 1;
    uint32_t bz = 1;
    uint32_t occupancy = 1;

    if(argc >= 2) bx = atoi(argv[1]);
    if(argc >= 3) by = atoi(argv[2]);
    if(argc >= 4) bz = atoi(argv[3]);
    if(argc >= 5) occupancy = atoi(argv[4]);

    printf("%ux%ux%u, occupancy:%u\n", bx, by, bz, occupancy);
    printf("bx , by , bz , xcc, se, cu, smid\n");

    float * A;
    float * B;
    CALL(hipMalloc(&A, bx * by * bz * 256 * sizeof(float)));
    CALL(hipMalloc(&B, bx * by * bz * 256 * sizeof(float)));

    if(occupancy == 1)
        smid_kernel<65536, 256><<<dim3(bx, by, bz), 256>>>(A, B);
    else if (occupancy == 2)
        smid_kernel<32768, 256><<<dim3(bx, by, bz), 256>>>(A, B);

    CALL(hipFree(A));
    CALL(hipFree(B));

    return 0;
}
