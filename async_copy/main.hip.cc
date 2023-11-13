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

#define BUFFER_LOAD_DWORD3 0x00020000   // This is valid for 
struct buffer_resource {
    const void * ptr;
    dword_t range;
    dword_t config;
};
__device__ dwordx4_t make_buffer_resource(const void * ptr)
{
    buffer_resource res {ptr, 0xffffffff, BUFFER_LOAD_DWORD3};
    return __builtin_bit_cast(dwordx4_t, res);
}

__device__ void init_m0(uint32_t m0_value)
{
    asm volatile("s_mov_b32 m0, %0" : : "s"(m0_value): "memory");
}

__device__ void inc_m0(uint32_t m0_inc)
{
    asm volatile("s_add_u32 m0, %0, m0" : : "n"(m0_inc): "memory");
}

__device__ void async_gld_dword(void * smem, dwordx4_t res/*buffer resource*/, index_t v_offset, index_t s_offset, index_t i_offset/*max 0xFFF*/, index_t /*flag*/ = 0){
    asm volatile("buffer_load_dword %1, %2, %3 offen offset:%4 lds"
        : "=r"(smem) /*dummy dependency for smem*/ : "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset) : "memory");
}

__device__ void async_gld_fence(index_t cnt)
{
    asm volatile("s_waitcnt vmcnt(%0)" : : "n" (cnt) : "memory");
}

__device__ void wave_barrier()
{
    asm volatile("s_barrier" : : : "memory");
}

template<typename T, index_t BLOCK_SIZE, index_t ROWS, index_t COLS>
__global__
void transpose_lds_kernel(T* __restrict__ dst, uint32_t dst_stride, const T* __restrict__ src, uint32_t src_stride){
    static_assert(sizeof(T) == 4);      // for simplicity, only support dword type
    static_assert((ROWS * COLS) % BLOCK_SIZE == 0);
    static_assert(ROWS == 16 || ROWS == 32 || ROWS == 64);  // TODO: this assume BLOCK_SIZE is 256
    static_assert(COLS == 16 || COLS == 32 || COLS == 64);

    /*
    * we use async copy to load data direct into LDS.
    * the data in LDS will always be contiguous within a wave, this is a big limitation    
    */
    constexpr index_t issues = (ROWS * COLS) / BLOCK_SIZE;
    constexpr index_t pad = 1;
    constexpr index_t lds_stride = 64 + pad;
    constexpr index_t waves = 4;

    constexpr index_t r0_len = issues;
    constexpr index_t r2_len = waves;
    constexpr index_t r1_len = 64 / COLS;
    constexpr index_t c_len = COLS;
    static_assert(ROWS == r0_len * r1_len * r2_len);

    //                 r0     , r2,      r1*c (+pad)
    __shared__ T smem[issues * waves * lds_stride];

    constexpr index_t src_rows_per_wave_per_issue = 64 / COLS;

    // Note: src offset is not linear, but that's fine
    index_t src_col_id = threadIdx.x % COLS;
    index_t src_row_id_0 = threadIdx.x / COLS;
    index_t src_row_id = src_row_id_0 / src_rows_per_wave_per_issue + (src_row_id_0 % src_rows_per_wave_per_issue) * waves;

    index_t src_vos = src_col_id + src_row_id * src_stride;

    // printf("tid:%d, src_vos:%d\n", (int)threadIdx.x, src_vos);

    uint32_t wave_id = threadIdx.x / 64;
    uint32_t m0 = __builtin_amdgcn_readfirstlane(wave_id * lds_stride * sizeof(T));
    init_m0(m0);
    constexpr index_t src_stride_per_issue = BLOCK_SIZE / COLS;
    for(int i = 0; i < issues; i++) {
        async_gld_dword(smem, make_buffer_resource(src), src_vos * sizeof(T), i * src_stride_per_issue * src_stride * sizeof(T), 0);
        inc_m0(waves *lds_stride * sizeof(T));  // async copy depends on m0 value, so better also use inline asm to update m0
    }
    async_gld_fence(0);
    wave_barrier();

    // Note: dst offset is linear
    constexpr index_t dst_stride_per_issue = BLOCK_SIZE / ROWS;

    index_t dst_col_id = threadIdx.x % ROWS;
    index_t dst_row_id = threadIdx.x / ROWS;

    index_t dst_vos = dst_row_id * dst_stride + dst_col_id;
    for(int i = 0 ; i < issues; i++){
        uint32_t r2 = dst_col_id % r2_len;
        uint32_t r1 = (dst_col_id / r2_len) % r1_len;
        uint32_t r0 = dst_col_id / (r2_len * r1_len);

        uint32_t lds_idx = (r0 * r2_len + r2) * lds_stride + r1 * c_len + dst_row_id + i * dst_stride_per_issue;
        // printf("tid:%d, lds_idx:%d\n", (int)threadIdx.x, lds_idx);

        uint32_t tmp = smem[lds_idx];
        *reinterpret_cast<uint32_t * >(dst + dst_vos + i * dst_stride_per_issue * dst_stride) = tmp;
    }
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
    uint32_t *A, *B;
    uint32_t n_row = 16;
    uint32_t n_col = 16;

    if(argc >= 2) n_row = atoi(argv[1]);
    if(argc >= 3) n_col = atoi(argv[2]);

    printf("transpose %dx%d ---- \n", n_row, n_col);

    int dwords = n_row * n_col;
    uint32_t * h_A = (uint32_t*)malloc(dwords*sizeof(uint32_t));
    uint32_t * h_B = (uint32_t*)malloc(dwords*sizeof(uint32_t));

    for (uint32_t r = 0 ; r < n_row; r++) {
        for (uint32_t c = 0 ; c < n_col; c++) {
            uint32_t value = (r << 8) | c;
            h_A[r * n_col + c] = value;
        }
    }

    CALL(hipMalloc(&A, dwords * sizeof(uint32_t)));
    CALL(hipMalloc(&B, dwords * sizeof(uint32_t)));
    CALL(hipMemcpy(A, h_A, dwords * sizeof(uint32_t), hipMemcpyHostToDevice));

    {
        if(n_row == 64 && n_col == 64 )      transpose_lds_kernel<uint32_t, 256, 64, 64><<<1, 256>>>(B, n_row, A, n_col);
        else if(n_row == 32 && n_col == 64 ) transpose_lds_kernel<uint32_t, 256, 32, 64><<<1, 256>>>(B, n_row, A, n_col);
        else if(n_row == 64 && n_col == 32 ) transpose_lds_kernel<uint32_t, 256, 64, 32><<<1, 256>>>(B, n_row, A, n_col);
        else if(n_row == 32 && n_col == 32 ) transpose_lds_kernel<uint32_t, 256, 32, 32><<<1, 256>>>(B, n_row, A, n_col);
        else if(n_row == 32 && n_col == 16 ) transpose_lds_kernel<uint32_t, 256, 32, 16><<<1, 256>>>(B, n_row, A, n_col);
        else if(n_row == 16 && n_col == 32 ) transpose_lds_kernel<uint32_t, 256, 16, 32><<<1, 256>>>(B, n_row, A, n_col);
        else if(n_row == 16 && n_col == 16 ) transpose_lds_kernel<uint32_t, 256, 16, 16><<<1, 256>>>(B, n_row, A, n_col);
        else { printf("===== not supported yet\n"); goto out;}
    }

    CALL(hipMemcpy(h_B, B, dwords * sizeof(uint32_t), hipMemcpyDeviceToHost));

    printf("src =======================================\n");
    for (uint32_t r = 0 ; r < n_row; r++) {
        printf("[%2d]", r);
        for (uint32_t c = 0 ; c < n_col; c++) {
            printf("0x%04x ", h_A[r * n_col + c]);
        }
        printf("\n");
    }

    printf("dst =======================================\n");
    for (uint32_t c = 0 ; c < n_col; c++) {
        printf("[%2d]", c);
        for (uint32_t r = 0 ; r < n_row; r++) {
            printf("0x%04x ", h_B[c * n_row + r]);
        }
        printf("\n");
    }

out:
    free(h_A);
    free(h_B);
    CALL(hipFree(A));
    CALL(hipFree(B));
    return 0;
}
