#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <random>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <numeric>
#include <list>
#include "opus/opus.hpp"

#ifndef FMT_LIMIT
#define FMT_LIMIT 1
#define FMT_LIMIT_MAX 16
#endif

__device__ int count_4_lane(int v)
{
    
    int y = 1;
    // make sure value is within 64
    int r = opus::mov_dpp(v, opus::number<0xb1>{}); /*quad_perm:[1,0,3,2]*/
    int s = r == v ? 0x2 : 0;
    y = s | y;

    r = opus::mov_dpp(v, opus::number<0x4e>{});     /*quad_perm:[2,3,0,1]*/
    s = r == v ? 0x4 : 0;
    y = s | y;

    r = opus::mov_dpp(v, opus::number<0x1b>{});     /*quad_perm:[3,2,1,0]*/
    s = r == v ? 0x8 : 0;
    y = s | y;

    // printf("[%2d] v:%d, y:%d\n", threadIdx.x, v, y);
    return y;
}

template<int bucket = 64>
__device__ __inline__ auto warp_histogram_step(int32_t value, int8_t* smem /*[bucket x bucket / 4]*/)
{
    static_assert(bucket == 64, "currently only 64 buckets supported");
    
    // reinterpret_cast<int32x4_t>(smem + xrow_id * bucket);
    int32_t v = value % bucket; // just make sure value is within bucket

    int c4 = count_4_lane(v);   // 4 bit
    

    int xrow_id = threadIdx.x % bucket;

    *reinterpret_cast<opus::i32x4_t*>(smem + xrow_id * bucket / 4) = 0;
    opus::s_waitcnt_lgkmcnt(opus::number<0>{});

    int row_id = v;
    int col_id = threadIdx.x / 4;

    union {
        struct {
            int8_t x;
            int8_t y;
            int8_t z;
            int8_t w;
        };
        int32_t v;
    } pool;

    pool.v = c4;

    smem[row_id * (bucket / 4) + col_id] = pool.x;

    opus::s_waitcnt_lgkmcnt(opus::number<0>{});

    
    opus::i32x4_t r = *reinterpret_cast<opus::i32x4_t*>(smem + xrow_id * bucket / 4);

    int cnt = 0;
    for(int i = 0; i < 4; i++) {
        cnt += __builtin_popcount(r[i]);
    }
    return cnt;
}

// TODO: bucket size is always 64 (same as warp size)
template<typename DType, typename IndexType, int WarpSize = 64>
__global__ void warp_histogram_kernel(DType* i_ptr, IndexType * o_ptr, int nelem)
{
    __shared__ int8_t smem[WarpSize * WarpSize / 4];   // ! our histogram using a 64x16 smem to reduce :)

    int cnt = 0;
    int loops = (nelem + WarpSize - 1) / WarpSize;
    for(int i = 0; i < loops; i++) {
        DType value = i_ptr[WarpSize * i + threadIdx.x];
        cnt += warp_histogram_step(value, smem);
    }
    o_ptr[threadIdx.x] = cnt;
}

// host c++ API
void warp_histogram(void* i_ptr, void* o_ptr, int num_element)
{
    using DType = int;
    using IndexType = int;
    warp_histogram_kernel<DType, IndexType><<<1, 64>>>(reinterpret_cast<DType*>(i_ptr), reinterpret_cast<IndexType*>(o_ptr), num_element);
}
