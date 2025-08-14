#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <random>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <numeric>
#include "ck_tile/core.hpp"


#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define HIP_CALL(call) do{  \
    hipError_t err = call;  \
    if(err != hipSuccess){  \
        printf("[hiperror](%d) fail to call %s",(int)err,#call);    \
        exit(0);            \
    }                       \
} while(0)

#define ABS(x) ((x) > 0 ? (x) : -(x))



template<typename T, int dpp_i>
__device__ __inline__ T mov_dpp_(T x, ck_tile::number<dpp_i>) {
    static_assert(sizeof(T) == 4);
    constexpr int row_mask    = 0xf;
    constexpr int bank_mask   = 0xf;
    constexpr bool bound_ctrl = true;   // ! out-of-bound is zero !
    return __builtin_bit_cast(T,
                        // __builtin_amdgcn_update_dpp(0,__builtin_bit_cast(int, x),
                        __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, x),
                                    dpp_i,
                                    row_mask,
                                    bank_mask,
                                    bound_ctrl));
}

template<typename T>
__device__ __inline__ T dev_max_(const T&a, const T&b)
{
    return a > b ? a : b;
}

template<>
__device__ __inline__ float dev_max_<float>(const float&a, const float&b)
{
    return __builtin_fmaxf(a, b);
}

template<typename T>
__device__ __inline__ T dev_min_(const T&a, const T&b)
{
    return a > b ? b : a;
}

template<>
__device__ __inline__ float dev_min_<float>(const float&a, const float&b)
{
    return __builtin_fminf(a, b);
}

// https://en.wikipedia.org/wiki/Batcher_odd%E2%80%93even_mergesort
// TODO: this is assuming descending order sort
// result store to smem :)
template <typename T, int lanegroup_size = 64>
__device__ __inline__ void warp_merge_sort_to_smem(T* smem, const T& x, ck_tile::number<lanegroup_size> = {})
{
    static_assert(sizeof(T) == 4);
    int lane_id = threadIdx.x % lanegroup_size;
    int group_id = threadIdx.x / lanegroup_size;

#define DPP_MERGE_2_()                                  \
    using vec2_t = ck_tile::ext_vector_t<T, 2>;         \
    vec2_t res2;                                        \
    T remote_x = mov_dpp_(x, ck_tile::number<0xb1>{}); /*quad_perm:[1,0,3,2]*/  \
    res2[0] = dev_max_(remote_x, x);    \
    res2[1] = dev_min_(remote_x, x);

#define DPP_MERGE_4_()                              \
    using vec4_t = ck_tile::ext_vector_t<T, 4>;     \
    vec4_t res4;                                    \
                                                    \
    T m_0 = mov_dpp_(res2[0],  ck_tile::number<0x4e>{}); /*quad_perm:[2,3,0,1]*/    \
    res4[0] = dev_max_(res2[0],  m_0);                                              \
    T m_1 = dev_min_(res2[0],  m_0);                                                \
                                                                                    \
    T m_3 = mov_dpp_(res2[1],  ck_tile::number<0x4e>{}); /*quad_perm:[2,3,0,1]*/    \
    T m_2 = dev_max_(res2[1],  m_3);                                                \
    res4[3] = dev_min_(res2[1],  m_3);                                              \
                                                                                    \
    res4[1] = dev_max_(m_1, m_2);                                                   \
    res4[2] = dev_min_(m_1, m_2);

#define DPP_MERGE_8_()                                  \
        using vec8_t = ck_tile::ext_vector_t<T, 8>;     \
        vec8_t res8;                                    \
        vec4_t res4_r;                                  \
                                                        \
        /* only lane 0,1,2,3 contain valid data */      \
        res4_r[0] = mov_dpp_(res4[0],  ck_tile::number<0x104>{}); /* row_shl:4 */ \
        res4_r[1] = mov_dpp_(res4[1],  ck_tile::number<0x104>{}); /* row_shl:4 */ \
        res4_r[2] = mov_dpp_(res4[2],  ck_tile::number<0x104>{}); /* row_shl:4 */ \
        res4_r[3] = mov_dpp_(res4[3],  ck_tile::number<0x104>{}); /* row_shl:4 */ \
        res8[0]      = dev_max_(res4[0], res4_r[0]);    \
        T res8_4_tmp = dev_min_(res4[0], res4_r[0]);    \
                                                        \
        T res8_1_tmp = dev_max_(res4[1], res4_r[1]);    \
        T res8_5_tmp = dev_min_(res4[1], res4_r[1]);    \
                                                        \
        T res8_2_tmp = dev_max_(res4[2], res4_r[2]);    \
        T res8_6_tmp = dev_min_(res4[2], res4_r[2]);    \
                                                        \
        T res8_3_tmp = dev_max_(res4[3], res4_r[3]);    \
        res8[7]      = dev_min_(res4[3], res4_r[3]);    \
                                                        \
        T res8_2_tmp_r = dev_max_(res8_2_tmp, res8_4_tmp);  \
        T res8_4_tmp_r = dev_min_(res8_2_tmp, res8_4_tmp);  \
                                                            \
        T res8_3_tmp_r = dev_max_(res8_3_tmp, res8_5_tmp);  \
        T res8_5_tmp_r = dev_min_(res8_3_tmp, res8_5_tmp);  \
                                                            \
        res8[1] = dev_max_(res8_1_tmp, res8_2_tmp_r);       \
        res8[2] = dev_min_(res8_1_tmp, res8_2_tmp_r);       \
                                                            \
        res8[3] = dev_max_(res8_3_tmp_r, res8_4_tmp_r);     \
        res8[4] = dev_min_(res8_3_tmp_r, res8_4_tmp_r);     \
                                                            \
        res8[5] = dev_max_(res8_5_tmp_r, res8_6_tmp);       \
        res8[6] = dev_min_(res8_5_tmp_r, res8_6_tmp);

#define DPP_MERGE_16_()                                         \
        using vec16_t = ck_tile::ext_vector_t<T, 16>;           \
        vec16_t res16;                                          \
        vec8_t res8_r;                                          \
        /* only lane 0,1,2,3 contain valid data */              \
        res8_r[0] = mov_dpp_(res8[0],  ck_tile::number<0x108>{}); /* row_shl:8 */   \
        res8_r[1] = mov_dpp_(res8[1],  ck_tile::number<0x108>{}); /* row_shl:8 */   \
        res8_r[2] = mov_dpp_(res8[2],  ck_tile::number<0x108>{}); /* row_shl:8 */   \
        res8_r[3] = mov_dpp_(res8[3],  ck_tile::number<0x108>{}); /* row_shl:8 */   \
        res8_r[4] = mov_dpp_(res8[4],  ck_tile::number<0x108>{}); /* row_shl:8 */   \
        res8_r[5] = mov_dpp_(res8[5],  ck_tile::number<0x108>{}); /* row_shl:8 */   \
        res8_r[6] = mov_dpp_(res8[6],  ck_tile::number<0x108>{}); /* row_shl:8 */   \
        res8_r[7] = mov_dpp_(res8[7],  ck_tile::number<0x108>{}); /* row_shl:8 */   \
                                                                                    \
        res16[0]      = dev_max_(res8[0], res8_r[0]);       \
        T res16_8_tmp = dev_min_(res8[0], res8_r[0]);       \
                                                            \
        T res16_1_tmp = dev_max_(res8[1], res8_r[1]);       \
        T res16_9_tmp = dev_min_(res8[1], res8_r[1]);       \
                                                            \
        T res16_2_tmp  = dev_max_(res8[2], res8_r[2]);      \
        T res16_10_tmp = dev_min_(res8[2], res8_r[2]);      \
                                                            \
        T res16_3_tmp  = dev_max_(res8[3], res8_r[3]);      \
        T res16_11_tmp = dev_min_(res8[3], res8_r[3]);      \
                                                            \
        T res16_4_tmp  = dev_max_(res8[4], res8_r[4]);      \
        T res16_12_tmp = dev_min_(res8[4], res8_r[4]);      \
                                                            \
        T res16_5_tmp  = dev_max_(res8[5], res8_r[5]);      \
        T res16_13_tmp = dev_min_(res8[5], res8_r[5]);      \
                                                            \
        T res16_6_tmp  = dev_max_(res8[6], res8_r[6]);      \
        T res16_14_tmp = dev_min_(res8[6], res8_r[6]);      \
                                                            \
        T res16_7_tmp  = dev_max_(res8[7], res8_r[7]);      \
        res16[15]      = dev_min_(res8[7], res8_r[7]);      \
                                                            \
                                                            \
        T res16_4_tmp_x = dev_max_(res16_4_tmp, res16_8_tmp);       \
        T res16_8_tmp_x = dev_min_(res16_4_tmp, res16_8_tmp);       \
                                                                    \
        T res16_5_tmp_x = dev_max_(res16_5_tmp, res16_9_tmp);       \
        T res16_9_tmp_x = dev_min_(res16_5_tmp, res16_9_tmp);       \
                                                                    \
        T res16_6_tmp_x  = dev_max_(res16_6_tmp, res16_10_tmp);     \
        T res16_10_tmp_x = dev_min_(res16_6_tmp, res16_10_tmp);     \
                                                                    \
        T res16_7_tmp_x  = dev_max_(res16_7_tmp, res16_11_tmp);     \
        T res16_11_tmp_x = dev_min_(res16_7_tmp, res16_11_tmp);     \
                                                                    \
                                                                    \
        T res16_2_tmp_x  = dev_max_(res16_2_tmp, res16_4_tmp_x);    \
        T res16_4_tmp_xx = dev_min_(res16_2_tmp, res16_4_tmp_x);    \
                                                                    \
        T res16_3_tmp_x  = dev_max_(res16_3_tmp, res16_5_tmp_x);    \
        T res16_5_tmp_xx = dev_min_(res16_3_tmp, res16_5_tmp_x);    \
                                                                    \
        T res16_6_tmp_xx = dev_max_(res16_6_tmp_x, res16_8_tmp_x);  \
        T res16_8_tmp_xx = dev_min_(res16_6_tmp_x, res16_8_tmp_x);  \
                                                                    \
        T res16_7_tmp_xx = dev_max_(res16_7_tmp_x, res16_9_tmp_x);  \
        T res16_9_tmp_xx = dev_min_(res16_7_tmp_x, res16_9_tmp_x);  \
                                                                    \
        T res16_10_tmp_xx = dev_max_(res16_10_tmp_x, res16_12_tmp);  \
        T res16_12_tmp_xx = dev_min_(res16_10_tmp_x, res16_12_tmp);  \
                                                                     \
        T res16_11_tmp_xx = dev_max_(res16_11_tmp_x, res16_13_tmp);  \
        T res16_13_tmp_xx = dev_min_(res16_11_tmp_x, res16_13_tmp);  \
                                                                    \
        res16[1]  = dev_max_(res16_1_tmp, res16_2_tmp_x);           \
        res16[2]  = dev_min_(res16_1_tmp, res16_2_tmp_x);           \
                                                                    \
        res16[3]  = dev_max_(res16_3_tmp_x, res16_4_tmp_xx);        \
        res16[4]  = dev_min_(res16_3_tmp_x, res16_4_tmp_xx);        \
                                                                    \
        res16[5]  = dev_max_(res16_5_tmp_xx, res16_6_tmp_xx);       \
        res16[6]  = dev_min_(res16_5_tmp_xx, res16_6_tmp_xx);       \
                                                                    \
        res16[7]  = dev_max_(res16_7_tmp_xx, res16_8_tmp_xx);       \
        res16[8]  = dev_min_(res16_7_tmp_xx, res16_8_tmp_xx);       \
                                                                    \
        res16[9]  = dev_max_(res16_9_tmp_xx, res16_10_tmp_xx);      \
        res16[10] = dev_min_(res16_9_tmp_xx, res16_10_tmp_xx);      \
                                                                    \
        res16[11] = dev_max_(res16_11_tmp_xx, res16_12_tmp_xx);     \
        res16[12] = dev_min_(res16_11_tmp_xx, res16_12_tmp_xx);     \
                                                                    \
        res16[13] = dev_max_(res16_13_tmp_xx, res16_14_tmp);        \
        res16[14] = dev_min_(res16_13_tmp_xx, res16_14_tmp);


    if constexpr (lanegroup_size == 2) {
        DPP_MERGE_2_();

        if(lane_id == 0) {
            reinterpret_cast<vec2_t*>(smem)[group_id] = res2;
        }
    } else if constexpr (lanegroup_size == 4) {
        DPP_MERGE_2_();
        DPP_MERGE_4_();

        if(lane_id == 0) {
            reinterpret_cast<vec4_t*>(smem)[group_id] = res4;
        }
    } else if constexpr (lanegroup_size == 8) {
        DPP_MERGE_2_();
        DPP_MERGE_4_();
        DPP_MERGE_8_();

        if(lane_id == 0) {
            union {
                struct  {
                    vec4_t x;
                    vec4_t y;
                };
                vec8_t value;
            } _tmp;
            _tmp.value = res8;
            reinterpret_cast<vec4_t*>(smem)[group_id * 2] = _tmp.x;
            reinterpret_cast<vec4_t*>(smem)[group_id * 2 + 1] = _tmp.y;
        }
    } else if constexpr (lanegroup_size == 16) {
        DPP_MERGE_2_();
        DPP_MERGE_4_();
        DPP_MERGE_8_();
        DPP_MERGE_16_();

        if(lane_id == 0) {
#if 0
            union {
                struct {
                    vec4_t x;
                    vec4_t y;
                    vec4_t z;
                    vec4_t w;
                };
                vec16_t value;
            } _tmp;
            _tmp.value = res16;
            reinterpret_cast<vec4_t*>(smem)[group_id * 4 + 0] = _tmp.x;
            __syncthreads();
            reinterpret_cast<vec4_t*>(smem)[group_id * 4 + 1] = _tmp.y;
            __syncthreads();
            reinterpret_cast<vec4_t*>(smem)[group_id * 4 + 2] = _tmp.z;
            __syncthreads();
            reinterpret_cast<vec4_t*>(smem)[group_id * 4 + 3] = _tmp.w;
#else
            reinterpret_cast<vec16_t*>(smem)[group_id] = res16;
#endif
        }
    }
#undef DPP_MERGE_2_
#undef DPP_MERGE_4_
#undef DPP_MERGE_8_
#undef DPP_MERGE_16_
}

template<typename T, int wave_size = 64, int lanegroup_size = 64>
__global__ void warp_sort_kernel(T* i_ptr, T* o_ptr)
{
    __shared__ T smem[wave_size / lanegroup_size];
    T data = -INFINITY;
    if(threadIdx.x < lanegroup_size) {
        data = i_ptr[threadIdx.x];
    }

    warp_merge_sort_to_smem(smem, data, ck_tile::number<lanegroup_size>{});

    __syncthreads();

    T sorted = smem[threadIdx.x];   // ignore out-of-bound check

    if(threadIdx.x < lanegroup_size) {
        o_ptr[threadIdx.x] = sorted;
    }
}

static inline float get_rand(){
    static int inited = 0;
    float v;
    if(!inited){ srand(time(NULL)); inited = 1; }
    v = rand() % 600 + 1;
    return v / 70.0f;
}

static inline void rand_vector(float* vec, int len) {
    for(int i =0; i < len; i++) {
        vec[i] = get_rand();
    }
}

static inline bool check_ordered(float* vec, int len) {
    bool rtn = true;
    for(int i = 0; i < len - 1; i++) {
        rtn &= vec[i] >= vec[i+1];
    }
    return rtn;
}

template<typename T, int wave_size = 64, int lanegroup_size = 64>
void run()
{
    T * input = reinterpret_cast<T*>(malloc(sizeof(T) * lanegroup_size));
    T * output = reinterpret_cast<T*>(malloc(sizeof(T) * lanegroup_size));

    T *dev_i, *dev_o;

    HIP_CALL(hipMalloc(&dev_i, sizeof(T) * lanegroup_size));
    HIP_CALL(hipMalloc(&dev_o, sizeof(T) * lanegroup_size));

    rand_vector(input, lanegroup_size);

    HIP_CALL(hipMemcpy(dev_i, input, sizeof(T) * lanegroup_size, hipMemcpyHostToDevice));

    auto gx = dim3(1);
    auto bx = dim3(wave_size);

    warp_sort_kernel<T, wave_size, lanegroup_size><<<gx, bx>>>(dev_i, dev_o);

    HIP_CALL(hipMemcpy(output, dev_o, sizeof(T) * lanegroup_size, hipMemcpyDeviceToHost));

    printf("[origin-%d]", lanegroup_size);
    for(int i = 0; i < lanegroup_size; i++) {
        printf("%.3f ", input[i]);
    }
    printf("\n");
    printf("[sorted-%d]", lanegroup_size);
    for(int i = 0; i < lanegroup_size; i++) {
        printf("%.3f ", output[i]);
    }
    printf("\n");
    bool is_ordered = check_ordered(output, lanegroup_size);
    printf("-------------------------------------- %s\n", is_ordered?"ordered":"non-order");

    free(input);
    free(output);

    HIP_CALL(hipFree(dev_i));
    HIP_CALL(hipFree(dev_o));
}

int main(int argc, char ** argv)
{
    run<float, 64, 2>();
    run<float, 64, 4>();
    run<float, 64, 8>();
    run<float, 64, 16>();
    // run<float, 64, 8>();
}
