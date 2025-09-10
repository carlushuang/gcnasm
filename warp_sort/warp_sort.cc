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

#define DPP_MERGE_2_CMP_(x_, y_)                        \
    using vec2_t = ck_tile::ext_vector_t<T, 2>;         \
    vec2_t res2;                                        \
    res2[0] = dev_max_(x_, y_);                         \
    res2[1] = dev_min_(x_, y_);

#define DPP_MERGE_2_DPP_()                                  \
    T res1_r = mov_dpp_(res1, ck_tile::number<0xb1>{}); /*quad_perm:[1,0,3,2]*/

#define DPP_ARG_MERGE_2_CMP_(x_, y_, ax_, ay_)          \
    using vec2_t = ck_tile::ext_vector_t<T, 2>;         \
    using aec2_t = ck_tile::ext_vector_t<V, 2>;         \
    vec2_t res2;                                        \
    aec2_t arg2;                                        \
    res2[0] = x_ > y_? x_ : y_;                         \
    res2[1] = x_ > y_? y_ : x_;                         \
    arg2[0] = x_ > y_? ax_ : ay_;                       \
    arg2[1] = x_ > y_? ay_ : ax_;

#define DPP_ARG_MERGE_2_DPP_()                          \
    T res1_r = mov_dpp_(res1, ck_tile::number<0xb1>{}); /*quad_perm:[1,0,3,2]*/ \
    V arg1_r = mov_dpp_(arg1, ck_tile::number<0xb1>{}); /*quad_perm:[1,0,3,2]*/

#define DPP_MERGE_4_CMP_(x_, y_)                    \
    using vec4_t = ck_tile::ext_vector_t<T, 4>;     \
    vec4_t res4;                                    \
                                                    \
    res4[0] = dev_max_(x_[0],  y_[0]);              \
    T m_1 = dev_min_(x_[0],  y_[0]);                \
                                                    \
    T m_2 = dev_max_(x_[1],  y_[1]);                \
    res4[3] = dev_min_(x_[1],  y_[1]);              \
                                                    \
    res4[1] = dev_max_(m_1, m_2);                   \
    res4[2] = dev_min_(m_1, m_2);

#define DPP_MERGE_4_DPP_()                                  \
    vec2_t res2_r;                                          \
    res2_r[0] = mov_dpp_(res2[0],  ck_tile::number<0x4e>{}); /*quad_perm:[2,3,0,1]*/    \
    res2_r[1] = mov_dpp_(res2[1],  ck_tile::number<0x4e>{}); /*quad_perm:[2,3,0,1]*/

#define DPP_ARG_MERGE_4_CMP_(x_, y_, ax_, ay_)      \
    using vec4_t = ck_tile::ext_vector_t<T, 4>;     \
    using aec4_t = ck_tile::ext_vector_t<V, 4>;     \
    vec4_t res4;                                    \
    aec4_t arg4;                                    \
                                                    \
    res4[0] = x_[0] > y_[0] ? x_[0] : y_[0];        \
    T m_1   = x_[0] > y_[0] ? y_[0] : x_[0];        \
    arg4[0] = x_[0] > y_[0] ? ax_[0] : ay_[0];      \
    V am_1  = x_[0] > y_[0] ? ay_[0] : ax_[0];      \
                                                    \
    T m_2 = x_[1] > y_[1] ? x_[1] : y_[1];          \
    res4[3] = x_[1] > y_[1] ? y_[1] : x_[1];        \
    V am_2 = x_[1] > y_[1] ? ax_[1] : ay_[1];       \
    arg4[3] = x_[1] > y_[1] ? ay_[1] : ax_[1];      \
                                                    \
    res4[1] = m_1 > m_2 ? m_1 : m_2;                \
    res4[2] = m_1 > m_2 ? m_2 : m_1;                \
    arg4[1] = m_1 > m_2 ? am_1 : am_2;              \
    arg4[2] = m_1 > m_2 ? am_2 : am_1;

#define DPP_ARG_MERGE_4_DPP_()                              \
    vec2_t res2_r;                                          \
    aec2_t arg2_r;                                          \
    res2_r[0] = mov_dpp_(res2[0],  ck_tile::number<0x4e>{}); /*quad_perm:[2,3,0,1]*/    \
    res2_r[1] = mov_dpp_(res2[1],  ck_tile::number<0x4e>{}); /*quad_perm:[2,3,0,1]*/    \
    arg2_r[0] = mov_dpp_(arg2[0],  ck_tile::number<0x4e>{}); /*quad_perm:[2,3,0,1]*/    \
    arg2_r[1] = mov_dpp_(arg2[1],  ck_tile::number<0x4e>{}); /*quad_perm:[2,3,0,1]*/

#define DPP_MERGE_8_CMP_(x_, y_)                            \
        using vec8_t = ck_tile::ext_vector_t<T, 8>;     \
        vec8_t res8;                                    \
                                                        \
        res8[0]      = dev_max_(x_[0], y_[0]);      \
        T res8_4_tmp = dev_min_(x_[0], y_[0]);      \
                                                    \
        T res8_1_tmp = dev_max_(x_[1], y_[1]);      \
        T res8_5_tmp = dev_min_(x_[1], y_[1]);      \
                                                    \
        T res8_2_tmp = dev_max_(x_[2], y_[2]);      \
        T res8_6_tmp = dev_min_(x_[2], y_[2]);      \
                                                    \
        T res8_3_tmp = dev_max_(x_[3], y_[3]);      \
        res8[7]      = dev_min_(x_[3], y_[3]);      \
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

#define DPP_MERGE_8_DPP_()                              \
        vec4_t res4_r;                                  \
                                                        \
        /* only lane 0,1,2,3 contain valid data */      \
        res4_r[0] = mov_dpp_(res4[0],  ck_tile::number<0x104>{}); /* row_shl:4 */ \
        res4_r[1] = mov_dpp_(res4[1],  ck_tile::number<0x104>{}); /* row_shl:4 */ \
        res4_r[2] = mov_dpp_(res4[2],  ck_tile::number<0x104>{}); /* row_shl:4 */ \
        res4_r[3] = mov_dpp_(res4[3],  ck_tile::number<0x104>{}); /* row_shl:4 */

#define DPP_ARG_MERGE_8_CMP_(x_, y_, ax_, ay_)              \
        using vec8_t = ck_tile::ext_vector_t<T, 8>;         \
        using aec8_t = ck_tile::ext_vector_t<V, 8>;         \
        vec8_t res8;                                        \
        aec8_t arg8;                                        \
                                                            \
        res8[0]      = x_[0] > y_[0] ? x_[0] : y_[0];       \
        T res8_4_tmp = x_[0] > y_[0] ? y_[0] : x_[0];       \
        arg8[0]      = x_[0] > y_[0] ? ax_[0] : ay_[0];     \
        V arg8_4_tmp = x_[0] > y_[0] ? ay_[0] : ax_[0];     \
                                                            \
        T res8_1_tmp = x_[1] > y_[1] ? x_[1] : y_[1];       \
        T res8_5_tmp = x_[1] > y_[1] ? y_[1] : x_[1];       \
        V arg8_1_tmp = x_[1] > y_[1] ? ax_[1] : ay_[1];     \
        V arg8_5_tmp = x_[1] > y_[1] ? ay_[1] : ax_[1];     \
                                                            \
        T res8_2_tmp = x_[2] > y_[2] ? x_[2] : y_[2];       \
        T res8_6_tmp = x_[2] > y_[2] ? y_[2] : x_[2];       \
        V arg8_2_tmp = x_[2] > y_[2] ? ax_[2] : ay_[2];     \
        V arg8_6_tmp = x_[2] > y_[2] ? ay_[2] : ax_[2];     \
                                                            \
        T res8_3_tmp = x_[3] > y_[3] ? x_[3] : y_[3];       \
        res8[7]      = x_[3] > y_[3] ? y_[3] : x_[3];       \
        V arg8_3_tmp = x_[3] > y_[3] ? ax_[3] : ay_[3];     \
        arg8[7]      = x_[3] > y_[3] ? ay_[3] : ax_[3];     \
                                                            \
        T res8_2_tmp_r = res8_2_tmp > res8_4_tmp ? res8_2_tmp :res8_4_tmp;  \
        T res8_4_tmp_r = res8_2_tmp > res8_4_tmp ? res8_4_tmp :res8_2_tmp;  \
        V arg8_2_tmp_r = res8_2_tmp > res8_4_tmp ? arg8_2_tmp :arg8_4_tmp;  \
        V arg8_4_tmp_r = res8_2_tmp > res8_4_tmp ? arg8_4_tmp :arg8_2_tmp;  \
                                                                            \
        T res8_3_tmp_r = res8_3_tmp > res8_5_tmp ? res8_3_tmp : res8_5_tmp; \
        T res8_5_tmp_r = res8_3_tmp > res8_5_tmp ? res8_5_tmp : res8_3_tmp; \
        V arg8_3_tmp_r = res8_3_tmp > res8_5_tmp ? arg8_3_tmp : arg8_5_tmp; \
        V arg8_5_tmp_r = res8_3_tmp > res8_5_tmp ? arg8_5_tmp : arg8_3_tmp; \
                                                                            \
        res8[1] = res8_1_tmp > res8_2_tmp_r ? res8_1_tmp : res8_2_tmp_r;  \
        res8[2] = res8_1_tmp > res8_2_tmp_r ? res8_2_tmp_r : res8_1_tmp;  \
        arg8[1] = res8_1_tmp > res8_2_tmp_r ? arg8_1_tmp : arg8_2_tmp_r;  \
        arg8[2] = res8_1_tmp > res8_2_tmp_r ? arg8_2_tmp_r : arg8_1_tmp;  \
                                                            \
        res8[3] = res8_3_tmp_r > res8_4_tmp_r ? res8_3_tmp_r : res8_4_tmp_r;  \
        res8[4] = res8_3_tmp_r > res8_4_tmp_r ? res8_4_tmp_r : res8_3_tmp_r;  \
        arg8[3] = res8_3_tmp_r > res8_4_tmp_r ? arg8_3_tmp_r : arg8_4_tmp_r;  \
        arg8[4] = res8_3_tmp_r > res8_4_tmp_r ? arg8_4_tmp_r : arg8_3_tmp_r;  \
                                                            \
        res8[5] = res8_5_tmp_r > res8_6_tmp ? res8_5_tmp_r: res8_6_tmp;    \
        res8[6] = res8_5_tmp_r > res8_6_tmp ? res8_6_tmp: res8_5_tmp_r;    \
        arg8[5] = res8_5_tmp_r > res8_6_tmp ? arg8_5_tmp_r: arg8_6_tmp;    \
        arg8[6] = res8_5_tmp_r > res8_6_tmp ? arg8_6_tmp: arg8_5_tmp_r; 

#define DPP_ARG_MERGE_8_DPP_()                          \
        vec4_t res4_r;                                  \
        aec4_t arg4_r;                                  \
                                                        \
        /* only lane 0,1,2,3 contain valid data */      \
        res4_r[0] = mov_dpp_(res4[0],  ck_tile::number<0x104>{}); /* row_shl:4 */ \
        res4_r[1] = mov_dpp_(res4[1],  ck_tile::number<0x104>{}); /* row_shl:4 */ \
        res4_r[2] = mov_dpp_(res4[2],  ck_tile::number<0x104>{}); /* row_shl:4 */ \
        res4_r[3] = mov_dpp_(res4[3],  ck_tile::number<0x104>{}); /* row_shl:4 */ \
        arg4_r[0] = mov_dpp_(arg4[0],  ck_tile::number<0x104>{}); /* row_shl:4 */ \
        arg4_r[1] = mov_dpp_(arg4[1],  ck_tile::number<0x104>{}); /* row_shl:4 */ \
        arg4_r[2] = mov_dpp_(arg4[2],  ck_tile::number<0x104>{}); /* row_shl:4 */ \
        arg4_r[3] = mov_dpp_(arg4[3],  ck_tile::number<0x104>{}); /* row_shl:4 */

#define DPP_MERGE_16_CMP_(x_, y_)                               \
        using vec16_t = ck_tile::ext_vector_t<T, 16>;       \
        vec16_t res16;                                      \
                                                        \
        res16[0]      = dev_max_(x_[0], y_[0]);         \
        T res16_8_tmp = dev_min_(x_[0], y_[0]);         \
                                                        \
        T res16_1_tmp = dev_max_(x_[1], y_[1]);         \
        T res16_9_tmp = dev_min_(x_[1], y_[1]);         \
                                                        \
        T res16_2_tmp  = dev_max_(x_[2], y_[2]);        \
        T res16_10_tmp = dev_min_(x_[2], y_[2]);        \
                                                        \
        T res16_3_tmp  = dev_max_(x_[3], y_[3]);        \
        T res16_11_tmp = dev_min_(x_[3], y_[3]);        \
                                                        \
        T res16_4_tmp  = dev_max_(x_[4], y_[4]);        \
        T res16_12_tmp = dev_min_(x_[4], y_[4]);        \
                                                        \
        T res16_5_tmp  = dev_max_(x_[5], y_[5]);        \
        T res16_13_tmp = dev_min_(x_[5], y_[5]);        \
                                                        \
        T res16_6_tmp  = dev_max_(x_[6], y_[6]);        \
        T res16_14_tmp = dev_min_(x_[6], y_[6]);        \
                                                        \
        T res16_7_tmp  = dev_max_(x_[7], y_[7]);        \
        res16[15]      = dev_min_(x_[7], y_[7]);        \
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

#define DPP_MERGE_16_DPP_()                                         \
        vec8_t res8_r;                                          \
        /* only lane 0,1,2,3 contain valid data */              \
        res8_r[0] = mov_dpp_(res8[0],  ck_tile::number<0x108>{}); /* row_shl:8 */   \
        res8_r[1] = mov_dpp_(res8[1],  ck_tile::number<0x108>{}); /* row_shl:8 */   \
        res8_r[2] = mov_dpp_(res8[2],  ck_tile::number<0x108>{}); /* row_shl:8 */   \
        res8_r[3] = mov_dpp_(res8[3],  ck_tile::number<0x108>{}); /* row_shl:8 */   \
        res8_r[4] = mov_dpp_(res8[4],  ck_tile::number<0x108>{}); /* row_shl:8 */   \
        res8_r[5] = mov_dpp_(res8[5],  ck_tile::number<0x108>{}); /* row_shl:8 */   \
        res8_r[6] = mov_dpp_(res8[6],  ck_tile::number<0x108>{}); /* row_shl:8 */   \
        res8_r[7] = mov_dpp_(res8[7],  ck_tile::number<0x108>{}); /* row_shl:8 */


#define DPP_ARG_MERGE_16_CMP_(x_, y_, ax_, ay_)             \
        using vec16_t = ck_tile::ext_vector_t<T, 16>;       \
        using aec16_t = ck_tile::ext_vector_t<V, 16>;       \
        vec16_t res16;                                      \
        aec16_t arg16;                                      \
                                                            \
        res16[0]      = x_[0] > y_[0] ? x_[0] : y_[0];      \
        T res16_8_tmp = x_[0] > y_[0] ? y_[0] : x_[0];      \
        arg16[0]      = x_[0] > y_[0] ? ax_[0] : ay_[0];    \
        V arg16_8_tmp = x_[0] > y_[0] ? ay_[0] : ax_[0];    \
                                                            \
        T res16_1_tmp = x_[1] > y_[1] ? x_[1] : y_[1];      \
        T res16_9_tmp = x_[1] > y_[1] ? y_[1] : x_[1];      \
        V arg16_1_tmp = x_[1] > y_[1] ? ax_[1] : ay_[1];    \
        V arg16_9_tmp = x_[1] > y_[1] ? ay_[1] : ax_[1];    \
                                                            \
        T res16_2_tmp  = x_[2] > y_[2] ? x_[2] : y_[2];     \
        T res16_10_tmp = x_[2] > y_[2] ? y_[2] : x_[2];     \
        V arg16_2_tmp  = x_[2] > y_[2] ? ax_[2] : ay_[2];   \
        V arg16_10_tmp = x_[2] > y_[2] ? ay_[2] : ax_[2];   \
                                                            \
        T res16_3_tmp  = x_[3] > y_[3] ? x_[3] : y_[3];     \
        T res16_11_tmp = x_[3] > y_[3] ? y_[3] : x_[3];     \
        V arg16_3_tmp  = x_[3] > y_[3] ? ax_[3] : ay_[3];   \
        V arg16_11_tmp = x_[3] > y_[3] ? ay_[3] : ax_[3];   \
                                                            \
        T res16_4_tmp  = x_[4] > y_[4] ? x_[4] : y_[4];     \
        T res16_12_tmp = x_[4] > y_[4] ? y_[4] : x_[4];     \
        V arg16_4_tmp  = x_[4] > y_[4] ? ax_[4] : ay_[4];   \
        V arg16_12_tmp = x_[4] > y_[4] ? ay_[4] : ax_[4];   \
                                                            \
        T res16_5_tmp  = x_[5] > y_[5] ? x_[5] : y_[5];     \
        T res16_13_tmp = x_[5] > y_[5] ? y_[5] : x_[5];     \
        V arg16_5_tmp  = x_[5] > y_[5] ? ax_[5] : ay_[5];   \
        V arg16_13_tmp = x_[5] > y_[5] ? ay_[5] : ax_[5];   \
                                                            \
        T res16_6_tmp  = x_[6] > y_[6] ? x_[6] : y_[6];     \
        T res16_14_tmp = x_[6] > y_[6] ? y_[6] : x_[6];     \
        V arg16_6_tmp  = x_[6] > y_[6] ? ax_[6] : ay_[6];   \
        V arg16_14_tmp = x_[6] > y_[6] ? ay_[6] : ax_[6];   \
                                                            \
        T res16_7_tmp  = x_[7] > y_[7] ? x_[7] : y_[7];     \
        res16[15]      = x_[7] > y_[7] ? y_[7] : x_[7];     \
        V arg16_7_tmp  = x_[7] > y_[7] ? ax_[7] : ay_[7];   \
        arg16[15]      = x_[7] > y_[7] ? ay_[7] : ax_[7];   \
                                                            \
        T res16_4_tmp_x = res16_4_tmp > res16_8_tmp ? res16_4_tmp : res16_8_tmp;        \
        T res16_8_tmp_x = res16_4_tmp > res16_8_tmp ? res16_8_tmp : res16_4_tmp;        \
        V arg16_4_tmp_x = res16_4_tmp > res16_8_tmp ? arg16_4_tmp : arg16_8_tmp;        \
        V arg16_8_tmp_x = res16_4_tmp > res16_8_tmp ? arg16_8_tmp : arg16_4_tmp;        \
                                                                                        \
        T res16_5_tmp_x = res16_5_tmp > res16_9_tmp ? res16_5_tmp : res16_9_tmp;        \
        T res16_9_tmp_x = res16_5_tmp > res16_9_tmp ? res16_9_tmp : res16_5_tmp;        \
        V arg16_5_tmp_x = res16_5_tmp > res16_9_tmp ? arg16_5_tmp : arg16_9_tmp;        \
        V arg16_9_tmp_x = res16_5_tmp > res16_9_tmp ? arg16_9_tmp : arg16_5_tmp;        \
                                                                                        \
        T res16_6_tmp_x  = res16_6_tmp > res16_10_tmp ? res16_6_tmp  : res16_10_tmp;    \
        T res16_10_tmp_x = res16_6_tmp > res16_10_tmp ? res16_10_tmp : res16_6_tmp;     \
        V arg16_6_tmp_x  = res16_6_tmp > res16_10_tmp ? arg16_6_tmp  : arg16_10_tmp;    \
        V arg16_10_tmp_x = res16_6_tmp > res16_10_tmp ? arg16_10_tmp : arg16_6_tmp;     \
                                                                                        \
        T res16_7_tmp_x  = res16_7_tmp > res16_11_tmp ? res16_7_tmp  : res16_11_tmp;    \
        T res16_11_tmp_x = res16_7_tmp > res16_11_tmp ? res16_11_tmp : res16_7_tmp;     \
        V arg16_7_tmp_x  = res16_7_tmp > res16_11_tmp ? arg16_7_tmp  : arg16_11_tmp;    \
        V arg16_11_tmp_x = res16_7_tmp > res16_11_tmp ? arg16_11_tmp : arg16_7_tmp;     \
                                                                                        \
        T res16_2_tmp_x  = res16_2_tmp > res16_4_tmp_x ? res16_2_tmp   : res16_4_tmp_x;     \
        T res16_4_tmp_xx = res16_2_tmp > res16_4_tmp_x ? res16_4_tmp_x : res16_2_tmp ;      \
        V arg16_2_tmp_x  = res16_2_tmp > res16_4_tmp_x ? arg16_2_tmp   : arg16_4_tmp_x;     \
        V arg16_4_tmp_xx = res16_2_tmp > res16_4_tmp_x ? arg16_4_tmp_x : arg16_2_tmp ;      \
                                                                                            \
        T res16_3_tmp_x  = res16_3_tmp > res16_5_tmp_x ? res16_3_tmp   : res16_5_tmp_x;     \
        T res16_5_tmp_xx = res16_3_tmp > res16_5_tmp_x ? res16_5_tmp_x : res16_3_tmp ;      \
        V arg16_3_tmp_x  = res16_3_tmp > res16_5_tmp_x ? arg16_3_tmp   : arg16_5_tmp_x;     \
        V arg16_5_tmp_xx = res16_3_tmp > res16_5_tmp_x ? arg16_5_tmp_x : arg16_3_tmp ;      \
                                                                                            \
        T res16_6_tmp_xx = res16_6_tmp_x > res16_8_tmp_x ? res16_6_tmp_x : res16_8_tmp_x;   \
        T res16_8_tmp_xx = res16_6_tmp_x > res16_8_tmp_x ? res16_8_tmp_x : res16_6_tmp_x;   \
        V arg16_6_tmp_xx = res16_6_tmp_x > res16_8_tmp_x ? arg16_6_tmp_x : arg16_8_tmp_x;   \
        V arg16_8_tmp_xx = res16_6_tmp_x > res16_8_tmp_x ? arg16_8_tmp_x : arg16_6_tmp_x;   \
                                                                                            \
        T res16_7_tmp_xx = res16_7_tmp_x > res16_9_tmp_x ? res16_7_tmp_x : res16_9_tmp_x;   \
        T res16_9_tmp_xx = res16_7_tmp_x > res16_9_tmp_x ? res16_9_tmp_x : res16_7_tmp_x;   \
        V arg16_7_tmp_xx = res16_7_tmp_x > res16_9_tmp_x ? arg16_7_tmp_x : arg16_9_tmp_x;   \
        V arg16_9_tmp_xx = res16_7_tmp_x > res16_9_tmp_x ? arg16_9_tmp_x : arg16_7_tmp_x;   \
                                                                                            \
        T res16_10_tmp_xx = res16_10_tmp_x > res16_12_tmp ? res16_10_tmp_x : res16_12_tmp  ;    \
        T res16_12_tmp_xx = res16_10_tmp_x > res16_12_tmp ? res16_12_tmp   : res16_10_tmp_x;    \
        V arg16_10_tmp_xx = res16_10_tmp_x > res16_12_tmp ? arg16_10_tmp_x : arg16_12_tmp  ;    \
        V arg16_12_tmp_xx = res16_10_tmp_x > res16_12_tmp ? arg16_12_tmp   : arg16_10_tmp_x;    \
                                                                                                \
        T res16_11_tmp_xx = res16_11_tmp_x > res16_13_tmp ? res16_11_tmp_x : res16_13_tmp  ;    \
        T res16_13_tmp_xx = res16_11_tmp_x > res16_13_tmp ? res16_13_tmp   : res16_11_tmp_x;    \
        V arg16_11_tmp_xx = res16_11_tmp_x > res16_13_tmp ? arg16_11_tmp_x : arg16_13_tmp  ;    \
        V arg16_13_tmp_xx = res16_11_tmp_x > res16_13_tmp ? arg16_13_tmp   : arg16_11_tmp_x;    \
                                                                                        \
        res16[1]  = res16_1_tmp > res16_2_tmp_x ? res16_1_tmp   : res16_2_tmp_x ;       \
        res16[2]  = res16_1_tmp > res16_2_tmp_x ? res16_2_tmp_x : res16_1_tmp   ;       \
        arg16[1]  = res16_1_tmp > res16_2_tmp_x ? arg16_1_tmp   : arg16_2_tmp_x ;       \
        arg16[2]  = res16_1_tmp > res16_2_tmp_x ? arg16_2_tmp_x : arg16_1_tmp   ;       \
                                                                                        \
        res16[3]  = res16_3_tmp_x > res16_4_tmp_xx ? res16_3_tmp_x  : res16_4_tmp_xx;   \
        res16[4]  = res16_3_tmp_x > res16_4_tmp_xx ? res16_4_tmp_xx : res16_3_tmp_x ;   \
        arg16[3]  = res16_3_tmp_x > res16_4_tmp_xx ? arg16_3_tmp_x  : arg16_4_tmp_xx;   \
        arg16[4]  = res16_3_tmp_x > res16_4_tmp_xx ? arg16_4_tmp_xx : arg16_3_tmp_x ;   \
                                                                                        \
        res16[5]  = res16_5_tmp_xx > res16_6_tmp_xx ? res16_5_tmp_xx : res16_6_tmp_xx;  \
        res16[6]  = res16_5_tmp_xx > res16_6_tmp_xx ? res16_6_tmp_xx : res16_5_tmp_xx;  \
        arg16[5]  = res16_5_tmp_xx > res16_6_tmp_xx ? arg16_5_tmp_xx : arg16_6_tmp_xx;  \
        arg16[6]  = res16_5_tmp_xx > res16_6_tmp_xx ? arg16_6_tmp_xx : arg16_5_tmp_xx;  \
                                                                                        \
        res16[7]  = res16_7_tmp_xx > res16_8_tmp_xx ? res16_7_tmp_xx : res16_8_tmp_xx;  \
        res16[8]  = res16_7_tmp_xx > res16_8_tmp_xx ? res16_8_tmp_xx : res16_7_tmp_xx;  \
        arg16[7]  = res16_7_tmp_xx > res16_8_tmp_xx ? arg16_7_tmp_xx : arg16_8_tmp_xx;  \
        arg16[8]  = res16_7_tmp_xx > res16_8_tmp_xx ? arg16_8_tmp_xx : arg16_7_tmp_xx;  \
                                                                                            \
        res16[9]  = res16_9_tmp_xx > res16_10_tmp_xx ? res16_9_tmp_xx  : res16_10_tmp_xx;   \
        res16[10] = res16_9_tmp_xx > res16_10_tmp_xx ? res16_10_tmp_xx : res16_9_tmp_xx ;   \
        arg16[9]  = res16_9_tmp_xx > res16_10_tmp_xx ? arg16_9_tmp_xx  : arg16_10_tmp_xx;   \
        arg16[10] = res16_9_tmp_xx > res16_10_tmp_xx ? arg16_10_tmp_xx : arg16_9_tmp_xx ;   \
                                                                                            \
        res16[11] = res16_11_tmp_xx > res16_12_tmp_xx ? res16_11_tmp_xx : res16_12_tmp_xx;  \
        res16[12] = res16_11_tmp_xx > res16_12_tmp_xx ? res16_12_tmp_xx : res16_11_tmp_xx;  \
        arg16[11] = res16_11_tmp_xx > res16_12_tmp_xx ? arg16_11_tmp_xx : arg16_12_tmp_xx;  \
        arg16[12] = res16_11_tmp_xx > res16_12_tmp_xx ? arg16_12_tmp_xx : arg16_11_tmp_xx;  \
                                                                                        \
        res16[13] = res16_13_tmp_xx > res16_14_tmp ? res16_13_tmp_xx : res16_14_tmp   ; \
        res16[14] = res16_13_tmp_xx > res16_14_tmp ? res16_14_tmp    : res16_13_tmp_xx; \
        arg16[13] = res16_13_tmp_xx > res16_14_tmp ? arg16_13_tmp_xx : arg16_14_tmp   ; \
        arg16[14] = res16_13_tmp_xx > res16_14_tmp ? arg16_14_tmp    : arg16_13_tmp_xx;

#define DPP_ARG_MERGE_16_DPP_()                                 \
        vec8_t res8_r;                                          \
        aec8_t arg8_r;                                          \
        /* only lane 0,1,2,3 contain valid data */              \
        res8_r[0] = mov_dpp_(res8[0],  ck_tile::number<0x108>{}); /* row_shl:8 */   \
        res8_r[1] = mov_dpp_(res8[1],  ck_tile::number<0x108>{}); /* row_shl:8 */   \
        res8_r[2] = mov_dpp_(res8[2],  ck_tile::number<0x108>{}); /* row_shl:8 */   \
        res8_r[3] = mov_dpp_(res8[3],  ck_tile::number<0x108>{}); /* row_shl:8 */   \
        res8_r[4] = mov_dpp_(res8[4],  ck_tile::number<0x108>{}); /* row_shl:8 */   \
        res8_r[5] = mov_dpp_(res8[5],  ck_tile::number<0x108>{}); /* row_shl:8 */   \
        res8_r[6] = mov_dpp_(res8[6],  ck_tile::number<0x108>{}); /* row_shl:8 */   \
        res8_r[7] = mov_dpp_(res8[7],  ck_tile::number<0x108>{}); /* row_shl:8 */   \
        arg8_r[0] = mov_dpp_(arg8[0],  ck_tile::number<0x108>{}); /* row_shl:8 */   \
        arg8_r[1] = mov_dpp_(arg8[1],  ck_tile::number<0x108>{}); /* row_shl:8 */   \
        arg8_r[2] = mov_dpp_(arg8[2],  ck_tile::number<0x108>{}); /* row_shl:8 */   \
        arg8_r[3] = mov_dpp_(arg8[3],  ck_tile::number<0x108>{}); /* row_shl:8 */   \
        arg8_r[4] = mov_dpp_(arg8[4],  ck_tile::number<0x108>{}); /* row_shl:8 */   \
        arg8_r[5] = mov_dpp_(arg8[5],  ck_tile::number<0x108>{}); /* row_shl:8 */   \
        arg8_r[6] = mov_dpp_(arg8[6],  ck_tile::number<0x108>{}); /* row_shl:8 */   \
        arg8_r[7] = mov_dpp_(arg8[7],  ck_tile::number<0x108>{}); /* row_shl:8 */ 


// https://en.wikipedia.org/wiki/Batcher_odd%E2%80%93even_mergesort
// TODO: this is assuming descending order sort
// result store to smem :)
template <typename T, int lanegroup_size = ck_tile::get_warp_size()>
__device__ __inline__ void warp_merge_sort_to_smem(T* smem, const T& x, ck_tile::number<lanegroup_size> = {})
{
    static_assert(sizeof(T) == 4);
    int lane_id = threadIdx.x % lanegroup_size;
    int group_id = threadIdx.x / lanegroup_size;
    T res1 = x;

    if constexpr (lanegroup_size == 2) {
        DPP_MERGE_2_DPP_();
        DPP_MERGE_2_CMP_(res1_r, res1);

        if(lane_id == 0) {
            reinterpret_cast<vec2_t*>(smem)[group_id] = res2;
        }
    } else if constexpr (lanegroup_size == 4) {
        DPP_MERGE_2_DPP_();
        DPP_MERGE_2_CMP_(res1_r, res1);
        DPP_MERGE_4_DPP_();
        DPP_MERGE_4_CMP_(res2_r, res2);

        if(lane_id == 0) {
            reinterpret_cast<vec4_t*>(smem)[group_id] = res4;
        }
    } else if constexpr (lanegroup_size == 8) {
        DPP_MERGE_2_DPP_();
        DPP_MERGE_2_CMP_(res1_r, res1);
        DPP_MERGE_4_DPP_();
        DPP_MERGE_4_CMP_(res2_r, res2);
        DPP_MERGE_8_DPP_();
        DPP_MERGE_8_CMP_(res4_r, res4);

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
        DPP_MERGE_2_DPP_();
        DPP_MERGE_2_CMP_(res1_r, res1);
        DPP_MERGE_4_DPP_();
        DPP_MERGE_4_CMP_(res2_r, res2);
        DPP_MERGE_8_DPP_();
        DPP_MERGE_8_CMP_(res4_r, res4);
        DPP_MERGE_16_DPP_();
        DPP_MERGE_16_CMP_(res8_r, res8);

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
}

template <typename T, int lanegroup_size = ck_tile::get_warp_size()>
__device__ __inline__ auto warp_merge_sort_to_reg(const T& x, ck_tile::number<lanegroup_size> = {})
{
    static_assert(sizeof(T) == 4);
    T res1 = x;

    if constexpr (lanegroup_size == 2) {
        DPP_MERGE_2_DPP_();
        DPP_MERGE_2_CMP_(res1, res1_r);
        return res2;
    } else if constexpr (lanegroup_size == 4) {
        DPP_MERGE_2_DPP_();
        DPP_MERGE_2_CMP_(res1_r, res1);
        DPP_MERGE_4_DPP_();
        DPP_MERGE_4_CMP_(res2_r, res2);
        return res4;
    } else if constexpr (lanegroup_size == 8) {
        DPP_MERGE_2_DPP_();
        DPP_MERGE_2_CMP_(res1_r, res1);
        DPP_MERGE_4_DPP_();
        DPP_MERGE_4_CMP_(res2_r, res2);
        DPP_MERGE_8_DPP_();
        DPP_MERGE_8_CMP_(res4_r, res4);
        // TODO: only lane:1,2,3,4 within 8 lanes does not have correct result !
        return res8;
    } else if constexpr (lanegroup_size == 16) {
        DPP_MERGE_2_DPP_();
        DPP_MERGE_2_CMP_(res1_r, res1);
        DPP_MERGE_4_DPP_();
        DPP_MERGE_4_CMP_(res2_r, res2);
        DPP_MERGE_8_DPP_();
        DPP_MERGE_8_CMP_(res4_r, res4);
        DPP_MERGE_16_DPP_();
        DPP_MERGE_16_CMP_(res8_r, res8);
        // TODO: only lane:1,2,3,4 within 16 lanes does not have correct result !
        return res16;
    } else {
        return 0;
    }
}

// sort based on x, and sort v
template <typename T, typename V, int lanegroup_size = ck_tile::get_warp_size()>
__device__ __inline__ auto warp_arg_merge_sort_to_reg(const T& x, const V& v, ck_tile::number<lanegroup_size> = {})
{
    static_assert(sizeof(T) == 4);
    T res1 = x;
    V arg1 = v;

    if constexpr (lanegroup_size == 2) {
        DPP_ARG_MERGE_2_DPP_();
        DPP_ARG_MERGE_2_CMP_(res1_r, res1, arg1_r, arg1);
        return ck_tile::make_tuple(res2, arg2);
    } else if constexpr (lanegroup_size == 4) {
        DPP_ARG_MERGE_2_DPP_();
        DPP_ARG_MERGE_2_CMP_(res1_r, res1, arg1_r, arg1);
        DPP_ARG_MERGE_4_DPP_();
        DPP_ARG_MERGE_4_CMP_(res2_r, res2, arg2_r, arg2);
        return ck_tile::make_tuple(res4, arg4);
    } else if constexpr (lanegroup_size == 8) {
        DPP_ARG_MERGE_2_DPP_();
        DPP_ARG_MERGE_2_CMP_(res1_r, res1, arg1_r, arg1);
        DPP_ARG_MERGE_4_DPP_();
        DPP_ARG_MERGE_4_CMP_(res2_r, res2, arg2_r, arg2);
        DPP_ARG_MERGE_8_DPP_();
        DPP_ARG_MERGE_8_CMP_(res4_r, res4, arg4_r, arg4);
        // TODO: only lane:1,2,3,4 within 8 lanes does not have correct result !
        return ck_tile::make_tuple(res8, arg8);
    } else if constexpr (lanegroup_size == 16) {
        DPP_ARG_MERGE_2_DPP_();
        DPP_ARG_MERGE_2_CMP_(res1_r, res1, arg1_r, arg1);
        DPP_ARG_MERGE_4_DPP_();
        DPP_ARG_MERGE_4_CMP_(res2_r, res2, arg2_r, arg2);
        DPP_ARG_MERGE_8_DPP_();
        DPP_ARG_MERGE_8_CMP_(res4_r, res4, arg4_r, arg4);
        DPP_ARG_MERGE_16_DPP_();
        DPP_ARG_MERGE_16_CMP_(res8_r, res8, arg8_r, arg8);
        // TODO: only lane:1,2,3,4 within 16 lanes does not have correct result !
        return ck_tile::make_tuple(res16, arg16);
    } else {
        return 0;
    }
}

#undef DPP_MERGE_2_DPP_
#undef DPP_MERGE_2_CMP_
#undef DPP_MERGE_4_DPP_
#undef DPP_MERGE_4_CMP_
#undef DPP_MERGE_8_DPP_
#undef DPP_MERGE_8_CMP_
#undef DPP_MERGE_16_DPP_
#undef DPP_MERGE_16_CMP_
#undef DPP_ARG_MERGE_2_DPP_
#undef DPP_ARG_MERGE_2_CMP_
#undef DPP_ARG_MERGE_4_DPP_
#undef DPP_ARG_MERGE_4_CMP_
#undef DPP_ARG_MERGE_8_DPP_
#undef DPP_ARG_MERGE_8_CMP_
#undef DPP_ARG_MERGE_16_DPP_
#undef DPP_ARG_MERGE_16_CMP_

template<typename T, int wave_size = 64, int lanegroup_size = 64>
__global__ void warp_sort_kernel_smem(T* i_ptr, T* o_ptr)
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

template<typename T, int wave_size = 64, int lanegroup_size = 64>
__global__ void warp_sort_kernel_reg(T* i_ptr, T* o_ptr)
{
    T data = -INFINITY;
    if(threadIdx.x < lanegroup_size) {
        data = i_ptr[threadIdx.x];
    }

    auto res = warp_merge_sort_to_reg(data, ck_tile::number<lanegroup_size>{});
    if(threadIdx.x == 0) {
        using final_vec_t = ck_tile::ext_vector_t<T, lanegroup_size>;
        * reinterpret_cast<final_vec_t*>(o_ptr) = res;
    }
}
#if 0
template<typename T, typename V, int wave_size = 64, int lanegroup_size = 64>
__global__ void warp_arg_sort_kernel_smem(T* i_ptr, V* ai_ptr, T* o_ptr, T* ao_ptr)
{
    __shared__ T smem[2 * wave_size / lanegroup_size];
    T data = -INFINITY;
    V valu = 0;
    if(threadIdx.x < lanegroup_size) {
        data = i_ptr[threadIdx.x];
        valu = ai_ptr[threadIdx.x];
    }

    warp_arg_merge_sort_to_smem(smem, data, valu, ck_tile::number<lanegroup_size>{});

    __syncthreads();

    T sorted = smem[threadIdx.x];   // ignore out-of-bound check
    V asorted = smem[lanegroup_size + threadIdx.x]

    if(threadIdx.x < lanegroup_size) {
        o_ptr[threadIdx.x] = sorted;
        ao_Ptr[threadIdx.x] = asorted;
    }
}
#endif

template<typename T, typename V, int wave_size = 64, int lanegroup_size = 64>
__global__ void warp_arg_sort_kernel_reg(T* i_ptr, V* ai_ptr, T* o_ptr, V* ao_ptr)
{
    T data = -INFINITY;
    V valu = 0;
    if(threadIdx.x < lanegroup_size) {
        data = i_ptr[threadIdx.x];
        valu = ai_ptr[threadIdx.x];
    }

    auto [res, arg] = warp_arg_merge_sort_to_reg(data, valu, ck_tile::number<lanegroup_size>{});
    if(threadIdx.x == 0) {
        using final_vec_t = ck_tile::ext_vector_t<T, lanegroup_size>;
        using final_arg_t = ck_tile::ext_vector_t<V, lanegroup_size>;
        *reinterpret_cast<final_vec_t*>(o_ptr) = res;
        *reinterpret_cast<final_arg_t*>(ao_ptr) = arg;
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

template<typename T, typename V, int kid, int wave_size = 64, int lanegroup_size = 64>
void run()
{
    T * input = reinterpret_cast<T*>(malloc(sizeof(T) * lanegroup_size));
    T * output = reinterpret_cast<T*>(malloc(sizeof(T) * lanegroup_size));
    V * ai     = reinterpret_cast<V*>(malloc(sizeof(V) * lanegroup_size));
    V * ao    = reinterpret_cast<V*>(malloc(sizeof(V) * lanegroup_size));

    T *dev_i, *dev_o;
    V * dev_ai, *dev_ao;

    HIP_CALL(hipMalloc(&dev_i, sizeof(T) * lanegroup_size));
    HIP_CALL(hipMalloc(&dev_o, sizeof(T) * lanegroup_size));
    HIP_CALL(hipMalloc(&dev_ai, sizeof(V) * lanegroup_size));
    HIP_CALL(hipMalloc(&dev_ao, sizeof(V) * lanegroup_size));

    rand_vector(input, lanegroup_size);
    for(int i = 0; i < lanegroup_size; i++) {
        ai[i] = static_cast<V>(i);
    }

    HIP_CALL(hipMemcpy(dev_i, input, sizeof(T) * lanegroup_size, hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(dev_ai, ai, sizeof(V) * lanegroup_size, hipMemcpyHostToDevice));

    auto gx = dim3(1);
    auto bx = dim3(wave_size);

    if constexpr(kid == 0)
        warp_sort_kernel_smem<T, wave_size, lanegroup_size><<<gx, bx>>>(dev_i, dev_o);
    else if constexpr(kid == 1)
        warp_sort_kernel_reg<T, wave_size, lanegroup_size><<<gx, bx>>>(dev_i, dev_o);

    else if constexpr(kid == 3)
        warp_arg_sort_kernel_reg<T, V, wave_size, lanegroup_size><<<gx, bx>>>(dev_i, dev_ai, dev_o, dev_ao);

    HIP_CALL(hipMemcpy(output, dev_o, sizeof(T) * lanegroup_size, hipMemcpyDeviceToHost));

    printf("[k%d|origin-%d]", kid, lanegroup_size);
    for(int i = 0; i < lanegroup_size; i++) {
        printf("%.3f ", input[i]);
    }
    printf("\n");
    printf("[k%d|sorted-%d]", kid, lanegroup_size);
    for(int i = 0; i < lanegroup_size; i++) {
        printf("%.3f ", output[i]);
    }
    printf("\n");
    if constexpr (kid == 2 || kid == 3) {
        HIP_CALL(hipMemcpy(ao, dev_ao, sizeof(V) * lanegroup_size, hipMemcpyDeviceToHost));
        printf("         ");
        for(int i = 0; i < lanegroup_size; i++) {
            printf("%5d ", ao[i]);
        }
        printf("\n");
        {
            struct bundle_type {
                T x;
                V v;
            };

            std::vector<bundle_type> arg_vec;
            for(int i =0 ; i < lanegroup_size; i++) {
                arg_vec.push_back({input[i], ai[i] });
            }
            std::sort(arg_vec.begin(), arg_vec.end(), [&](auto a, auto b){return a.x > b.x; });

            int invalid_idx_cnt = [&] (){
                int c_ = 0;
                for(int i = 0; i < lanegroup_size; i++) {
                    if(arg_vec[i].v != ao[i])
                        c_++;
                }
                return c_;
            }();
            int duplicated_key_pair = 0;
            if(invalid_idx_cnt != 0) {
                // need to filter out duplication case, but key value are different
                // due to sorting order
                // e.g
                // value:[0.5, 0.2, 0.5, 0.7, 0.4]
                // key  :[  0,   1,   2,   3,   4]
                //
                // sorting
                //
                // value:[0.7, 0.5, 0.5, 0.4, 0.2]
                // key  :[  3,   0,   2,   4,   1]  => case-1's key
                // key  :[  3,   2,   0,   4,   1]  => case-2's key
                //
                // case-1/case-2 should both consider correct
                std::list<int> different_keys;
                for(int i = 0; i < lanegroup_size; i++) {
                    if(arg_vec[i].v != ao[i])
                        different_keys.push_back(arg_vec[i].v);
                }
                if(different_keys.size() % 2 == 0) {
                    // in case any invalid case that is not paired keys
                    while(true) {
                        auto target_key = different_keys.front();
                        different_keys.pop_front();
                        bool found_duplication = false;
                        for(auto itr = different_keys.begin(); itr != different_keys.end(); itr++) {
                            // printf("xxx %d:%f(%x), %d:%f(%x)\n", target_key, input[target_key], *reinterpret_cast<uint32_t*>(&input[target_key]),
                            //                                 *itr, input[*itr],*reinterpret_cast<uint32_t*>(&input[*itr]) );
                            if(input[target_key] == input[*itr]) {
                                found_duplication = true;
                                duplicated_key_pair++;
                                different_keys.erase(itr);
                                break;
                            }
                        }
                        if(!found_duplication)
                            break;  // invalid return
                        if(different_keys.size() == 0)
                            break;
                    }
                }
            }
            printf("         ");
            for(int i = 0; i < lanegroup_size; i++) {
                printf("%5d ", arg_vec[i].v);
            }
            printf("%s", invalid_idx_cnt == 0 ? "[y]" : (invalid_idx_cnt == 2*duplicated_key_pair ? "[y, duplicated value]" : "[n]"));
            printf("\n");
        }
    }
    bool is_ordered = check_ordered(output, lanegroup_size);
    printf("-------------------------------------- %s\n", is_ordered?"ordered":"non-order");

    free(input);
    free(output);
    free(ai);
    free(ao);

    HIP_CALL(hipFree(dev_i));
    HIP_CALL(hipFree(dev_o));
    HIP_CALL(hipFree(dev_ai));
    HIP_CALL(hipFree(dev_ao));
}

int main(int argc, char ** argv)
{
    printf("[TEST SORT TO SMEM]___________________________________________\n");
    run<float, int, 0, 64, 2>();
    run<float, int, 0, 64, 4>();
    run<float, int, 0, 64, 8>();
    run<float, int, 0, 64, 16>();

    printf("[TEST SORT TO REG]____________________________________________\n");
    run<float, int, 1, 64, 2>();
    run<float, int, 1, 64, 4>();
    run<float, int, 1, 64, 8>();
    run<float, int, 1, 64, 16>();

    printf("[TEST ARG SORT TO REG]________________________________________\n");
    run<float, int, 3, 64, 2>();
    run<float, int, 3, 64, 4>();
    run<float, int, 3, 64, 8>();
    run<float, int, 3, 64, 16>();
    // run<float, 64, 8>();
}
