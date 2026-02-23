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

#ifndef FMT_LIMIT
#define FMT_LIMIT 1
#define FMT_LIMIT_MAX 16
#endif

template <typename T, int dpp_i, int row_mask = 0xf, int bank_mask = 0xf, bool bound_ctrl = true>
__device__ __inline__ T mov_dpp_(T x,
                                 ck_tile::number<dpp_i>,
                                 ck_tile::number<row_mask>          = {},
                                 ck_tile::number<bank_mask>         = {},
                                 ck_tile::bool_constant<bound_ctrl> = {})
{
    static_assert(sizeof(T) == 4);
    return __builtin_bit_cast(
        T,
        // __builtin_amdgcn_update_dpp(0,__builtin_bit_cast(int, x),
        __builtin_amdgcn_mov_dpp(
            __builtin_bit_cast(int, x), dpp_i, row_mask, bank_mask, bound_ctrl));
}

template<typename O, typename T, int dpp_i, int row_mask = 0xf, int bank_mask = 0xf, bool bound_ctrl = true>
__device__ __inline__ T upd_dpp_(const O& old, T x, ck_tile::number<dpp_i>,
                                ck_tile::number<row_mask>          = {},
                                 ck_tile::number<bank_mask>         = {},
                                 ck_tile::bool_constant<bound_ctrl> = {}) {
    static_assert(sizeof(T) == 4);
    return __builtin_bit_cast(T,
                        __builtin_amdgcn_update_dpp(__builtin_bit_cast(int, old), __builtin_bit_cast(int, x),
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

template<typename T>
__device__ __inline__ T dev_med3_(const T&a, const T&b, const T&c)
{
    if constexpr(std::is_same_v<T, float>) {
        return __builtin_amdgcn_fmed3f(a, b, c);
    }
#if 0
    else if constexpr(std::is_same_v<T, half>) {

    }
#endif
    auto max_0 = dev_max_(a, b);
    auto min_0 = dev_min_(a, b);
    return dev_min_(max_0, dev_max_(min_0, c));
}

// swap lo/hi half within a lanegroup
template <typename T, int lanegroup_size>
__device__ __inline__ auto warp_swap_(const T& x, int lane_idx, ck_tile::number<lanegroup_size> = {})
{
    if constexpr (lanegroup_size == 1) {
        // just return same value if groupsize is 1(no dpp, no permute)
        return x;
    }
    if constexpr (lanegroup_size == 2) {
        return mov_dpp_(x, ck_tile::number<0xb1>{}); /*quad_perm:[1,0,3,2]*/
    } else if constexpr (lanegroup_size == 4) {
        return mov_dpp_(x,  ck_tile::number<0x4e>{}); /*quad_perm:[2,3,0,1]*/
    } else if constexpr(lanegroup_size == 8) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wuninitialized"
        // this builtin require the old value, and
        // will generate a v_mov_b32 vxxx [old] before cvt, which result in unwanted ISA
        // so we prepare an uninitialized variable purposely, and turn off the warning
        //
        // note the 2nd operation, we need it as old value to prevent compiler optimize out for multi assignement
        //
        // NOTE: we can also use volatile, but compiler will generate scratch (it's memory operation?)
        T r;
        r = upd_dpp_(r, x, ck_tile::number<260>{}, ck_tile::number<0xf>{}, ck_tile::number<0b0101>{}); /*row_shl:4*/
        r = upd_dpp_(r, x, ck_tile::number<276>{}, ck_tile::number<0xf>{}, ck_tile::number<0b1010>{}); /*row_shr:4*/
#pragma clang diagnostic pop
        return  r;
    } else if constexpr(lanegroup_size == 16) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wuninitialized"
        T r;
        r = upd_dpp_(r, x, ck_tile::number<264>{}, ck_tile::number<0xf>{}, ck_tile::number<0b0011>{}); /*row_shl:8*/
        r = upd_dpp_(r, x, ck_tile::number<280>{}, ck_tile::number<0xf>{}, ck_tile::number<0b1100>{}); /*row_shr:8*/
#pragma clang diagnostic pop
        return r;
    } else if constexpr(lanegroup_size == 32) {
        return __shfl(x, lane_idx ^ 16);    // consume LDS
    } else if constexpr(lanegroup_size == 64) {
        return __shfl(x, lane_idx ^ 32);    // consume LDS
    }
}

// This is the core function to build the construct/combine stage of bitonic merge sor
template <typename T, int lanegroup_size = ck_tile::get_warp_size(), int is_descending = 1>
__device__ __inline__ auto warp_bitonic_merge_sort_step_(const T& x, const T& y, int lane_idx, int twiddle, ck_tile::number<lanegroup_size> = {}, ck_tile::number<is_descending> = {})
{
    auto guard = [&](auto div_) {
            if constexpr(is_descending) return  (((lane_idx / div_.value) & 1) ^ twiddle) == 0 ? INFINITY : -INFINITY;
            else return                         (((lane_idx / div_.value) & 1) ^ twiddle) == 0 ? -INFINITY : INFINITY;
    };

    // compare and swap within lanegroup_size lo/hi half
    auto g = guard(ck_tile::number<lanegroup_size / 2>{});
    return dev_med3_(x, y, g);
}

// this version the return value will be stored into per-lane register
template <typename T, int lanegroup_size = ck_tile::get_warp_size(), int is_descending = 1>
__device__ __inline__ auto warp_bitonic_merge_sort_build(const T& x, int lane_idx, ck_tile::number<lanegroup_size> = {}, ck_tile::number<is_descending> = {})
{
    if constexpr (lanegroup_size == 2) {
        // TODO:!!! if 2, always use combine, not build
        // here we just return the original value
        return x;
    }
    else if constexpr (lanegroup_size == 4) {
        T y =  warp_swap_(x, lane_idx, ck_tile::number<2>{});
        T o = warp_bitonic_merge_sort_step_(x, y, lane_idx, (lane_idx / 2) & 1 , ck_tile::number<2>{}, ck_tile::number<is_descending>{});
        return o;
    }
    else if constexpr (lanegroup_size == 8) {
        T y =  warp_swap_(x, lane_idx, ck_tile::number<2>{});
        T o = warp_bitonic_merge_sort_step_(x, y, lane_idx, (lane_idx / 2) & 1 , ck_tile::number<2>{}, ck_tile::number<is_descending>{});

        y   = warp_swap_(o, lane_idx, ck_tile::number<4>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 4) & 1 , ck_tile::number<4>{}, ck_tile::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, ck_tile::number<2>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 4) & 1 , ck_tile::number<2>{}, ck_tile::number<is_descending>{});
        return o;
    }
    else if constexpr (lanegroup_size == 16) {
        T y =  warp_swap_(x, lane_idx, ck_tile::number<2>{});
        T o = warp_bitonic_merge_sort_step_(x, y, lane_idx, (lane_idx / 2) & 1 , ck_tile::number<2>{}, ck_tile::number<is_descending>{});

        y   = warp_swap_(o, lane_idx, ck_tile::number<4>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 4) & 1 , ck_tile::number<4>{}, ck_tile::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, ck_tile::number<2>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 4) & 1 , ck_tile::number<2>{}, ck_tile::number<is_descending>{});

        y   = warp_swap_(o, lane_idx, ck_tile::number<8>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 8) & 1 , ck_tile::number<8>{}, ck_tile::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, ck_tile::number<4>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 8) & 1 , ck_tile::number<4>{}, ck_tile::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, ck_tile::number<2>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 8) & 1 , ck_tile::number<2>{}, ck_tile::number<is_descending>{});
        return o;
    }
    else if constexpr (lanegroup_size == 32) {
        T y =  warp_swap_(x, lane_idx, ck_tile::number<2>{});
        T o = warp_bitonic_merge_sort_step_(x, y, lane_idx, (lane_idx / 2) & 1 , ck_tile::number<2>{}, ck_tile::number<is_descending>{});

        y   = warp_swap_(o, lane_idx, ck_tile::number<4>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 4) & 1 , ck_tile::number<4>{}, ck_tile::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, ck_tile::number<2>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 4) & 1 , ck_tile::number<2>{}, ck_tile::number<is_descending>{});

        y   = warp_swap_(o, lane_idx, ck_tile::number<8>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 8) & 1 , ck_tile::number<8>{}, ck_tile::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, ck_tile::number<4>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 8) & 1 , ck_tile::number<4>{}, ck_tile::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, ck_tile::number<2>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 8) & 1 , ck_tile::number<2>{}, ck_tile::number<is_descending>{});

        y   = warp_swap_(o, lane_idx, ck_tile::number<16>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 16) & 1 , ck_tile::number<16>{}, ck_tile::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, ck_tile::number<8>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 16) & 1 , ck_tile::number<8>{}, ck_tile::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, ck_tile::number<4>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 16) & 1 , ck_tile::number<4>{}, ck_tile::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, ck_tile::number<2>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 16) & 1 , ck_tile::number<2>{}, ck_tile::number<is_descending>{});
        return o;
    }
    else if constexpr (lanegroup_size == 64) {
        T y =  warp_swap_(x, lane_idx, ck_tile::number<2>{});
        T o = warp_bitonic_merge_sort_step_(x, y, lane_idx, (lane_idx / 2) & 1 , ck_tile::number<2>{}, ck_tile::number<is_descending>{});

        y   = warp_swap_(o, lane_idx, ck_tile::number<4>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 4) & 1 , ck_tile::number<4>{}, ck_tile::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, ck_tile::number<2>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 4) & 1 , ck_tile::number<2>{}, ck_tile::number<is_descending>{});

        y   = warp_swap_(o, lane_idx, ck_tile::number<8>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 8) & 1 , ck_tile::number<8>{}, ck_tile::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, ck_tile::number<4>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 8) & 1 , ck_tile::number<4>{}, ck_tile::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, ck_tile::number<2>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 8) & 1 , ck_tile::number<2>{}, ck_tile::number<is_descending>{});

        y   = warp_swap_(o, lane_idx, ck_tile::number<16>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 16) & 1 , ck_tile::number<16>{}, ck_tile::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, ck_tile::number<8>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 16) & 1 , ck_tile::number<8>{}, ck_tile::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, ck_tile::number<4>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 16) & 1 , ck_tile::number<4>{}, ck_tile::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, ck_tile::number<2>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 16) & 1 , ck_tile::number<2>{}, ck_tile::number<is_descending>{});

        y   = warp_swap_(o, lane_idx, ck_tile::number<32>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 32) & 1 , ck_tile::number<32>{}, ck_tile::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, ck_tile::number<16>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 32) & 1 , ck_tile::number<16>{}, ck_tile::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, ck_tile::number<8>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 32) & 1 , ck_tile::number<8>{}, ck_tile::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, ck_tile::number<4>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 32) & 1 , ck_tile::number<4>{}, ck_tile::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, ck_tile::number<2>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 32) & 1 , ck_tile::number<2>{}, ck_tile::number<is_descending>{});
        return o;
    }
    else if constexpr (lanegroup_size == 128) {
        T y =  warp_swap_(x, lane_idx, ck_tile::number<2>{});
        T o = warp_bitonic_merge_sort_step_(x, y, lane_idx, (lane_idx / 2) & 1 , ck_tile::number<2>{}, ck_tile::number<is_descending>{});

        y   = warp_swap_(o, lane_idx, ck_tile::number<4>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 4) & 1 , ck_tile::number<4>{}, ck_tile::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, ck_tile::number<2>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 4) & 1 , ck_tile::number<2>{}, ck_tile::number<is_descending>{});

        y   = warp_swap_(o, lane_idx, ck_tile::number<8>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 8) & 1 , ck_tile::number<8>{}, ck_tile::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, ck_tile::number<4>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 8) & 1 , ck_tile::number<4>{}, ck_tile::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, ck_tile::number<2>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 8) & 1 , ck_tile::number<2>{}, ck_tile::number<is_descending>{});

        y   = warp_swap_(o, lane_idx, ck_tile::number<16>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 16) & 1 , ck_tile::number<16>{}, ck_tile::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, ck_tile::number<8>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 16) & 1 , ck_tile::number<8>{}, ck_tile::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, ck_tile::number<4>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 16) & 1 , ck_tile::number<4>{}, ck_tile::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, ck_tile::number<2>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 16) & 1 , ck_tile::number<2>{}, ck_tile::number<is_descending>{});

        y   = warp_swap_(o, lane_idx, ck_tile::number<32>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 32) & 1 , ck_tile::number<32>{}, ck_tile::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, ck_tile::number<16>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 32) & 1 , ck_tile::number<16>{}, ck_tile::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, ck_tile::number<8>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 32) & 1 , ck_tile::number<8>{}, ck_tile::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, ck_tile::number<4>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 32) & 1 , ck_tile::number<4>{}, ck_tile::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, ck_tile::number<2>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 32) & 1 , ck_tile::number<2>{}, ck_tile::number<is_descending>{});

        y   = warp_swap_(o, lane_idx, ck_tile::number<64>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 64) & 1 , ck_tile::number<64>{}, ck_tile::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, ck_tile::number<32>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 64) & 1 , ck_tile::number<32>{}, ck_tile::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, ck_tile::number<16>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 64) & 1 , ck_tile::number<16>{}, ck_tile::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, ck_tile::number<8>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 64) & 1 , ck_tile::number<8>{}, ck_tile::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, ck_tile::number<4>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 64) & 1 , ck_tile::number<4>{}, ck_tile::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, ck_tile::number<2>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 64) & 1 , ck_tile::number<2>{}, ck_tile::number<is_descending>{});
        return o;
    }
}

// this version the return value will be stored into per-lane register
template <typename T, int lanegroup_size = ck_tile::get_warp_size(), int is_descending = 1>
__device__ __inline__ auto warp_bitonic_merge_sort_combine(const T& x, const T& y, int lane_idx, int twiddle, ck_tile::number<lanegroup_size> = {}, ck_tile::number<is_descending> = {})
{
    if constexpr (lanegroup_size == 2) {
        T o = warp_bitonic_merge_sort_step_(x, y, lane_idx, twiddle, ck_tile::number<2>{}, ck_tile::number<is_descending>{});
        return o;
    }
    else if constexpr (lanegroup_size == 4) {
        T o = warp_bitonic_merge_sort_step_(x, y, lane_idx, twiddle, ck_tile::number<4>{}, ck_tile::number<is_descending>{});
        T z = warp_swap_(o, lane_idx, ck_tile::number<2>{});
        o   = warp_bitonic_merge_sort_step_(o, z, lane_idx, twiddle, ck_tile::number<2>{}, ck_tile::number<is_descending>{});
        return o;
    }
    else if constexpr (lanegroup_size == 8) {
        T o = warp_bitonic_merge_sort_step_(x, y, lane_idx, twiddle, ck_tile::number<8>{}, ck_tile::number<is_descending>{});
        T z = warp_swap_(o, lane_idx, ck_tile::number<4>{});

        o   = warp_bitonic_merge_sort_step_(o, z, lane_idx, twiddle, ck_tile::number<4>{}, ck_tile::number<is_descending>{});
        z   = warp_swap_(o, lane_idx, ck_tile::number<2>{});
        o   = warp_bitonic_merge_sort_step_(o, z, lane_idx, twiddle, ck_tile::number<2>{}, ck_tile::number<is_descending>{});
        return o;
    }
    else if constexpr (lanegroup_size == 16) {
        T o = warp_bitonic_merge_sort_step_(x, y, lane_idx, twiddle, ck_tile::number<16>{}, ck_tile::number<is_descending>{});
        T z = warp_swap_(o, lane_idx, ck_tile::number<8>{});

        o   = warp_bitonic_merge_sort_step_(o, z, lane_idx, twiddle, ck_tile::number<8>{}, ck_tile::number<is_descending>{});
        z   = warp_swap_(o, lane_idx, ck_tile::number<4>{});

        o   = warp_bitonic_merge_sort_step_(o, z, lane_idx, twiddle, ck_tile::number<4>{}, ck_tile::number<is_descending>{});
        z   = warp_swap_(o, lane_idx, ck_tile::number<2>{});
        o   = warp_bitonic_merge_sort_step_(o, z, lane_idx, twiddle, ck_tile::number<2>{}, ck_tile::number<is_descending>{});
        return o;
    }
    else if constexpr (lanegroup_size == 32) {
        T o = warp_bitonic_merge_sort_step_(x, y, lane_idx, twiddle, ck_tile::number<32>{}, ck_tile::number<is_descending>{});
        T z = warp_swap_(o, lane_idx, ck_tile::number<16>{});

        o   = warp_bitonic_merge_sort_step_(o, z, lane_idx, twiddle, ck_tile::number<16>{}, ck_tile::number<is_descending>{});
        z   = warp_swap_(o, lane_idx, ck_tile::number<8>{});

        o   = warp_bitonic_merge_sort_step_(o, z, lane_idx, twiddle, ck_tile::number<8>{}, ck_tile::number<is_descending>{});
        z   = warp_swap_(o, lane_idx, ck_tile::number<4>{});

        o   = warp_bitonic_merge_sort_step_(o, z, lane_idx, twiddle, ck_tile::number<4>{}, ck_tile::number<is_descending>{});
        z   = warp_swap_(o, lane_idx, ck_tile::number<2>{});
        o   = warp_bitonic_merge_sort_step_(o, z, lane_idx, twiddle, ck_tile::number<2>{}, ck_tile::number<is_descending>{});
        return o;
    }
    else if constexpr (lanegroup_size == 64) {
        T o = warp_bitonic_merge_sort_step_(x, y, lane_idx, twiddle, ck_tile::number<64>{}, ck_tile::number<is_descending>{});
        T z = warp_swap_(o, lane_idx, ck_tile::number<32>{});

        o   = warp_bitonic_merge_sort_step_(o, z, lane_idx, twiddle, ck_tile::number<32>{}, ck_tile::number<is_descending>{});
        z   = warp_swap_(o, lane_idx, ck_tile::number<16>{});

        o   = warp_bitonic_merge_sort_step_(o, z, lane_idx, twiddle, ck_tile::number<16>{}, ck_tile::number<is_descending>{});
        z   = warp_swap_(o, lane_idx, ck_tile::number<8>{});

        o   = warp_bitonic_merge_sort_step_(o, z, lane_idx, twiddle, ck_tile::number<8>{}, ck_tile::number<is_descending>{});
        z   = warp_swap_(o, lane_idx, ck_tile::number<4>{});

        o   = warp_bitonic_merge_sort_step_(o, z, lane_idx, twiddle, ck_tile::number<4>{}, ck_tile::number<is_descending>{});
        z   = warp_swap_(o, lane_idx, ck_tile::number<2>{});
        o   = warp_bitonic_merge_sort_step_(o, z, lane_idx, twiddle, ck_tile::number<2>{}, ck_tile::number<is_descending>{});
        return o;
    }
    else if constexpr (lanegroup_size == 128) {
        T o = warp_bitonic_merge_sort_step_(x, y, lane_idx, twiddle, ck_tile::number<128>{}, ck_tile::number<is_descending>{});
        T z = warp_swap_(o, lane_idx, ck_tile::number<64>{});

        o   = warp_bitonic_merge_sort_step_(o, z, lane_idx, twiddle, ck_tile::number<64>{}, ck_tile::number<is_descending>{});
        z   = warp_swap_(o, lane_idx, ck_tile::number<32>{});

        o   = warp_bitonic_merge_sort_step_(o, z, lane_idx, twiddle, ck_tile::number<32>{}, ck_tile::number<is_descending>{});
        z   = warp_swap_(o, lane_idx, ck_tile::number<16>{});

        o   = warp_bitonic_merge_sort_step_(o, z, lane_idx, twiddle, ck_tile::number<16>{}, ck_tile::number<is_descending>{});
        z   = warp_swap_(o, lane_idx, ck_tile::number<8>{});

        o   = warp_bitonic_merge_sort_step_(o, z, lane_idx, twiddle, ck_tile::number<8>{}, ck_tile::number<is_descending>{});
        z   = warp_swap_(o, lane_idx, ck_tile::number<4>{});

        o   = warp_bitonic_merge_sort_step_(o, z, lane_idx, twiddle, ck_tile::number<4>{}, ck_tile::number<is_descending>{});
        z   = warp_swap_(o, lane_idx, ck_tile::number<2>{});
        o   = warp_bitonic_merge_sort_step_(o, z, lane_idx, twiddle, ck_tile::number<2>{}, ck_tile::number<is_descending>{});
        return o;
    }
}
// this version the return value will be stored into per-lane register
template <typename T, int lanegroup_size = ck_tile::get_warp_size(), int is_descending = 1>
__device__ __inline__ auto warp_bitonic_merge_sort_to_reg(const T& x, ck_tile::number<lanegroup_size> = {}, ck_tile::number<is_descending> = {})
{
    static_assert(lanegroup_size <= ck_tile::get_warp_size());
    int lane_idx = threadIdx.x;
    T c = warp_bitonic_merge_sort_build(x, lane_idx, ck_tile::number<lanegroup_size>{}, ck_tile::number<is_descending>{});
    T r = warp_swap_(c, lane_idx, ck_tile::number<lanegroup_size>{});
    // if(threadIdx.x < lanegroup_size) printf("[%2d] c:%f, r:%f\n", threadIdx.x, c, r);
    T o = warp_bitonic_merge_sort_combine(c, r, lane_idx, 0, ck_tile::number<lanegroup_size>{}, ck_tile::number<is_descending>{});
    return o;
}

template <typename T, int lanegroup_size = ck_tile::get_warp_size(), int is_descending = 1>
__device__ __inline__ auto block_bitonic_merge_sort_to_reg(void* smem, const T& x, ck_tile::number<lanegroup_size> = {}, ck_tile::number<is_descending> = {})
{
    // need make sure smem before this function is ready to use
    // need guarantee smem usage, will not if...else... write smem inside this kernel
    // smem require sizeof(T) * lanegroup_size
    static_assert(lanegroup_size > ck_tile::get_warp_size());
    int lane_idx = threadIdx.x;
    if constexpr (lanegroup_size == 128) {
        T c = warp_bitonic_merge_sort_build(x, lane_idx, ck_tile::number<128>{}, ck_tile::number<is_descending>{});
    
        reinterpret_cast<T*>(smem)[lane_idx] = c;
        __syncthreads();
        T r = reinterpret_cast<T*>(smem)[lane_idx ^ 64];

        T o = warp_bitonic_merge_sort_combine(c, r, lane_idx, 0, ck_tile::number<128>{}, ck_tile::number<is_descending>{});
        return o;
    }
    else if constexpr (lanegroup_size == 256) {
        T c = warp_bitonic_merge_sort_build(x, lane_idx, ck_tile::number<128>{}, ck_tile::number<is_descending>{});

        reinterpret_cast<T*>(smem)[lane_idx] = c;
        __syncthreads();
        T r = reinterpret_cast<T*>(smem)[lane_idx ^ 64];

        // using combine to simulate build stage
        T o  = warp_bitonic_merge_sort_combine(c, r, lane_idx, (lane_idx / 128) & 1, ck_tile::number<128>{}, ck_tile::number<is_descending>{});

        // start to combine
        __syncthreads();
        reinterpret_cast<T*>(smem)[lane_idx] = o;
        __syncthreads();
        r   = reinterpret_cast<T*>(smem)[lane_idx ^ 128];
        c   = warp_bitonic_merge_sort_step_(o, r, lane_idx, 0, ck_tile::number<256>{}, ck_tile::number<is_descending>{});

        __syncthreads();
        reinterpret_cast<T*>(smem)[lane_idx] = c;
        __syncthreads();
        r   = reinterpret_cast<T*>(smem)[lane_idx ^ 64];
        o   = warp_bitonic_merge_sort_combine(c, r, lane_idx, 0, ck_tile::number<128>{}, ck_tile::number<is_descending>{});

        return o;
    }
    else if constexpr (lanegroup_size == 512) {
        // little bit complex
#if 0
        T c = warp_bitonic_merge_sort_build(x, lane_idx, ck_tile::number<128>{}, ck_tile::number<is_descending>{});

        reinterpret_cast<T*>(smem)[lane_idx] = c;
        __syncthreads();
        T r = reinterpret_cast<T*>(smem)[lane_idx ^ 64];

        // using combine to simulate build stage
        T o  = warp_bitonic_merge_sort_combine(c, r, lane_idx, (lane_idx / 128) & 1, ck_tile::number<128>{}, ck_tile::number<is_descending>{});

        __syncthreads();
        reinterpret_cast<T*>(smem)[lane_idx] = o;
        __syncthreads();
        r   = reinterpret_cast<T*>(smem)[lane_idx ^ 128];

        c   = warp_bitonic_merge_sort_step_(o, r, lane_idx, (lane_idx / 256) & 1, ck_tile::number<256>{}, ck_tile::number<is_descending>{});
        __syncthreads();
        reinterpret_cast<T*>(smem)[lane_idx] = o;
        __syncthreads();
        r   = reinterpret_cast<T*>(smem)[lane_idx ^ 64];
        o  = warp_bitonic_merge_sort_combine(c, r, lane_idx, (lane_idx / 128) & 1, ck_tile::number<128>{}, ck_tile::number<is_descending>{});

        // using combine to simulate build stage
        __syncthreads();
        reinterpret_cast<T*>(smem)[lane_idx] = o;
        __syncthreads();
        r   = reinterpret_cast<T*>(smem)[lane_idx ^ 256];

        // start to combine
        c   = warp_bitonic_merge_sort_step_(o, r, lane_idx, 0, ck_tile::number<512>{}, ck_tile::number<is_descending>{});
        __syncthreads();
        reinterpret_cast<T*>(smem)[lane_idx] = c;
        __syncthreads();
        r   = reinterpret_cast<T*>(smem)[lane_idx ^ 128];

        c   = warp_bitonic_merge_sort_step_(o, r, lane_idx, 0, ck_tile::number<256>{}, ck_tile::number<is_descending>{});
        __syncthreads();
        reinterpret_cast<T*>(smem)[lane_idx] = c;
        __syncthreads();

        r   = reinterpret_cast<T*>(smem)[lane_idx ^ 64];
        o   = warp_bitonic_merge_sort_combine(c, r, lane_idx, 0, ck_tile::number<128>{}, ck_tile::number<is_descending>{});

        return o;
#endif
    }
}

template<typename T, int block_size = 64, int lanegroup_size = 64, int is_descending = 1>
__global__ void bitonic_merge_sort_kernel(T* i_ptr, T* o_ptr)
{
    __shared__ T smem[block_size];  // smem will only be used for block sort
    T data = -INFINITY;
    if(threadIdx.x < lanegroup_size) {
        data = i_ptr[threadIdx.x];
    }

    auto res = [&](){
        if constexpr(block_size <= 64)
            return warp_bitonic_merge_sort_to_reg(data, ck_tile::number<lanegroup_size>{}, ck_tile::number<is_descending>{});
        else
            return block_bitonic_merge_sort_to_reg(smem, data, ck_tile::number<lanegroup_size>{}, ck_tile::number<is_descending>{});
    }(); 
    if(threadIdx.x  < lanegroup_size) {
        o_ptr[threadIdx.x] = res;
    }
}

#define DISPATCH_()                                     \
    auto gx = dim3(1);                                  \
    auto bx = dim3(block_size);                         \
    if(is_descending == 1) {                            \
        bitonic_merge_sort_kernel<T, block_size, lanegroup_size, 1><<<gx, bx>>>(\
                    reinterpret_cast<T*>(i_ptr), reinterpret_cast<T*>(o_ptr));  \
    } else {                                            \
        bitonic_merge_sort_kernel<T, block_size, lanegroup_size, 0><<<gx, bx>>>(\
                    reinterpret_cast<T*>(i_ptr), reinterpret_cast<T*>(o_ptr));  \
    }

// host c++ API
void warp_bitonic_sort_kernel(void* i_ptr, void* o_ptr, int num_element, int is_descending)
{
    using T = float;    // TODO: only support float input/output
    if(num_element == 2) {
        constexpr int block_size = 64;
        constexpr int lanegroup_size = 2;
        DISPATCH_()
    }
    else if(num_element == 4) {
        constexpr int block_size = 64;
        constexpr int lanegroup_size = 4;
        DISPATCH_()
    }
    else if(num_element == 8) {
        constexpr int block_size = 64;
        constexpr int lanegroup_size = 8;
        DISPATCH_()
    }
    else if(num_element == 16) {
        constexpr int block_size = 64;
        constexpr int lanegroup_size = 16;
        DISPATCH_()
    }
    else if(num_element == 32) {
        constexpr int block_size = 64;
        constexpr int lanegroup_size = 32;
        DISPATCH_()
    }
    else if(num_element == 64) {
        constexpr int block_size = 64;
        constexpr int lanegroup_size = 64;
        DISPATCH_()
    }
    else if(num_element == 128) {
        constexpr int block_size = 128;
        constexpr int lanegroup_size = 128;
        DISPATCH_()
    }
    else if(num_element == 256) {
        constexpr int block_size = 256;
        constexpr int lanegroup_size = 256;
        DISPATCH_()
    }
}
