// pyhip_v2: Zero <<<>>> in C++. Kernel launch done entirely from Python via
// hipLaunchKernel in libamdhip64.so. hip_minimal.h provides only __shfl.
// All other HIP primitives replaced with compiler builtins:
//   threadIdx.x   → __builtin_amdgcn_workitem_id_x()
//   __syncthreads → __builtin_amdgcn_s_barrier()
//   warpSize      → __builtin_amdgcn_wavefrontsize()
#include "hip_minimal.h"

#ifdef __HIP_DEVICE_COMPILE__
// ── Device pass: full implementation ─────────────────────────────────────────
#include "opus/opus.hpp"

#ifndef FMT_LIMIT
#define FMT_LIMIT 1
#define FMT_LIMIT_MAX 16
#endif

// swap lo/hi half within a lanegroup
template <typename T, int lanegroup_size>
__device__ __inline__ auto warp_swap_(const T& x, int lane_idx, opus::number<lanegroup_size> = {})
{
    if constexpr (lanegroup_size == 1) {
        // just return same value if groupsize is 1(no dpp, no permute)
        return x;
    }
    if constexpr (lanegroup_size == 2) {
        return opus::mov_dpp(x, opus::number<0xb1>{}); /*quad_perm:[1,0,3,2]*/
    } else if constexpr (lanegroup_size == 4) {
        return opus::mov_dpp(x,  opus::number<0x4e>{}); /*quad_perm:[2,3,0,1]*/
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
        r = opus::upd_dpp(r, x, opus::number<260>{}, opus::number<0xf>{}, opus::number<0b0101>{}); /*row_shl:4*/
        r = opus::upd_dpp(r, x, opus::number<276>{}, opus::number<0xf>{}, opus::number<0b1010>{}); /*row_shr:4*/
#pragma clang diagnostic pop
        return  r;
    } else if constexpr(lanegroup_size == 16) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wuninitialized"
        T r;
        r = opus::upd_dpp(r, x, opus::number<264>{}, opus::number<0xf>{}, opus::number<0b0011>{}); /*row_shl:8*/
        r = opus::upd_dpp(r, x, opus::number<280>{}, opus::number<0xf>{}, opus::number<0b1100>{}); /*row_shr:8*/
#pragma clang diagnostic pop
        return r;
    } else if constexpr(lanegroup_size == 32) {
        return __shfl(x, lane_idx ^ 16);    // consume LDS
    } else if constexpr(lanegroup_size == 64) {
        return __shfl(x, lane_idx ^ 32);    // consume LDS
    }
}

// This is the core function to build the construct/combine stage of bitonic merge sor
template <typename T, int lanegroup_size = opus::get_warp_size(), int is_descending = 1>
__device__ __inline__ auto warp_bitonic_merge_sort_step_(const T& x, const T& y, int lane_idx, int twiddle, opus::number<lanegroup_size> = {}, opus::number<is_descending> = {})
{
    auto guard = [&](auto div_) {
            if constexpr(is_descending) return  (((lane_idx / div_.value) & 1) ^ twiddle) == 0 ? INFINITY : -INFINITY;
            else return                         (((lane_idx / div_.value) & 1) ^ twiddle) == 0 ? -INFINITY : INFINITY;
    };

    // compare and swap within lanegroup_size lo/hi half
    auto g = guard(opus::number<lanegroup_size / 2>{});
    return opus::med3(x, y, g);
}

// this version the return value will be stored into per-lane register
template <typename T, int lanegroup_size = opus::get_warp_size(), int is_descending = 1>
__device__ __inline__ auto warp_bitonic_merge_sort_build(const T& x, int lane_idx, opus::number<lanegroup_size> = {}, opus::number<is_descending> = {})
{
    if constexpr (lanegroup_size == 2) {
        // TODO:!!! if 2, always use combine, not build
        // here we just return the original value
        return x;
    }
    else if constexpr (lanegroup_size == 4) {
        T y =  warp_swap_(x, lane_idx, opus::number<2>{});
        T o = warp_bitonic_merge_sort_step_(x, y, lane_idx, (lane_idx / 2) & 1 , opus::number<2>{}, opus::number<is_descending>{});
        return o;
    }
    else if constexpr (lanegroup_size == 8) {
        T y =  warp_swap_(x, lane_idx, opus::number<2>{});
        T o = warp_bitonic_merge_sort_step_(x, y, lane_idx, (lane_idx / 2) & 1 , opus::number<2>{}, opus::number<is_descending>{});

        y   = warp_swap_(o, lane_idx, opus::number<4>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 4) & 1 , opus::number<4>{}, opus::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, opus::number<2>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 4) & 1 , opus::number<2>{}, opus::number<is_descending>{});
        return o;
    }
    else if constexpr (lanegroup_size == 16) {
        T y =  warp_swap_(x, lane_idx, opus::number<2>{});
        T o = warp_bitonic_merge_sort_step_(x, y, lane_idx, (lane_idx / 2) & 1 , opus::number<2>{}, opus::number<is_descending>{});

        y   = warp_swap_(o, lane_idx, opus::number<4>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 4) & 1 , opus::number<4>{}, opus::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, opus::number<2>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 4) & 1 , opus::number<2>{}, opus::number<is_descending>{});

        y   = warp_swap_(o, lane_idx, opus::number<8>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 8) & 1 , opus::number<8>{}, opus::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, opus::number<4>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 8) & 1 , opus::number<4>{}, opus::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, opus::number<2>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 8) & 1 , opus::number<2>{}, opus::number<is_descending>{});
        return o;
    }
    else if constexpr (lanegroup_size == 32) {
        T y =  warp_swap_(x, lane_idx, opus::number<2>{});
        T o = warp_bitonic_merge_sort_step_(x, y, lane_idx, (lane_idx / 2) & 1 , opus::number<2>{}, opus::number<is_descending>{});

        y   = warp_swap_(o, lane_idx, opus::number<4>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 4) & 1 , opus::number<4>{}, opus::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, opus::number<2>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 4) & 1 , opus::number<2>{}, opus::number<is_descending>{});

        y   = warp_swap_(o, lane_idx, opus::number<8>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 8) & 1 , opus::number<8>{}, opus::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, opus::number<4>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 8) & 1 , opus::number<4>{}, opus::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, opus::number<2>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 8) & 1 , opus::number<2>{}, opus::number<is_descending>{});

        y   = warp_swap_(o, lane_idx, opus::number<16>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 16) & 1 , opus::number<16>{}, opus::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, opus::number<8>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 16) & 1 , opus::number<8>{}, opus::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, opus::number<4>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 16) & 1 , opus::number<4>{}, opus::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, opus::number<2>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 16) & 1 , opus::number<2>{}, opus::number<is_descending>{});
        return o;
    }
    else if constexpr (lanegroup_size == 64) {
        T y =  warp_swap_(x, lane_idx, opus::number<2>{});
        T o = warp_bitonic_merge_sort_step_(x, y, lane_idx, (lane_idx / 2) & 1 , opus::number<2>{}, opus::number<is_descending>{});

        y   = warp_swap_(o, lane_idx, opus::number<4>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 4) & 1 , opus::number<4>{}, opus::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, opus::number<2>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 4) & 1 , opus::number<2>{}, opus::number<is_descending>{});

        y   = warp_swap_(o, lane_idx, opus::number<8>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 8) & 1 , opus::number<8>{}, opus::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, opus::number<4>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 8) & 1 , opus::number<4>{}, opus::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, opus::number<2>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 8) & 1 , opus::number<2>{}, opus::number<is_descending>{});

        y   = warp_swap_(o, lane_idx, opus::number<16>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 16) & 1 , opus::number<16>{}, opus::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, opus::number<8>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 16) & 1 , opus::number<8>{}, opus::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, opus::number<4>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 16) & 1 , opus::number<4>{}, opus::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, opus::number<2>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 16) & 1 , opus::number<2>{}, opus::number<is_descending>{});

        y   = warp_swap_(o, lane_idx, opus::number<32>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 32) & 1 , opus::number<32>{}, opus::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, opus::number<16>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 32) & 1 , opus::number<16>{}, opus::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, opus::number<8>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 32) & 1 , opus::number<8>{}, opus::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, opus::number<4>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 32) & 1 , opus::number<4>{}, opus::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, opus::number<2>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 32) & 1 , opus::number<2>{}, opus::number<is_descending>{});
        return o;
    }
    else if constexpr (lanegroup_size == 128) {
        T y =  warp_swap_(x, lane_idx, opus::number<2>{});
        T o = warp_bitonic_merge_sort_step_(x, y, lane_idx, (lane_idx / 2) & 1 , opus::number<2>{}, opus::number<is_descending>{});

        y   = warp_swap_(o, lane_idx, opus::number<4>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 4) & 1 , opus::number<4>{}, opus::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, opus::number<2>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 4) & 1 , opus::number<2>{}, opus::number<is_descending>{});

        y   = warp_swap_(o, lane_idx, opus::number<8>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 8) & 1 , opus::number<8>{}, opus::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, opus::number<4>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 8) & 1 , opus::number<4>{}, opus::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, opus::number<2>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 8) & 1 , opus::number<2>{}, opus::number<is_descending>{});

        y   = warp_swap_(o, lane_idx, opus::number<16>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 16) & 1 , opus::number<16>{}, opus::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, opus::number<8>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 16) & 1 , opus::number<8>{}, opus::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, opus::number<4>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 16) & 1 , opus::number<4>{}, opus::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, opus::number<2>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 16) & 1 , opus::number<2>{}, opus::number<is_descending>{});

        y   = warp_swap_(o, lane_idx, opus::number<32>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 32) & 1 , opus::number<32>{}, opus::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, opus::number<16>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 32) & 1 , opus::number<16>{}, opus::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, opus::number<8>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 32) & 1 , opus::number<8>{}, opus::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, opus::number<4>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 32) & 1 , opus::number<4>{}, opus::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, opus::number<2>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 32) & 1 , opus::number<2>{}, opus::number<is_descending>{});

        y   = warp_swap_(o, lane_idx, opus::number<64>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 64) & 1 , opus::number<64>{}, opus::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, opus::number<32>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 64) & 1 , opus::number<32>{}, opus::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, opus::number<16>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 64) & 1 , opus::number<16>{}, opus::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, opus::number<8>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 64) & 1 , opus::number<8>{}, opus::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, opus::number<4>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 64) & 1 , opus::number<4>{}, opus::number<is_descending>{});
        y   = warp_swap_(o, lane_idx, opus::number<2>{});
        o   = warp_bitonic_merge_sort_step_(o, y, lane_idx, (lane_idx / 64) & 1 , opus::number<2>{}, opus::number<is_descending>{});
        return o;
    }
}

// this version the return value will be stored into per-lane register
template <typename T, int lanegroup_size = opus::get_warp_size(), int is_descending = 1>
__device__ __inline__ auto warp_bitonic_merge_sort_combine(const T& x, const T& y, int lane_idx, int twiddle, opus::number<lanegroup_size> = {}, opus::number<is_descending> = {})
{
    if constexpr (lanegroup_size == 2) {
        T o = warp_bitonic_merge_sort_step_(x, y, lane_idx, twiddle, opus::number<2>{}, opus::number<is_descending>{});
        return o;
    }
    else if constexpr (lanegroup_size == 4) {
        T o = warp_bitonic_merge_sort_step_(x, y, lane_idx, twiddle, opus::number<4>{}, opus::number<is_descending>{});
        T z = warp_swap_(o, lane_idx, opus::number<2>{});
        o   = warp_bitonic_merge_sort_step_(o, z, lane_idx, twiddle, opus::number<2>{}, opus::number<is_descending>{});
        return o;
    }
    else if constexpr (lanegroup_size == 8) {
        T o = warp_bitonic_merge_sort_step_(x, y, lane_idx, twiddle, opus::number<8>{}, opus::number<is_descending>{});
        T z = warp_swap_(o, lane_idx, opus::number<4>{});

        o   = warp_bitonic_merge_sort_step_(o, z, lane_idx, twiddle, opus::number<4>{}, opus::number<is_descending>{});
        z   = warp_swap_(o, lane_idx, opus::number<2>{});
        o   = warp_bitonic_merge_sort_step_(o, z, lane_idx, twiddle, opus::number<2>{}, opus::number<is_descending>{});
        return o;
    }
    else if constexpr (lanegroup_size == 16) {
        T o = warp_bitonic_merge_sort_step_(x, y, lane_idx, twiddle, opus::number<16>{}, opus::number<is_descending>{});
        T z = warp_swap_(o, lane_idx, opus::number<8>{});

        o   = warp_bitonic_merge_sort_step_(o, z, lane_idx, twiddle, opus::number<8>{}, opus::number<is_descending>{});
        z   = warp_swap_(o, lane_idx, opus::number<4>{});

        o   = warp_bitonic_merge_sort_step_(o, z, lane_idx, twiddle, opus::number<4>{}, opus::number<is_descending>{});
        z   = warp_swap_(o, lane_idx, opus::number<2>{});
        o   = warp_bitonic_merge_sort_step_(o, z, lane_idx, twiddle, opus::number<2>{}, opus::number<is_descending>{});
        return o;
    }
    else if constexpr (lanegroup_size == 32) {
        T o = warp_bitonic_merge_sort_step_(x, y, lane_idx, twiddle, opus::number<32>{}, opus::number<is_descending>{});
        T z = warp_swap_(o, lane_idx, opus::number<16>{});

        o   = warp_bitonic_merge_sort_step_(o, z, lane_idx, twiddle, opus::number<16>{}, opus::number<is_descending>{});
        z   = warp_swap_(o, lane_idx, opus::number<8>{});

        o   = warp_bitonic_merge_sort_step_(o, z, lane_idx, twiddle, opus::number<8>{}, opus::number<is_descending>{});
        z   = warp_swap_(o, lane_idx, opus::number<4>{});

        o   = warp_bitonic_merge_sort_step_(o, z, lane_idx, twiddle, opus::number<4>{}, opus::number<is_descending>{});
        z   = warp_swap_(o, lane_idx, opus::number<2>{});
        o   = warp_bitonic_merge_sort_step_(o, z, lane_idx, twiddle, opus::number<2>{}, opus::number<is_descending>{});
        return o;
    }
    else if constexpr (lanegroup_size == 64) {
        T o = warp_bitonic_merge_sort_step_(x, y, lane_idx, twiddle, opus::number<64>{}, opus::number<is_descending>{});
        T z = warp_swap_(o, lane_idx, opus::number<32>{});

        o   = warp_bitonic_merge_sort_step_(o, z, lane_idx, twiddle, opus::number<32>{}, opus::number<is_descending>{});
        z   = warp_swap_(o, lane_idx, opus::number<16>{});

        o   = warp_bitonic_merge_sort_step_(o, z, lane_idx, twiddle, opus::number<16>{}, opus::number<is_descending>{});
        z   = warp_swap_(o, lane_idx, opus::number<8>{});

        o   = warp_bitonic_merge_sort_step_(o, z, lane_idx, twiddle, opus::number<8>{}, opus::number<is_descending>{});
        z   = warp_swap_(o, lane_idx, opus::number<4>{});

        o   = warp_bitonic_merge_sort_step_(o, z, lane_idx, twiddle, opus::number<4>{}, opus::number<is_descending>{});
        z   = warp_swap_(o, lane_idx, opus::number<2>{});
        o   = warp_bitonic_merge_sort_step_(o, z, lane_idx, twiddle, opus::number<2>{}, opus::number<is_descending>{});
        return o;
    }
    else if constexpr (lanegroup_size == 128) {
        T o = warp_bitonic_merge_sort_step_(x, y, lane_idx, twiddle, opus::number<128>{}, opus::number<is_descending>{});
        T z = warp_swap_(o, lane_idx, opus::number<64>{});

        o   = warp_bitonic_merge_sort_step_(o, z, lane_idx, twiddle, opus::number<64>{}, opus::number<is_descending>{});
        z   = warp_swap_(o, lane_idx, opus::number<32>{});

        o   = warp_bitonic_merge_sort_step_(o, z, lane_idx, twiddle, opus::number<32>{}, opus::number<is_descending>{});
        z   = warp_swap_(o, lane_idx, opus::number<16>{});

        o   = warp_bitonic_merge_sort_step_(o, z, lane_idx, twiddle, opus::number<16>{}, opus::number<is_descending>{});
        z   = warp_swap_(o, lane_idx, opus::number<8>{});

        o   = warp_bitonic_merge_sort_step_(o, z, lane_idx, twiddle, opus::number<8>{}, opus::number<is_descending>{});
        z   = warp_swap_(o, lane_idx, opus::number<4>{});

        o   = warp_bitonic_merge_sort_step_(o, z, lane_idx, twiddle, opus::number<4>{}, opus::number<is_descending>{});
        z   = warp_swap_(o, lane_idx, opus::number<2>{});
        o   = warp_bitonic_merge_sort_step_(o, z, lane_idx, twiddle, opus::number<2>{}, opus::number<is_descending>{});
        return o;
    }
}
// this version the return value will be stored into per-lane register
template <typename T, int lanegroup_size = opus::get_warp_size(), int is_descending = 1>
__device__ __inline__ auto warp_bitonic_merge_sort_to_reg(const T& x, opus::number<lanegroup_size> = {}, opus::number<is_descending> = {})
{
    static_assert(lanegroup_size <= opus::get_warp_size());
    int lane_idx = __builtin_amdgcn_workitem_id_x();
    T c = warp_bitonic_merge_sort_build(x, lane_idx, opus::number<lanegroup_size>{}, opus::number<is_descending>{});
    T r = warp_swap_(c, lane_idx, opus::number<lanegroup_size>{});
    // if(__builtin_amdgcn_workitem_id_x() < lanegroup_size) printf("[%2d] c:%f, r:%f\n", __builtin_amdgcn_workitem_id_x(), c, r);
    T o = warp_bitonic_merge_sort_combine(c, r, lane_idx, 0, opus::number<lanegroup_size>{}, opus::number<is_descending>{});
    return o;
}

template <typename T, int lanegroup_size = opus::get_warp_size(), int is_descending = 1>
__device__ __inline__ auto block_bitonic_merge_sort_to_reg(void* smem, const T& x, opus::number<lanegroup_size> = {}, opus::number<is_descending> = {})
{
    // need make sure smem before this function is ready to use
    // need guarantee smem usage, will not if...else... write smem inside this kernel
    // smem require sizeof(T) * lanegroup_size
    static_assert(lanegroup_size > opus::get_warp_size());
    int lane_idx = __builtin_amdgcn_workitem_id_x();
    if constexpr (lanegroup_size == 128) {
        T c = warp_bitonic_merge_sort_build(x, lane_idx, opus::number<128>{}, opus::number<is_descending>{});
    
        reinterpret_cast<T*>(smem)[lane_idx] = c;
        __builtin_amdgcn_s_barrier();
        T r = reinterpret_cast<T*>(smem)[lane_idx ^ 64];

        T o = warp_bitonic_merge_sort_combine(c, r, lane_idx, 0, opus::number<128>{}, opus::number<is_descending>{});
        return o;
    }
    else if constexpr (lanegroup_size == 256) {
        T c = warp_bitonic_merge_sort_build(x, lane_idx, opus::number<128>{}, opus::number<is_descending>{});

        reinterpret_cast<T*>(smem)[lane_idx] = c;
        __builtin_amdgcn_s_barrier();
        T r = reinterpret_cast<T*>(smem)[lane_idx ^ 64];

        // using combine to simulate build stage
        T o  = warp_bitonic_merge_sort_combine(c, r, lane_idx, (lane_idx / 128) & 1, opus::number<128>{}, opus::number<is_descending>{});

        // start to combine
        __builtin_amdgcn_s_barrier();
        reinterpret_cast<T*>(smem)[lane_idx] = o;
        __builtin_amdgcn_s_barrier();
        r   = reinterpret_cast<T*>(smem)[lane_idx ^ 128];
        c   = warp_bitonic_merge_sort_step_(o, r, lane_idx, 0, opus::number<256>{}, opus::number<is_descending>{});

        __builtin_amdgcn_s_barrier();
        reinterpret_cast<T*>(smem)[lane_idx] = c;
        __builtin_amdgcn_s_barrier();
        r   = reinterpret_cast<T*>(smem)[lane_idx ^ 64];
        o   = warp_bitonic_merge_sort_combine(c, r, lane_idx, 0, opus::number<128>{}, opus::number<is_descending>{});

        return o;
    }
    else if constexpr (lanegroup_size == 512) {
        // little bit complex
#if 0
        T c = warp_bitonic_merge_sort_build(x, lane_idx, opus::number<128>{}, opus::number<is_descending>{});

        reinterpret_cast<T*>(smem)[lane_idx] = c;
        __builtin_amdgcn_s_barrier();
        T r = reinterpret_cast<T*>(smem)[lane_idx ^ 64];

        // using combine to simulate build stage
        T o  = warp_bitonic_merge_sort_combine(c, r, lane_idx, (lane_idx / 128) & 1, opus::number<128>{}, opus::number<is_descending>{});

        __builtin_amdgcn_s_barrier();
        reinterpret_cast<T*>(smem)[lane_idx] = o;
        __builtin_amdgcn_s_barrier();
        r   = reinterpret_cast<T*>(smem)[lane_idx ^ 128];

        c   = warp_bitonic_merge_sort_step_(o, r, lane_idx, (lane_idx / 256) & 1, opus::number<256>{}, opus::number<is_descending>{});
        __builtin_amdgcn_s_barrier();
        reinterpret_cast<T*>(smem)[lane_idx] = o;
        __builtin_amdgcn_s_barrier();
        r   = reinterpret_cast<T*>(smem)[lane_idx ^ 64];
        o  = warp_bitonic_merge_sort_combine(c, r, lane_idx, (lane_idx / 128) & 1, opus::number<128>{}, opus::number<is_descending>{});

        // using combine to simulate build stage
        __builtin_amdgcn_s_barrier();
        reinterpret_cast<T*>(smem)[lane_idx] = o;
        __builtin_amdgcn_s_barrier();
        r   = reinterpret_cast<T*>(smem)[lane_idx ^ 256];

        // start to combine
        c   = warp_bitonic_merge_sort_step_(o, r, lane_idx, 0, opus::number<512>{}, opus::number<is_descending>{});
        __builtin_amdgcn_s_barrier();
        reinterpret_cast<T*>(smem)[lane_idx] = c;
        __builtin_amdgcn_s_barrier();
        r   = reinterpret_cast<T*>(smem)[lane_idx ^ 128];

        c   = warp_bitonic_merge_sort_step_(o, r, lane_idx, 0, opus::number<256>{}, opus::number<is_descending>{});
        __builtin_amdgcn_s_barrier();
        reinterpret_cast<T*>(smem)[lane_idx] = c;
        __builtin_amdgcn_s_barrier();

        r   = reinterpret_cast<T*>(smem)[lane_idx ^ 64];
        o   = warp_bitonic_merge_sort_combine(c, r, lane_idx, 0, opus::number<128>{}, opus::number<is_descending>{});

        return o;
#endif
    }
}

template<typename T, int block_size = 64, int lanegroup_size = 64, int is_descending = 1>
__global__ void bitonic_merge_sort_kernel(T* i_ptr, T* o_ptr)
{
    __shared__ T smem[block_size];  // smem will only be used for block sort
    T data = -INFINITY;
    if(__builtin_amdgcn_workitem_id_x() < lanegroup_size) {
        data = i_ptr[__builtin_amdgcn_workitem_id_x()];
    }

    auto res = [&](){
        if constexpr(block_size <= 64)
            return warp_bitonic_merge_sort_to_reg(data, opus::number<lanegroup_size>{}, opus::number<is_descending>{});
        else
            return block_bitonic_merge_sort_to_reg(smem, data, opus::number<lanegroup_size>{}, opus::number<is_descending>{});
    }(); 
    if(__builtin_amdgcn_workitem_id_x()  < lanegroup_size) {
        o_ptr[__builtin_amdgcn_workitem_id_x()] = res;
    }
}

#else
// ── Host pass: empty kernel body so hipcc generates the launch stub ──────────
template<typename T, int block_size = 64, int lanegroup_size = 64, int is_descending = 1>
__global__ void bitonic_merge_sort_kernel(T* i_ptr, T* o_ptr) {}
#endif // __HIP_DEVICE_COMPILE__

// ── Explicit instantiations (forces hipcc to generate device+host stubs) ─────
#define INSTANTIATE(BS, LS, D) \
    template __global__ void bitonic_merge_sort_kernel<float, BS, LS, D>(float*, float*);

INSTANTIATE(64,   2, 1) INSTANTIATE(64,   2, 0)
INSTANTIATE(64,   4, 1) INSTANTIATE(64,   4, 0)
INSTANTIATE(64,   8, 1) INSTANTIATE(64,   8, 0)
INSTANTIATE(64,  16, 1) INSTANTIATE(64,  16, 0)
INSTANTIATE(64,  32, 1) INSTANTIATE(64,  32, 0)
INSTANTIATE(64,  64, 1) INSTANTIATE(64,  64, 0)
INSTANTIATE(128,128, 1) INSTANTIATE(128,128, 0)
INSTANTIATE(256,256, 1) INSTANTIATE(256,256, 0)
#undef INSTANTIATE

// Export kernel host-stub addresses for Python to pass to hipLaunchKernel.
// No <<<>>>, no dim3, no C++ dispatch code.
struct kernel_entry_t {
    int         lanegroup_size;
    int         block_size;
    const void* desc;
    const void* asc;
};

extern "C" const kernel_entry_t kernel_table[] = {
    {  2,  64, (const void*)&bitonic_merge_sort_kernel<float, 64,   2, 1>,
               (const void*)&bitonic_merge_sort_kernel<float, 64,   2, 0>},
    {  4,  64, (const void*)&bitonic_merge_sort_kernel<float, 64,   4, 1>,
               (const void*)&bitonic_merge_sort_kernel<float, 64,   4, 0>},
    {  8,  64, (const void*)&bitonic_merge_sort_kernel<float, 64,   8, 1>,
               (const void*)&bitonic_merge_sort_kernel<float, 64,   8, 0>},
    { 16,  64, (const void*)&bitonic_merge_sort_kernel<float, 64,  16, 1>,
               (const void*)&bitonic_merge_sort_kernel<float, 64,  16, 0>},
    { 32,  64, (const void*)&bitonic_merge_sort_kernel<float, 64,  32, 1>,
               (const void*)&bitonic_merge_sort_kernel<float, 64,  32, 0>},
    { 64,  64, (const void*)&bitonic_merge_sort_kernel<float, 64,  64, 1>,
               (const void*)&bitonic_merge_sort_kernel<float, 64,  64, 0>},
    {128, 128, (const void*)&bitonic_merge_sort_kernel<float,128, 128, 1>,
               (const void*)&bitonic_merge_sort_kernel<float,128, 128, 0>},
    {256, 256, (const void*)&bitonic_merge_sort_kernel<float,256, 256, 1>,
               (const void*)&bitonic_merge_sort_kernel<float,256, 256, 0>},
    {  0,   0, nullptr, nullptr}
};
