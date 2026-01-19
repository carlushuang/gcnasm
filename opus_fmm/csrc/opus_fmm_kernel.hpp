// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include "opus/opus.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace opus {
/*******************************************************************************************************************/
template<typename Traits> struct gemm_tile_scheduler_bmn;
template<typename Traits> struct gemm_tile_scheduler_bnm;

enum class gemm_tile_scheduler_enum { BMN = 0, BNM,  };
template<gemm_tile_scheduler_enum ID, typename Traits>   struct gemm_tile_scheduler_dispatch;
template<typename T> struct gemm_tile_scheduler_dispatch<gemm_tile_scheduler_enum::BMN, T> { using type = gemm_tile_scheduler_bmn<T>; };
template<typename T> struct gemm_tile_scheduler_dispatch<gemm_tile_scheduler_enum::BNM, T> { using type = gemm_tile_scheduler_bnm<T>; };
template<gemm_tile_scheduler_enum ID, typename Traits> using gemm_tile_scheduler_dispatch_t = gemm_tile_scheduler_dispatch<ID, Traits>::type;

template<int BLOCK_M_, int BLOCK_N_, int BLOCK_K_, int BLOCK_BATCH_>
struct gemm_tile_scheduler_traits {
    static constexpr int B_M = BLOCK_M_;
    static constexpr int B_N = BLOCK_N_;
    static constexpr int B_K = BLOCK_K_;
    static constexpr int B_B = BLOCK_BATCH_;
};

struct gemm_tile_scheduler_args { int m; int n; int k; int b; };
struct gemm_tile_scheduler_tile_ids { int i_m; int i_n; int i_k; int i_b; };

using gemm_tile_scheduler_hargs = gemm_tile_scheduler_args;

template<typename Traits>
struct gemm_tile_scheduler_bmn {
    __device__ operator()(gemm_tile_scheduler_args a) {
        (void) a;
        int i_m = blockIdx.y;
        int i_n = blockIdx.x;
        int i_k = 0;
        int i_b = blockIdx.z;
        return gemm_tile_scheduler_tile_ids{i_m, i_n, i_k, i_b};
    }
    static __host__ auto gridsize(gemm_tile_scheduler_hargs a) {
        using T = opus::remove_cvref_t<Traits>;
        int tiles_m = (a.m + T::B_M - 1) / T::B_M;
        int tiles_n = (a.n + T::B_N - 1) / T::B_N;
        int tiles_b = (a.b + T::B_B - 1) / T::B_B;
        return dim3(tiles_n, tiles_m, tiles_b);
    }
};

template<typename Traits>
struct gemm_tile_scheduler_bnm {
    __device__ operator()(gemm_tile_scheduler_args a) {
        (void) a;
        int i_m = blockIdx.x;
        int i_n = blockIdx.y;
        int i_k = 0;
        int i_b = blockIdx.z;
        return gemm_tile_scheduler_kretn{i_m, i_n, i_k, i_b};
    }
    static __host__ auto gridsize(gemm_tile_scheduler_hargs a) {
        using T = opus::remove_cvref_t<Traits>;
        int tiles_m = (a.m + T::B_M - 1) / T::B_M;
        int tiles_n = (a.n + T::B_N - 1) / T::B_N;
        int tiles_b = (a.b + T::B_B - 1) / T::B_B;
        return dim3(tiles_m, tiles_n, tiles_b);
    }
};
/******************************************************************************************************************* */
// clang-format off
// batch/bias gemm with rcr layout
// A: [Batch, M, K], with stride for Batch and M
// B: [Batch, N, K], with stride for Batch and N
// C: [Batch, M, N], with stride for Batch and M
// Bias: [Batch, N], no stride

template<typename WG_       // opus::seq<x, y, z>, workgroup size
        typename BLOCK_,    // opus::seq<m, n, k>, block_tile m/n/k
        typename DTYPE_,    // opus::tuple<d_a, d_b, d_c, d_acc, d_bias>, data type
        typename VEC_,      // opus::seq<a, b, c>, fast changing dim vector size, for global load/store
        int TILE_SCHEDULER_ID_, //  
        bool HAS_BIAS_      // has bias with size [1, N] or not
        >
struct opus_flatmm_traits {
    // always use 16x16 mfma
    // always 4 wave
    // always 32x64x[64DW] WG tile size for one repeat(expand)
    using WG    = opus::remove_cvref_t<WG_>;
    using BLOCK = opus::remove_cvref_t<BLOCK_>;
    using TILE  = opus::remove_cvref_t<TILE_>;
    using WAVE  = opus::remove_cvref_t<WAVE>;
    using DTYPE = opus::remove_cvref_t<DTYPE_>;
    using VEC   = opus::remove_cvref_t<VEC_>;

    static constexpr int WG_X   = opus::get<0>(WG{});
    static constexpr int WG_Y   = opus::get<1>(WG{});
    static constexpr int WG_Z   = opus::get<2>(WG{});

    static constexpr int BLOCK_SIZE = WG_X * WG_Y * WG_Z;

    static constexpr int B_M    = opus::get<0>(BLOCK{});
    static constexpr int B_N    = opus::get<1>(BLOCK{});
    static constexpr int B_K    = opus::get<2>(BLOCK{});

    using D_A    = opus::tuple_element_t<0, DTYPE>;
    using D_B    = opus::tuple_element_t<1, DTYPE>;
    using D_C    = opus::tuple_element_t<2, DTYPE>;
    using D_ACC  = opus::tuple_element_t<3, DTYPE>;
    using D_BIAS = opus::tuple_element_t<4, DTYPE>;

    // TODO
    using mfma = opus::mfma_f32_16x16x32_bf16;

    static constexpr int T_M    = 1; // waves along M
    static constexpr int T_N    = 4; // waves along N
    static constexpr int T_K    = 1; // waves along K

    static constexpr int W_M    = 16; // wave gemm size M
    static constexpr int W_N    = 16; // wave gemm size N
    static constexpr int W_K    = 32; // wave gemm size K

    static_assert(B_M % (W_M * T_M) == 0);
    static_assert(B_N % (W_N * T_N) == 0);
    static_assert(B_K % (W_K * T_K) == 0);

    static constexpr int E_M = B_M / (W_M * T_M);   // expand, repeat how many times along each dim
    static constexpr int E_N = B_N / (W_N * T_N);   // expand, repeat how many times along each dim
    static constexpr int E_K = B_K / (W_K * T_K);   // expand, repeat how many times along each dim

    using EXPAND = opus::seq<E_M, E_N, E_K>;

    static constexpr int V_A = opus::get<0>(VEC{});
    static constexpr int V_B = opus::get<1>(VEC{});
    static constexpr int V_C = opus::get<2>(VEC{});

    static constexpr int TILE_SCHEDULER_ID = TILE_SCHEDULER_ID_;

    using TILE_SCHEDULER_TRAITS = gemm_tile_scheduler_traits<B_M, B_N, B_K, 1>;
    using TILE_SCHEDULER = gemm_tile_scheduler_dispatch_t<static_cast<gemm_tile_scheduler_enum>(TILE_SCHEDULER_ID), TILE_SCHEDULER_TRAITS>;

    static constexpr bool HAS_BIAS = HAS_BIAS_;

    // minimal compact pixels for async copy for one wave
    constexpr int smem_linear_wave = opus::get_warp_size() * 16 / sizeof(D_A);
    constexpr int smem_m_sub = smem_linear_wave / B_K;
    constexpr int smem_m_rep = B_M / smem_m_rep;
    constexpr int smem_padding = 2 * 16 / sizeof(D_A);
};

struct opus_fmm_kargs {
    const void* __restrict__ ptr_a;
    const void* __restrict__ ptr_b;
    void* __restrict__ ptr_c;
    const void* __restrict__ ptr_bias;
    int m;
    int n;
    int k;
    int batch;
    int stride_a; // stride in unit of pixel
    int stride_b;
    int stride_c;
    int stride_a_batch;
    int stride_b_batch;
    int stride_c_batch;
};

using opus_fmm_hargs = opus_fmm_kargs;

namespace impl {
OPUS_USING_COMMON_TYPES_ALL;
// kernel entry point
template<typename Traits>
__global__ void opus_fmm(opus_fmm_kargs kargs) {
    using T = opus::remove_cvref_t<Traits>;

    auto tile_ids = T::TILE_SCHEDULER{}(gemm_tile_scheduler_args{kargs.m, kargs.n, kargs.k, kargs.batch});

    int lane_id = threadIdx.x % opus::get_warp_size();
    int wave_id = threadIdx.x / opus::get_warp_size();
    int g_im = blockIdx.x * BLOCK_M;
    int g_in = blockIdx.y * BLOCK_N;

    // NOTE: the shape merge is per-dim
    // global load a/b, we see the WG as a whole
    // A:[(step_a<y>, wavs_a<p>, lgpm_a<p>), (lgpk_a<p>, vect_a<y>)] => [B_M<x>, B_K<x>], different wave along different M, not K
    // B:[(step_b<y>, thdn_b<p>), (thdk_b<p>, vect_b<y>)] => [B_N<x>, B_K<x>]
    constexpr int vect_a = T::V_A;
    constexpr int lgpk_a = T::B_K / T::V_A;
    static_assert(opus::get_warp_size() % lgpk_a == 0);
    constexpr int lgpm_a = opus::get_warp_size() / lgpk_a;
    constexpr int wavs_a = T::BLOCK_SIZE / opus::get_warp_size();

    constexpr int lgpm_a = T::BLOCK_SIZE / lgpk_a;
    constexpr int step_a = T::B_M / lgpm_a;

    constexpr int vect_b = T::V_B;
    constexpr int thdk_b = T::B_K / T::V_B;
    constexpr int thdn_b = T::BLOCK_SIZE / thdk_b;
    constexpr int step_b = T::B_N / thdn_b;

    // minimal compact pixels for async copy for one wave
    // constexpr int smem_linear_wave = opus::get_warp_size() * 16 / sizeof(D_A);
    // constexpr int smem_m_sub = smem_linear_wave / T::B_K;
    // constexpr int smem_m_rep = T::B_M / smem_m_rep;
    // constexpr int smem_padding = 2 * 16 / sizeof(D_A);
    constexpr int w = T::W_N * T::W_K;
    int nr = kargs.n / T::W_N;
    int kr = kargs.k / T::W_K;

    auto x_ga = opus::make_layout(opus::tuple{kargs.m, kargs.k}, opus::tuple{kargs.stride_a, 1_I});
    auto u_ga = opus::partition_layout<vect_a>(x_ga, tup<tup<y_dim, y_dim, p_dim, p_dim>, tup<p_dim, y_dim>>{}, opus::tuple{lane_id / lgpk_a, wave_id, lane_id % lgpk_a});


    int addr = u_ga(0_I, 1_I, 0_I)


    auto x_sa = opus::make_layout(opus::tuple{num<T::smem_m_rep>{}, num<T::smem_m_sub>{}, num<T::B_K>{}}, opus::tuple{num<T::smem_linear_wave + T::smem_padding>{}, num<T::B_K>{}, 1_I});
    auto u_sa = opus::partition_layout<vect_a>(x_sa, tup<tup<y_dim, p_dim>, tup<y_dim>, tup<p_dim, y_dim>>{}, opus::tuple{lane_id / lgpk_a, wave_id, lane_id % lgpk_a});
    auto v_sa = opus::partition_layout<16 / sizeof(T::D_A)>(x_sa, tup<tup<y_dim, p_dim>, tup<y_dim, p_dim, y_dim>>{}, opus::tuple{lane_id / lgpk_a, wave_id, lane_id % lgpk_a});

    auto x_gb = opus::make_layout(opue::tuple{nr, kr, w}, opue::tuple{kr*w, w, 1_I});
    auto u_gb = opus::partition_layout<vect_b>(x_gb, tup<tup<y_dim, p_dim>, tup<y_dim>, tup<p_dim, y_dim>>{}, opus::tuple{wave_id, lane_id});

    auto x_sa = opus::make_layout(opus::make_tuple(), opus::make_tuple(kargs.stride_a, 1_I));

    //
    // mma a/b/c, we need explicitly illustrate dims within-warp and cross-warp
    //    (cross-warp)                                       (within-warp)
    // A:[(expd_m<y>, tile_m<p>), (expd_k<y>, tile_k<p>)] * [(grpm_a<p>), (rept_a<y>, grpk_a<p>, pack_a<y>)]
    // B:[(expd_n<y>, tile_n<p>), (expd_k<y>, tile_k<p>)] * [(grpn_b<p>), (rept_b<y>, grpk_b<p>, pack_b<y>)]
    // C:[(expd_m<y>, tile_m<p>), (expd_n<y>, tile_n<p>)] * [(grpn_c<p>), (rept_c<y>, grpm_c<p>, pack_c<y>)]
    //
    //    (embed together)
    // A:[(expd_m<y>, tile_m<p>, grpm_a<p>), (expd_k<y>, tile_k<p>, rept_a<y>, grpk_a<p>, pack_a<y>)] => [B_M<x>, B_K<x>]
    // B:[(expd_n<y>, tile_n<p>, grpn_b<p>), (expd_k<y>, tile_k<p>, rept_b<y>, grpk_b<p>, pack_b<y>)] => [B_N<x>, B_K<x>]
    // C:[(expd_m<y>, tile_m<p>, grpn_c<p>), (expd_n<y>, tile_n<p>, rept_c<y>, grpm_c<p>, pack_c<y>)] => [B_M<x>, B_N<x>]
    //
    auto mma  = opus::make_tiled_mma<D_A, D_B, D_ACC>(seq<T::E_M, T::E_N, T::E_K>{}, seq<T::T_M, T::T_N, T::T_K>{}, seq<T::W_M, T::W_N, T::W_K>{}, opus::mfma_adaptor_swap_ab{});

    // auto u_a = opus::partition_layout_a<4>(mma, opus::make_tuple(stride_a, 1_I), opus::make_tuple(wave_id / 2, lane_id % mma.grpm_a, 0_I, lane_id / mma.grpm_a) /*tile_m<p>, grpm_a<p>, tile_k<p>, grpk_a<p>*/);
    // auto u_b = opus::partition_layout_b<4>(mma, opus::make_tuple(stride_b, 1_I), opus::make_tuple(wave_id % 2, lane_id % mma.grpn_b, 0_I, lane_id / mma.grpn_b) /*tile_n<p>, grpn_b<p>, tile_k<p>, grpk_b<p>*/);
    // auto u_c = opus::partition_layout_c(mma, opus::make_tuple(stride_c, 1_I), opus::make_tuple(wave_id / 2, lane_id % mma.grpn_c, wave_id % 2, lane_id / mma.grpn_c) /*tile_m<p>, grpn_c<p> tile_n<p>, grpm_c<p>*/);
    auto g_a = opus::make_gmem(reinterpret_cast<const d_a*>(ptr_a) + g_im * stride_a);
    auto g_b = opus::make_gmem(reinterpret_cast<const d_b*>(ptr_b) + g_in * stride_b);
    auto g_c = opus::make_gmem(reinterpret_cast<opus::fp16_t*>(ptr_c) + g_im * stride_c + g_in);

    // start of kernel
    int loops = (k + BLOCK_K - 1) / BLOCK_K;
#if 1
    typename decltype(mma)::vtype_c v_c;
    opus::clear(v_c);

    for(auto i = 0; i < loops; i++ ) {
        auto v_a = g_a.load<4>(u_a);  u_a += BLOCK_K;
        auto v_b = g_b.load<4>(u_b);  u_b += BLOCK_K;
        v_c = mma(v_a, v_b, v_c);
    }

    auto v_c_f16 = opus::cast<fp16_t>(v_c);
    g_c.store<4>(v_c_f16, u_c);
#else
    auto v_a = g_a.load<4>(u_a);  u_a += BLOCK_K;
    auto v_b = g_b.load<4>(u_b);  u_b += BLOCK_K;
    auto v_c = mma(v_a, v_b);   // first time, C is always zero

    for(auto i = 0; i < loops - 1; i++ ) {
        v_a = g_a.load<4>(u_a);  u_a += BLOCK_K;
        v_b = g_b.load<4>(u_b);  u_b += BLOCK_K;
        v_c = mma(v_a, v_b, v_c);
    }

    auto v_c_f16 = opus::cast<fp16_t>(v_c);
    g_c.store<4>(v_c_f16, u_c);
#endif
}
}   // namespace impl

template<typename Traits>
struct opus_fmm_kernel {
    using kernel = impl::opus_fmm<Traits>;

    bool is_applicable(opus_fmm_hargs args) const { (void)args; return true; }

    void prepare(opus_fmm_hargs args) {
        kargs = args;
        gx = Traits::TILE_SCHEDULER{}.gridsize(gemm_tile_scheduler_args{args.m, args.n, args.k, args.batch});
        bx = dim3(Traits::WG_X, Traits::WG_Y, Traits::WG_Z);
    }

    void operator()(hipStream_t s = nullptr) { kernel<<<gx, bx, 0, s>>>(kargs); }

    opus_fmm_kargs kargs;
    dim3 gx;
    dim3 bx;
};
// clang-format on
}   // namespace opus
