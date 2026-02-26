#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <random>
#include <iostream>
#include <numeric>
#include <memory>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <omp.h>

#include <opus/opus.hpp>

using fp16_t = __fp16;
using opus::operator""_I;

#define CHECK_HIP(call)                                                                                   \
    do {                                                                                                  \
        hipError_t status_ = call;                                                                        \
        if (status_ != hipSuccess) {                                                                      \
            fprintf(stderr, "HIP error (%s:%d): %s\n", __FILE__, __LINE__, hipGetErrorString(status_));   \
            exit(1);                                                                                      \
        }                                                                                                 \
    } while(0)

#define CHECK_HIP_KERNEL_LAUNCH() CHECK_HIP(hipGetLastError())

__host__ __device__ inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

// Configuration traits for FlatMM kernel: defines tile sizes, data types, and MFMA layout
template<int BLOCK_SIZE_,   // workgroup size
        typename BLOCK_,    // opus::seq<m, n, k>, block_tile m/n/k
        typename DTYPE_,    // opus::tuple<d_a, d_b, d_c, d_acc, d_bias>, data type
        typename VEC_,      // opus::seq<a, b, c>, fast changing dim vector size, for global load/store
        bool HAS_BIAS_      // has bias with size [1, N] or not
        >
struct opus_flatmm_traits {
    // always use 16x16 mfma
    using BLOCK = opus::remove_cvref_t<BLOCK_>;
    using DTYPE = opus::remove_cvref_t<DTYPE_>;
    using VEC   = opus::remove_cvref_t<VEC_>;

    static constexpr int BLOCK_SIZE = BLOCK_SIZE_;

    static constexpr int B_M = opus::get<0>(BLOCK{});
    static constexpr int B_N = opus::get<1>(BLOCK{});
    static constexpr int B_K = opus::get<2>(BLOCK{});

    using D_A    = opus::tuple_element_t<0, DTYPE>;
    using D_B    = opus::tuple_element_t<1, DTYPE>;
    using D_C    = opus::tuple_element_t<2, DTYPE>;
    using D_ACC  = opus::tuple_element_t<3, DTYPE>;
    using D_BIAS = opus::tuple_element_t<4, DTYPE>;

    static constexpr int T_M = 2; // waves along M
    static constexpr int T_N = 4; // waves along N
    static constexpr int T_K = 1; // waves along K

    static_assert(BLOCK_SIZE / opus::get_warp_size() == T_M * T_N * T_K);
    static_assert(T_K == 1);

    static constexpr int W_M = 32; // wave gemm size M
    static constexpr int W_N = 32; // wave gemm size N
    static constexpr int W_K = 16; // wave gemm size K

    static constexpr int REG_BLOCK_M = B_M / 4;
    static constexpr int REG_BLOCK_N = B_N / 4;
    static constexpr int REG_SLICE_K = 16;

    static constexpr int E_M = REG_BLOCK_M / W_M;
    static constexpr int E_N = REG_BLOCK_N / W_N;
    static constexpr int E_K = REG_SLICE_K / W_K;

    static constexpr int VEC_A = opus::get<0>(VEC{});
    static constexpr int VEC_B = opus::get<1>(VEC{});
    static constexpr int VEC_C = opus::get<2>(VEC{});

    static constexpr bool HAS_BIAS = HAS_BIAS_;

    // flatmm shape
    static constexpr int flat_k_per_wave  = W_N * W_K;
    static_assert(opus::get_warp_size() * VEC_B == flat_k_per_wave);
    static constexpr int flat_k_per_block = flat_k_per_wave * B_K / W_K;
    static constexpr int flat_n_per_block = B_N / W_N;

    // minimal compact pixels for async copy for one wave
    static_assert(VEC_A == 16 / sizeof(D_A));
    static constexpr int smem_linear_wave = opus::get_warp_size() * 16 / sizeof(D_A);
    static constexpr int smem_m_sub = smem_linear_wave / B_K;
    static constexpr int smem_m_rep = B_M / smem_m_sub;
    static constexpr int smem_padding = 16 / sizeof(D_A);

    // number of assembly instructions
    static constexpr int a_buffer_load_insts = B_M * B_K / (BLOCK_SIZE * VEC_A);
    static constexpr int b_buffer_load_insts = flat_n_per_block * flat_k_per_block / (BLOCK_SIZE * VEC_B);
    static constexpr int a_ds_read_insts = (E_M * E_K * W_M * W_K) / (opus::get_warp_size() * VEC_A);
    static constexpr int b_ds_read_insts = (E_N * E_K * W_N * W_K) / (opus::get_warp_size() * VEC_B);
};

// Kernel arguments structure for FlatMM operation
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

// Create layout for loading A matrix from global memory
template<typename T>
inline __device__ auto make_layout_ga(int lane_id, int wave_id, int stride_a) {
    constexpr int num_waves = T::BLOCK_SIZE / opus::get_warp_size();
    constexpr int threads_k = T::B_K / T::VEC_A;
    constexpr int threads_m_per_block = T::BLOCK_SIZE / threads_k;
    constexpr int threads_m_per_wave = opus::get_warp_size() / threads_k;

    constexpr auto ga_block_shape = opus::make_tuple(
        opus::number<T::B_M / threads_m_per_block>{},
        opus::number<threads_m_per_wave>{},
        opus::number<num_waves>{},
        opus::number<threads_k>{},
        opus::number<T::VEC_A>{});

    constexpr auto ga_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_A>(
        ga_block_shape,
        opus::unfold_x_stride(ga_block_dim, ga_block_shape, opus::tuple{stride_a, 1_I}),
        opus::unfold_p_coord(ga_block_dim, opus::tuple{lane_id / threads_k, wave_id, lane_id % threads_k}));
}

// Create layout for storing A matrix to shared memory
template<typename T>
inline __device__ auto make_layout_sa(int lane_id, int wave_id) {
    constexpr int num_waves = T::BLOCK_SIZE / opus::get_warp_size();

    constexpr auto sa_block_shape = opus::make_tuple(
        opus::number<T::smem_m_rep / num_waves>{},
        opus::number<num_waves>{},
        opus::number<opus::get_warp_size()>{},
        opus::number<T::VEC_A>{});

    constexpr auto sa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_A>(
        sa_block_shape,
        opus::unfold_x_stride(sa_block_dim, sa_block_shape, opus::tuple{T::smem_linear_wave + T::smem_padding, 1_I}),
        opus::unfold_p_coord(sa_block_dim, opus::tuple{wave_id, lane_id}));
}

// Create layout for reading A matrix from shared memory to registers
template<typename T>
inline __device__ auto make_layout_ra(int lane_id, int wave_id_m) {
    constexpr int num_waves = T::BLOCK_SIZE / opus::get_warp_size();

    constexpr auto ra_block_shape = opus::make_tuple(
        opus::number<T::E_M>{},
        opus::number<num_waves>{},
        opus::number<T::T_M>{},
        opus::number<T::W_M / num_waves>{},
        opus::number<opus::get_warp_size() / T::W_M>{},
        opus::number<T::VEC_A>{});

    constexpr auto ra_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    auto lane_id_m = lane_id % T::W_M;

    return opus::make_layout<T::VEC_A>(
        ra_block_shape,
        opus::unfold_x_stride(ra_block_dim, ra_block_shape, opus::tuple{T::smem_linear_wave + T::smem_padding, T::B_K, 1_I}),
        opus::unfold_p_coord(ra_block_dim, opus::tuple{lane_id_m % num_waves, wave_id_m, lane_id_m / num_waves, lane_id / T::W_M}));
}

// Create layout for loading B matrix from global memory to shared memory
template<typename T>
inline __device__ auto make_layout_gb_sb(int lane_id, int wave_id, int stride) {
    constexpr int num_waves = T::BLOCK_SIZE / opus::get_warp_size();
    constexpr int waves_k = T::B_K / T::W_K;
    constexpr int waves_n = num_waves / waves_k;
    
    constexpr auto flat_b_block_shape = opus::make_tuple(
        opus::number<T::flat_n_per_block / waves_n>{},
        opus::number<waves_n>{},
        opus::number<waves_k>{},
        opus::number<opus::get_warp_size()>{},
        opus::number<T::VEC_B>{});

    constexpr auto flat_b_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_B>(
        flat_b_block_shape,
        opus::unfold_x_stride(flat_b_block_dim, flat_b_block_shape, opus::tuple{stride, 1_I}),
        opus::unfold_p_coord(flat_b_block_dim, opus::tuple{wave_id / waves_k, wave_id % waves_k, lane_id}));
}

// Create layout for reading B matrix from shared memory to registers
template<typename T>
inline __device__ auto make_layout_rb(int lane_id, int wave_id_n, int flat_k) {
    constexpr auto flat_b_block_shape = opus::make_tuple(
        opus::number<T::E_N>{},
        opus::number<T::T_N>{},
        opus::number<opus::get_warp_size()>{},
        opus::number<T::VEC_B>{});

    constexpr auto flat_b_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_B>(
        flat_b_block_shape,
        opus::unfold_x_stride(flat_b_block_dim, flat_b_block_shape, opus::tuple{flat_k, 1_I}),
        opus::unfold_p_coord(flat_b_block_dim, opus::tuple{wave_id_n, lane_id}));
}

// FlatMM kernel
template<typename Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 2) void flatmm_kernel(opus_fmm_kargs kargs) {
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    using D_A = typename T::D_A;
    using D_B = typename T::D_B;
    using D_C = typename T::D_C;
    using D_ACC = typename T::D_ACC;

    // Calculate global workgroup and tile indices
    int wgid = (blockIdx.y * gridDim.x) + blockIdx.x;
    const int num_tiles_m = ceil_div(kargs.m, T::B_M);
    int row = (wgid % num_tiles_m) * T::B_M;
    int col = (wgid / num_tiles_m) * T::B_N;
    int flat_col = col / T::W_N;

    int batch_id = blockIdx.z;
    int wave_id = __builtin_amdgcn_readfirstlane(threadIdx.x / get_warp_size());
    int lane_id = threadIdx.x % get_warp_size();

    int flat_k = T::flat_k_per_wave * kargs.k / T::W_K;
    // int flat_n = kargs.n / T::W_N;

    // Setup global memory pointers for A, B, C matrices
    auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + batch_id*kargs.stride_a_batch + row*kargs.stride_a);
    auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + batch_id*kargs.stride_b_batch + flat_col*flat_k);
    auto g_c = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id*kargs.stride_c_batch + row*kargs.stride_c + col);

    // Calculate wave position in the output tile
    int wave_id_m = wave_id / T::T_N;
    int wave_id_n = wave_id % T::T_N;

    // Create memory layouts for loading A, B matrices
    auto u_ga = make_layout_ga<T>(lane_id, wave_id, kargs.stride_a);
    auto u_sa = make_layout_sa<T>(lane_id, wave_id);
    auto u_ra = make_layout_ra<T>(lane_id, wave_id_m);
    auto u_gb = make_layout_gb_sb<T>(lane_id, wave_id, flat_k);
    auto u_sb = make_layout_gb_sb<T>(lane_id, wave_id, T::flat_k_per_block);
    auto u_rb = make_layout_rb<T>(lane_id, wave_id_n, T::flat_k_per_block);

    // Allocate shared memory for A/B subtiles and double buffer
    constexpr int smem_a_byte = T::smem_m_rep * (T::smem_linear_wave + T::smem_padding) * sizeof(D_A);
    constexpr int smem_b_byte = T::flat_n_per_block * T::flat_k_per_block * sizeof(D_B);
    __shared__ char smem_buf[smem_a_byte + smem_b_byte];
    auto s_a = make_smem(reinterpret_cast<D_A*>(smem_buf));
    auto s_b = make_smem(reinterpret_cast<D_B*>(smem_buf + smem_a_byte));

    // Create tiled MFMA operation with specified tile sizes and types
    auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
        seq<T::E_M, T::E_N, T::E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});

    typename decltype(mma)::vtype_a v_a[5];
    typename decltype(mma)::vtype_b v_b[3];
    typename decltype(mma)::vtype_c v_c[2];
    clear(v_c[0]);
    clear(v_c[1]);

    // Calculate offsets for A, B matrices based on tile indices
    auto a_offset = [&](int reg_tile_m, int reg_tile_k) {
        return reg_tile_m * (T::smem_m_rep / 2) * (T::smem_linear_wave + T::smem_padding)  + reg_tile_k * T::REG_SLICE_K;
    };
    auto b_offset = [&](int reg_tile_k) {
        return reg_tile_k * T::flat_k_per_wave;
    };

    const int loops = ceil_div(kargs.k, T::B_K);
    
    // Load first tile into shared memory
    {
        auto a_block_tile = load<T::VEC_A>(g_a, u_ga);
        store<T::VEC_A>(s_a, a_block_tile, u_sa);
        auto b_block_tile = load<T::VEC_B>(g_b, u_gb);
        store<T::VEC_B>(s_b, b_block_tile, u_sb);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
    }

    if (wave_id_m == 1) {
        __builtin_amdgcn_s_barrier();
    }

    // Main loop
    for(int tile = 0; tile < loops - 1; tile ++) {
        // Cluster 0
        auto a_buffer_next = load<T::VEC_A>(g_a, u_ga, (tile + 1) * T::B_K);
        v_a[0] = load<T::VEC_A>(s_a, u_ra + a_offset(0, 0));
        v_a[1] = load<T::VEC_A>(s_a, u_ra + a_offset(1, 0));
        v_b[0] = load<T::VEC_B>(s_b, u_rb + b_offset(0));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 1
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[0] = mma(v_a[0], v_b[0], v_c[0]);
        v_c[1] = mma(v_a[1], v_b[0], v_c[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 2
        v_b[1] = load<T::VEC_B>(s_b, u_rb + b_offset(1));
        v_a[2] = load<T::VEC_A>(s_a, u_ra + a_offset(0, 1));
        v_a[3] = load<T::VEC_A>(s_a, u_ra + a_offset(1, 1));
        v_b[0] = load<T::VEC_B>(s_b, u_rb + b_offset(2));
        v_a[0] = load<T::VEC_A>(s_a, u_ra + a_offset(0, 2));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 3
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[0] = mma(v_a[2], v_b[1], v_c[0]);
        v_c[1] = mma(v_a[3], v_b[1], v_c[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 4
        auto b_buffer_next = load<T::VEC_B>(g_b, u_gb, (tile + 1) * T::flat_k_per_block);
        v_a[1] = load<T::VEC_A>(s_a, u_ra + a_offset(1, 2));
        v_b[2] = load<T::VEC_B>(s_b, u_rb + b_offset(3));
        v_a[4] = load<T::VEC_A>(s_a, u_ra + a_offset(0, 3));
        v_a[3] = load<T::VEC_A>(s_a, u_ra + a_offset(1, 3));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 5
        __builtin_amdgcn_s_setprio(1);
        v_c[0] = mma(v_a[0], v_b[0], v_c[0]);
        v_c[1] = mma(v_a[1], v_b[0], v_c[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 6
        s_waitcnt_lgkmcnt(0_I);
        store<T::VEC_A>(s_a, a_buffer_next, u_sa);
        store<T::VEC_B>(s_b, b_buffer_next, u_sb);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 7
        __builtin_amdgcn_s_setprio(1);
        v_c[0] = mma(v_a[4], v_b[2], v_c[0]);
        v_c[1] = mma(v_a[3], v_b[2], v_c[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }

    // Epilogue
    // Cluster 0
    __builtin_amdgcn_sched_barrier(0);
    v_b[0] = load<T::VEC_B>(s_b, u_rb + b_offset(0));
    v_a[0] = load<T::VEC_A>(s_a, u_ra + a_offset(0, 0));
    v_a[1] = load<T::VEC_A>(s_a, u_ra + a_offset(1, 0));
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 1
    __builtin_amdgcn_s_setprio(1);
    v_c[0] = mma(v_a[0], v_b[0], v_c[0]);
    v_c[1] = mma(v_a[1], v_b[0], v_c[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 2
    v_b[1] = load<T::VEC_B>(s_b, u_rb + b_offset(1));
    v_a[2] = load<T::VEC_A>(s_a, u_ra + a_offset(0, 1));
    v_a[3] = load<T::VEC_A>(s_a, u_ra + a_offset(1, 1));
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 3
    __builtin_amdgcn_s_setprio(1);
    v_c[0] = mma(v_a[2], v_b[1], v_c[0]);
    v_c[1] = mma(v_a[3], v_b[1], v_c[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 4
    v_b[0] = load<T::VEC_B>(s_b, u_rb + b_offset(2));
    v_a[0] = load<T::VEC_A>(s_a, u_ra + a_offset(0, 2));
    v_a[1] = load<T::VEC_A>(s_a, u_ra + a_offset(1, 2));
    v_b[1] = load<T::VEC_B>(s_b, u_rb + b_offset(3));
    v_a[2] = load<T::VEC_A>(s_a, u_ra + a_offset(0, 3));
    v_a[3] = load<T::VEC_A>(s_a, u_ra + a_offset(1, 3));
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 5
    __builtin_amdgcn_s_setprio(1);
    v_c[0] = mma(v_a[0], v_b[0], v_c[0]);
    v_c[1] = mma(v_a[1], v_b[0], v_c[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 7
    __builtin_amdgcn_s_setprio(1);
    v_c[0] = mma(v_a[2], v_b[1], v_c[0]);
    v_c[1] = mma(v_a[3], v_b[1], v_c[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    if (wave_id_m == 0) {
        __builtin_amdgcn_s_barrier();
    }

    // Store results to global memory
    auto u_gc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(kargs.stride_c, 1_I), opus::make_tuple(wave_id_m, lane_id % mma.grpn_c, wave_id_n, lane_id / mma.grpn_c));

    auto c_offset = [&](int reg_tile_m) {
        return reg_tile_m * T::REG_BLOCK_M * T::T_M * kargs.stride_c;
    };

    auto v_c_f16 = cast<opus::fp16_t>(v_c[0]);
    store<T::VEC_C>(g_c, v_c_f16, u_gc + c_offset(0));
    v_c_f16 = cast<opus::fp16_t>(v_c[1]);
    store<T::VEC_C>(g_c, v_c_f16, u_gc + c_offset(1));
}

// Fill 2D matrix with random values in specified range
template<typename T>
void rand_vector_2d(T* ptr, int m, int n, int ld, float min_val = 0.0f, float max_val = 1.0f) {
    #pragma omp parallel
    {
        std::random_device rd;
        std::mt19937 gen(rd() + omp_get_thread_num());
        std::uniform_real_distribution<float> dis(min_val, max_val);
        #pragma omp for collapse(2)
        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) {
                ptr[i * ld + j] = static_cast<T>(dis(gen));
            }
        }
    }
}

// Validate computed results against reference with error threshold
template<typename T>
bool valid_vector(const T* ref, const T* result, int n, float threshold = 1e-3f) {
    int errors = 0;
    for(int i = 0; i < n; i++) {
        float diff = std::abs(static_cast<float>(ref[i]) - static_cast<float>(result[i]));
        if(diff > threshold) {
            if(errors < 10) {
                printf("Error at %d: ref=%.6f, result=%.6f, diff=%.6f\n", 
                       i, static_cast<float>(ref[i]), static_cast<float>(result[i]), diff);
            }
            errors++;
            if(errors >= 10) break;
        }
    }
    return errors == 0;
}

// CPU reference GEMM implementation for validation
void gemm_ref(const fp16_t* a, const fp16_t* b, fp16_t* c, int m, int n, int k, int lda, int ldb, int ldc) {
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            float sum = 0.0f;
            for(int p = 0; p < k; p++) {
                sum += static_cast<float>(a[i * lda + p]) * static_cast<float>(b[p * ldb + j]);
            }
            c[i * ldc + j] = static_cast<fp16_t>(sum);
        }
    }
}

// Benchmark kernel performance with warm-up and timing
template<typename Traits>
void benchmark_kernel(const opus_fmm_kargs& kargs, dim3 grid, dim3 block, int warmup = 50, int iterations = 100) {
    // Warm up
    for (int i = 0; i < warmup; ++i) {
        flatmm_kernel<Traits><<<grid, block>>>(kargs);
        CHECK_HIP_KERNEL_LAUNCH();
    }

    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));

    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipEventRecord(start));

    // Timed iterations
    for (int i = 0; i < iterations; ++i) {
        flatmm_kernel<Traits><<<grid, block>>>(kargs);
        CHECK_HIP_KERNEL_LAUNCH();
    }

    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));

    float total_time = 0;
    CHECK_HIP(hipEventElapsedTime(&total_time, start, stop));

    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));

    const float avg_time = total_time / iterations;
    const std::size_t flop = std::size_t(2) * kargs.m * kargs.n * kargs.k * kargs.batch;
    const float tflops = static_cast<float>(flop) / 1.0e9f / avg_time;

    printf("Kernel Performance: avg_time=%.4f ms, %.2f TFlops\n", avg_time, tflops);
}

// Shuffle B matrix for optimized memory access pattern in FlatMM kernel
template<typename DataType, typename Traits>
void shuffle_b(const DataType* b_input, DataType* b_output, int N, int K) {
    using T = opus::remove_cvref_t<Traits>;
    
    constexpr int WarpSize      = opus::get_warp_size();
    constexpr int N_Warp_Tile   = T::W_N;
    constexpr int K_Warp_Tile   = T::W_K;
    constexpr int MaxVecSize    = 16 / sizeof(DataType);
    constexpr int KLane         = WarpSize / N_Warp_Tile;
    constexpr int ItemsPerAccess = std::min(MaxVecSize, K_Warp_Tile / KLane);

    static_assert(WarpSize % N_Warp_Tile == 0, "WarpSize must be divisible by N_Warp_Tile");
    assert(N % N_Warp_Tile == 0 && "N must be divisible by N_Warp_Tile");
    assert(K % ItemsPerAccess == 0 && "K must be divisible by ItemsPerAccess");

    const int dim0 = N / N_Warp_Tile;      // Number of N tiles
    const int dim1 = N_Warp_Tile;          // Lanes per N tile
    const int dim2 = K / ItemsPerAccess;   // Number of K blocks
    const int dim3 = ItemsPerAccess;       // Items per access

    // Output layout: [dim0, dim2, dim1, dim3] = [N/N_tile, K/items, N_tile, items]
    // This matches the flat K layout expected by the kernel
    #pragma omp parallel for collapse(2)
    for(int d0 = 0; d0 < dim0; ++d0) {
        for(int d2 = 0; d2 < dim2; ++d2) {
            const int n_base = d0 * N_Warp_Tile;
            const int k_base = d2 * ItemsPerAccess;
            
            for(int d1 = 0; d1 < dim1; ++d1) {
                const int n = n_base + d1;
                const int k_start = k_base;
                
                const DataType* src = &b_input[k_start * N + n];
                DataType* dst = &b_output[d0 * (dim2 * dim1 * dim3) + 
                                         d2 * (dim1 * dim3) + 
                                         d1 * dim3];
                
                for(int d3 = 0; d3 < dim3; ++d3) {
                    dst[d3] = src[d3 * N];
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    constexpr int BLOCK_SIZE = 512;
    constexpr int BLOCK_M = 256;
    constexpr int BLOCK_N = 128;
    constexpr int BLOCK_K = 64;

    using Traits = opus_flatmm_traits<
        BLOCK_SIZE,
        opus::seq<BLOCK_M, BLOCK_N, BLOCK_K>,
        opus::tuple<fp16_t, fp16_t, fp16_t, float, fp16_t>,
        opus::seq<8, 8, 4>,
        false
    >;

    // Default problem sizes
    int M = 256;
    int N = 512;
    int K = 128;
    int batch = 8;

    // Parse command line arguments: -m -n -k -b
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if ((std::strcmp(arg, "-m") == 0 || std::strcmp(arg, "--m") == 0) && i + 1 < argc) {
            M = std::atoi(argv[++i]);
        } else if ((std::strcmp(arg, "-n") == 0 || std::strcmp(arg, "--n") == 0) && i + 1 < argc) {
            N = std::atoi(argv[++i]);
        } else if ((std::strcmp(arg, "-k") == 0 || std::strcmp(arg, "--k") == 0) && i + 1 < argc) {
            K = std::atoi(argv[++i]);
        } else if ((std::strcmp(arg, "-b") == 0 || std::strcmp(arg, "--b") == 0) && i + 1 < argc) {
            batch = std::atoi(argv[++i]);
        }
    }

    if (M <= 0 || N <= 0 || K <= 0 || batch <= 0) {
        std::cerr << "Invalid problem size: M,N,K and batch must be positive.\n";
        return 1;
    }

    // Allocate host memory
    auto host_a = std::make_unique<fp16_t[]>(batch * M * K);
    auto host_b = std::make_unique<fp16_t[]>(batch * N * K);
    auto host_c = std::make_unique<fp16_t[]>(batch * M * N);
    auto host_b_shuffled = std::make_unique<fp16_t[]>(batch * N * K);
    auto host_c_out = std::make_unique<fp16_t[]>(batch * M * N);

    // Initialize data
    for(int b = 0; b < batch; b++) {
        rand_vector_2d(host_a.get() + b * M * K, M, K, K, 0.0f, 1.0f);
        rand_vector_2d(host_b.get() + b * N * K, K, N, N, -0.5f, 0.5f);
    }

    // Shuffle B matrix for optimized GPU memory access
    printf("Shuffling B matrix for FlatMM layout (batch=%d)...\n", batch);
    for(int b = 0; b < batch; b++) {
        shuffle_b<fp16_t, Traits>(
            host_b.get() + b * N * K,
            host_b_shuffled.get() + b * N * K,
            N, K);
    }

    // Allocate device memory
    fp16_t *dev_a, *dev_b, *dev_c;
    CHECK_HIP(hipMalloc(&dev_a, batch * M * K * sizeof(fp16_t)));
    CHECK_HIP(hipMalloc(&dev_b, batch * N * K * sizeof(fp16_t)));
    CHECK_HIP(hipMalloc(&dev_c, batch * M * N * sizeof(fp16_t)));

    // Copy to device (use shuffled B matrix)
    CHECK_HIP(hipMemcpy(dev_a, host_a.get(), batch * M * K * sizeof(fp16_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dev_b, host_b_shuffled.get(), batch * N * K * sizeof(fp16_t), hipMemcpyHostToDevice));

    // Setup kernel arguments
    opus_fmm_kargs kargs{};
    kargs.ptr_a = dev_a;
    kargs.ptr_b = dev_b;
    kargs.ptr_c = dev_c;
    kargs.ptr_bias = nullptr;
    kargs.m = M;
    kargs.n = N;
    kargs.k = K;
    kargs.batch = batch;
    kargs.stride_a = K;
    kargs.stride_b = N;
    kargs.stride_c = N;
    kargs.stride_a_batch = M * K;
    kargs.stride_b_batch = N * K;
    kargs.stride_c_batch = M * N;

    // Calculate grid dimensions and launch kernel
    const int num_tiles_m = ceil_div(M, BLOCK_M);
    const int num_tiles_n = ceil_div(N, BLOCK_N);
    dim3 grid(num_tiles_m * num_tiles_n, 1, batch);
    dim3 block(BLOCK_SIZE);

    printf("Launching FlatMM kernel: M=%d, N=%d, K=%d, grid=(%d,%d,%d), block=%d\n",
           M, N, K, grid.x, grid.y, grid.z, BLOCK_SIZE);

    flatmm_kernel<Traits><<<grid, block>>>(kargs);
    
    CHECK_HIP_KERNEL_LAUNCH();

    // Copy results back to host for validation
    CHECK_HIP(hipMemcpy(host_c_out.get(), dev_c, batch * M * N * sizeof(fp16_t), hipMemcpyDeviceToHost));

    // Verify each batch against CPU reference implementation
    bool all_valid = true;
    for(int b = 0; b < batch; b++) {
        gemm_ref(
            host_a.get() + b * M * K,
            host_b.get() + b * N * K,
            host_c.get() + b * M * N,
            M, N, K, K, N, N);
        bool valid = valid_vector(
            host_c.get() + b * M * N,
            host_c_out.get() + b * M * N,
            M * N, 5e-1f);
        printf("[FlatMM batch %d/%d: %dx%dx%d, block_%dx%dx%d] %s\n", 
               b + 1, batch, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K,
               valid ? "✓ VALID" : "✗ FAIL");
        all_valid = all_valid && valid;
    }

    printf("\n[Overall] %s\n", all_valid ? "✓ ALL BATCHES VALID" : "✗ SOME BATCHES FAILED");

    // Benchmark kernel performance
    printf("\n");
    benchmark_kernel<Traits>(kargs, grid, block);
    printf("\n");

    // Cleanup
    CHECK_HIP(hipFree(dev_a));
    CHECK_HIP(hipFree(dev_b));
    CHECK_HIP(hipFree(dev_c));

    return all_valid ? 0 : 1;
}