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

// sched barrier mask values
#define MFMA_MASK 0x08
#define VMEM_MASK 0x20
#define DS_READ_MASK 0x100
#define DS_WRITE_MASK 0x200

__host__ __device__ inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

/**
 * @brief Transform a workgroup ID to a new workgroup ID based on the chunk size and number of XCDs.
 * @param workgroup_id The original workgroup ID.
 * @param num_workgroups The total number of workgroups.
 * @param num_xcds The number of XCDs.
 * @param chunk_size The chunk size.
 * @return The new workgroup ID.
 */
__host__ __device__ inline int chiplet_transform_chunked(
    int workgroup_id, 
    int num_workgroups,
    int num_xcds,
    int chunk_size 
) {
    // Current XCD
    int xcd = workgroup_id % num_xcds;

    // Largest full (NUM_XCDS*CHUNK_SIZE)-aligned block
    int block = num_xcds * chunk_size;
    int limit = (num_workgroups / block) * block;

    // If pid beyond the last full block, leave unchanged
    if (workgroup_id > limit) return workgroup_id;

    // Local PID (within round-robin assignment)
    int local_pid    = workgroup_id / num_xcds;
    int chunk_idx    = local_pid / chunk_size;
    int pos_in_chunk = local_pid % chunk_size;

    // New PID
    return chunk_idx * block + xcd * chunk_size + pos_in_chunk;
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

    static constexpr int T_M = 1; // waves along M
    static constexpr int T_N = 4; // waves along N
    static constexpr int T_K = 1; // waves along K

    static_assert(BLOCK_SIZE / opus::get_warp_size() == T_M * T_N * T_K);
    static_assert(T_K == 1);

    static constexpr int W_M = 16; // wave gemm size M
    static constexpr int W_N = 16; // wave gemm size N
    static constexpr int W_K = 32; // wave gemm size K

    static_assert(B_M % (W_M * T_M) == 0);
    static_assert(B_N % (W_N * T_N) == 0);
    static_assert(B_K % (W_K * T_K) == 0);

    static constexpr int E_M = B_M / (W_M * T_M);   // expand, repeat how many times along each dim
    static constexpr int E_N = B_N / (W_N * T_N);   // expand, repeat how many times along each dim
    static constexpr int E_K = B_K / (W_K * T_K);   // expand, repeat how many times along each dim

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
    static constexpr int smem_padding = 2 * 16 / sizeof(D_A);

    // number of instructions
    static constexpr int a_buffer_load_insts = B_M * B_K / (BLOCK_SIZE * VEC_A);
    static constexpr int b_buffer_load_insts = flat_n_per_block * flat_k_per_block / (opus::get_warp_size() * T_N * VEC_B);   
    static constexpr int c_mfma_insts = E_M * E_N * E_K;
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
inline __device__ auto make_layout_ga(int lane_id, int wave_id_m, int wave_id_n, int stride_a) {
    constexpr int threads_k = T::B_K / T::VEC_A;
    constexpr int threads_m_per_block = T::BLOCK_SIZE / threads_k;
    constexpr int threads_m_per_wave = opus::get_warp_size() / threads_k;

    constexpr auto ga_block_shape = opus::make_tuple(
        opus::number<T::B_M / threads_m_per_block>{},
        opus::number<T::T_M>{},
        opus::number<threads_m_per_wave>{},
        opus::number<T::T_N>{},
        opus::number<threads_k>{},
        opus::number<T::VEC_A>{});

    constexpr auto ga_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_A>(
        ga_block_shape,
        opus::unfold_x_stride(ga_block_dim, ga_block_shape, opus::tuple{stride_a, 1_I}),
        opus::unfold_p_coord(ga_block_dim, opus::tuple{wave_id_m, lane_id / threads_k, wave_id_n, lane_id % threads_k}));
}

// Create layout for storing A matrix to shared memory
template<typename T>
inline __device__ auto make_layout_sa(int lane_id, int wave_id_m, int wave_id_n) {
    constexpr int num_waves = T::BLOCK_SIZE / opus::get_warp_size();

    constexpr auto sa_block_shape = opus::make_tuple(
        opus::number<T::smem_m_rep / num_waves>{},
        opus::number<T::T_M>{},
        opus::number<T::T_N>{},
        opus::number<opus::get_warp_size()>{},
        opus::number<T::VEC_A>{});

    constexpr auto sa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_A>(
        sa_block_shape,
        opus::unfold_x_stride(sa_block_dim, sa_block_shape, opus::tuple{T::smem_linear_wave + T::smem_padding, 1_I}),
        opus::unfold_p_coord(sa_block_dim, opus::tuple{wave_id_m, wave_id_n, lane_id}));
}

// Create layout for reading A matrix from shared memory to registers
template<typename T>
inline __device__ auto make_layout_ra(int lane_id, int wave_id) {
    constexpr int threads_k = T::B_K / T::VEC_A;
    constexpr int threads_m_per_block = T::BLOCK_SIZE / threads_k;
    constexpr int num_waves = T::BLOCK_SIZE / opus::get_warp_size();
    constexpr auto ra_block_shape = opus::make_tuple(
        opus::number<T::E_M / (threads_m_per_block / T::W_M)>{},
        opus::number<num_waves>{},
        opus::number<threads_m_per_block / T::W_M>{},
        opus::number<T::W_M / num_waves>{},
        opus::number<T::E_K>{},
        opus::number<opus::get_warp_size() / T::W_M>{},
        opus::number<T::VEC_A>{});

    constexpr auto ra_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    auto lane_id_m = lane_id % T::W_M;

    return opus::make_layout<T::VEC_A>(
        ra_block_shape,
        opus::unfold_x_stride(ra_block_dim, ra_block_shape, opus::tuple{T::smem_linear_wave + T::smem_padding, 1_I}),
        opus::unfold_p_coord(ra_block_dim, opus::tuple{lane_id_m % num_waves, lane_id_m / num_waves, lane_id / T::W_M}));
}

// Create layout for loading flat B matrix from global memory
template<typename T>
inline __device__ auto make_layout_gb(int lane_id, int wave_id_n, int flat_k) {
    constexpr auto flat_b_block_shape = opus::make_tuple(
        opus::number<T::E_N>{},
        opus::number<T::T_N>{},
        opus::number<T::E_K>{},
        opus::number<opus::get_warp_size()>{},
        opus::number<T::VEC_B>{});
    constexpr auto flat_b_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<T::VEC_B>(
        flat_b_block_shape,
        opus::unfold_x_stride(flat_b_block_dim, flat_b_block_shape, opus::tuple{flat_k, 1_I}),
        opus::unfold_p_coord(flat_b_block_dim, opus::tuple{wave_id_n, lane_id}));
}

// FlatMM kernel
template<typename Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 2) void flatmm_kernel(opus_fmm_kargs kargs) {
    using T = opus::remove_cvref_t<Traits>;
    using D_A = typename T::D_A;
    using D_B = typename T::D_B;
    using D_C = typename T::D_C;
    using D_ACC = typename T::D_ACC;

    // Calculate global workgroup and tile indices
    int wgid = (blockIdx.y * gridDim.x) + blockIdx.x;
#if 0
    const int num_tiles_m = ceil_div(kargs.m, T::B_M);
    int row = (wgid % num_tiles_m) * T::B_M;
    int col = (wgid / num_tiles_m) * T::B_N;
    int flat_col = col / T::W_N;
#else
    const int NUM_WGS = gridDim.x * gridDim.y;
    const int NUM_XCDS = 8;  // Number of XCDs (chiplets)
    const int WGM = 8;       // Workgroup tile size for M dimension grouping
    
    // Swizzle chiplet so that wgids are in the same XCD
    wgid = chiplet_transform_chunked(wgid, NUM_WGS, NUM_XCDS, 64);
    
    // Swizzle for better L2 within the same XCD
    const int num_tiles_m = ceil_div(kargs.m, T::B_M);
    const int num_tiles_n = ceil_div(kargs.n, T::B_N);
    const int num_wgid_in_group = WGM * num_tiles_n;
    int group_id = wgid / num_wgid_in_group;
    int first_pid_m = group_id * WGM;
    int group_size_m = min(num_tiles_m - first_pid_m, WGM);
    int pid_m = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int pid_n = (wgid % num_wgid_in_group) / group_size_m;
    
    // Assign the tile's row/column based on the pid_m and pid_n
    int row = pid_m * T::B_M;
    int col = pid_n * T::B_N;
    int flat_col = col / T::W_N;
#endif

    int batch_id = blockIdx.z;
    int wave_id = __builtin_amdgcn_readfirstlane(threadIdx.x / opus::get_warp_size());
    int lane_id = threadIdx.x % opus::get_warp_size();

    int flat_k = T::flat_k_per_wave * kargs.k / T::W_K;
    // int flat_n = kargs.n / T::W_N;

    // Setup global memory pointers for A, B, C matrices
    auto g_a = opus::make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + batch_id*kargs.stride_a_batch + row*kargs.stride_a);
    auto g_b = opus::make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + batch_id*kargs.stride_b_batch + flat_col*flat_k);
    auto g_c = opus::make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id*kargs.stride_c_batch + row*kargs.stride_c + col);

    // Calculate wave position in the output tile
    int wave_id_m = wave_id / T::T_N;
    int wave_id_n = wave_id % T::T_N;

    // Create memory layouts for loading A, B matrices
    auto u_ga = make_layout_ga<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
    auto u_sa = make_layout_sa<T>(lane_id, wave_id_m, wave_id_n);
    auto u_ra = make_layout_ra<T>(lane_id, wave_id);
    auto u_gb = make_layout_gb<T>(lane_id, wave_id_n, flat_k);

    // Allocate shared memory for A matrix with padding to avoid bank conflicts
    constexpr int smem_a_byte = T::smem_m_rep * (T::smem_linear_wave + T::smem_padding) * sizeof(D_A);
    __shared__ char smem_a[smem_a_byte * 2];
    opus::smem<D_A> s_a[2] = {
        opus::make_smem(reinterpret_cast<D_A*>(smem_a)),
        opus::make_smem(reinterpret_cast<D_A*>(smem_a + smem_a_byte))
    };

    // Create tiled MFMA operation with specified tile sizes and types
    auto mma = opus::make_tiled_mma<D_A, D_B, D_ACC>(
        opus::seq<T::E_M, T::E_N, T::E_K>{},
        opus::seq<T::T_M, T::T_N, T::T_K>{},
        opus::seq<T::W_M, T::W_N, T::W_K>{},
        opus::mfma_adaptor_swap_ab{});

    typename decltype(mma)::vtype_a v_a;
    typename decltype(mma)::vtype_b v_b[2];
    typename decltype(mma)::vtype_c v_c;
    opus::clear(v_c);

    int loops = (kargs.k + T::B_K - 1) / T::B_K;

    // Prologue
    v_b[0] = g_b.template load<T::VEC_B>(u_gb);
    u_gb += T::flat_k_per_block;
    __builtin_amdgcn_sched_group_barrier(VMEM_MASK, T::b_buffer_load_insts, 0);

    g_a.template async_load<T::VEC_A>(s_a[0].ptr, u_ga, u_sa);
    u_ga += T::B_K;
    __builtin_amdgcn_sched_group_barrier(VMEM_MASK, T::a_buffer_load_insts, 0);

    if (loops > 1) {
        v_b[1] = g_b.template load<T::VEC_B>(u_gb);
        u_gb += T::flat_k_per_block;
        __builtin_amdgcn_sched_group_barrier(VMEM_MASK, T::b_buffer_load_insts, 0);

        g_a.template async_load<T::VEC_A>(s_a[1].ptr, u_ga, u_sa);
        u_ga += T::B_K;
        __builtin_amdgcn_sched_group_barrier(VMEM_MASK, T::a_buffer_load_insts, 0);
    }

    // Main Loop - process 2 tiles per iteration
    for(auto i = 0; i < loops - 2; i += 2) {
        // Process tile i
        opus::s_waitcnt_vmcnt(opus::number<T::a_buffer_load_insts + T::b_buffer_load_insts>{});
        __builtin_amdgcn_s_barrier();

        v_a = s_a[0].template load<T::VEC_A>(u_ra);
        v_c = mma(v_a, v_b[0], v_c);

        // Load tile i+2
        v_b[0] = g_b.template load<T::VEC_B>(u_gb);
        u_gb += T::flat_k_per_block;
        __builtin_amdgcn_sched_group_barrier(VMEM_MASK, T::b_buffer_load_insts, 0);

        opus::s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();

        g_a.template async_load<T::VEC_A>(s_a[0].ptr, u_ga, u_sa);
        u_ga += T::B_K;
        __builtin_amdgcn_sched_group_barrier(VMEM_MASK, T::a_buffer_load_insts, 0);

        // Process tile i+1
        opus::s_waitcnt_vmcnt(opus::number<T::a_buffer_load_insts + T::b_buffer_load_insts>{});
        __builtin_amdgcn_s_barrier();

        v_a = s_a[1].template load<T::VEC_A>(u_ra);
        v_c = mma(v_a, v_b[1], v_c);

        // Load tile i+3
        v_b[1] = g_b.template load<T::VEC_B>(u_gb);
        u_gb += T::flat_k_per_block;
        __builtin_amdgcn_sched_group_barrier(VMEM_MASK, T::b_buffer_load_insts, 0);

        opus::s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        
        g_a.template async_load<T::VEC_A>(s_a[1].ptr, u_ga, u_sa);
        u_ga += T::B_K;
        __builtin_amdgcn_sched_group_barrier(VMEM_MASK, T::a_buffer_load_insts, 0);
    }

    // Epilogue
    if (loops >= 2) {
        // Tile loops-2
        opus::s_waitcnt_vmcnt(opus::number<T::a_buffer_load_insts + T::b_buffer_load_insts>{});
        __builtin_amdgcn_s_barrier();

        v_a = s_a[0].template load<T::VEC_A>(u_ra);
        v_c = mma(v_a, v_b[0], v_c);

        // Tile loops-1 (last tile)
        opus::s_waitcnt_vmcnt(0_I);
        __builtin_amdgcn_s_barrier();

        v_a = s_a[1].template load<T::VEC_A>(u_ra);
        v_c = mma(v_a, v_b[1], v_c);
    } else if (loops == 1) {
        // Handle single tile case
        opus::s_waitcnt_vmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        
        v_a = s_a[0].template load<T::VEC_A>(u_ra);
        v_c = mma(v_a, v_b[0], v_c);
    }

    auto u_gc = opus::partition_layout_c<T::VEC_C>(
        mma,
        opus::make_tuple(kargs.stride_c, 1_I),
        opus::make_tuple(wave_id_m, lane_id % mma.grpn_c, wave_id_n, lane_id / mma.grpn_c));

    auto v_c_f16 = opus::cast<D_C>(v_c);
    g_c.template store<T::VEC_C>(v_c_f16, u_gc);
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
    constexpr int BLOCK_SIZE = 256;
    constexpr int BLOCK_M = 128;
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
            M * N, 5e-2f);
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