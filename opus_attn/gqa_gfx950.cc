#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <random>
#include <iostream>
#include <numeric>
#include <memory>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <omp.h>

#include <opus/opus.hpp>

using bf16_t = __bf16;
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

// Kernel arguments for GQA attention
struct opus_gqa_kargs {
    const void* __restrict__ ptr_q;  // [B, N, H, D]
    const void* __restrict__ ptr_k;  // [B, N, H_KV, D]
    const void* __restrict__ ptr_v;  // [B, N, H_KV, D]
    void* __restrict__ ptr_o;        // [B, N, H, D]
    int B;
    int N;
    int H;
    int H_KV;
    int D;
};

// Configuration traits for GQA kernel (tile sizes, data types, vector lengths, MFMA config)
template<int Q_TILE_SIZE_ = 32,
        int KV_TILE_SIZE_ = 64,
        int D_TILE_SIZE_ = 128,
        int NUM_WARPS_ = 8>
struct opus_gqa_traits {
    static constexpr int Q_TILE_SIZE = Q_TILE_SIZE_;
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;
    static constexpr int D_TILE_SIZE = D_TILE_SIZE_;
    static constexpr int NUM_WARPS = NUM_WARPS_;

    static constexpr int WARP_SIZE = opus::get_warp_size();
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;

    // Data types: Q/K/V/O share one bf16 type; accumulation fp32
    using D_ATTN = bf16_t;
    using D_ACC  = float;

    // MFMA wave layout
    static constexpr int T_M = NUM_WARPS; // waves along M
    static constexpr int T_N = 1;         // waves along N
    static constexpr int T_K = 1;         // waves along K

    // MFMA base tile (bf16 32x32x16 on gfx950)
    static constexpr int W_M = 32;
    static constexpr int W_N = 32;
    static constexpr int W_K = 16;

    // GEMM0: S[Q_TILE x KV_TILE] = Q[Q_TILE x D] @ K^T[D x KV_TILE]
    // Per-wave (T=1,1,1), expand covers the full tile
    static constexpr int GEMM0_E_M = Q_TILE_SIZE / W_M;   // 1
    static constexpr int GEMM0_E_N = KV_TILE_SIZE / W_N;  // 2
    static constexpr int GEMM0_E_K = D_TILE_SIZE / W_K;   // 8

    // GEMM1: O[Q_TILE x D] = P[Q_TILE x KV_TILE] @ V[KV_TILE x D]
    static constexpr int GEMM1_E_M = Q_TILE_SIZE / W_M;   // 1
    static constexpr int GEMM1_E_N = D_TILE_SIZE / W_N;   // 4
    static constexpr int GEMM1_E_K = KV_TILE_SIZE / W_K;  // 4

    // Vector lengths for global load/store
    static constexpr int VEC_Q    = 8;
    static constexpr int VEC_KV   = 8;
    static constexpr int VEC_TR_V = 4;
    static constexpr int VEC_O    = 4;

    // Minimal compact pixels for async copy for one wave
    static constexpr int D_128B_SIZE = 128 / sizeof(D_ATTN);
    static_assert(VEC_KV == 16 / sizeof(D_ATTN));
    static constexpr int smem_linear_wave = opus::get_warp_size() * 16 / sizeof(D_ATTN);
    static constexpr int smem_n_sub = smem_linear_wave / D_128B_SIZE;
    static constexpr int smem_n_rpt = KV_TILE_SIZE / smem_n_sub;
    static constexpr int smem_d_rpt = D_TILE_SIZE / D_128B_SIZE;
    static constexpr int smem_padding = 16 / sizeof(D_ATTN);

    static constexpr size_t smem_size_bytes() {
        return (smem_n_rpt * smem_d_rpt * (smem_linear_wave + smem_padding) + KV_TILE_SIZE * D_TILE_SIZE) * sizeof(D_ATTN);
    }
};

// Create layout for loading Q matrix from global memory
template<class T>
inline __device__ auto make_layout_q(int warp_id, int lane_id, int stride_q_n) {
    constexpr auto q_block_shape = opus::make_tuple(
        opus::number<T::GEMM0_E_M>{},
        opus::number<T::T_M>{},
        opus::number<T::W_M>{},
        opus::number<T::GEMM0_E_K>{},
        opus::number<T::WARP_SIZE / T::W_M>{},
        opus::number<T::VEC_Q>{});

    constexpr auto q_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_Q>(
        q_block_shape,
        opus::unfold_x_stride(q_block_dim, q_block_shape, opus::tuple{stride_q_n, 1_I}),
        opus::unfold_p_coord(q_block_dim, opus::tuple{warp_id, lane_id % T::W_M, lane_id / T::W_M}));
}

// Create layout for storing O matrix to global memory
template<class T>
inline __device__ auto make_layout_o(int warp_id, int lane_id, int stride_o_n) {
    constexpr auto o_block_shape = opus::make_tuple(
        opus::number<T::GEMM1_E_M>{},
        opus::number<T::T_M>{},
        opus::number<T::W_M>{},
        opus::number<T::GEMM1_E_N>{},
        opus::number<T::W_M * T::W_N / T::WARP_SIZE / T::VEC_O>{},
        opus::number<T::WARP_SIZE / T::W_M>{},
        opus::number<T::VEC_O>{});

    constexpr auto o_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_O>(
        o_block_shape,
        opus::unfold_x_stride(o_block_dim, o_block_shape, opus::tuple{stride_o_n, 1_I}),
        opus::unfold_p_coord(o_block_dim, opus::tuple{warp_id, lane_id % T::W_M, lane_id / T::W_M}));
}

// Create layout for loading K matrix from global memory
template<typename T>
inline __device__ auto make_layout_gk(int warp_id, int lane_id, int stride_kv_n) {
    constexpr int threads_d = T::D_128B_SIZE / T::VEC_KV;
    constexpr int threads_n_per_block = T::BLOCK_SIZE / threads_d;
    constexpr int threads_n_per_wave = opus::get_warp_size() / threads_d;

    constexpr auto gk_block_shape = opus::make_tuple(
        opus::number<T::smem_d_rpt>{},
        opus::number<T::KV_TILE_SIZE / threads_n_per_block>{},
        opus::number<threads_n_per_wave>{},
        opus::number<T::NUM_WARPS>{},
        opus::number<threads_d>{},
        opus::number<T::VEC_KV>{});

    constexpr auto gk_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_KV>(
        gk_block_shape,
        opus::unfold_x_stride(gk_block_dim, gk_block_shape, opus::tuple{T::D_128B_SIZE, stride_kv_n, 1_I}),
        opus::unfold_p_coord(gk_block_dim, opus::tuple{lane_id / threads_d, warp_id, lane_id % threads_d}));
}

// Create layout for storing K matrix to shared memory
template<typename T>
inline __device__ auto make_layout_sk(int warp_id, int lane_id) {
    constexpr auto sk_block_shape = opus::make_tuple(
        opus::number<T::smem_d_rpt>{},
        opus::number<T::smem_n_rpt / T::NUM_WARPS>{},
        opus::number<T::NUM_WARPS>{},
        opus::number<opus::get_warp_size()>{},
        opus::number<T::VEC_KV>{});

    constexpr auto sk_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_KV>(
        sk_block_shape,
        opus::unfold_x_stride(sk_block_dim, sk_block_shape, opus::tuple{T::smem_linear_wave + T::smem_padding, 1_I}),
        opus::unfold_p_coord(sk_block_dim, opus::tuple{warp_id, lane_id}));
}

// Create layout for reading K matrix from shared memory to registers
template<typename T>
inline __device__ auto make_layout_rk(int lane_id) {
    constexpr int n_per_wave = opus::get_warp_size() / (T::D_128B_SIZE / T::VEC_KV);
    constexpr int n_grp = n_per_wave / (T::W_N / T::NUM_WARPS);

    constexpr auto rk_block_shape = opus::make_tuple(
        opus::number<T::GEMM0_E_N / n_grp>{},
        opus::number<T::NUM_WARPS>{},
        opus::number<n_grp>{},
        opus::number<T::W_N / T::NUM_WARPS>{},
        opus::number<T::smem_d_rpt>{},
        opus::number<T::GEMM0_E_K / T::smem_d_rpt>{},
        opus::number<opus::get_warp_size() / T::W_N>{},
        opus::number<T::VEC_KV>{});

    constexpr auto rk_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    auto lane_id_n = lane_id % T::W_N;

    return opus::make_layout<T::VEC_KV>(
        rk_block_shape,
        opus::unfold_x_stride(rk_block_dim, rk_block_shape, opus::tuple{T::smem_linear_wave + T::smem_padding, T::D_128B_SIZE, T::smem_n_rpt * (T::smem_linear_wave + T::smem_padding), 1_I}),
        opus::unfold_p_coord(rk_block_dim, opus::tuple{lane_id_n % T::NUM_WARPS, lane_id_n / T::NUM_WARPS, lane_id / T::W_N}));
}

template<class T>
inline __device__ auto make_layout_gv_sv(int thread_id, int stride) {
    constexpr int threads_d = T::D_TILE_SIZE / T::VEC_KV;
    constexpr int threads_n = T::BLOCK_SIZE / threads_d;

    constexpr auto v_block_shape = opus::make_tuple(
        opus::number<T::KV_TILE_SIZE / threads_n>{},
        opus::number<threads_n>{},
        opus::number<threads_d>{},
        opus::number<T::VEC_KV>{});
    
    constexpr auto v_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    
    return opus::make_layout<T::VEC_KV>(
        v_block_shape,
        opus::unfold_x_stride(v_block_dim, v_block_shape, opus::tuple{stride, 1_I}),
        opus::unfold_p_coord(v_block_dim, opus::tuple{thread_id / threads_d, thread_id % threads_d}));
}

template<class T>
inline __device__ auto make_layout_rv(int lane_id) {
    constexpr int lane_per_grp = 16;
    constexpr int lane_lo = 4;
    constexpr int lane_hi = lane_per_grp / lane_lo;

    constexpr int num_grps = T::WARP_SIZE / lane_per_grp;
    constexpr int grp_n = T::W_N / (lane_lo * T::VEC_TR_V);
    constexpr int grp_k = num_grps / grp_n;

    constexpr auto rv_block_shape = opus::make_tuple(
        opus::number<T::GEMM1_E_N>{},
        opus::number<T::GEMM1_E_K>{},
        opus::number<T::W_K / (lane_hi * grp_k)>{},
        opus::number<grp_k>{},
        opus::number<lane_hi>{},
        opus::number<grp_n>{},
        opus::number<lane_lo>{},
        opus::number<T::VEC_TR_V>{});

    constexpr auto rv_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{})); 

    int grp_id = lane_id / lane_per_grp;
    int lane_in_grp = lane_id % lane_per_grp;

    return opus::make_layout<T::VEC_TR_V>(
        rv_block_shape,
        opus::unfold_x_stride(rv_block_dim, rv_block_shape, opus::tuple{grp_n * lane_lo * T::VEC_TR_V, T::D_TILE_SIZE, 1_I}),
        opus::unfold_p_coord(rv_block_dim, opus::tuple{grp_id / grp_n, lane_in_grp / lane_lo, grp_id % grp_n, lane_in_grp % lane_lo}));
}

// ─── GQA kernel: template on traits; K/V in shared, Q in registers, Flash Attention online softmax ───
template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 2) void gqa_kernel(opus_gqa_kargs kargs) {
    using namespace opus;
    using T = opus::remove_cvref_t<Traits>;
    using D_ATTN = typename T::D_ATTN;
    using D_ACC = typename T::D_ACC;

    const int GROUP_SIZE = kargs.H / kargs.H_KV;
    const float scale = 1.0f / sqrtf(static_cast<float>(kargs.D));

    const int h = (blockIdx.x % kargs.H_KV) * GROUP_SIZE + (blockIdx.x / kargs.H_KV);
    const int block_tile_idx = blockIdx.y;
    const int b = blockIdx.z;
    const int h_kv = h / GROUP_SIZE;

    const int warp_id = threadIdx.x / T::WARP_SIZE;
    const int lane_id = threadIdx.x % T::WARP_SIZE;

    const int q_start = block_tile_idx * T::NUM_WARPS * T::Q_TILE_SIZE;

    const int stride_q_b = kargs.N * kargs.H * kargs.D;
    const int stride_q_n = kargs.H * kargs.D;
    const int stride_q_h = kargs.D;
    const int stride_kv_b = kargs.N * kargs.H_KV * kargs.D;
    const int stride_kv_n = kargs.H_KV * kargs.D;
    const int stride_kv_h = kargs.D;

    // Create global memory tensors
    auto g_q = opus::make_gmem(reinterpret_cast<const D_ATTN*>(kargs.ptr_q) + b * stride_q_b + q_start * stride_q_n + h * stride_q_h);
    auto g_k = opus::make_gmem(reinterpret_cast<const D_ATTN*>(kargs.ptr_k) + b * stride_kv_b + h_kv * stride_kv_h);
    auto g_v = opus::make_gmem(reinterpret_cast<const D_ATTN*>(kargs.ptr_v) + b * stride_kv_b + h_kv * stride_kv_h);
    auto g_o = opus::make_gmem(reinterpret_cast<D_ATTN*>(kargs.ptr_o) + b * stride_q_b + q_start * stride_q_n + h * stride_q_h);

    // Shared memory for K and V tiles
    extern __shared__ char smem[];
    auto s_k = opus::make_smem(reinterpret_cast<D_ATTN*>(smem));
    auto s_v = opus::make_smem(reinterpret_cast<D_ATTN*>(smem) + T::smem_n_rpt * T::smem_d_rpt * (T::smem_linear_wave + T::smem_padding));

    // GEMM0: S = Q @ K^T
    auto mma0 = make_tiled_mma<D_ATTN, D_ATTN, D_ACC>(
        seq<T::GEMM0_E_M, T::GEMM0_E_N, T::GEMM0_E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});
    // GEMM1: O = P @ V
    auto mma1 = opus::make_tiled_mma<D_ATTN, D_ATTN, D_ACC>(
        seq<T::GEMM1_E_M, T::GEMM1_E_N, T::GEMM1_E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});

    // ──── Partition layouts ────
    auto u_q  = make_layout_q<T>(warp_id, lane_id, stride_q_n);
    auto u_gk = make_layout_gk<T>(warp_id, lane_id, stride_kv_n);
    auto u_sk = make_layout_sk<T>(warp_id, lane_id);
    auto u_rk = make_layout_rk<T>(lane_id);
    auto u_gv = make_layout_gv_sv<T>(threadIdx.x, stride_kv_n);
    auto u_sv = make_layout_gv_sv<T>(threadIdx.x, T::D_TILE_SIZE);
    auto u_rv = make_layout_rv<T>(lane_id);
    auto u_o  = make_layout_o<T>(warp_id, lane_id, stride_q_n);

    // ──── Vector registers ────
    typename decltype(mma0)::vtype_a v_q;
    typename decltype(mma0)::vtype_b v_k;
    typename decltype(mma0)::vtype_c v_s;
    typename decltype(mma1)::vtype_a v_p;
    typename decltype(mma1)::vtype_b v_v;
    typename decltype(mma1)::vtype_c v_o;

    constexpr index_t s_len = vector_traits<typename decltype(mma0)::vtype_c>::size();
    constexpr index_t o_len = vector_traits<typename decltype(mma1)::vtype_c>::size();

    D_ACC m_row = -1e30f;
    D_ACC l_row = 0.0f;

    v_q = load<T::VEC_Q>(g_q, u_q);
    opus::clear(v_o);

    const int num_kv_tiles = ceil_div(kargs.N, T::KV_TILE_SIZE);

    auto kv_offset = [&](int tile_idx) {
        return tile_idx * T::KV_TILE_SIZE * stride_kv_n;
    };

    for (int tile = 0; tile < num_kv_tiles; tile++) {
        async_load<T::VEC_KV>(g_k, s_k.ptr, u_gk, u_sk, kv_offset(tile));
        opus::s_waitcnt_vmcnt(0_I);
        __builtin_amdgcn_s_barrier();

        // ──── GEMM0: S = Q @ K^T via MFMA ────
        v_k = load<T::VEC_KV>(s_k, u_rk);
        v_s = mma0(v_q, v_k);

        // Scale by 1/sqrt(D)
        static_for<s_len>([&](auto i) { v_s[i.value] *= scale; });

        D_ACC row_max = -1e30f;
        static_for<s_len>([&](auto i) { row_max = max(row_max, v_s[i.value]); });
        row_max = max(row_max, __shfl_xor(row_max, 32));

        D_ACC scale_old = expf(m_row - row_max);

        // Rescale running O accumulator
        static_for<o_len>([&](auto i) { v_o[i.value] *= scale_old; });

        // P = exp(S - row_max), accumulate row sum
        D_ACC row_sum = 0.0f;
        static_for<s_len>([&](auto i) {
            v_s[i.value] = expf(v_s[i.value] - row_max);
            row_sum += v_s[i.value];
        });
        row_sum += __shfl_xor(row_sum, 32);

        D_ACC l_new = scale_old * l_row + row_sum;
        m_row = row_max;
        l_row = l_new;

        // Convert P to bf16 for GEMM1
        v_p = opus::cast<D_ATTN>(v_s);

        async_load<T::VEC_KV>(g_v, s_v.ptr, u_gv, u_sv, kv_offset(tile));
        opus::s_waitcnt_vmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        
        // ──── GEMM1: O += P @ V via MFMA ────
        v_v = tr_load<T::VEC_TR_V>(s_v, u_rv);
        v_o = mma1(v_p, v_v, v_o);
    }

    // ──── Normalize O and store to gmem ────
    D_ACC l_inv = (l_row > D_ACC(0.0f)) ? (D_ACC(1.0f) / l_row) : D_ACC(0.0f);
    static_for<o_len>([&](auto i) { v_o[i.value] *= l_inv; });

    auto v_o_bf16 = opus::cast<D_ATTN>(v_o);
    store<T::VEC_O>(g_o, v_o_bf16, u_o);
}

// Fill a contiguous vector with random values
template<typename T>
void rand_vector(T* ptr, size_t size, float min_val = 0.0f, float max_val = 1.0f) {
    #pragma omp parallel
    {
        std::random_device rd;
        std::mt19937 gen(rd() + omp_get_thread_num());
        std::uniform_real_distribution<float> dis(min_val, max_val);
        #pragma omp for
        for (size_t i = 0; i < size; i++) {
            ptr[i] = static_cast<T>(dis(gen));
        }
    }
}

// Benchmark GQA kernel performance with warm-up and timing
template<class Traits>
void benchmark_gqa_kernel(const opus_gqa_kargs& kargs, dim3 grid, dim3 block, size_t smem_size,
                          int warmup = 10, int iterations = 50) {
    for (int i = 0; i < warmup; ++i) {
        gqa_kernel<Traits><<<grid, block, smem_size>>>(kargs);
        CHECK_HIP_KERNEL_LAUNCH();
    }
    CHECK_HIP(hipDeviceSynchronize());

    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));

    CHECK_HIP(hipEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        gqa_kernel<Traits><<<grid, block, smem_size>>>(kargs);
        CHECK_HIP_KERNEL_LAUNCH();
    }
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));

    float total_time = 0;
    CHECK_HIP(hipEventElapsedTime(&total_time, start, stop));

    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));

    const float avg_time = total_time / iterations;
    // Approximate FLOPs: 2*B*H*N*N*D (QK^T) + 2*B*H*N*D*N (softmax+PV)
    const double flops = 4.0 * kargs.B * kargs.H * kargs.N * kargs.N * kargs.D;
    const double tflops = flops / (avg_time * 1e-3) / 1e12;

    printf("GQA Kernel Performance: avg_time=%.3f ms, %.2f TFlops\n", avg_time, tflops);
}

// Validate GQA GPU results against CPU reference
bool validate_gqa_results(const bf16_t* ref, const bf16_t* gpu, 
                          int B, int N, int H, int D, float threshold = 5e-2f) {
    bool all_valid = true;
    int total_errors = 0;
    
    // Sample-based validation (check a subset to avoid too much output)
    const int sample_heads = std::min(4, H);
    const int sample_queries = std::min(8, N);
    
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < sample_heads; h++) {
            for (int i = 0; i < sample_queries; i++) {
                int offset = b * N * H * D + i * H * D + h * D;
                
                // Check element-wise
                int local_errors = 0;
                float max_diff = 0.0f;
                for (int d = 0; d < D; d++) {
                    float ref_val = static_cast<float>(ref[offset + d]);
                    float gpu_val = static_cast<float>(gpu[offset + d]);
                    float diff = std::abs(ref_val - gpu_val);
                    max_diff = std::max(max_diff, diff);
                    if (diff > threshold) {
                        local_errors++;
                        total_errors++;
                    }
                }
                
                if (local_errors > 0) {
                    printf("  [b=%d,h=%d,n=%d] max_diff=%.6f, errors=%d/%d\n",
                           b, h, i, max_diff, local_errors, D);
                    all_valid = false;
                }
            }
        }
    }
    
    if (all_valid) {
        printf("✓ Sample validation passed (checked %d samples)\n", 
               B * sample_heads * sample_queries);
    } else {
        printf("✗ Validation failed with %d total errors\n", total_errors);
    }
    
    return all_valid;
}

// ─── CPU reference: Grouped-Query Attention (GQA) ──────────────────────────
//
// Q  layout: [B, N, H,    D]   (row-major, contiguous in D)
// K  layout: [B, N, H_KV, D]
// V  layout: [B, N, H_KV, D]
// O  layout: [B, N, H,    D]
//
// Standard scaled-dot-product attention with online softmax:
//   S[i,j]  = sum_d Q[b,i,h,d] * K[b,j,h_kv,d]   (h_kv = h / group_size)
//   P[i,:]  = softmax( S[i,:] / sqrt(D) )
//   O[i,d]  = sum_j P[i,j] * V[b,j,h_kv,d]
//
void gqa_attention_ref(
    const bf16_t* Q,  // [B, N, H, D]
    const bf16_t* K,  // [B, N, H_KV, D]
    const bf16_t* V,  // [B, N, H_KV, D]
    bf16_t*       O,  // [B, N, H, D]
    int B, int N, int H, int H_KV, int D)
{
    const int GROUP_SIZE = H / H_KV;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // Strides (row-major, last dim = D is contiguous)
    const int stride_q_b = N * H * D;
    const int stride_q_n = H * D;
    const int stride_q_h = D;

    const int stride_kv_b = N * H_KV * D;
    const int stride_kv_n = H_KV * D;
    const int stride_kv_h = D;

    #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            for (int i = 0; i < N; i++) {
                const int h_kv = h / GROUP_SIZE;
                const bf16_t* q_row = Q + b * stride_q_b + i * stride_q_n + h * stride_q_h;

                // ---- Compute attention scores S[j] = Q[b,i,h,:] . K[b,j,h_kv,:] ----
                std::vector<float> scores(N);
                for (int j = 0; j < N; j++) {
                    const bf16_t* k_row = K + b * stride_kv_b + j * stride_kv_n + h_kv * stride_kv_h;
                    float dot = 0.0f;
                    for (int d = 0; d < D; d++) {
                        dot += static_cast<float>(q_row[d]) * static_cast<float>(k_row[d]);
                    }
                    scores[j] = dot * scale;
                }

                // ---- Softmax ----
                float max_score = *std::max_element(scores.begin(), scores.end());
                float sum_exp = 0.0f;
                for (int j = 0; j < N; j++) {
                    scores[j] = std::exp(scores[j] - max_score);
                    sum_exp += scores[j];
                }
                for (int j = 0; j < N; j++) {
                    scores[j] /= sum_exp;
                }

                // ---- Output: O[b,i,h,d] = sum_j P[j] * V[b,j,h_kv,d] ----
                bf16_t* o_row = O + b * stride_q_b + i * stride_q_n + h * stride_q_h;
                for (int d = 0; d < D; d++) {
                    float acc = 0.0f;
                    for (int j = 0; j < N; j++) {
                        const bf16_t* v_row = V + b * stride_kv_b + j * stride_kv_n + h_kv * stride_kv_h;
                        acc += scores[j] * static_cast<float>(v_row[d]);
                    }
                    o_row[d] = static_cast<bf16_t>(acc);
                }
            }
        }
    }
}

// ─── main ───────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    int B    = 16;    // batch size
    int H    = 64;    // query heads
    int H_KV = 8;     // key/value heads
    int N    = 1024;  // sequence length
    int D    = 128;   // head dimension

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if ((std::strcmp(arg, "-b") == 0 || std::strcmp(arg, "--batch") == 0) && i + 1 < argc)
            B = std::atoi(argv[++i]);
        else if ((std::strcmp(arg, "-h") == 0 || std::strcmp(arg, "--heads") == 0) && i + 1 < argc)
            H = std::atoi(argv[++i]);
        else if ((std::strcmp(arg, "--hkv") == 0) && i + 1 < argc)
            H_KV = std::atoi(argv[++i]);
        else if ((std::strcmp(arg, "-n") == 0 || std::strcmp(arg, "--seq") == 0) && i + 1 < argc)
            N = std::atoi(argv[++i]);
        else if ((std::strcmp(arg, "-d") == 0 || std::strcmp(arg, "--dim") == 0) && i + 1 < argc)
            D = std::atoi(argv[++i]);
    }

    if (B <= 0 || H <= 0 || H_KV <= 0 || N <= 0 || D <= 0 || H % H_KV != 0) {
        std::cerr << "Invalid parameters. B,H,H_KV,N,D must be positive and H must be divisible by H_KV.\n";
        return 1;
    }

    const int GROUP_SIZE = H / H_KV;
    printf("GQA Attention: B=%d, H=%d, H_KV=%d, GROUP_SIZE=%d, N=%d, D=%d\n",
           B, H, H_KV, GROUP_SIZE, N, D);

    // Allocate host memory
    const size_t q_size = (size_t)B * N * H * D;
    const size_t kv_size = (size_t)B * N * H_KV * D;
    auto host_q = std::make_unique<bf16_t[]>(q_size);
    auto host_k = std::make_unique<bf16_t[]>(kv_size);
    auto host_v = std::make_unique<bf16_t[]>(kv_size);
    auto host_o_ref = std::make_unique<bf16_t[]>(q_size);  // CPU reference output
    auto host_o_gpu = std::make_unique<bf16_t[]>(q_size);  // GPU output

    // Initialize with random data
    rand_vector(host_q.get(), q_size, -2.f, 2.f);
    rand_vector(host_k.get(), kv_size, -2.f, 2.f);
    rand_vector(host_v.get(), kv_size, -2.f, 2.f);

    // Allocate device memory
    bf16_t *dev_q, *dev_k, *dev_v, *dev_o;
    CHECK_HIP(hipMalloc(&dev_q, q_size * sizeof(bf16_t)));
    CHECK_HIP(hipMalloc(&dev_k, kv_size * sizeof(bf16_t)));
    CHECK_HIP(hipMalloc(&dev_v, kv_size * sizeof(bf16_t)));
    CHECK_HIP(hipMalloc(&dev_o, q_size * sizeof(bf16_t)));

    CHECK_HIP(hipMemcpy(dev_q, host_q.get(), q_size * sizeof(bf16_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dev_k, host_k.get(), kv_size * sizeof(bf16_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dev_v, host_v.get(), kv_size * sizeof(bf16_t), hipMemcpyHostToDevice));

    // Setup kernel arguments
    opus_gqa_kargs kargs{};
    kargs.ptr_q = dev_q;
    kargs.ptr_k = dev_k;
    kargs.ptr_v = dev_v;
    kargs.ptr_o = dev_o;
    kargs.B = B;
    kargs.N = N;
    kargs.H = H;
    kargs.H_KV = H_KV;
    kargs.D = D;

    using GqaTraits = opus_gqa_traits<32, 64, 128, 8>;
    if (D != GqaTraits::D_TILE_SIZE) {
        std::cerr << "This kernel only supports head dimension D=" << GqaTraits::D_TILE_SIZE << ", got D=" << D << "\n";
        return 1;
    }
    const int num_q_tiles = ceil_div(N, GqaTraits::Q_TILE_SIZE);
    const int num_q_tile_blocks = ceil_div(num_q_tiles, GqaTraits::NUM_WARPS);
    dim3 grid(H, num_q_tile_blocks, B);
    dim3 block(GqaTraits::BLOCK_SIZE);
    size_t smem_size = GqaTraits::smem_size_bytes();

    printf("GQA kernel launch config: grid=(%d,%d,%d), block=%d (NUM_WARPS=%d), smem=%zu bytes (K/V tiles)\n",
           grid.x, grid.y, grid.z, (int)block.x, GqaTraits::NUM_WARPS, smem_size);

    gqa_kernel<GqaTraits><<<grid, block, smem_size>>>(kargs);
    CHECK_HIP_KERNEL_LAUNCH();

#if 1
    printf("\nValidating GPU results against CPU reference...\n");
    CHECK_HIP(hipMemcpy(host_o_gpu.get(), dev_o, q_size * sizeof(bf16_t), hipMemcpyDeviceToHost));
    gqa_attention_ref(host_q.get(), host_k.get(), host_v.get(), host_o_ref.get(),
                      B, N, H, H_KV, D);

    bool all_valid = validate_gqa_results(host_o_ref.get(), host_o_gpu.get(), B, N, H, D);
    printf("\n[Overall] %s\n", all_valid ? "✓ GPU KERNEL VALID" : "✗ GPU KERNEL FAILED");
#endif

#if 1
    printf("\n");
    benchmark_gqa_kernel<GqaTraits>(kargs, grid, block, smem_size);
    printf("\n");
#endif

    // Cleanup
    CHECK_HIP(hipFree(dev_q));
    CHECK_HIP(hipFree(dev_k));
    CHECK_HIP(hipFree(dev_v));
    CHECK_HIP(hipFree(dev_o));

    return 0;
}
