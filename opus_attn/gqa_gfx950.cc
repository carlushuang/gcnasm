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

// Configuration traits for GQA kernel (tile sizes, data types, vector lengths)
template<int Q_TILE_SIZE_ = 32,
        int KV_TILE_SIZE_ = 64,
        int D_TILE_SIZE_ = 128,
        int NUM_WARPS_ = 4>
struct opus_gqa_traits {
    static constexpr int Q_TILE_SIZE = Q_TILE_SIZE_;
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;
    static constexpr int D_TILE_SIZE = D_TILE_SIZE_;
    static constexpr int NUM_WARPS = NUM_WARPS_;

    static constexpr int WARP_SIZE = opus::get_warp_size();
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;

    // Data types: Q/K/V/O share one bf16 type; accumulation fp32
    using D_ATTN = bf16_t;  // Q, K, V, O
    using D_ACC  = float;

    // Vector lengths for global load (bf16 max 8)
    static constexpr int Q_VEC  = 8;
    static constexpr int KV_VEC = 8;
    static constexpr int O_VEC  = 8;

    static constexpr size_t smem_size_bytes() {
        return 2 * (size_t)KV_TILE_SIZE * D_TILE_SIZE * sizeof(D_ATTN);
    }
};

// ─── GQA kernel: template on traits; K/V in shared, Q in registers, Flash Attention online softmax ───
template<class Traits>
__global__ void gqa_kernel(opus_gqa_kargs kargs) {
    using namespace opus;
    using T = opus::remove_cvref_t<Traits>;
    using D_ATTN = typename T::D_ATTN;
    using D_ACC = typename T::D_ACC;

    const int GROUP_SIZE = kargs.H / kargs.H_KV;
    const float scale = 1.0f / sqrtf(static_cast<float>(kargs.D));

    const int h = blockIdx.x;
    const int block_tile_idx = blockIdx.y;
    const int b = blockIdx.z;
    const int h_kv = h / GROUP_SIZE;

    const int warp_id = threadIdx.x / T::WARP_SIZE;
    const int lane_id = threadIdx.x % T::WARP_SIZE;

    const int q_tile = block_tile_idx * T::NUM_WARPS + warp_id;
    const int q_start = q_tile * T::Q_TILE_SIZE;
    if (q_start >= kargs.N) return;
    const int q_end = min(q_start + T::Q_TILE_SIZE, kargs.N);
    const int q_rows = q_end - q_start;

    const int stride_q_b = kargs.N * kargs.H * kargs.D;
    const int stride_q_n = kargs.H * kargs.D;
    const int stride_q_h = kargs.D;
    const int stride_kv_b = kargs.N * kargs.H_KV * kargs.D;
    const int stride_kv_n = kargs.H_KV * kargs.D;
    const int stride_kv_h = kargs.D;

    const D_ATTN* Q = reinterpret_cast<const D_ATTN*>(kargs.ptr_q);
    const D_ATTN* K = reinterpret_cast<const D_ATTN*>(kargs.ptr_k);
    const D_ATTN* V = reinterpret_cast<const D_ATTN*>(kargs.ptr_v);
    D_ATTN* O = reinterpret_cast<D_ATTN*>(kargs.ptr_o);

    const int row_in_tile = lane_id / 2;
    const int half = lane_id % 2;
    constexpr int D_HALF = T::D_TILE_SIZE / 2;
    D_ATTN q_reg[D_HALF];
    {
        const D_ATTN* q_base = Q + b * stride_q_b + q_start * stride_q_n + h * stride_q_h;
        auto g_q = opus::make_gmem(q_base);
        if (row_in_tile < q_rows) {
            for (int i = 0; i < D_HALF / T::Q_VEC; i++) {
                auto v = g_q.template load<T::Q_VEC>(row_in_tile * stride_q_n + half * D_HALF + i * T::Q_VEC);
                for (int k = 0; k < T::Q_VEC; k++)
                    q_reg[i * T::Q_VEC + k] = v[k];
            }
        } else {
            for (int d = 0; d < D_HALF; d++) q_reg[d] = D_ATTN(0.0f);
        }
    }

    extern __shared__ char smem[];
    D_ATTN* s_K = reinterpret_cast<D_ATTN*>(smem);
    D_ATTN* s_V = reinterpret_cast<D_ATTN*>(smem + T::KV_TILE_SIZE * T::D_TILE_SIZE * sizeof(D_ATTN));

    D_ACC o_reg[D_HALF];
    for (int d = 0; d < D_HALF; d++) o_reg[d] = 0.0f;
    D_ACC m_row = -1e30f;
    D_ACC l_row = 0.0f;

    const D_ATTN* K_base = K + b * stride_kv_b + h_kv * stride_kv_h;
    const D_ATTN* V_base = V + b * stride_kv_b + h_kv * stride_kv_h;

    const int num_kv_tiles = ceil_div(kargs.N, T::KV_TILE_SIZE);
    constexpr int KV_TILE_VEC = (T::KV_TILE_SIZE * T::D_TILE_SIZE + T::KV_VEC - 1) / T::KV_VEC;
    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        const int kv_start = kv_tile * T::KV_TILE_SIZE;
        const int kv_count = min(T::KV_TILE_SIZE, kargs.N - kv_start);

        auto g_K = opus::make_gmem(K_base + kv_start * stride_kv_n);
        auto g_V = opus::make_gmem(V_base + kv_start * stride_kv_n);
        for (int vec_idx = threadIdx.x; vec_idx < KV_TILE_VEC; vec_idx += blockDim.x) {
            int row = (vec_idx * T::KV_VEC) / T::D_TILE_SIZE;
            if (row < kv_count) {
                g_K.template async_load<T::KV_VEC>(reinterpret_cast<void*>(s_K + vec_idx * T::KV_VEC), vec_idx * T::KV_VEC);
                g_V.template async_load<T::KV_VEC>(reinterpret_cast<void*>(s_V + vec_idx * T::KV_VEC), vec_idx * T::KV_VEC);
            }
        }
        opus::s_waitcnt_vmcnt(0_I);
        __syncthreads();

        if (row_in_tile >= q_rows) continue;

        D_ACC s_scores[T::KV_TILE_SIZE];
        for (int j = 0; j < T::KV_TILE_SIZE; j++) {
            D_ACC dot = 0.0f;
            for (int d = 0; d < D_HALF; d++)
                dot += float(q_reg[d]) * float(s_K[j * T::D_TILE_SIZE + half * D_HALF + d]);
            D_ACC partner = __shfl_xor(dot, 1);
            dot = (dot + partner) * scale;
            s_scores[j] = (j < kv_count) ? dot : D_ACC(-1e30f);
        }

        D_ACC block_max = -1e30f;
        for (int j = 0; j < kv_count; j++) block_max = fmaxf(block_max, s_scores[j]);
        for (int offset = T::WARP_SIZE / 2; offset > 0; offset >>= 1)
            block_max = fmaxf(block_max, __shfl_xor(block_max, offset));

        D_ACC scale_old = expf(m_row - block_max);
        for (int d = 0; d < D_HALF; d++) o_reg[d] *= scale_old;

        D_ACC local_p_sum = 0.0f;
        for (int j = 0; j < kv_count; j++) {
            D_ACC p = expf(s_scores[j] - block_max);
            local_p_sum += p;
            for (int d = 0; d < D_HALF; d++)
                o_reg[d] += p * float(s_V[j * T::D_TILE_SIZE + half * D_HALF + d]);
        }
        D_ACC l_new = scale_old * l_row + local_p_sum + __shfl_xor(local_p_sum, 1);
        m_row = block_max;
        l_row = l_new;
    }

    if (row_in_tile >= q_rows) return;
    D_ACC l_inv = (l_row > 0.0f) ? (D_ACC(1.0f) / l_row) : D_ACC(0.0f);
    D_ATTN* o_base = O + b * stride_q_b + (q_start + row_in_tile) * stride_q_n + h * stride_q_h;
    auto g_o = opus::make_gmem(o_base);
    for (int i = 0; i < D_HALF / T::O_VEC; i++) {
        opus::vector_t<D_ATTN, T::O_VEC> ov;
        for (int k = 0; k < T::O_VEC; k++)
            ov[k] = D_ATTN(o_reg[i * T::O_VEC + k] * l_inv);
        g_o.template store<T::O_VEC>(ov, half * D_HALF + i * T::O_VEC);
    }
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
    // Default GQA dimensions (matching kernel_hipkittens.cpp defaults)
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
    printf("Initializing random data...\n");
    rand_vector(host_q.get(), q_size, -0.5f, 0.5f);
    rand_vector(host_k.get(), kv_size, -0.5f, 0.5f);
    rand_vector(host_v.get(), kv_size, -0.5f, 0.5f);

    // ---- CPU reference ----
    printf("Computing CPU reference attention...\n");
    double t0 = omp_get_wtime();
    gqa_attention_ref(host_q.get(), host_k.get(), host_v.get(), host_o_ref.get(),
                      B, N, H, H_KV, D);
    double t1 = omp_get_wtime();
    printf("CPU reference done in %.3f s\n", t1 - t0);

    // ---- GPU kernel ----
    printf("\nAllocating device memory and launching GPU kernel...\n");
    
    // Allocate device memory
    bf16_t *dev_q, *dev_k, *dev_v, *dev_o;
    CHECK_HIP(hipMalloc(&dev_q, q_size * sizeof(bf16_t)));
    CHECK_HIP(hipMalloc(&dev_k, kv_size * sizeof(bf16_t)));
    CHECK_HIP(hipMalloc(&dev_v, kv_size * sizeof(bf16_t)));
    CHECK_HIP(hipMalloc(&dev_o, q_size * sizeof(bf16_t)));
    
    // Copy to device
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
    
    // Launch configuration via traits
    using GqaTraits = opus_gqa_traits<32, 64, 128, 4>;
    if (D != GqaTraits::D_TILE_SIZE) {
        std::cerr << "This kernel only supports head dimension D=" << GqaTraits::D_TILE_SIZE << ", got D=" << D << "\n";
        return 1;
    }
    const int num_q_tiles = ceil_div(N, GqaTraits::Q_TILE_SIZE);
    const int num_q_tile_blocks = ceil_div(num_q_tiles, GqaTraits::NUM_WARPS);
    dim3 grid(H, num_q_tile_blocks, B);
    dim3 block(GqaTraits::BLOCK_SIZE);
    size_t smem_size = GqaTraits::smem_size_bytes();

    printf("Launching GQA kernel: grid=(%d,%d,%d), block=%d (NUM_WARPS=%d), smem=%zu bytes (K/V tiles)\n",
           grid.x, grid.y, grid.z, (int)block.x, GqaTraits::NUM_WARPS, smem_size);

    gqa_kernel<GqaTraits><<<grid, block, smem_size>>>(kargs);
    CHECK_HIP_KERNEL_LAUNCH();
    CHECK_HIP(hipDeviceSynchronize());
    
    // Copy results back for validation
    CHECK_HIP(hipMemcpy(host_o_gpu.get(), dev_o, q_size * sizeof(bf16_t), hipMemcpyDeviceToHost));
    
    // ---- Validation ----
    printf("\nValidating GPU results against CPU reference...\n");
    bool all_valid = validate_gqa_results(host_o_ref.get(), host_o_gpu.get(), B, N, H, D);
    
    printf("\n[Overall] %s\n", all_valid ? "✓ GPU KERNEL VALID" : "✗ GPU KERNEL FAILED");
    
    // ---- Benchmark ----
    if (all_valid) {
        printf("\n");
        benchmark_gqa_kernel<GqaTraits>(kargs, grid, block, smem_size);
        printf("\n");
    }
    
    // Cleanup
    CHECK_HIP(hipFree(dev_q));
    CHECK_HIP(hipFree(dev_k));
    CHECK_HIP(hipFree(dev_v));
    CHECK_HIP(hipFree(dev_o));

    return all_valid ? 0 : 1;
}
