#include <hip/hip_fp8.h>
#include <opus/hip_minimal.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <omp.h>

#include "gemm_a8w8_mxfp8_scale_common.h"
#include "gemm_a8w8_mxfp8_scale_repack.hpp"

template<class Traits>
__global__ void gemm_a8w8_mxfp8_scale_kernel(opus_gemm_scale_kargs kargs);

#define CHECK_HIP(call)                                                                                   \
    do {                                                                                                  \
        hipError_t status_ = call;                                                                        \
        if (status_ != hipSuccess) {                                                                      \
            fprintf(stderr, "HIP error (%s:%d): %s\n", __FILE__, __LINE__, hipGetErrorString(status_));   \
            exit(1);                                                                                      \
        }                                                                                                 \
    } while(0)

#define CHECK_HIP_KERNEL_LAUNCH() CHECK_HIP(hipGetLastError())

// The two instantiations provided by the device TU.  They differ only in
// OUTPUT_TILES_PER_WG, so every tile/scale constant below can be read off
// either one; GemmTraits names the persistent form for that purpose.
using GemmTraitsPersist = gemm_a8w8_mxfp8_scale_traits<256, 256, 128, 1, 1, 32, 4>;
using GemmTraitsSingle  = gemm_a8w8_mxfp8_scale_traits<256, 256, 128, 1, 1, 32, 1>;
using GemmTraits = GemmTraitsPersist;

// The persistent fixed-B tile divides the workgroup count by four.  That is a
// win only once there are more workgroups than CUs -- below that it just parks
// CUs.  LDS is 139264 B, so occupancy is one workgroup per CU and the compare
// is directly against the CU count.
inline int pick_output_tiles_per_wg(int num_tiles_m, int num_tiles_n) {
    int cus = 0;
    CHECK_HIP(hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, 0));
    const int persist_wgs =
        ceil_div_scale(num_tiles_m, GemmTraitsPersist::OUTPUT_TILES_PER_WG) * num_tiles_n;
    return persist_wgs >= cus ? GemmTraitsPersist::OUTPUT_TILES_PER_WG
                              : GemmTraitsSingle::OUTPUT_TILES_PER_WG;
}
using host_fp8_t = __hip_fp8_e4m3;
using fp32_t = float;
using e8m0_t = uint8_t;

// E8M0: 8-bit exponent-only scale, bias 127. value = 2^(byte - 127); 0x7F = 1.0.
inline fp32_t e8m0_to_f32(e8m0_t e) {
    return std::ldexp(1.0f, static_cast<int>(e) - 127);
}

inline e8m0_t f32_to_e8m0(fp32_t v) {
    if (v <= 0.0f) return 0;
    int exp;
    std::frexp(v, &exp);      // v in [0.5, 1) * 2^exp -> nearest power-of-two exponent
    int biased = (exp - 1) + 127;
    if (biased < 0) biased = 0;
    if (biased > 255) biased = 255;
    return static_cast<e8m0_t>(biased);
}

template<typename T>
void rand_vector(T* ptr, std::size_t size, fp32_t min_val = 0.0f, fp32_t max_val = 1.0f) {
    #pragma omp parallel
    {
        std::random_device rd;
        std::mt19937 gen(rd() + omp_get_thread_num());
        std::uniform_real_distribution<fp32_t> dis(min_val, max_val);
        #pragma omp for
        for (std::size_t i = 0; i < size; ++i) {
            ptr[i] = static_cast<T>(dis(gen));
        }
    }
}

// Random E8M0 scales drawn as power-of-two exponents around 1.0 (byte in [lo, hi]).
void rand_scale_e8m0(e8m0_t* ptr, std::size_t size, int lo = 124, int hi = 130) {
    #pragma omp parallel
    {
        std::random_device rd;
        std::mt19937 gen(rd() + omp_get_thread_num());
        std::uniform_int_distribution<int> dis(lo, hi);
        #pragma omp for
        for (std::size_t i = 0; i < size; ++i) {
            ptr[i] = static_cast<e8m0_t>(dis(gen));
        }
    }
}

// Tile-major consumer order matching the LDS image:
// [consumer_wave_m][r][q][m_call]. The packed tile is a byte-for-byte image
// of the consumer-facing LDS tile.
template<class Traits>
void pack_sfa_consumer_major(
    const e8m0_t* src,
    e8m0_t* dst,
    int batch,
    int m,
    int k) {
    constexpr int scale_count = Traits::SCALE_KGROUPS_PER_MFMA;
    constexpr int tile_elems = Traits::packed_sfa_tile_elem;

    const int num_groups_k = k / Traits::GROUP_K;
    const int num_tiles_m = m / Traits::B_M;
    const int num_tiles_k = k / Traits::B_K;

    // Cap the thread count: each iteration packs one 1 KB tile, so with all
    // 256 cores of this box the fork/join cost swamps the work -- 8192^3 takes
    // 16.8 ms at 256 threads against 0.057 ms at 32.
    #pragma omp parallel for collapse(3) num_threads(std::min(32, omp_get_max_threads()))
    for (int b = 0; b < batch; ++b) {
        for (int mb = 0; mb < num_tiles_m; ++mb) {
            for (int kt = 0; kt < num_tiles_k; ++kt) {
                const std::size_t dst_tile =
                    (static_cast<std::size_t>(b) * num_tiles_m * num_tiles_k
                     + static_cast<std::size_t>(mb) * num_tiles_k + kt)
                    * tile_elems;
                for (int consumer_wave_m = 0; consumer_wave_m < Traits::T_M;
                     ++consumer_wave_m) {
                    for (int r = 0; r < Traits::W_M; ++r) {
                        for (int q = 0; q < scale_count; ++q) {
                            for (int m_call = 0; m_call < Traits::SCALE_M_CALLS;
                                 ++m_call) {
                                const int src_m =
                                    mb * Traits::B_M
                                    + m_call * Traits::T_M * Traits::W_M
                                    + consumer_wave_m * Traits::W_M + r;
                                const int src_q = kt * Traits::NUM_KGROUPS + q;
                                const std::size_t dst_offset =
                                    (((consumer_wave_m * Traits::W_M + r) * scale_count + q)
                                     * Traits::SCALE_M_CALLS)
                                    + m_call;
                                dst[dst_tile + dst_offset] =
                                    src[(static_cast<std::size_t>(b) * m + src_m)
                                        * num_groups_k + src_q];
                            }
                        }
                    }
                }
            }
        }
    }
}

// SFB consumer order is
// [half_n][consumer_wave_n][r][q][n_call]. The packed tile is copied
// linearly to LDS; producer vector width does not change this layout.
template<class Traits>
void pack_sfb_consumer_major(
    const e8m0_t* src,
    e8m0_t* dst,
    int batch,
    int n,
    int k) {
    constexpr int scale_count = Traits::SCALE_KGROUPS_PER_MFMA;
    constexpr int tile_elems = Traits::packed_sfb_tile_elem;

    const int num_groups_k = k / Traits::GROUP_K;
    const int num_tiles_n = n / Traits::B_N;
    const int num_tiles_k = k / Traits::B_K;

    // Cap the thread count: each iteration packs one 1 KB tile, so with all
    // 256 cores of this box the fork/join cost swamps the work -- 8192^3 takes
    // 16.8 ms at 256 threads against 0.057 ms at 32.
    #pragma omp parallel for collapse(3) num_threads(std::min(32, omp_get_max_threads()))
    for (int b = 0; b < batch; ++b) {
        for (int nb = 0; nb < num_tiles_n; ++nb) {
            for (int kt = 0; kt < num_tiles_k; ++kt) {
                const std::size_t dst_tile =
                    (static_cast<std::size_t>(b) * num_tiles_n * num_tiles_k
                     + static_cast<std::size_t>(nb) * num_tiles_k + kt)
                    * tile_elems;
                for (int half_n = 0; half_n < Traits::SCALE_N_HALVES; ++half_n) {
                    for (int consumer_wave_n = 0; consumer_wave_n < Traits::T_N;
                         ++consumer_wave_n) {
                        for (int r = 0; r < Traits::W_N; ++r) {
                            for (int q = 0; q < scale_count; ++q) {
                                for (int n_call = 0; n_call < Traits::SCALE_N_CALLS;
                                     ++n_call) {
                                    const int src_n =
                                        nb * Traits::B_N + half_n * Traits::HALF_B_N
                                        + n_call * Traits::T_N * Traits::W_N
                                        + consumer_wave_n * Traits::W_N + r;
                                    const int src_q = kt * Traits::NUM_KGROUPS + q;
                                    const std::size_t dst_offset =
                                        (((((half_n * Traits::T_N + consumer_wave_n)
                                             * Traits::W_N)
                                            + r)
                                           * scale_count
                                           + q)
                                          * Traits::SCALE_N_CALLS)
                                        + n_call;
                                    dst[dst_tile + dst_offset] =
                                        src[(static_cast<std::size_t>(b) * n + src_n)
                                            * num_groups_k + src_q];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

template<typename T>
bool valid_vector(const T* ref, const T* result, const double* mag, int n,
                  fp32_t rel_mag = 5e-5f, fp32_t abs_floor = 1e-4f) {
    int errors = 0;
    int max_idx = -1;
    int max_ratio_idx = -1;
    fp32_t max_diff = 0.0f;
    fp32_t max_tol = 0.0f;
    fp32_t max_ratio = 0.0f;
    for (int i = 0; i < n; ++i) {
        const fp32_t r = static_cast<fp32_t>(ref[i]);
        const fp32_t got = static_cast<fp32_t>(result[i]);
        const fp32_t diff = std::abs(r - got);
        const fp32_t tol = abs_floor + rel_mag * static_cast<fp32_t>(mag[i]);
        const fp32_t ratio = diff / tol;
        if (diff > max_diff) {
            max_diff = diff;
            max_tol = tol;
            max_idx = i;
        }
        if (ratio > max_ratio) {
            max_ratio = ratio;
            max_ratio_idx = i;
        }
        if (diff > tol) {
            if (errors < 10) {
                std::printf("Error at %d: ref=%.6f, result=%.6f, diff=%.6f, tol=%.6f (sum_abs_term=%.2f)\n",
                            i, r, got, diff, tol, mag[i]);
            }
            ++errors;
        }
    }
    if (max_idx >= 0) {
        std::printf("Validation stats: rel_mag=%.2e, abs_floor=%.2e, errors=%d/%d, max_diff=%.6f at %d (tol=%.6f, ref=%.6f, result=%.6f), max_ratio=%.3f at %d\n",
                    rel_mag, abs_floor, errors, n, max_diff, max_idx, max_tol,
                    static_cast<fp32_t>(ref[max_idx]), static_cast<fp32_t>(result[max_idx]),
                    max_ratio, max_ratio_idx);
    }
    return errors == 0;
}

// CPU reference MXFP8 GEMM: fp8 inputs, fp32 output, per-32-K E8M0 scales per row.
// SFA: [M, num_groups_k], SFB: [N, num_groups_k]. Accumulate in double and
// emit sum(abs(term)) per element for a condition-aware fp32 validation bound.
void gemm_ref(const host_fp8_t* a, const host_fp8_t* b, const e8m0_t* sfa, const e8m0_t* sfb, fp32_t* c,
              double* mag, int m, int n, int k, int lda, int ldb, int ldc, int stride_sfa, int stride_sfb,
              int group_k) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            const host_fp8_t* a_row = a + i * lda;
            const host_fp8_t* b_row = b + j * ldb;
            const e8m0_t* sfa_row = sfa + i * stride_sfa;
            const e8m0_t* sfb_row = sfb + j * stride_sfb;
            double sum = 0.0;
            double sum_abs_term = 0.0;
            for (int k_group_idx = 0; k_group_idx < k / group_k; ++k_group_idx) {
                const double scale = static_cast<double>(e8m0_to_f32(sfa_row[k_group_idx]))
                                   * static_cast<double>(e8m0_to_f32(sfb_row[k_group_idx]));
                const int p_begin = k_group_idx * group_k;
                const int p_end = p_begin + group_k;
                for (int p = p_begin; p < p_end; ++p) {
                    const double term = static_cast<double>(static_cast<fp32_t>(a_row[p]))
                                      * static_cast<double>(static_cast<fp32_t>(b_row[p]))
                                      * scale;
                    sum += term;
                    sum_abs_term += std::abs(term);
                }
            }
            c[i * ldc + j] = static_cast<fp32_t>(sum);
            mag[i * ldc + j] = sum_abs_term;
        }
    }
}

template<class Traits>
void benchmark_kernel(
    const opus_gemm_scale_kargs& kargs,
    dim3 grid,
    dim3 block,
    int warmup,
    int iterations) {
    for (int i = 0; i < warmup; ++i) {
        gemm_a8w8_mxfp8_scale_kernel<Traits><<<grid, block>>>(kargs);
        CHECK_HIP_KERNEL_LAUNCH();
    }

    hipEvent_t start;
    hipEvent_t stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));

    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipEventRecord(start));

    for (int i = 0; i < iterations; ++i) {
        gemm_a8w8_mxfp8_scale_kernel<Traits><<<grid, block>>>(kargs);
        CHECK_HIP_KERNEL_LAUNCH();
    }

    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));

    fp32_t total_time = 0.0f;
    CHECK_HIP(hipEventElapsedTime(&total_time, start, stop));

    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));

    const fp32_t avg_time = total_time / iterations;
    const std::size_t flop = static_cast<std::size_t>(2) * kargs.m * kargs.n * kargs.k * kargs.batch;
    const fp32_t tflops = static_cast<fp32_t>(flop) / 1.0e9f / avg_time;

    std::printf("Kernel Performance: avg_time=%.4f ms, %.2f TFlops\n", avg_time, tflops);
}

template<class Traits>
void benchmark_kernel_timeline(
    const opus_gemm_scale_kargs& kargs,
    dim3 grid,
    dim3 block) {
    // Record all boundaries in one uninterrupted stream. There is deliberately
    // no warmup and no host-side synchronization between ranges.
    constexpr int range_ends[] = {25, 50, 100, 200, 500, 1000};
    constexpr int num_ranges = sizeof(range_ends) / sizeof(range_ends[0]);
    hipEvent_t marks[num_ranges + 1];
    for (int i = 0; i <= num_ranges; ++i) {
        CHECK_HIP(hipEventCreate(&marks[i]));
    }

    CHECK_HIP(hipDeviceSynchronize());
    const auto host_start = std::chrono::steady_clock::now();
    CHECK_HIP(hipEventRecord(marks[0]));

    int launched = 0;
    for (int range = 0; range < num_ranges; ++range) {
        while (launched < range_ends[range]) {
            gemm_a8w8_mxfp8_scale_kernel<Traits><<<grid, block>>>(kargs);
            CHECK_HIP_KERNEL_LAUNCH();
            ++launched;
        }
        CHECK_HIP(hipEventRecord(marks[range + 1]));
    }

    CHECK_HIP(hipEventSynchronize(marks[num_ranges]));
    const auto host_end = std::chrono::steady_clock::now();
    const auto host_start_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        host_start.time_since_epoch()).count();
    const auto host_end_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        host_end.time_since_epoch()).count();

    const std::size_t flop =
        static_cast<std::size_t>(2) * kargs.m * kargs.n * kargs.k * kargs.batch;
    int range_begin = 1;
    fp32_t cumulative_time = 0.0f;
    std::printf("Timeline: warmup=0, consecutive_kernels=%d, host_start_ns=%lld, host_end_ns=%lld\n",
                range_ends[num_ranges - 1],
                static_cast<long long>(host_start_ns),
                static_cast<long long>(host_end_ns));
    for (int range = 0; range < num_ranges; ++range) {
        fp32_t range_time = 0.0f;
        CHECK_HIP(hipEventElapsedTime(&range_time, marks[range], marks[range + 1]));
        const int range_iterations = range_ends[range] - range_begin + 1;
        const fp32_t avg_time = range_time / range_iterations;
        const fp32_t tflops = static_cast<fp32_t>(flop) / 1.0e9f / avg_time;
        cumulative_time += range_time;
        std::printf(
            "Timeline kernels %d-%d: total=%.4f ms, avg=%.4f ms, %.2f TFlops (%.4f PFlop/s), cumulative=%.4f ms\n",
            range_begin,
            range_ends[range],
            range_time,
            avg_time,
            tflops,
            tflops / 1000.0f,
            cumulative_time);
        range_begin = range_ends[range] + 1;
    }

    for (int i = 0; i <= num_ranges; ++i) {
        CHECK_HIP(hipEventDestroy(marks[i]));
    }
}

int main(int argc, char** argv) {
    constexpr int BLOCK_M = GemmTraits::B_M;
    constexpr int BLOCK_N = GemmTraits::B_N;
    constexpr int BLOCK_K = GemmTraits::B_K;
    constexpr int BLOCK_SIZE = GemmTraits::BLOCK_SIZE;

    int M = 256;
    int N = 512;
    int K = 256;
    int batch = 8;
    int verify = 0;
    int warmup = 200;
    int iterations = 100;
    int timeline = 0;
    int device_repack = 0;

    auto parse_val = [](const char* arg, const char* flag) -> const char* {
        const std::size_t len = std::strlen(flag);
        if (std::strncmp(arg, flag, len) == 0) {
            if (arg[len] == '=') {
                return arg + len + 1;
            }
            if (arg[len] == '\0') {
                return reinterpret_cast<const char*>(1);
            }
        }
        return nullptr;
    };
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        const char* val = nullptr;
        auto try_parse = [&](int& target, const char* short_flag, const char* long_flag) {
            if ((val = parse_val(arg, short_flag)) || (long_flag && (val = parse_val(arg, long_flag)))) {
                if (val == reinterpret_cast<const char*>(1)) {
                    if (i + 1 < argc) {
                        target = std::atoi(argv[++i]);
                    }
                } else {
                    target = std::atoi(val);
                }
                return true;
            }
            return false;
        };
        if (try_parse(M, "-m", "--m")) continue;
        if (try_parse(N, "-n", "--n")) continue;
        if (try_parse(K, "-k", "--k")) continue;
        if (try_parse(batch, "-b", "--b")) continue;
        if (try_parse(verify, "-v", "--verify")) continue;
        if (try_parse(warmup, "-w", "--warmup")) continue;
        if (try_parse(iterations, "-i", "--iterations")) continue;
        if (std::strcmp(arg, "--timeline") == 0) {
            timeline = 1;
            continue;
        }
        if (std::strcmp(arg, "--device-repack") == 0) {
            device_repack = 1;
            continue;
        }
    }

    if (M <= 0 || N <= 0 || K <= 0 || batch <= 0 || warmup < 0 || iterations <= 0) {
        std::cerr << "Invalid arguments: M/N/K/batch/iterations must be positive and warmup must be non-negative.\n";
        return 1;
    }
    if (timeline && verify) {
        std::cerr << "--timeline requires --verify 0 so that no validation launch precedes kernel 1.\n";
        return 1;
    }

    constexpr int GROUP_K = GemmTraits::GROUP_K;
    if (M % BLOCK_M != 0 || N % BLOCK_N != 0 || K % BLOCK_K != 0) {
        std::cerr << "M/N/K must be multiples of BLOCK_M/BLOCK_N/BLOCK_K ("
                  << BLOCK_M << "," << BLOCK_N << "," << BLOCK_K << ").\n";
        return 1;
    }
    if (K % GROUP_K != 0) {
        std::cerr << "K must be a multiple of GROUP_K (" << GROUP_K << ").\n";
        return 1;
    }

    const int num_groups_k = K / GROUP_K;
    const int num_tiles_m = M / BLOCK_M;
    const int num_tiles_n = N / BLOCK_N;
    const int num_tiles_k = K / BLOCK_K;

    auto host_a = std::make_unique<host_fp8_t[]>(static_cast<std::size_t>(batch) * M * K);
    auto host_b = std::make_unique<host_fp8_t[]>(static_cast<std::size_t>(batch) * N * K);
    std::unique_ptr<fp32_t[]> host_c;
    std::unique_ptr<fp32_t[]> host_c_out;
    std::unique_ptr<double[]> host_c_mag;
    if (verify) {
        host_c = std::make_unique<fp32_t[]>(static_cast<std::size_t>(batch) * M * N);
        host_c_out = std::make_unique<fp32_t[]>(static_cast<std::size_t>(batch) * M * N);
        host_c_mag = std::make_unique<double[]>(static_cast<std::size_t>(M) * N);
    }

    const std::size_t sfa_count = static_cast<std::size_t>(batch) * M * num_groups_k;
    const std::size_t sfb_count = static_cast<std::size_t>(batch) * N * num_groups_k;
    auto host_sfa = std::make_unique<e8m0_t[]>(sfa_count);
    auto host_sfb = std::make_unique<e8m0_t[]>(sfb_count);
    auto host_sfa_packed = std::make_unique<e8m0_t[]>(sfa_count);
    auto host_sfb_packed = std::make_unique<e8m0_t[]>(sfb_count);

    rand_vector(host_a.get(), static_cast<std::size_t>(batch) * M * K, 0.0f, 1.0f);
    rand_vector(host_b.get(), static_cast<std::size_t>(batch) * N * K, -0.5f, 0.5f);
    rand_scale_e8m0(host_sfa.get(), sfa_count);
    rand_scale_e8m0(host_sfb.get(), sfb_count);
    if (const char* u = std::getenv("MXFP8_UNIT_SCALE")) {
        if (std::atoi(u)) {
            std::fill_n(host_sfa.get(), sfa_count, static_cast<e8m0_t>(127));
            std::fill_n(host_sfb.get(), sfb_count, static_cast<e8m0_t>(127));
        }
    }
    if (const char* sfa_value = std::getenv("MXFP8_SFA_VALUE")) {
        std::fill_n(host_sfa.get(), sfa_count, static_cast<e8m0_t>(std::atoi(sfa_value)));
    }
    if (const char* sfb_value = std::getenv("MXFP8_SFB_VALUE")) {
        std::fill_n(host_sfb.get(), sfb_count, static_cast<e8m0_t>(std::atoi(sfb_value)));
    }
    if (std::getenv("MXFP8_SFA_ROW_PATTERN")) {
        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < M; ++i) {
                const e8m0_t value = static_cast<e8m0_t>(124 + (i % 7));
                std::fill_n(host_sfa.get() + static_cast<std::size_t>(b) * M * num_groups_k + i * num_groups_k,
                            num_groups_k, value);
            }
        }
    }
    if (std::getenv("MXFP8_SFA_K_PATTERN")) {
        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < M; ++i) {
                for (int g = 0; g < num_groups_k; ++g) {
                    host_sfa[static_cast<std::size_t>(b) * M * num_groups_k + i * num_groups_k + g] =
                        static_cast<e8m0_t>(124 + (g % 7));
                }
            }
        }
    }
    if (std::getenv("MXFP8_SFB_ROW_PATTERN")) {
        for (int b = 0; b < batch; ++b) {
            for (int j = 0; j < N; ++j) {
                const e8m0_t value = static_cast<e8m0_t>(124 + (j % 7));
                std::fill_n(host_sfb.get() + static_cast<std::size_t>(b) * N * num_groups_k + j * num_groups_k,
                            num_groups_k, value);
            }
        }
    }
    if (std::getenv("MXFP8_SFB_K_PATTERN")) {
        for (int b = 0; b < batch; ++b) {
            for (int j = 0; j < N; ++j) {
                for (int g = 0; g < num_groups_k; ++g) {
                    host_sfb[static_cast<std::size_t>(b) * N * num_groups_k + j * num_groups_k + g] =
                        static_cast<e8m0_t>(124 + (g % 7));
                }
            }
        }
    }

    // The default path uploads this; --device-repack only needs it as the
    // oracle to compare against, which is a -v 1 affair.
    if (!device_repack || verify) {
        pack_sfa_consumer_major<GemmTraits>(
            host_sfa.get(), host_sfa_packed.get(), batch, M, K);
        pack_sfb_consumer_major<GemmTraits>(
            host_sfb.get(), host_sfb_packed.get(), batch, N, K);
    }

    void* dev_a = nullptr;
    void* dev_b = nullptr;
    void* dev_sfa = nullptr;
    void* dev_sfb = nullptr;
    fp32_t* dev_c = nullptr;
    CHECK_HIP(hipMalloc(&dev_a, static_cast<std::size_t>(batch) * M * K * sizeof(host_fp8_t)));
    CHECK_HIP(hipMalloc(&dev_b, static_cast<std::size_t>(batch) * N * K * sizeof(host_fp8_t)));
    CHECK_HIP(hipMalloc(&dev_c, static_cast<std::size_t>(batch) * M * N * sizeof(fp32_t)));
    CHECK_HIP(hipMalloc(&dev_sfa, sfa_count * sizeof(e8m0_t)));
    CHECK_HIP(hipMalloc(&dev_sfb, sfb_count * sizeof(e8m0_t)));

    CHECK_HIP(hipMemcpy(dev_a, host_a.get(), static_cast<std::size_t>(batch) * M * K * sizeof(host_fp8_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dev_b, host_b.get(), static_cast<std::size_t>(batch) * N * K * sizeof(host_fp8_t), hipMemcpyHostToDevice));
    if (device_repack) {
        // Upload the scales in their logical layout and repack on the GPU, the
        // way a real pipeline would -- there the scales come out of a
        // quantization kernel and are already in device memory.  The result is
        // compared byte-for-byte against the host packer below, which stays in
        // the build as the correctness oracle.
        void* dev_sfa_raw = nullptr;
        void* dev_sfb_raw = nullptr;
        CHECK_HIP(hipMalloc(&dev_sfa_raw, sfa_count * sizeof(e8m0_t)));
        CHECK_HIP(hipMalloc(&dev_sfb_raw, sfb_count * sizeof(e8m0_t)));
        CHECK_HIP(hipMemcpy(dev_sfa_raw, host_sfa.get(), sfa_count * sizeof(e8m0_t), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(dev_sfb_raw, host_sfb.get(), sfb_count * sizeof(e8m0_t), hipMemcpyHostToDevice));

        const float repack_ms = launch_scale_repack<GemmTraits>(
            static_cast<const e8m0_t*>(dev_sfa_raw), static_cast<e8m0_t*>(dev_sfa),
            static_cast<const e8m0_t*>(dev_sfb_raw), static_cast<e8m0_t*>(dev_sfb),
            batch, M, N, K);

        // The repack writes straight into dev_sfa / dev_sfb, which is what the
        // kernel reads -- nothing is copied again.  Reading it back to compare
        // against the host packer is a self-check on the lane mapping and has
        // no place in a timed run, so it only happens under -v 1.
        if (verify) {
            auto gpu_sfa = std::make_unique<e8m0_t[]>(sfa_count);
            auto gpu_sfb = std::make_unique<e8m0_t[]>(sfb_count);
            CHECK_HIP(hipMemcpy(gpu_sfa.get(), dev_sfa, sfa_count * sizeof(e8m0_t), hipMemcpyDeviceToHost));
            CHECK_HIP(hipMemcpy(gpu_sfb.get(), dev_sfb, sfb_count * sizeof(e8m0_t), hipMemcpyDeviceToHost));
            const bool sfa_match = std::memcmp(gpu_sfa.get(), host_sfa_packed.get(), sfa_count) == 0;
            const bool sfb_match = std::memcmp(gpu_sfb.get(), host_sfb_packed.get(), sfb_count) == 0;
            std::printf("Device scale repack: %.4f ms, SFA %s, SFB %s vs the host packer\n",
                        repack_ms, sfa_match ? "match" : "MISMATCH", sfb_match ? "match" : "MISMATCH");
            if (!sfa_match || !sfb_match) {
                return 1;
            }
        } else {
            std::printf("Device scale repack: %.4f ms\n", repack_ms);
        }

        CHECK_HIP(hipFree(dev_sfa_raw));
        CHECK_HIP(hipFree(dev_sfb_raw));
    } else {
        CHECK_HIP(hipMemcpy(dev_sfa, host_sfa_packed.get(), sfa_count * sizeof(e8m0_t), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(dev_sfb, host_sfb_packed.get(), sfb_count * sizeof(e8m0_t), hipMemcpyHostToDevice));
    }

    opus_gemm_scale_kargs kargs{};
    kargs.ptr_a = dev_a;
    kargs.ptr_b = dev_b;
    kargs.ptr_c = dev_c;
    kargs.m = M;
    kargs.n = N;
    kargs.k = K;
    kargs.batch = batch;
    kargs.stride_a = K;
    kargs.stride_b = K;
    kargs.stride_c = N;
    kargs.stride_a_batch = M * K;
    kargs.stride_b_batch = N * K;
    kargs.stride_c_batch = M * N;
    kargs.ptr_sfa = dev_sfa;
    kargs.ptr_sfb = dev_sfb;
    kargs.stride_sfa = GemmTraits::packed_sfa_tile_elem;
    kargs.stride_sfb = GemmTraits::packed_sfb_tile_elem;
    kargs.stride_sfa_batch = num_tiles_m * num_tiles_k * kargs.stride_sfa;
    kargs.stride_sfb_batch = num_tiles_n * num_tiles_k * kargs.stride_sfb;

    const int output_tiles_per_wg = pick_output_tiles_per_wg(num_tiles_m, num_tiles_n);
    const bool persist = output_tiles_per_wg == GemmTraitsPersist::OUTPUT_TILES_PER_WG;
    const int m_tile_groups =
        ceil_div_scale(num_tiles_m, output_tiles_per_wg);
    dim3 grid(m_tile_groups * num_tiles_n, 1, batch);
    dim3 block(BLOCK_SIZE);

    std::printf("Launching MXFP8 scaled-MFMA GEMM: M=%d, N=%d, K=%d, grid=(%u,%u,%u), block=%d, output_tiles_per_wg=%d\n",
                M, N, K, grid.x, grid.y, grid.z, BLOCK_SIZE,
                output_tiles_per_wg);

    if (verify) {
        if (persist) {
            gemm_a8w8_mxfp8_scale_kernel<GemmTraitsPersist><<<grid, block>>>(kargs);
        } else {
            gemm_a8w8_mxfp8_scale_kernel<GemmTraitsSingle><<<grid, block>>>(kargs);
        }
        CHECK_HIP_KERNEL_LAUNCH();
        std::printf("\nValidating GPU results against CPU reference...\n");
        CHECK_HIP(hipMemcpy(host_c_out.get(), dev_c, static_cast<std::size_t>(batch) * M * N * sizeof(fp32_t),
                            hipMemcpyDeviceToHost));

        bool all_valid = true;
        for (int b = 0; b < batch; ++b) {
            gemm_ref(host_a.get() + static_cast<std::size_t>(b) * M * K,
                     host_b.get() + static_cast<std::size_t>(b) * N * K,
                     host_sfa.get() + static_cast<std::size_t>(b) * M * num_groups_k,
                     host_sfb.get() + static_cast<std::size_t>(b) * N * num_groups_k,
                     host_c.get() + static_cast<std::size_t>(b) * M * N,
                     host_c_mag.get(),
                     M, N, K, K, K, N, num_groups_k, num_groups_k, GROUP_K);
            const bool valid = valid_vector(host_c.get() + static_cast<std::size_t>(b) * M * N,
                                            host_c_out.get() + static_cast<std::size_t>(b) * M * N,
                                            host_c_mag.get(), M * N);
            std::printf("[GEMM batch %d/%d: %dx%dx%d, block_%dx%dx%d] %s\n",
                        b + 1, batch, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, valid ? "VALID" : "FAIL");
            all_valid = all_valid && valid;
        }

        std::printf("\n[Overall] %s\n", all_valid ? "ALL BATCHES VALID" : "SOME BATCHES FAILED");
    }

    std::printf("\n");
    if (timeline) {
        if (persist) {
            benchmark_kernel_timeline<GemmTraitsPersist>(kargs, grid, block);
        } else {
            benchmark_kernel_timeline<GemmTraitsSingle>(kargs, grid, block);
        }
    } else {
        if (persist) {
            benchmark_kernel<GemmTraitsPersist>(kargs, grid, block, warmup, iterations);
        } else {
            benchmark_kernel<GemmTraitsSingle>(kargs, grid, block, warmup, iterations);
        }
    }
    std::printf("\n");

    CHECK_HIP(hipFree(dev_a));
    CHECK_HIP(hipFree(dev_b));
    CHECK_HIP(hipFree(dev_c));
    CHECK_HIP(hipFree(dev_sfa));
    CHECK_HIP(hipFree(dev_sfb));

    return 0;
}
