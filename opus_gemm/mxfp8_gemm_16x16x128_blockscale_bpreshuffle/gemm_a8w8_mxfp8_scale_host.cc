#include <hip/hip_fp8.h>
#include <opus/hip_minimal.hpp>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "gemm_a8w8_mxfp8_scale_common.h"

template<class Traits>
__global__ void gemm_a8w8_mxfp8_scale_kernel(opus_gemm_scale_kargs kargs);

#define CHECK_HIP(call)                                                                        \
    do {                                                                                       \
        const hipError_t status_ = call;                                                       \
        if (status_ != hipSuccess) {                                                           \
            std::fprintf(stderr, "HIP error (%s:%d): %s\n", __FILE__, __LINE__,                 \
                         hipGetErrorString(status_));                                          \
            std::exit(1);                                                                      \
        }                                                                                      \
    } while (0)

using GemmTraitsPersist = gemm_a8w8_mxfp8_scale_traits<256, 256, 128, 1, 128, 128, 4, false>;
using GemmTraitsSingle = gemm_a8w8_mxfp8_scale_traits<256, 256, 128, 1, 128, 128, 1, false>;
using GemmTraitsPersistBF16 = gemm_a8w8_mxfp8_scale_traits<256, 256, 128, 1, 128, 128, 4, true>;
using GemmTraitsSingleBF16 = gemm_a8w8_mxfp8_scale_traits<256, 256, 128, 1, 128, 128, 1, true>;
using GemmTraits = GemmTraitsPersist;
using host_fp8_t = __hip_fp8_e4m3;
using fp32_t = float;
using e8m0_t = std::uint8_t;
using host_bf16_t = std::uint16_t;
static_assert(sizeof(host_fp8_t) == 1, "The preshuffle copies one-byte FP8 values");

struct Options {
    int m = 256;
    int n = 512;
    int k = 256;
    int batch = 8;
    int verify = 0;
    int warmup = 200;
    int iterations = 100;
    int tiles = 0;
    int seed = 1;
    bool bf16 = true;
};

void print_usage(const char* program) {
    std::printf(
        "Usage: %s [-m M] [-n N] [-k K] [-b BATCH] [-v 0|1] [-w WARMUP] [-i ITERATIONS]\n"
        "          [--dtype bf16|fp32] [--tiles 0|1|4] [--seed SEED]\n"
        "M/N must be multiples of 256; K must be a multiple of 128.\n"
        "Defaults: M=256 N=512 K=256 batch=8 verify=0 warmup=200 iterations=100,\n"
        "          dtype=bf16 tiles=0 (auto) seed=1. Options also accept --name=value.\n",
        program);
}

int parse_int(const std::string& text, const std::string& option) {
    errno = 0;
    char* end = nullptr;
    const long long value = std::strtoll(text.c_str(), &end, 10);
    if (text.empty() || end == text.c_str() || *end != '\0' || errno == ERANGE ||
        value < INT_MIN || value > INT_MAX) {
        throw std::invalid_argument(option + " requires a signed 32-bit integer, got '" + text + "'");
    }
    return static_cast<int>(value);
}

Options parse_options(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        const auto equal = arg.find('=');
        const std::string name = arg.substr(0, equal);
        if (name == "-h" || name == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        }
        const bool known = name == "-m" || name == "--m" || name == "-n" || name == "--n" ||
            name == "-k" || name == "--k" || name == "-b" || name == "--b" || name == "--batch" ||
            name == "-v" || name == "--verify" || name == "-w" || name == "--warmup" ||
            name == "-i" || name == "--iterations" || name == "--dtype" || name == "--tiles" ||
            name == "--seed";
        if (!known) throw std::invalid_argument("Unknown argument: " + arg);
        std::string value;
        if (equal != std::string::npos) {
            value = arg.substr(equal + 1);
        } else {
            if (i + 1 == argc) throw std::invalid_argument("Missing value for " + name);
            value = argv[++i];
        }
        if (name == "--dtype") {
            if (value != "bf16" && value != "fp32") {
                throw std::invalid_argument("--dtype must be bf16 or fp32");
            }
            opts.bf16 = value == "bf16";
            continue;
        }
        const int v = parse_int(value, name);
        if (name == "-m" || name == "--m") opts.m = v;
        else if (name == "-n" || name == "--n") opts.n = v;
        else if (name == "-k" || name == "--k") opts.k = v;
        else if (name == "-b" || name == "--b" || name == "--batch") opts.batch = v;
        else if (name == "-v" || name == "--verify") opts.verify = v;
        else if (name == "-w" || name == "--warmup") opts.warmup = v;
        else if (name == "-i" || name == "--iterations") opts.iterations = v;
        else if (name == "--tiles") opts.tiles = v;
        else if (name == "--seed") opts.seed = v;
    }
    if (opts.m <= 0 || opts.n <= 0 || opts.k <= 0 || opts.batch <= 0 || opts.iterations <= 0 ||
        opts.warmup < 0 || opts.seed < 0) {
        throw std::invalid_argument("M/N/K/batch/iterations must be positive; warmup and seed must be non-negative");
    }
    if (opts.verify != 0 && opts.verify != 1) throw std::invalid_argument("--verify must be 0 or 1");
    if (opts.tiles != 0 && opts.tiles != 1 && opts.tiles != 4) {
        throw std::invalid_argument("--tiles must be 0 (auto), 1, or 4");
    }
    if (opts.m % GemmTraits::B_M || opts.n % GemmTraits::B_N || opts.k % GemmTraits::B_K) {
        throw std::invalid_argument("M/N/K must be multiples of 256/256/128");
    }
    return opts;
}

// The kernel's strides, batch offsets, and buffer byte offsets use signed int.
// Reject overflow before multiplication, allocation, or a HIP call.
int checked_count(const char* name, int a, int b) {
    if (a <= 0 || b <= 0 || a > INT_MAX / b) {
        throw std::invalid_argument(std::string(name) + " exceeds the signed 32-bit indexing limit");
    }
    return a * b;
}

// Counter-based initialization is reproducible across OpenMP thread counts.
std::uint64_t mix_bits(std::uint64_t x) {
    x += UINT64_C(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

void fill_fp8(host_fp8_t* ptr, std::size_t count, std::uint64_t seed) {
    #pragma omp parallel for
    for (std::size_t i = 0; i < count; ++i) {
        const float unit = static_cast<float>(mix_bits(seed + i) >> 40) * 0x1p-24f;
        ptr[i] = static_cast<host_fp8_t>(2.0f * unit - 1.0f);
    }
}

void fill_scales(e8m0_t* ptr, std::size_t count, std::uint64_t seed) {
    #pragma omp parallel for
    for (std::size_t i = 0; i < count; ++i) {
        ptr[i] = static_cast<e8m0_t>(124 + mix_bits(seed + i) % 7);
    }
}

// Standard shuffle_weight(layout=(16,16)): [N/16,K/16,16,16].
// Keep raw B for the reference; only this separate packed buffer reaches the GPU.
void pack_weight_16x16(const host_fp8_t* raw, host_fp8_t* packed, int batches, int n, int k) {
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < batches; ++b) {
        for (int nb = 0; nb < n / 16; ++nb) {
            for (int kb = 0; kb < k / 16; ++kb) {
                const std::size_t batch_base = static_cast<std::size_t>(b) * n * k;
                const std::size_t tile_base = batch_base +
                    (static_cast<std::size_t>(nb) * (k / 16) + kb) * 256;
                for (int ni = 0; ni < 16; ++ni) {
                    const std::size_t raw_base = batch_base +
                        static_cast<std::size_t>(nb * 16 + ni) * k + kb * 16;
                    std::memcpy(packed + tile_base + ni * 16, raw + raw_base, 16);
                }
            }
        }
    }
}

float e8m0_to_f32(e8m0_t e) {
    return std::ldexp(1.0f, static_cast<int>(e) - 127);
}

host_bf16_t f32_to_bf16_rne(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & UINT32_C(0x7fffffff)) > UINT32_C(0x7f800000)) {
        return static_cast<host_bf16_t>((bits >> 16) | UINT32_C(0x0040));
    }
    bits += UINT32_C(0x7fff) + ((bits >> 16) & 1u);
    return static_cast<host_bf16_t>(bits >> 16);
}

float bf16_to_f32(host_bf16_t value) {
    const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

float round_bf16(float value) {
    return bf16_to_f32(f32_to_bf16_rne(value));
}

// Bit tests keep validation meaningful even when the harness uses -ffast-math.
bool finite_f32(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}

bool finite_f64(double value) {
    std::uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

// Raw A/B are [M,K]/[N,K]. SFA is physically [K/128,M], while SFB is
// [N/128,K/128]. Double accumulation supplies an independent FP32 reference.
void gemm_ref(const host_fp8_t* a, const host_fp8_t* b, const e8m0_t* sfa, const e8m0_t* sfb,
              float* c, double* mag, int m, int n, int k) {
    constexpr int GROUP_K = GemmTraits::GROUP_K;
    constexpr int GROUP_N = GemmTraits::GROUP_N;
    const int groups_k = k / GROUP_K;
    #pragma omp parallel for collapse(2)
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            const host_fp8_t* a_row = a + static_cast<std::size_t>(row) * k;
            const host_fp8_t* b_row = b + static_cast<std::size_t>(col) * k;
            double sum = 0.0;
            double sum_abs_term = 0.0;
            for (int kg = 0; kg < groups_k; ++kg) {
                const double scale = static_cast<double>(e8m0_to_f32(sfa[static_cast<std::size_t>(kg) * m + row])) *
                    static_cast<double>(e8m0_to_f32(sfb[static_cast<std::size_t>(col / GROUP_N) * groups_k + kg]));
                for (int p = kg * GROUP_K; p < (kg + 1) * GROUP_K; ++p) {
                    const double term = static_cast<double>(static_cast<float>(a_row[p])) *
                        static_cast<double>(static_cast<float>(b_row[p])) * scale;
                    sum += term;
                    sum_abs_term += std::abs(term);
                }
            }
            const std::size_t index = static_cast<std::size_t>(row) * n + col;
            c[index] = static_cast<float>(sum);
            mag[index] = sum_abs_term;
        }
    }
}

bool valid_vector(const float* ref, const void* output, const double* mag, std::size_t count, bool bf16) {
    constexpr double rel_mag = 5e-5;
    constexpr double abs_floor = 1e-4;
    std::size_t errors = 0;
    std::size_t max_index = 0;
    double max_diff = 0.0;
    double max_ratio = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        const float raw_ref = ref[i];
        const float expected = bf16 ? round_bf16(raw_ref) : raw_ref;
        const float got = bf16 ? bf16_to_f32(static_cast<const host_bf16_t*>(output)[i])
                               : static_cast<const float*>(output)[i];
        const double fp32_error = abs_floor + rel_mag * mag[i];
        // Compare BF16 output to an RNE-rounded FP32 reference. Round the bounds
        // too, so an FP32 accumulation perturbation across a BF16 midpoint can
        // select the neighboring BF16 value without allowing arbitrary ULPs.
        const double lo = bf16 ? round_bf16(static_cast<float>(raw_ref - fp32_error)) : raw_ref - fp32_error;
        const double hi = bf16 ? round_bf16(static_cast<float>(raw_ref + fp32_error)) : raw_ref + fp32_error;
        const bool finite = finite_f32(raw_ref) && finite_f32(expected) && finite_f32(got) &&
            finite_f64(mag[i]) && finite_f64(lo) && finite_f64(hi);
        const double diff = finite ? std::abs(static_cast<double>(got) - expected)
                                   : std::numeric_limits<double>::infinity();
        const double tol = std::max(static_cast<double>(expected) - lo, hi - expected);
        const double ratio = !finite ? std::numeric_limits<double>::infinity()
            : (tol > 0.0 ? diff / tol : (diff == 0.0 ? 0.0 : std::numeric_limits<double>::infinity()));
        if (diff > max_diff) {
            max_diff = diff;
            max_index = i;
        }
        max_ratio = std::max(max_ratio, ratio);
        if (!finite || got < lo || got > hi) {
            if (errors < 10) {
                std::printf("Error at %zu: ref=%.9g, result=%.9g, allowed=[%.9g,%.9g], sum_abs_term=%.9g%s\n",
                            i, expected, got, lo, hi, mag[i], finite ? "" : " (non-finite value)");
            }
            ++errors;
        }
    }
    std::printf("Validation stats (%s): errors=%zu/%zu, max_diff=%.9g at %zu, max_ratio=%.3f, rel_mag=%.2e, abs_floor=%.2e\n",
                bf16 ? "bf16 RNE" : "fp32", errors, count, max_diff, max_index, max_ratio, rel_mag, abs_floor);
    return errors == 0;
}

int pick_output_tiles_per_wg(int m_tiles, int n_tiles, int batches) {
    int device = 0;
    int cus = 0;
    CHECK_HIP(hipGetDevice(&device));
    CHECK_HIP(hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, device));
    const std::int64_t persist_wgs = static_cast<std::int64_t>(ceil_div_scale(m_tiles, 4)) * n_tiles * batches;
    return persist_wgs >= cus ? 4 : 1;
}

template<class Traits>
void run_kernel(const opus_gemm_scale_kargs& kargs, dim3 grid, dim3 block,
                bool timed, int warmup, int iterations) {
    if (!timed) {
        gemm_a8w8_mxfp8_scale_kernel<Traits><<<grid, block>>>(kargs);
        CHECK_HIP(hipGetLastError());
        return;
    }
    for (int i = 0; i < warmup; ++i) {
        gemm_a8w8_mxfp8_scale_kernel<Traits><<<grid, block>>>(kargs);
        CHECK_HIP(hipGetLastError());
    }
    hipEvent_t start;
    hipEvent_t stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        gemm_a8w8_mxfp8_scale_kernel<Traits><<<grid, block>>>(kargs);
        CHECK_HIP(hipGetLastError());
    }
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));
    float total_time = 0.0f;
    CHECK_HIP(hipEventElapsedTime(&total_time, start, stop));
    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));
    const double avg_time = static_cast<double>(total_time) / iterations;
    const double flop = 2.0 * kargs.m * kargs.n * kargs.k * kargs.batch;
    std::printf("Kernel Performance: avg_time=%.4f ms, %.2f TFlops\n", avg_time, flop / 1.0e9 / avg_time);
}

void dispatch_kernel(const opus_gemm_scale_kargs& kargs, dim3 grid, dim3 block,
                     const Options& opts, int output_tiles, bool timed) {
    if (opts.bf16 && output_tiles == 4) {
        run_kernel<GemmTraitsPersistBF16>(kargs, grid, block, timed, opts.warmup, opts.iterations);
    } else if (opts.bf16) {
        run_kernel<GemmTraitsSingleBF16>(kargs, grid, block, timed, opts.warmup, opts.iterations);
    } else if (output_tiles == 4) {
        run_kernel<GemmTraitsPersist>(kargs, grid, block, timed, opts.warmup, opts.iterations);
    } else {
        run_kernel<GemmTraitsSingle>(kargs, grid, block, timed, opts.warmup, opts.iterations);
    }
}

int run(const Options& opts) {
    const int M = opts.m, N = opts.n, K = opts.k, batch = opts.batch;
    const int groups_k = K / GemmTraits::GROUP_K;
    const int groups_n = N / GemmTraits::GROUP_N;
    const int a_batch = checked_count("A batch stride", M, K);
    const int b_batch = checked_count("B batch stride", N, K);
    const int c_batch = checked_count("C batch stride", M, N);
    const int sfa_batch = checked_count("SFA batch stride", M, groups_k);
    const int sfb_batch = checked_count("SFB batch stride", groups_n, groups_k);
    const int a_count = checked_count("A element count", batch, a_batch);
    const int b_count = checked_count("B element count", batch, b_batch);
    const int c_count = checked_count("C element count", batch, c_batch);
    const int sfa_count = checked_count("SFA element count", batch, sfa_batch);
    const int sfb_count = checked_count("SFB element count", batch, sfb_batch);
    const int c_bytes = checked_count("C byte count", c_count,
                                     opts.bf16 ? sizeof(host_bf16_t) : sizeof(float));
    const int m_tiles = M / GemmTraits::B_M;
    const int n_tiles = N / GemmTraits::B_N;
    checked_count("Launch workgroup count", checked_count("M*N tile count", m_tiles, n_tiles), batch);

    auto host_a = std::make_unique<host_fp8_t[]>(a_count);
    auto host_b = std::make_unique<host_fp8_t[]>(b_count);
    auto host_b_packed = std::make_unique<host_fp8_t[]>(b_count);
    auto host_sfa = std::make_unique<e8m0_t[]>(sfa_count);
    auto host_sfb = std::make_unique<e8m0_t[]>(sfb_count);
    const std::uint64_t seed = static_cast<std::uint64_t>(opts.seed);
    fill_fp8(host_a.get(), a_count, seed);
    fill_fp8(host_b.get(), b_count, seed ^ UINT64_C(0x3141592653589793));
    pack_weight_16x16(host_b.get(), host_b_packed.get(), batch, N, K);
    // Generate the consumer's exact physical scale layouts directly. No scale
    // conversion, transpose, or repack is performed in this harness.
    // SFA[batch][K/128][M], SFB[batch][N/128][K/128].
    fill_scales(host_sfa.get(), sfa_count, seed ^ UINT64_C(0x2718281828459045));
    fill_scales(host_sfb.get(), sfb_count, seed ^ UINT64_C(0x6a09e667f3bcc909));

    std::unique_ptr<float[]> host_ref;
    std::unique_ptr<double[]> host_mag;
    std::unique_ptr<float[]> host_out_fp32;
    std::unique_ptr<host_bf16_t[]> host_out_bf16;
    if (opts.verify) {
        host_ref = std::make_unique<float[]>(c_batch);
        host_mag = std::make_unique<double[]>(c_batch);
        if (opts.bf16) host_out_bf16 = std::make_unique<host_bf16_t[]>(c_count);
        else host_out_fp32 = std::make_unique<float[]>(c_count);
    }

    void* dev_a = nullptr;
    void* dev_b = nullptr;
    void* dev_c = nullptr;
    void* dev_sfa = nullptr;
    void* dev_sfb = nullptr;
    CHECK_HIP(hipMalloc(&dev_a, a_count));
    CHECK_HIP(hipMalloc(&dev_b, b_count));
    CHECK_HIP(hipMalloc(&dev_c, c_bytes));
    CHECK_HIP(hipMalloc(&dev_sfa, sfa_count));
    CHECK_HIP(hipMalloc(&dev_sfb, sfb_count));
    CHECK_HIP(hipMemcpy(dev_a, host_a.get(), a_count, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dev_b, host_b_packed.get(), b_count, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dev_sfa, host_sfa.get(), sfa_count, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dev_sfb, host_sfb.get(), sfb_count, hipMemcpyHostToDevice));
    host_b_packed.reset();

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
    kargs.stride_a_batch = a_batch;
    kargs.stride_b_batch = b_batch;
    kargs.stride_c_batch = c_batch;
    kargs.ptr_sfa = dev_sfa;
    kargs.ptr_sfb = dev_sfb;
    kargs.stride_sfa = M;
    kargs.stride_sfb = groups_k;
    kargs.stride_sfa_batch = sfa_batch;
    kargs.stride_sfb_batch = sfb_batch;

    const int output_tiles = opts.tiles == 0 ? pick_output_tiles_per_wg(m_tiles, n_tiles, batch) : opts.tiles;
    const int grid_x = checked_count("Grid X", ceil_div_scale(m_tiles, output_tiles), n_tiles);
    const dim3 grid(grid_x, 1, batch);
    const dim3 block(GemmTraits::BLOCK_SIZE);
    std::printf("Launching blockscale bpreshuffle GEMM: M=%d, N=%d, K=%d, batch=%d, dtype=%s, seed=%d, "
                "grid=(%u,%u,%u), block=%d, output_tiles_per_wg=%d%s\n",
                M, N, K, batch, opts.bf16 ? "bf16" : "fp32", opts.seed, grid.x, grid.y, grid.z,
                GemmTraits::BLOCK_SIZE, output_tiles, opts.tiles == 0 ? " (auto)" : " (forced)");
    std::printf("Input layout: A raw [B,M,K], B packed (16,16), SFA [B,K/128,M], SFB [B,N/128,K/128]\n");

    bool all_valid = true;
    if (opts.verify) {
        // NaN poison makes a missing output store a deterministic validation failure.
        CHECK_HIP(hipMemset(dev_c, 0xff, c_bytes));
        dispatch_kernel(kargs, grid, block, opts, output_tiles, false);
        void* host_output = opts.bf16 ? static_cast<void*>(host_out_bf16.get())
                                     : static_cast<void*>(host_out_fp32.get());
        CHECK_HIP(hipMemcpy(host_output, dev_c, c_bytes, hipMemcpyDeviceToHost));
        std::printf("\nValidating GPU results against CPU reference...\n");
        for (int b = 0; b < batch; ++b) {
            gemm_ref(host_a.get() + static_cast<std::size_t>(b) * a_batch,
                     host_b.get() + static_cast<std::size_t>(b) * b_batch,
                     host_sfa.get() + static_cast<std::size_t>(b) * sfa_batch,
                     host_sfb.get() + static_cast<std::size_t>(b) * sfb_batch,
                     host_ref.get(), host_mag.get(), M, N, K);
            const void* result = opts.bf16
                ? static_cast<const void*>(host_out_bf16.get() + static_cast<std::size_t>(b) * c_batch)
                : static_cast<const void*>(host_out_fp32.get() + static_cast<std::size_t>(b) * c_batch);
            const bool valid = valid_vector(host_ref.get(), result, host_mag.get(), c_batch, opts.bf16);
            std::printf("[GEMM batch %d/%d: %dx%dx%d, dtype=%s, tiles=%d] %s\n",
                        b + 1, batch, M, N, K, opts.bf16 ? "bf16" : "fp32", output_tiles, valid ? "VALID" : "FAIL");
            all_valid = all_valid && valid;
        }
        std::printf("\n[Overall] %s\n", all_valid ? "ALL BATCHES VALID" : "SOME BATCHES FAILED");
    }
    if (all_valid) {
        std::printf("\n");
        dispatch_kernel(kargs, grid, block, opts, output_tiles, true);
    }
    CHECK_HIP(hipFree(dev_a));
    CHECK_HIP(hipFree(dev_b));
    CHECK_HIP(hipFree(dev_c));
    CHECK_HIP(hipFree(dev_sfa));
    CHECK_HIP(hipFree(dev_sfb));
    return all_valid ? 0 : 2;
}

int main(int argc, char** argv) {
    try {
        return run(parse_options(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
