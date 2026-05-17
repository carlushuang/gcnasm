// SPDX-License-Identifier: MIT
// opus_attn_gfx1201 — host driver + correctness check + benchmark.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>
#include "attn_common.h"

#define HIP_CALL(c) do { \
    hipError_t e = (c); \
    if (e != hipSuccess) { fprintf(stderr, "HIP %s @ %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); std::exit(1); } \
} while(0)

// Forward-declare kernel symbols from per-version .cc files.
template<class T> __global__ void opus_attn_gfx1201_kernel   (opus_attn_kargs);  // v0
template<class T> __global__ void opus_attn_gfx1201_kernel_v1(opus_attn_kargs);  // v1
template<class T> __global__ void opus_attn_gfx1201_kernel_v2(opus_attn_kargs);  // v2
template<class T> __global__ void opus_attn_gfx1201_kernel_v3(opus_attn_kargs);  // v3
template<class T> __global__ void opus_attn_gfx1201_kernel_v4(opus_attn_kargs);  // v4
template<class T> __global__ void opus_attn_gfx1201_kernel_v5(opus_attn_kargs);  // v5
template<class T> __global__ void opus_attn_gfx1201_kernel_v6(opus_attn_kargs);  // v6

template<int BM, int BN, class K>
static void launch_(opus_attn_kargs k, K kern) {
    using T = opus_attn_traits<BM, BN, 128>;
    const int n_blocks = k.N / T::BLOCK_M;
    const dim3 grid(n_blocks, k.H, k.B);
    const dim3 block(T::BLOCK_SIZE);
    kern<<<grid, block, 0, 0>>>(k);
}

static void run_opus_attn_gfx1201(int version, opus_attn_kargs k) {
    switch (version) {
        case 0: launch_<16, 16>(k, opus_attn_gfx1201_kernel   <opus_attn_traits<16, 16, 128>>); break;
        case 1: launch_<64, 16>(k, opus_attn_gfx1201_kernel_v1<opus_attn_traits<64, 16, 128>>); break;
        case 2: launch_<64, 64>(k, opus_attn_gfx1201_kernel_v2<opus_attn_traits<64, 64, 128>>); break;
        case 3: launch_<64, 16>(k, opus_attn_gfx1201_kernel_v3<opus_attn_traits<64, 16, 128>>); break;
        case 4: launch_<16, 16>(k, opus_attn_gfx1201_kernel_v4<opus_attn_traits<16, 16, 128>>); break;
        case 5: launch_<16, 32>(k, opus_attn_gfx1201_kernel_v5<opus_attn_traits<16, 32, 128>>); break;
        case 6: launch_<16, 16>(k, opus_attn_gfx1201_kernel_v6<opus_attn_traits<16, 16, 128>>); break;
        default: fprintf(stderr, "unknown --version=%d\n", version); std::exit(1);
    }
}

static void cpu_reference(int B, int H, int N, int D,
                          const fp16_t* Q, const fp16_t* K, const fp16_t* V,
                          fp16_t* O, fp32_t scale)
{
    std::vector<fp32_t> S(N), P(N);
    for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h) {
        const fp16_t* Qbh = Q + (b * H + h) * N * D;
        const fp16_t* Kbh = K + (b * H + h) * N * D;
        const fp16_t* Vbh = V + (b * H + h) * N * D;
        fp16_t*       Obh = O + (b * H + h) * N * D;
        for (int m = 0; m < N; ++m) {
            fp32_t row_max = -3.4e38f;
            for (int n = 0; n < N; ++n) {
                fp32_t s = 0.0f;
                for (int d = 0; d < D; ++d) s += (fp32_t)Qbh[m*D+d] * (fp32_t)Kbh[n*D+d];
                s *= scale;
                S[n] = s;
                if (s > row_max) row_max = s;
            }
            fp32_t row_sum = 0.0f;
            for (int n = 0; n < N; ++n) {
                P[n] = std::exp(S[n] - row_max);
                row_sum += P[n];
            }
            const fp32_t inv = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
            for (int d = 0; d < D; ++d) {
                fp32_t o = 0.0f;
                for (int n = 0; n < N; ++n) o += P[n] * (fp32_t)Vbh[n*D+d];
                Obh[m*D+d] = (fp16_t)(o * inv);
            }
        }
    }
}

int main(int argc, char** argv) {
    int B = 1, H = 1, N = 256, D = 128;
    int verify = 1, warmups = 5, iters = 100, version = 1;
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        auto eq = [&](const char* k, int& dst) {
            auto kn = std::strlen(k);
            if (std::strncmp(a, k, kn) == 0) { dst = std::atoi(a[kn] == '=' ? a + kn + 1 : argv[++i]); return true; }
            return false;
        };
        if (eq("-b", B) || eq("--batch", B)) continue;
        if (eq("-h", H) || eq("--heads", H)) continue;
        if (eq("-n", N) || eq("--seq",   N)) continue;
        if (eq("-d", D) || eq("--dim",   D)) continue;
        if (eq("--verify", verify)) continue;
        if (eq("--iters",  iters))  continue;
        if (eq("--version", version)) continue;
    }
    if (D != 128) { fprintf(stderr, "only D=128 supported (got %d)\n", D); return 1; }
    if (N % 64)   { fprintf(stderr, "N must be a multiple of 64 (got %d)\n", N); return 1; }
    printf("running version v%d  B=%d H=%d N=%d D=%d\n", version, B, H, N, D);

    fp32_t scale = 1.0f / std::sqrt((fp32_t)D);
    size_t sz_qkvo = (size_t)B * H * N * D;
    std::vector<fp16_t> hQ(sz_qkvo), hK(sz_qkvo), hV(sz_qkvo), hO(sz_qkvo), hRef(sz_qkvo);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u(-0.5f, 0.5f);
    for (auto& x : hQ) x = (fp16_t)u(rng);
    for (auto& x : hK) x = (fp16_t)u(rng);
    for (auto& x : hV) x = (fp16_t)u(rng);

    fp16_t *dQ, *dK, *dV, *dO;
    HIP_CALL(hipMalloc(&dQ, sz_qkvo * sizeof(fp16_t)));
    HIP_CALL(hipMalloc(&dK, sz_qkvo * sizeof(fp16_t)));
    HIP_CALL(hipMalloc(&dV, sz_qkvo * sizeof(fp16_t)));
    HIP_CALL(hipMalloc(&dO, sz_qkvo * sizeof(fp16_t)));
    HIP_CALL(hipMemcpy(dQ, hQ.data(), sz_qkvo * sizeof(fp16_t), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(dK, hK.data(), sz_qkvo * sizeof(fp16_t), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(dV, hV.data(), sz_qkvo * sizeof(fp16_t), hipMemcpyHostToDevice));

    opus_attn_kargs kargs{};
    kargs.ptr_q = dQ; kargs.ptr_k = dK; kargs.ptr_v = dV; kargs.ptr_o = dO;
    kargs.B = B; kargs.H = H; kargs.N = N; kargs.D = D; kargs.scale = scale;

    // Warmup
    for (int i = 0; i < warmups; ++i) run_opus_attn_gfx1201(version, kargs);
    HIP_CALL(hipDeviceSynchronize());

    // Verify
    if (verify) {
        HIP_CALL(hipMemcpy(hO.data(), dO, sz_qkvo * sizeof(fp16_t), hipMemcpyDeviceToHost));
        cpu_reference(B, H, N, D, hQ.data(), hK.data(), hV.data(), hRef.data(), scale);
        double max_abs = 0, mean_abs = 0, max_rel = 0;
        int n_bad = 0;
        for (size_t i = 0; i < sz_qkvo; ++i) {
            double a = (double)(fp32_t)hO[i], r = (double)(fp32_t)hRef[i];
            double d = std::abs(a - r);
            max_abs = std::max(max_abs, d);
            mean_abs += d;
            double rel = std::abs(r) > 1e-3 ? d / std::abs(r) : 0.0;
            max_rel = std::max(max_rel, rel);
            if (d > 0.05) ++n_bad;
        }
        mean_abs /= sz_qkvo;
        printf("VERIFY B=%d H=%d N=%d D=%d  max_abs=%.4f  mean_abs=%.5f  max_rel=%.4f  n_bad(>0.05)=%d/%zu\n",
               B, H, N, D, max_abs, mean_abs, max_rel, n_bad, sz_qkvo);
    }

    // Bench
    hipEvent_t ev0, ev1;
    HIP_CALL(hipEventCreate(&ev0));
    HIP_CALL(hipEventCreate(&ev1));
    HIP_CALL(hipEventRecord(ev0));
    for (int i = 0; i < iters; ++i) run_opus_attn_gfx1201(version, kargs);
    HIP_CALL(hipEventRecord(ev1));
    HIP_CALL(hipEventSynchronize(ev1));
    float ms = 0;
    HIP_CALL(hipEventElapsedTime(&ms, ev0, ev1));
    ms /= iters;
    // FLOPS: 4 * B * H * N * N * D (2 matmuls, mul+add each)
    double tflops = 4.0 * B * H * (double)N * N * D / (ms * 1e9);
    printf("BENCH  iters=%d  avg=%.3f ms  %.2f TFLOPS\n", iters, ms, tflops);

    HIP_CALL(hipFree(dQ));
    HIP_CALL(hipFree(dK));
    HIP_CALL(hipFree(dV));
    HIP_CALL(hipFree(dO));
    return 0;
}
