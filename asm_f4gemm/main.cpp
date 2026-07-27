// SPDX-License-Identifier: MIT
// Standalone driver for the ROCm/aiter gfx950 "f4gemm" MXFP4 assembly kernels.
//
//   D[M, N] (bf16) = alpha * A[M, K] (mxfp4) * B[N, K] (mxfp4)^T
//
// The code objects are not shipped here -- point --co-dir at
// <aiter>/hsa/gfx950/f4gemm (see README.md).
#include "f4gemm.hpp"
#include "f4gemm_ref.hpp"

#include <cstdio>
#include <cstring>
#include <string>

using namespace f4gemm;

struct Options
{
    int M            = 512;
    int N            = 1024;
    int K            = 2048;
    int bpreshuffle  = 1;
    int log2_k_split = 0;  // <0 -> let the heuristic try 2/4/8/16
    int iters        = 20;
    int warmup       = 5;
    bool verify      = true;
    bool list        = false;
    std::string kernel;
    std::string co_dir;
};

static void usage(const char* exe)
{
    printf(
        "usage: %s [options]\n"
        "  --co-dir DIR     directory holding the .co files + manifest CSV\n"
        "                   (default $AITER_F4GEMM_DIR, else $AITER_ASM_DIR/gfx950/f4gemm)\n"
        "  -m, -n, -k INT   GEMM shape (default 512 1024 2048; K must be a multiple of 256)\n"
        "  --kernel NAME    force a kernel by mangled name (default: heuristic)\n"
        "  --bpreshuffle 0|1  use the 16x16-preshuffled-B kernels (default 1)\n"
        "  --splitk L       log2 of the K split; -1 lets the heuristic choose (default 0)\n"
        "  --iters N        timed iterations (default 20)\n"
        "  --no-verify      skip the CPU reference check\n"
        "  --list           print the kernel manifest and exit\n",
        exe);
}

static std::string default_co_dir()
{
    if(const char* d = getenv("AITER_F4GEMM_DIR"))
        return d;
    if(const char* d = getenv("AITER_ASM_DIR"))
        return std::string(d) + "/gfx950/f4gemm";
    return "";
}

static bool parse_args(int argc, char** argv, Options& o)
{
    o.co_dir = default_co_dir();
    for(int i = 1; i < argc; i++)
    {
        std::string a = argv[i];
        auto next     = [&](const char* what) -> const char* {
            if(i + 1 >= argc)
            {
                printf("[f4gemm] missing value for %s\n", what);
                exit(EXIT_FAILURE);
            }
            return argv[++i];
        };
        if(a == "-m")
            o.M = atoi(next("-m"));
        else if(a == "-n")
            o.N = atoi(next("-n"));
        else if(a == "-k")
            o.K = atoi(next("-k"));
        else if(a == "--co-dir")
            o.co_dir = next("--co-dir");
        else if(a == "--kernel")
            o.kernel = next("--kernel");
        else if(a == "--bpreshuffle")
            o.bpreshuffle = atoi(next("--bpreshuffle"));
        else if(a == "--splitk")
            o.log2_k_split = atoi(next("--splitk"));
        else if(a == "--iters")
            o.iters = atoi(next("--iters"));
        else if(a == "--no-verify")
            o.verify = false;
        else if(a == "--list")
            o.list = true;
        else if(a == "-h" || a == "--help")
        {
            usage(argv[0]);
            exit(0);
        }
        else
        {
            printf("[f4gemm] unknown option %s\n", a.c_str());
            usage(argv[0]);
            return false;
        }
    }
    if(o.co_dir.empty())
    {
        printf("[f4gemm] no code-object directory: pass --co-dir or set AITER_ASM_DIR\n");
        return false;
    }
    return true;
}

int main(int argc, char** argv)
{
    Options opt;
    if(!parse_args(argc, argv, opt))
        return 1;

    const std::string csv = opt.co_dir + "/f4gemm_bf16_per1x32Fp4.csv";
    std::vector<Config> cfgs = load_configs(csv);
    printf("[f4gemm] manifest %s: %zu kernels\n", csv.c_str(), cfgs.size());

    if(opt.list)
    {
        printf("%-6s %-6s %-7s %-12s %s\n", "tileM", "tileN", "splitK", "bpreshuffle", "co_name");
        for(const auto& c : cfgs)
            printf("%-6d %-6d %-7d %-12d %s\n",
                   c.tile_M, c.tile_N, c.splitK, c.bpreshuffle, c.co_name.c_str());
        return 0;
    }

    hipDeviceProp_t prop;
    int dev;
    HIP_CALL(hipGetDevice(&dev));
    HIP_CALL(hipGetDeviceProperties(&prop, dev));
    std::string arch = prop.gcnArchName;
    arch             = arch.substr(0, arch.find(':'));
    printf("[f4gemm] device: %s (%s, %d CUs)\n", prop.name, arch.c_str(), prop.multiProcessorCount);
    if(arch != "gfx950")
        printf("[f4gemm] WARNING: these code objects are gfx950-only\n");

    const int M = opt.M, N = opt.N, K = opt.K;
    if(K % 256 != 0)
    {
        printf("[f4gemm] K must be a multiple of 256 (got %d)\n", K);
        return 1;
    }
    if(N % 16 != 0)
    {
        printf("[f4gemm] N must be a multiple of 16 for the B preshuffle (got %d)\n", N);
        return 1;
    }

    // ---- kernel selection -------------------------------------------------
    int idx = -1, log2_k_split = (opt.log2_k_split >= 0) ? opt.log2_k_split : 0;
    if(!opt.kernel.empty())
    {
        for(size_t i = 0; i < cfgs.size(); i++)
            if(cfgs[i].knl_name == opt.kernel || cfgs[i].co_name == opt.kernel)
                idx = (int)i;
        if(idx < 0)
        {
            printf("[f4gemm] kernel '%s' not in the manifest\n", opt.kernel.c_str());
            return 1;
        }
    }
    else
    {
        auto sel = select_kernel(
            M, N, K, prop.multiProcessorCount, opt.log2_k_split, opt.bpreshuffle, cfgs);
        idx          = sel.first;
        log2_k_split = sel.second;
        if(idx < 0)
        {
            printf("[f4gemm] heuristic found no kernel for this shape\n");
            return 1;
        }
    }
    const Config& cfg = cfgs[idx];
    LaunchDesc ld     = make_launch(cfg, M, N, K, log2_k_split);
    printf("[f4gemm] shape M=%d N=%d K=%d, kernel %s (tile %dx%d, splitK=%d, bpreshuffle=%d)\n",
           M, N, K, cfg.co_name.c_str(), cfg.tile_M, cfg.tile_N, cfg.splitK, cfg.bpreshuffle);
    printf("[f4gemm] grid = (%d, %d, %d) x 256 threads, log2_k_split=%d\n",
           ld.gdx, ld.gdy, ld.gdz, ld.log2_k_split);

    Kernel kern;
    kern.load(opt.co_dir, cfg);

    // ---- host operands ----------------------------------------------------
    Shape sh{M, N, K};
    Operand A = make_operand(M, K, 1234);
    Operand B = make_operand(N, K, 5678);

    std::vector<uint8_t> h_A((size_t)sh.a_rows_pad() * sh.Kp(), 0);
    memcpy(h_A.data(), A.packed.data(), A.packed.size());

    std::vector<uint8_t> h_B((size_t)sh.b_rows_pad() * sh.Kp(), 0);
    if(cfg.bpreshuffle)
    {
        auto shuf = shuffle_weight_16x16(B.packed, N, sh.Kp());
        memcpy(h_B.data(), shuf.data(), shuf.size());
    }
    else
        memcpy(h_B.data(), B.packed.data(), B.packed.size());

    std::vector<uint8_t> h_SA =
        shuffle_scale(A.scales, M, sh.Ks(), sh.scale_a_rows(), sh.scale_cols());
    std::vector<uint8_t> h_SB =
        shuffle_scale(B.scales, N, sh.Ks(), sh.scale_b_rows(), sh.scale_cols());

    // ---- device buffers ---------------------------------------------------
    void *d_A = nullptr, *d_B = nullptr, *d_SA = nullptr, *d_SB = nullptr, *d_D = nullptr;
    size_t d_bytes = (size_t)sh.d_rows_pad() * N * sizeof(uint16_t);
    HIP_CALL(hipMalloc(&d_A, h_A.size()));
    HIP_CALL(hipMalloc(&d_B, h_B.size()));
    HIP_CALL(hipMalloc(&d_SA, h_SA.size()));
    HIP_CALL(hipMalloc(&d_SB, h_SB.size()));
    HIP_CALL(hipMalloc(&d_D, d_bytes));
    HIP_CALL(hipMemcpy(d_A, h_A.data(), h_A.size(), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_B, h_B.data(), h_B.size(), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_SA, h_SA.data(), h_SA.size(), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_SB, h_SB.data(), h_SB.size(), hipMemcpyHostToDevice));
    HIP_CALL(hipMemset(d_D, 0, d_bytes));

    // beta / C are left at 0 / nullptr: that is the path aiter's gemm_a4w4 takes
    // by default, and the only one exercised by aiter's own op tests.
    const float alpha = 1.0f, beta = 0.0f;
    KernelArgs args;
    fill_args(args, d_D, nullptr, d_A, d_B, d_SA, d_SB, M, N, K,
              /*stride_D0=*/N,
              /*stride_ScaleA0=*/sh.scale_cols(),
              /*stride_ScaleB0=*/sh.scale_cols(),
              alpha, beta, ld.log2_k_split);

    auto run_once = [&](hipStream_t stream) {
        if(ld.zero_out)
            HIP_CALL(hipMemsetAsync(d_D, 0, d_bytes, stream));
        kern.launch(args, ld.gdx, ld.gdy, ld.gdz, stream);
    };

    run_once(nullptr);
    HIP_CALL(hipDeviceSynchronize());

    // ---- verify -----------------------------------------------------------
    int rc = 0;
    if(opt.verify)
    {
        std::vector<uint16_t> h_D((size_t)sh.d_rows_pad() * N);
        HIP_CALL(hipMemcpy(h_D.data(), d_D, d_bytes, hipMemcpyDeviceToHost));

        std::vector<float> ref =
            reference_gemm(A.dequant, B.dequant, M, N, K, alpha, {}, 0.0f);

        double max_abs_ref = 0.0;
        for(float v : ref)
            max_abs_ref = std::max(max_abs_ref, (double)std::fabs(v));

        // bf16 carries ~8 mantissa bits (rel eps 2^-8); allow a few ulp on top
        // of that for the kernel's f32 accumulation order.
        //
        // With log2_k_split > 0 the epilogue writes each K-chunk's partial sum
        // with buffer_atomic_pk_add_bf16, i.e. the cross-chunk reduction itself
        // happens in bf16. The error then scales with the magnitude of the
        // partials rather than with the final element, so the bound has to be
        // absolute (a few bf16 ulp of max|ref|), not per-element relative.
        const bool split_accum = ld.gdz > 1;
        const double rtol      = 2e-2;
        const double atol      = (split_accum ? 2e-2 : 2e-4) * max_abs_ref;
        if(split_accum)
            printf("[f4gemm] note: %d-way splitK reduces in bf16 (buffer_atomic_pk_add_bf16),"
                   " using abs tol %.4g\n", ld.gdz, atol);

        double worst_abs = 0.0, worst_rel = 0.0;
        size_t bad = 0, first_bad = 0;
        for(int m = 0; m < M; m++)
            for(int n = 0; n < N; n++)
            {
                size_t i  = (size_t)m * N + n;
                double got = bf16_to_f32(h_D[i]);
                double exp = ref[i];
                double d   = std::fabs(got - exp);
                double rel = d / std::max(std::fabs(exp), 1e-30);
                worst_abs  = std::max(worst_abs, d);
                if(std::fabs(exp) > atol)
                    worst_rel = std::max(worst_rel, rel);
                if(d > atol + rtol * std::fabs(exp))
                {
                    if(bad == 0)
                        first_bad = i;
                    bad++;
                }
            }

        printf("[f4gemm] verify: max|ref|=%.4g  max_abs_err=%.4g  max_rel_err=%.4g  mismatches=%zu/%zu\n",
               max_abs_ref, worst_abs, worst_rel, bad, (size_t)M * N);
        if(bad)
        {
            printf("[f4gemm]   first mismatch at (m=%zu, n=%zu): got %.6g, want %.6g\n",
                   first_bad / N, first_bad % N,
                   (double)bf16_to_f32(h_D[first_bad]), (double)ref[first_bad]);
            printf("[f4gemm] FAILED\n");
            rc = 1;
        }
        else
            printf("[f4gemm] PASSED\n");
    }

    // ---- benchmark --------------------------------------------------------
    if(opt.iters > 0)
    {
        hipStream_t stream;
        HIP_CALL(hipStreamCreate(&stream));
        for(int i = 0; i < opt.warmup; i++)
            run_once(stream);
        HIP_CALL(hipStreamSynchronize(stream));

        hipEvent_t beg, end;
        HIP_CALL(hipEventCreate(&beg));
        HIP_CALL(hipEventCreate(&end));
        HIP_CALL(hipEventRecord(beg, stream));
        for(int i = 0; i < opt.iters; i++)
            run_once(stream);
        HIP_CALL(hipEventRecord(end, stream));
        HIP_CALL(hipEventSynchronize(end));

        float ms = 0.f;
        HIP_CALL(hipEventElapsedTime(&ms, beg, end));
        double us     = (double)ms * 1000.0 / opt.iters;
        double tflops = 2.0 * M * N * K / us / 1e6;
        double bytes  = (double)M * K / 2 + (double)N * K / 2 + (double)M * N * 2;
        printf("[f4gemm] %8.2f us   %8.2f TFLOP/s   %7.2f TB/s\n", us, tflops, bytes / us / 1e6);
        HIP_CALL(hipStreamDestroy(stream));
    }

    HIP_CALL(hipFree(d_A));
    HIP_CALL(hipFree(d_B));
    HIP_CALL(hipFree(d_SA));
    HIP_CALL(hipFree(d_SB));
    HIP_CALL(hipFree(d_D));
    return rc;
}
