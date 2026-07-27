// SPDX-License-Identifier: MIT
// Host-side launch logic for the ROCm/aiter gfx950 "f4gemm" assembly kernels.
//
// The code objects themselves are NOT part of gcnasm -- they live in the aiter
// repo under hsa/gfx950/f4gemm/ (see README.md). This header re-implements the
// launch path of aiter's csrc/py_itfs_cu/asm_gemm_a4w4.cu with no torch / no
// aiter dependency, so the kernels can be driven from a plain HIP program.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#define HIP_CALL(call)                                                                  \
    do                                                                                  \
    {                                                                                   \
        hipError_t err = call;                                                          \
        if(err != hipSuccess)                                                           \
        {                                                                               \
            printf("[HIP ERROR] (%d) %s  at %s:%d\n",                                   \
                   (int)err,                                                            \
                   hipGetErrorString(err),                                              \
                   __FILE__,                                                            \
                   __LINE__);                                                           \
            exit(EXIT_FAILURE);                                                         \
        }                                                                               \
    } while(0)

namespace f4gemm {

// ---------------------------------------------------------------------------
// Kernel argument block
// ---------------------------------------------------------------------------
// Every scalar is padded out to a 16-byte slot: the kernels were generated with
// one "argument" per 16 bytes and load them with s_load_dword at fixed offsets.
// Verified against the .co metadata (llvm-readelf --notes), e.g.
//   D 0x00, C 0x10, A 0x20, B 0x30, alpha 0x40, beta 0x50,
//   strideD0 0x60, strideD1 0x70, strideC0 0x80, strideC1 0x90,
//   strideA0 0xa0, strideA1 0xb0, strideB0 0xc0, strideB1 0xd0,
//   M 0xe0, N 0xf0, K 0x100, ScaleA 0x110, ScaleB 0x120,
//   strideScaleA0 0x130, strideScaleA1 0x140,
//   strideScaleB0 0x150, strideScaleB1 0x160, log2_k_split 0x170
// Disassembly shows the kernel only ever s_loads:
//   0x00 0x10 0x20 0x30 0x40 0x50 0x80 0xa0 0xc0 0xe0 0xf0 0x100
//   0x110 0x120 0x130 0x150 and (splitK kernels only) 0x170.
// The unread slots are kept so the offsets line up.
struct p2
{
    unsigned int _p0, _p1;
};
struct p3
{
    unsigned int _p0, _p1, _p2;
};

struct __attribute__((packed)) KernelArgs
{
    void* ptr_D;
    p2 _p0;
    void* ptr_C;
    p2 _p1;
    void* ptr_A;
    p2 _p2;
    void* ptr_B;
    p2 _p3;
    float alpha;
    p3 _p4;
    float beta;
    p3 _p5;
    unsigned int stride_D0;
    p3 _p6;
    unsigned int stride_D1;
    p3 _p7;
    unsigned int stride_C0;
    p3 _p8;
    unsigned int stride_C1;
    p3 _p9;
    unsigned int stride_A0;
    p3 _p10;
    unsigned int stride_A1;
    p3 _p11;
    unsigned int stride_B0;
    p3 _p12;
    unsigned int stride_B1;
    p3 _p13;
    unsigned int M;
    p3 _p14;
    unsigned int N;
    p3 _p15;
    unsigned int K;
    p3 _p16;
    void* ptr_ScaleA;
    p2 _p17;
    void* ptr_ScaleB;
    p2 _p18;
    unsigned int stride_ScaleA0;
    p3 _p19;
    unsigned int stride_ScaleA1;
    p3 _p20;
    unsigned int stride_ScaleB0;
    p3 _p21;
    unsigned int stride_ScaleB1;
    p3 _p22;
    int log2_k_split;
};

static_assert(sizeof(KernelArgs) == 0x174, "f4gemm kernarg layout mismatch");

// ---------------------------------------------------------------------------
// Kernel manifest (hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4.csv)
// ---------------------------------------------------------------------------
struct Config
{
    int tile_M     = 0;
    int tile_N     = 0;
    int splitK     = 0; // 1 -> kernel understands log2_k_split / gdz
    int bpreshuffle = 0; // 1 -> B must be 16x16-tile preshuffled
    std::string knl_name; // mangled symbol inside the .co
    std::string co_name;  // file name of the code object
};

inline std::string trim(const std::string& s)
{
    size_t b = s.find_first_not_of(" \t\r\n");
    if(b == std::string::npos)
        return "";
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

// Parse the manifest that ships next to the code objects. Column order is taken
// from the header row so a reordered/extended CSV still works.
inline std::vector<Config> load_configs(const std::string& csv_path)
{
    std::ifstream f(csv_path);
    if(!f.is_open())
    {
        printf("[f4gemm] cannot open manifest %s\n", csv_path.c_str());
        exit(EXIT_FAILURE);
    }

    std::string line;
    if(!std::getline(f, line))
    {
        printf("[f4gemm] empty manifest %s\n", csv_path.c_str());
        exit(EXIT_FAILURE);
    }

    std::vector<std::string> cols;
    {
        std::stringstream ss(line);
        std::string tok;
        while(std::getline(ss, tok, ','))
            cols.push_back(trim(tok));
    }

    std::vector<Config> out;
    while(std::getline(f, line))
    {
        if(trim(line).empty())
            continue;
        std::vector<std::string> vals;
        std::stringstream ss(line);
        std::string tok;
        while(std::getline(ss, tok, ','))
            vals.push_back(trim(tok));
        if(vals.size() != cols.size())
            continue;

        Config c;
        for(size_t i = 0; i < cols.size(); i++)
        {
            const std::string& k = cols[i];
            const std::string& v = vals[i];
            if(k == "tile_M")
                c.tile_M = std::atoi(v.c_str());
            else if(k == "tile_N")
                c.tile_N = std::atoi(v.c_str());
            else if(k == "splitK")
                c.splitK = std::atoi(v.c_str());
            else if(k == "bpreshuffle")
                c.bpreshuffle = std::atoi(v.c_str());
            else if(k == "knl_name")
                c.knl_name = v;
            else if(k == "co_name")
                c.co_name = v;
        }
        if(!c.knl_name.empty() && !c.co_name.empty())
            out.push_back(c);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Heuristic tile pick -- port of get_heuristic_kernel() in asm_gemm_a4w4.cu
// ---------------------------------------------------------------------------
// aiter iterates an unordered_map, so its tie-breaking is not reproducible; here
// the manifest order (CSV order) decides ties, which makes the pick stable.
//
// log2_k_split < 0 means "let the heuristic also try splitK 2/4/8/16".
// Returns the index into `cfgs` and the chosen log2 split, or index -1.
inline std::pair<int, int> select_kernel(
    int M, int N, int /*K*/, int num_cu, int log2_k_split, int bpreshuffle,
    const std::vector<Config>& cfgs)
{
    uint32_t empty_cu        = (uint32_t)num_cu;
    uint32_t round           = 0xffffffffu;
    float compute2mem_effi   = 1.0f;
    int log2_k_split_en      = (log2_k_split > 0) ? 1 : 0;
    int bpreshuffle_en       = (bpreshuffle == 0) ? 0 : 1;
    int selected             = -1;
    int selected_splitK      = 1;

    for(size_t i = 0; i < cfgs.size(); i++)
    {
        const Config& cfg = cfgs[i];
        if(cfg.bpreshuffle != bpreshuffle_en || cfg.splitK < log2_k_split_en)
            continue;
        // tile128x512 may not support N % tile_N != 0
        if(cfg.tile_M == 128 && cfg.tile_N == 512 && (N % cfg.tile_N) != 0)
            continue;

        std::vector<int> splitK_list;
        if(log2_k_split >= 0 && cfg.splitK)
            splitK_list = {1 << log2_k_split};
        else if(cfg.splitK)
            splitK_list = {2, 4, 8, 16};
        else
            splitK_list = {1};

        for(int splitK : splitK_list)
        {
            int tg_num_M         = (M + cfg.tile_M - 1) / cfg.tile_M;
            int tg_num_N         = (N + cfg.tile_N - 1) / cfg.tile_N;
            uint32_t tg_num      = (uint32_t)(tg_num_M * tg_num_N * splitK);
            uint32_t local_round = (tg_num + num_cu - 1) / num_cu;

            float local_effi = (float)cfg.tile_M * cfg.tile_N / (cfg.tile_M + cfg.tile_N);

            bool earlier   = local_round < round;
            bool same      = local_round == round;
            bool more_busy = empty_cu > (local_round * (uint32_t)num_cu - tg_num);
            bool better    = local_effi > compute2mem_effi;
            if(earlier || (same && (more_busy || better)))
            {
                round            = local_round;
                empty_cu         = local_round * (uint32_t)num_cu - tg_num;
                compute2mem_effi = local_effi;
                selected         = (int)i;
                selected_splitK  = splitK;
            }
        }
    }

    int log2_result = 0;
    while(selected_splitK >>= 1)
        ++log2_result;
    return {selected, log2_result};
}

// ---------------------------------------------------------------------------
// Code-object loading
// ---------------------------------------------------------------------------
class Kernel
{
    public:
    Kernel() = default;

    void load(const std::string& co_dir, const Config& cfg)
    {
        std::string path = co_dir + "/" + cfg.co_name;
        // hipModuleLoad only reports "file not found" without saying which file,
        // which is the single most common way to get the --co-dir wrong.
        {
            std::ifstream probe(path, std::ios::binary);
            if(!probe.is_open())
            {
                printf("[f4gemm] no such code object: %s\n", path.c_str());
                printf("[f4gemm]   the manifest lists this kernel but --co-dir does not"
                       " contain it -- point --co-dir at a directory holding it\n");
                exit(EXIT_FAILURE);
            }
        }
        HIP_CALL(hipModuleLoad(&module_, path.c_str()));
        HIP_CALL(hipModuleGetFunction(&func_, module_, cfg.knl_name.c_str()));
        path_ = path;
    }

    void launch(KernelArgs& args, int gdx, int gdy, int gdz, hipStream_t stream) const
    {
        size_t arg_size = sizeof(args);
        void* config[]  = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                           &args,
                           HIP_LAUNCH_PARAM_BUFFER_SIZE,
                           &arg_size,
                           HIP_LAUNCH_PARAM_END};
        // 256 threads = 4 wave64. LDS (160 KB) is static in the code object, so
        // sharedMemBytes stays 0.
        HIP_CALL(hipModuleLaunchKernel(
            func_, gdx, gdy, gdz, 256, 1, 1, 0, stream, nullptr, (void**)&config));
    }

    const std::string& path() const { return path_; }

    private:
    hipModule_t module_ = nullptr;
    hipFunction_t func_ = nullptr;
    std::string path_;
};

// ---------------------------------------------------------------------------
// Grid setup -- port of the entrypoint body in asm_gemm_a4w4.cu
// ---------------------------------------------------------------------------
struct LaunchDesc
{
    int gdx = 0, gdy = 0, gdz = 1;
    int log2_k_split = 0;
    bool zero_out    = false; // splitK > 1 accumulates into D, so D must be zeroed
};

inline LaunchDesc make_launch(const Config& cfg, int M, int N, int K, int log2_k_split)
{
    LaunchDesc d;
    if(cfg.splitK == 1)
    {
        d.log2_k_split = log2_k_split;
        int k_num      = 1 << d.log2_k_split;
        if(K % k_num != 0)
        {
            printf("[f4gemm] K(%d) %% (1 << log2_k_split)(%d) != 0\n", K, k_num);
            exit(EXIT_FAILURE);
        }
        d.zero_out   = k_num > 1;
        int k_per_tg = K / k_num;
        k_per_tg     = ((k_per_tg + 255) / 256) * 256;
        d.gdz        = (K + k_per_tg - 1) / k_per_tg;
    }
    d.gdx = (N + cfg.tile_N - 1) / cfg.tile_N;
    d.gdy = (M + cfg.tile_M - 1) / cfg.tile_M;
    return d;
}

// D[M, N] bf16 = alpha * A[M, K] * B[N, K]^T + beta * C[M, N]
//
// Strides are element counts (fp4 elements for A/B), matching what aiter feeds
// the kernel: stride_A0 = A.stride(0) * 2 because A is stored as fp4x2 bytes.
inline void fill_args(KernelArgs& args,
                      void* d_D,
                      void* d_C,
                      void* d_A,
                      void* d_B,
                      void* d_ScaleA,
                      void* d_ScaleB,
                      int M,
                      int N,
                      int K,
                      int stride_D0,      // bf16 elements per D row
                      int stride_ScaleA0, // bytes per A-scale row
                      int stride_ScaleB0, // bytes per B-scale row
                      float alpha,
                      float beta,
                      int log2_k_split)
{
    memset(&args, 0, sizeof(args));
    args.ptr_D          = d_D;
    args.ptr_C          = d_C;
    args.ptr_A          = d_A;
    args.ptr_B          = d_B;
    args.alpha          = alpha;
    args.beta           = beta;
    // The kernel reads stride_C0 only -- it uses it for both C and D.
    args.stride_C0      = (unsigned)stride_D0;
    args.stride_D0      = (unsigned)stride_D0;
    args.stride_A0      = (unsigned)K;
    args.stride_B0      = (unsigned)K;
    args.M              = (unsigned)M;
    args.N              = (unsigned)N;
    args.K              = (unsigned)K;
    args.ptr_ScaleA     = d_ScaleA;
    args.ptr_ScaleB     = d_ScaleB;
    args.stride_ScaleA0 = (unsigned)stride_ScaleA0;
    args.stride_ScaleB0 = (unsigned)stride_ScaleB0;
    args.log2_k_split   = log2_k_split;
}

} // namespace f4gemm
