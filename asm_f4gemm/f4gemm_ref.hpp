// SPDX-License-Identifier: MIT
// Data layout helpers + CPU reference for the aiter gfx950 f4gemm kernels.
//
// Everything here mirrors the aiter python side that prepares the tensors:
//   A / A_scale : aiter.utility.fp4_utils.dynamic_mxfp4_quant(x, shuffle=True)
//   B           : aiter.ops.shuffle.shuffle_weight(w, layout=(16, 16))
//   B_scale     : same scale shuffle as A_scale
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

namespace f4gemm {

// ---------------------------------------------------------------------------
// MXFP4 (OCP e2m1) and E8M0 scalars
// ---------------------------------------------------------------------------
// Nibble -> value. Byte i of a packed row holds element 2*i in the low nibble
// and element 2*i+1 in the high nibble.
static const float kFp4Lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

// E8M0 is a bare biased exponent: value = 2^(e - 127). 0 -> 2^-126, 0xFF -> NaN.
inline float e8m0_to_f32(uint8_t e)
{
    uint32_t bits;
    if(e == 0)
        bits = 0x00400000u;
    else if(e == 0xFF)
        bits = 0x7F800001u;
    else
        bits = (uint32_t)e << 23;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

inline float bf16_to_f32(uint16_t h)
{
    uint32_t bits = (uint32_t)h << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

inline int cdiv(int a, int b) { return (a + b - 1) / b; }
inline int round_up(int a, int b) { return cdiv(a, b) * b; }

// ---------------------------------------------------------------------------
// Padded buffer geometry
// ---------------------------------------------------------------------------
// The kernel reads whole tiles, so every buffer is over-allocated the same way
// aiter over-allocates: scale rows to a multiple of 256, scale cols to a
// multiple of 8, and the output rows to a multiple of 32. A/B rows are padded to
// 256 as well so an edge tile can never touch unmapped memory.
struct Shape
{
    int M, N, K;

    int Kp() const { return K / 2; }             // packed fp4 bytes per row
    int Ks() const { return K / 32; }            // e8m0 scales per row
    int scale_cols() const { return round_up(Ks(), 8); }
    int a_rows_pad() const { return round_up(M, 256); }
    int b_rows_pad() const { return round_up(N, 256); }
    int scale_a_rows() const { return round_up(M, 256); }
    int scale_b_rows() const { return round_up(N, 256); }
    int d_rows_pad() const { return round_up(M, 32); }
};

// ---------------------------------------------------------------------------
// B preshuffle: shuffle_weight(w, layout=(16, 16)) on the packed byte buffer
// ---------------------------------------------------------------------------
// src.view(N/16, 16, Kp/32, 2, 16).permute(0, 2, 3, 1, 4)
//   src[n0*16 + n1][k0*32 + k1*16 + k2]
//     -> dst[(((n0 * (Kp/32) + k0) * 2 + k1) * 16 + n1) * 16 + k2]
inline std::vector<uint8_t> shuffle_weight_16x16(const std::vector<uint8_t>& src, int N, int Kp)
{
    std::vector<uint8_t> dst((size_t)N * Kp);
    const int nk = Kp / 32;
    for(int n0 = 0; n0 < N / 16; n0++)
        for(int n1 = 0; n1 < 16; n1++)
            for(int k0 = 0; k0 < nk; k0++)
                for(int k1 = 0; k1 < 2; k1++)
                    for(int k2 = 0; k2 < 16; k2++)
                    {
                        size_t s = (size_t)(n0 * 16 + n1) * Kp + k0 * 32 + k1 * 16 + k2;
                        size_t d = ((((size_t)n0 * nk + k0) * 2 + k1) * 16 + n1) * 16 + k2;
                        dst[d]   = src[s];
                    }
    return dst;
}

// ---------------------------------------------------------------------------
// Scale preshuffle: aiter.ops.shuffle.shuffle_scale (non-guinterleave path)
// ---------------------------------------------------------------------------
// pad to [sm = round_up(rows, 256), sn = round_up(cols, 8)], then
//   view(sm/32, 2, 16, sn/8, 2, 4).permute(0, 3, 5, 2, 4, 1)
//   padded[d0*32 + d1*16 + d2][d3*8 + d4*4 + d5]
//     -> dst[((((d0 * (sn/8) + d3) * 4 + d5) * 16 + d2) * 2 + d4) * 2 + d1]
//
// `src` is the unpadded [rows, cols] e8m0 buffer; padding is filled with 0x7F
// (2^0 == 1.0) so a garbage scale can never turn into a NaN/Inf.
inline std::vector<uint8_t>
shuffle_scale(const std::vector<uint8_t>& src, int rows, int cols, int sm, int sn)
{
    std::vector<uint8_t> padded((size_t)sm * sn, 0x7F);
    for(int r = 0; r < rows; r++)
        memcpy(&padded[(size_t)r * sn], &src[(size_t)r * cols], cols);

    std::vector<uint8_t> dst((size_t)sm * sn);
    const int n3 = sn / 8;
    for(int d0 = 0; d0 < sm / 32; d0++)
        for(int d1 = 0; d1 < 2; d1++)
            for(int d2 = 0; d2 < 16; d2++)
                for(int d3 = 0; d3 < n3; d3++)
                    for(int d4 = 0; d4 < 2; d4++)
                        for(int d5 = 0; d5 < 4; d5++)
                        {
                            size_t s = (size_t)(d0 * 32 + d1 * 16 + d2) * sn + d3 * 8 + d4 * 4 + d5;
                            size_t d =
                                (((((size_t)d0 * n3 + d3) * 4 + d5) * 16 + d2) * 2 + d4) * 2 + d1;
                            dst[d] = padded[s];
                        }
    return dst;
}

// ---------------------------------------------------------------------------
// Random MXFP4 operand generation
// ---------------------------------------------------------------------------
// Random nibbles + a narrow exponent window. Keeping the exponents near 127
// keeps every partial product exactly representable in f32, so the only error
// left to measure is the kernel's own f32 accumulate + bf16 round.
struct Operand
{
    std::vector<uint8_t> packed; // [rows, K/2] fp4x2
    std::vector<uint8_t> scales; // [rows, K/32] e8m0, unpadded
    std::vector<float> dequant;  // [rows, K] f32, for the reference
};

inline Operand make_operand(int rows, int K, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> nib(0, 15);
    std::uniform_int_distribution<int> exp_dist(124, 130); // 2^-3 .. 2^3

    Operand op;
    op.packed.resize((size_t)rows * (K / 2));
    op.scales.resize((size_t)rows * (K / 32));
    op.dequant.resize((size_t)rows * K);

    for(int r = 0; r < rows; r++)
    {
        for(int b = 0; b < K / 2; b++)
        {
            uint8_t lo                          = (uint8_t)nib(rng);
            uint8_t hi                          = (uint8_t)nib(rng);
            op.packed[(size_t)r * (K / 2) + b] = (uint8_t)((hi << 4) | lo);
        }
        for(int s = 0; s < K / 32; s++)
            op.scales[(size_t)r * (K / 32) + s] = (uint8_t)exp_dist(rng);

        for(int k = 0; k < K; k++)
        {
            uint8_t byte = op.packed[(size_t)r * (K / 2) + k / 2];
            uint8_t nibv = (k & 1) ? (byte >> 4) : (byte & 0xF);
            float sc     = e8m0_to_f32(op.scales[(size_t)r * (K / 32) + k / 32]);
            op.dequant[(size_t)r * K + k] = kFp4Lut[nibv] * sc;
        }
    }
    return op;
}

// ---------------------------------------------------------------------------
// CPU reference: D = alpha * A * B^T + beta * C
// ---------------------------------------------------------------------------
inline std::vector<float> reference_gemm(const std::vector<float>& A,
                                         const std::vector<float>& B,
                                         int M,
                                         int N,
                                         int K,
                                         float alpha,
                                         const std::vector<float>& C,
                                         float beta)
{
    std::vector<float> D((size_t)M * N);
    unsigned nthreads = std::max(1u, std::thread::hardware_concurrency());
    nthreads          = std::min<unsigned>(nthreads, (unsigned)M);

    auto worker = [&](int m_begin, int m_end) {
        for(int m = m_begin; m < m_end; m++)
            for(int n = 0; n < N; n++)
            {
                double acc = 0.0;
                const float* a = &A[(size_t)m * K];
                const float* b = &B[(size_t)n * K];
                for(int k = 0; k < K; k++)
                    acc += (double)a[k] * (double)b[k];
                float v = alpha * (float)acc;
                if(beta != 0.0f && !C.empty())
                    v += beta * C[(size_t)m * N + n];
                D[(size_t)m * N + n] = v;
            }
    };

    std::vector<std::thread> pool;
    int chunk = cdiv(M, (int)nthreads);
    for(unsigned t = 0; t < nthreads; t++)
    {
        int b = (int)t * chunk;
        int e = std::min(M, b + chunk);
        if(b < e)
            pool.emplace_back(worker, b, e);
    }
    for(auto& th : pool)
        th.join();
    return D;
}

} // namespace f4gemm
