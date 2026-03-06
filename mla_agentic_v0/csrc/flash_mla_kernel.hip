/***************************************************************************************************
 * Flash MLA Decode Kernel — MODEL1 FP8 (head_dim=512, absorbed, K=V)
 *
 * Implements flash-attention-style online softmax decode for Multi-head Latent Attention
 * using gfx942 MFMA 16x16x16 bf16 matrix cores.
 *
 * In the absorbed MLA formulation, K = V = c_t (the compressed latent vector), so we
 * load each KV token once and use it for both the attention score and value accumulation.
 *
 * MODEL1 FP8 layout:
 *   d_qk = 512,  d_nope = 448 (FP8 e4m3fnuz),  d_rope = 64 (BF16)
 *   tile_size = 64,  num_tiles = 7  (per-tile dequant scales)
 *
 * Uses opus.hpp (from aiter) for AMD GPU type definitions and utilities.
 * Target: gfx942 (MI300X), ROCm/HIP
 **************************************************************************************************/
#include <hip/hip_runtime.h>
#include <opus/opus.hpp>

using namespace opus;

// ── Kernel configuration ────────────────────────────────────────────────────
constexpr int HEAD_DIM       = 512;

// ── MODEL1 FP8 dimensions ──────────────────────────────────────────────────
constexpr int D_NOPE         = 448;
constexpr int D_ROPE         = 64;
constexpr int TILE_SIZE      = 64;
constexpr int NUM_TILES      = D_NOPE / TILE_SIZE;           // 7
constexpr int I4_NOPE        = D_NOPE / 4;                    // 112 int32 per token

// ── MFMA tile configuration ────────────────────────────────────────────────
using MFMA_BF16 = opus::mfma<bf16_t, bf16_t, fp32_t, 16, 16, 16>;

constexpr int MFMA_M         = 16;
constexpr int MFMA_N         = 16;
constexpr int MFMA_K         = 16;
constexpr int MFMA_HD        = 512;
constexpr int MFMA_KV_TILE   = 16;
constexpr int MFMA_WARP      = 64;
constexpr int MFMA_QK_ITERS  = MFMA_HD / MFMA_K;     // 32
constexpr int MFMA_PV_CHUNKS = MFMA_HD / MFMA_N;     // 32

// LDS padding to eliminate bank conflicts on gfx942 (32 banks, 4-byte width).
constexpr int SQ_PAD         = 2;
constexpr int SKV_PAD        = 1;
constexpr int SQ_STRIDE      = MFMA_HD + SQ_PAD;       // 514
constexpr int SKV_STRIDE     = MFMA_KV_TILE + SKV_PAD;  // 17


/***************************************************************************************************
 * flash_mla_decode_kernel — MODEL1 FP8 with MFMA 16x16x16 bf16 (CSR indptr)
 *
 * Uses MFMA matrix cores for both QK and PV GEMMs.
 * Dequantizes FP8 nope→BF16 using per-tile scales and loads BF16 rope
 * into shared memory, then performs BF16 MFMA.
 *
 * Block: 64 threads (1 wavefront), processes 16 Q heads at once
 * Grid:  (ceil(num_heads/16), batch_size)
 **************************************************************************************************/
__global__ __launch_bounds__(64)
void flash_mla_decode_kernel(
    const bf16_t*  __restrict__ Q,          // [batch, num_heads, 512] bf16
    const int32_t* __restrict__ KV_nope,    // [total_tokens, 112] int32 (packed fp8)
    const bf16_t*  __restrict__ KV_rope,    // [total_tokens, 64] bf16
    const float*   __restrict__ KV_scales,  // [total_tokens, 7] float32
    bf16_t*        __restrict__ O,          // [batch, num_heads, 512] bf16
    const int*     __restrict__ kv_indptr,
    int num_heads,
    float sm_scale)
{
    const int batch_idx  = blockIdx.y;
    const int head_block = blockIdx.x;
    const int head_start = head_block * MFMA_M;
    const int lid        = threadIdx.x;

    if (head_start >= num_heads) return;

    const int valid_heads = min(MFMA_M, num_heads - head_start);

    const int mfma_row      = lid & 15;
    const int mfma_col_base = (lid >> 4) << 2;

    __shared__ bf16_t sQ   [MFMA_M  * SQ_STRIDE];
    __shared__ bf16_t sKV_T[MFMA_HD * SKV_STRIDE];

    // ── Load Q into shared memory (nope + rope) ─────────────────────────
    const bf16_t* q_base =
        Q + (static_cast<int64_t>(batch_idx) * num_heads + head_start) * MFMA_HD;

    for (int i = lid; i < MFMA_M * MFMA_HD; i += MFMA_WARP) {
        int h = i / MFMA_HD;
        int d = i % MFMA_HD;
        sQ[h * SQ_STRIDE + d] = (h < valid_heads) ? q_base[i] : fp32_to_bf16(0.0f);
    }
    __syncthreads();

    MFMA_BF16 mma;

    MFMA_BF16::vtype_c o_acc[MFMA_PV_CHUNKS];
    #pragma unroll
    for (int c = 0; c < MFMA_PV_CHUNKS; c++)
        opus::clear(o_acc[c]);

    float m_val = -INFINITY;
    float l_val = 0.0f;

    const int kv_begin = kv_indptr[batch_idx];
    const int kv_end   = kv_indptr[batch_idx + 1];

    // ═════════════════════════════════════════════════════════════════════
    // Main loop: process 16 KV tokens per tile
    // ═════════════════════════════════════════════════════════════════════
    for (int kv_tile = kv_begin; kv_tile < kv_end; kv_tile += MFMA_KV_TILE) {
        const int tile_count = min(MFMA_KV_TILE, kv_end - kv_tile);

        // ── Load FP8 nope with per-tile dequant → sKV_T[dim, token] ────
        for (int idx = lid; idx < MFMA_KV_TILE * I4_NOPE; idx += MFMA_WARP) {
            int token = idx / I4_NOPE;
            int i4    = idx % I4_NOPE;
            int dim   = i4 * 4;
            int scale_idx = dim / TILE_SIZE;

            int32_t packed = 0;
            float tile_scale = 0.0f;
            if (token < tile_count) {
                int gtok = kv_tile + token;
                packed = KV_nope[static_cast<int64_t>(gtok) * I4_NOPE + i4];
                tile_scale = KV_scales[static_cast<int64_t>(gtok) * NUM_TILES + scale_idx];
            }

            auto f01 = __builtin_amdgcn_cvt_pk_f32_fp8(packed, 0);
            auto f23 = __builtin_amdgcn_cvt_pk_f32_fp8(packed, 1);

            sKV_T[(dim + 0) * SKV_STRIDE + token] = fp32_to_bf16(f01[0] * tile_scale);
            sKV_T[(dim + 1) * SKV_STRIDE + token] = fp32_to_bf16(f01[1] * tile_scale);
            sKV_T[(dim + 2) * SKV_STRIDE + token] = fp32_to_bf16(f23[0] * tile_scale);
            sKV_T[(dim + 3) * SKV_STRIDE + token] = fp32_to_bf16(f23[1] * tile_scale);
        }

        // ── Load BF16 rope directly → sKV_T[448..511, token] ───────────
        for (int idx = lid; idx < MFMA_KV_TILE * D_ROPE; idx += MFMA_WARP) {
            int token  = idx / D_ROPE;
            int rope_d = idx % D_ROPE;
            int dim    = D_NOPE + rope_d;

            bf16_t val = fp32_to_bf16(0.0f);
            if (token < tile_count) {
                int gtok = kv_tile + token;
                val = KV_rope[static_cast<int64_t>(gtok) * D_ROPE + rope_d];
            }
            sKV_T[dim * SKV_STRIDE + token] = val;
        }
        __syncthreads();

        // ── QK GEMM: S[16,16] = Q[16,512] @ KV[16,512]^T ──────────────
        MFMA_BF16::vtype_c s_acc;
        opus::clear(s_acc);

        for (int k = 0; k < MFMA_QK_ITERS; k++) {
            MFMA_BF16::vtype_a a_val;
            int a_off = mfma_row * SQ_STRIDE + k * MFMA_K + mfma_col_base;
            a_val[0] = sQ[a_off + 0];
            a_val[1] = sQ[a_off + 1];
            a_val[2] = sQ[a_off + 2];
            a_val[3] = sQ[a_off + 3];

            MFMA_BF16::vtype_b b_val;
            int b_base = (k * MFMA_K + mfma_col_base) * SKV_STRIDE + mfma_row;
            b_val[0] = sKV_T[b_base + 0 * SKV_STRIDE];
            b_val[1] = sKV_T[b_base + 1 * SKV_STRIDE];
            b_val[2] = sKV_T[b_base + 2 * SKV_STRIDE];
            b_val[3] = sKV_T[b_base + 3 * SKV_STRIDE];

            s_acc = mma(b_val, a_val, s_acc);
        }

        s_acc[0] *= sm_scale;
        s_acc[1] *= sm_scale;
        s_acc[2] *= sm_scale;
        s_acc[3] *= sm_scale;

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            if (mfma_col_base + i >= tile_count)
                s_acc[i] = -INFINITY;
        }

        // ── Online softmax ──────────────────────────────────────────────
        float local_max = fmaxf(fmaxf(s_acc[0], s_acc[1]),
                                fmaxf(s_acc[2], s_acc[3]));
        local_max = fmaxf(local_max, __shfl_xor(local_max, 16, MFMA_WARP));
        local_max = fmaxf(local_max, __shfl_xor(local_max, 32, MFMA_WARP));

        float m_new = fmaxf(m_val, local_max);
        float correction = __expf(m_val - m_new);

        #pragma unroll
        for (int c = 0; c < MFMA_PV_CHUNKS; c++) {
            o_acc[c][0] *= correction;
            o_acc[c][1] *= correction;
            o_acc[c][2] *= correction;
            o_acc[c][3] *= correction;
        }

        float ev[4], local_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            ev[i] = __expf(s_acc[i] - m_new);
            local_sum += ev[i];
        }
        local_sum += __shfl_xor(local_sum, 16, MFMA_WARP);
        local_sum += __shfl_xor(local_sum, 32, MFMA_WARP);

        l_val = l_val * correction + local_sum;
        m_val = m_new;

        MFMA_BF16::vtype_a p_reg;
        p_reg[0] = fp32_to_bf16(ev[0]);
        p_reg[1] = fp32_to_bf16(ev[1]);
        p_reg[2] = fp32_to_bf16(ev[2]);
        p_reg[3] = fp32_to_bf16(ev[3]);

        // ── PV GEMM: O[16,512] += P[16,16] @ V[16,512] ────────────────
        #pragma unroll
        for (int c = 0; c < MFMA_PV_CHUNKS; c++) {
            MFMA_BF16::vtype_b b_pv;
            int vt_off = (c * MFMA_N + mfma_row) * SKV_STRIDE + mfma_col_base;
            b_pv[0] = sKV_T[vt_off + 0];
            b_pv[1] = sKV_T[vt_off + 1];
            b_pv[2] = sKV_T[vt_off + 2];
            b_pv[3] = sKV_T[vt_off + 3];

            o_acc[c] = mma(b_pv, p_reg, o_acc[c]);
        }

        __syncthreads();
    }

    // ── Finalize ────────────────────────────────────────────────────────
    float inv_l = (l_val > 0.0f) ? (1.0f / l_val) : 0.0f;

    bf16_t* o_base =
        O + (static_cast<int64_t>(batch_idx) * num_heads + head_start) * MFMA_HD;

    #pragma unroll
    for (int c = 0; c < MFMA_PV_CHUNKS; c++) {
        int head_local = mfma_row;
        int dim = c * MFMA_N + mfma_col_base;

        if (head_local < valid_heads) {
            bf16_t* out = o_base + head_local * MFMA_HD + dim;
            out[0] = fp32_to_bf16(o_acc[c][0] * inv_l);
            out[1] = fp32_to_bf16(o_acc[c][1] * inv_l);
            out[2] = fp32_to_bf16(o_acc[c][2] * inv_l);
            out[3] = fp32_to_bf16(o_acc[c][3] * inv_l);
        }
    }
}


/***************************************************************************************************
 * Sparse decode — packed FP8 KV cache
 *
 * Matches FlashMLA's sparse_attn_decode_interface for MODEL1 (d_qk=512).
 *
 * Packed KV layout per token (AoS, 584 bytes):
 *   [0:448)    FP8 e4m3fnuz nope  (448 bytes = 112 int32)
 *   [448:576)  BF16 rope          (128 bytes = 64 bf16)
 *   [576:583)  e8m0 scales        (7 bytes, one per 64-elem tile)
 *   [583:584)  padding            (1 byte)
 *
 * e8m0 scale format: value = 2^(byte - 127).
 *
 * Grid:  (ceil(h_q / MFMA_M_TOTAL), b * s_q)
 * Block: 64 threads (1 wavefront)
 * Template: HEADS_PER_BLOCK = 1 or 2 (number of MFMA head groups per block)
 **************************************************************************************************/

constexpr int SPARSE_SCALE_BYTES     = NUM_TILES + 1;                          // 8
constexpr int SPARSE_BYTES_PER_TOKEN = D_NOPE + 2 * D_ROPE + SPARSE_SCALE_BYTES; // 584
constexpr int SPARSE_NOPE_OFFSET     = 0;
constexpr int SPARSE_ROPE_OFFSET     = D_NOPE;             // 448
constexpr int SPARSE_SCALE_OFFSET    = D_NOPE + 2 * D_ROPE; // 576

__device__ __forceinline__ float e8m0_to_float(uint32_t e8m0_byte) {
    union { uint32_t u; float f; } conv;
    conv.u = e8m0_byte << 23;
    return conv.f;
}

constexpr int SPARSE_BLOCK     = 64;    // 1 wavefront per block
constexpr int MI308_NUM_CUS    = 304;   // MI308X CU count for dispatch heuristic

template <int HEADS_PER_BLOCK>
__global__ __launch_bounds__(SPARSE_BLOCK, 3)
void flash_mla_sparse_decode_kernel(
    const bf16_t*  __restrict__ Q,              // [b, s_q, h_q, 512] bf16
    const uint8_t* __restrict__ KV_packed,      // packed FP8 KV (584 bytes/token)
    const int32_t* __restrict__ indices,        // [b, s_q, topk] int32
    const int32_t* __restrict__ topk_length_ptr,// [b] int32 or nullptr
    bf16_t*        __restrict__ O,              // [b, s_q, h_q, 512] bf16
    float*         __restrict__ lse_out,        // [b, s_q, h_q] float32
    int num_heads,
    int topk,
    int s_q,
    float sm_scale)
{
    constexpr int MFMA_M_TOTAL = HEADS_PER_BLOCK * MFMA_M;

    const int batch_sq_idx = blockIdx.y;
    const int batch_idx    = batch_sq_idx / s_q;
    const int sq_idx       = batch_sq_idx % s_q;
    const int head_block   = blockIdx.x;
    const int head_start   = head_block * MFMA_M_TOTAL;
    const int lid          = threadIdx.x;

    if (head_start >= num_heads) return;

    const int mfma_row     = lid & 15;
    const int mfma_col_base= (lid >> 4) << 2;

    __shared__ bf16_t sKV_T[MFMA_HD * SKV_STRIDE];
    __shared__ const uint8_t* sTokPtrs[MFMA_KV_TILE];

    // ── Load Q for all head groups into registers via LDS staging ──
    const int64_t q_bsq = (static_cast<int64_t>(batch_idx) * s_q + sq_idx)
                           * num_heads * MFMA_HD;
    bf16_t* sQ_tmp = sKV_T;

    bf16_t q_reg[HEADS_PER_BLOCK][MFMA_QK_ITERS * 4];

    #pragma unroll
    for (int g = 0; g < HEADS_PER_BLOCK; g++) {
        int g_head_start = head_start + g * MFMA_M;
        int g_valid = min(MFMA_M, max(0, num_heads - g_head_start));

        if (g_valid > 0) {
            const bf16_t* q_ptr = Q + q_bsq + static_cast<int64_t>(g_head_start) * MFMA_HD;
            for (int i = lid; i < MFMA_M * MFMA_HD; i += SPARSE_BLOCK) {
                int h = i / MFMA_HD;
                sQ_tmp[i] = (h < g_valid) ? q_ptr[i] : fp32_to_bf16(0.0f);
            }
        } else {
            for (int i = lid; i < MFMA_M * MFMA_HD; i += SPARSE_BLOCK)
                sQ_tmp[i] = fp32_to_bf16(0.0f);
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < MFMA_QK_ITERS; k++) {
            int base = mfma_row * MFMA_HD + k * MFMA_K + mfma_col_base;
            q_reg[g][k * 4 + 0] = sQ_tmp[base + 0];
            q_reg[g][k * 4 + 1] = sQ_tmp[base + 1];
            q_reg[g][k * 4 + 2] = sQ_tmp[base + 2];
            q_reg[g][k * 4 + 3] = sQ_tmp[base + 3];
        }
        __syncthreads();
    }

    // ── Effective topk ──
    int eff_topk = topk;
    if (topk_length_ptr != nullptr) {
        eff_topk = min(topk, static_cast<int>(topk_length_ptr[batch_idx]));
        if (eff_topk < 0) eff_topk = 0;
    }

    const int32_t* idx_base = indices
        + (static_cast<int64_t>(batch_idx) * s_q + sq_idx) * topk;

    MFMA_BF16 mma;

    MFMA_BF16::vtype_c o_acc[HEADS_PER_BLOCK][MFMA_PV_CHUNKS];
    float m_val[HEADS_PER_BLOCK];
    float l_val[HEADS_PER_BLOCK];

    #pragma unroll
    for (int g = 0; g < HEADS_PER_BLOCK; g++) {
        #pragma unroll
        for (int c = 0; c < MFMA_PV_CHUNKS; c++)
            opus::clear(o_acc[g][c]);
        m_val[g] = -INFINITY;
        l_val[g] = 0.0f;
    }

    // ═════════════════════════════════════════════════════════════════════
    // Main loop: load KV once, process all head groups
    // ═════════════════════════════════════════════════════════════════════
    for (int tile_start = 0; tile_start < eff_topk; tile_start += MFMA_KV_TILE) {
        const int tile_count = min(MFMA_KV_TILE, eff_topk - tile_start);

        if (lid < MFMA_KV_TILE) {
            const uint8_t* ptr = nullptr;
            if (lid < tile_count) {
                int32_t flat_idx = idx_base[tile_start + lid];
                if (flat_idx >= 0)
                    ptr = KV_packed + static_cast<int64_t>(flat_idx) * SPARSE_BYTES_PER_TOKEN;
            }
            sTokPtrs[lid] = ptr;
        }
        __syncthreads();

        // ── Load FP8 nope → sKV_T[dim, token] ──
        for (int idx = lid; idx < MFMA_KV_TILE * I4_NOPE; idx += SPARSE_BLOCK) {
            int token_in_tile = idx / I4_NOPE;
            int i4            = idx % I4_NOPE;
            int dim           = i4 * 4;
            int scale_idx     = dim / TILE_SIZE;

            int32_t packed    = 0;
            float tile_scale  = 0.0f;

            const uint8_t* tok = sTokPtrs[token_in_tile];
            if (tok != nullptr) {
                packed     = reinterpret_cast<const int32_t*>(tok + SPARSE_NOPE_OFFSET)[i4];
                tile_scale = e8m0_to_float(tok[SPARSE_SCALE_OFFSET + scale_idx]);
            }

            auto f01 = __builtin_amdgcn_cvt_pk_f32_fp8(packed, 0);
            auto f23 = __builtin_amdgcn_cvt_pk_f32_fp8(packed, 1);

            sKV_T[(dim + 0) * SKV_STRIDE + token_in_tile] = fp32_to_bf16(f01[0] * tile_scale);
            sKV_T[(dim + 1) * SKV_STRIDE + token_in_tile] = fp32_to_bf16(f01[1] * tile_scale);
            sKV_T[(dim + 2) * SKV_STRIDE + token_in_tile] = fp32_to_bf16(f23[0] * tile_scale);
            sKV_T[(dim + 3) * SKV_STRIDE + token_in_tile] = fp32_to_bf16(f23[1] * tile_scale);
        }

        // ── Load BF16 rope → sKV_T[448..511, token] ──
        for (int idx = lid; idx < MFMA_KV_TILE * D_ROPE; idx += SPARSE_BLOCK) {
            int token_in_tile = idx / D_ROPE;
            int rope_d        = idx % D_ROPE;
            int dim           = D_NOPE + rope_d;

            bf16_t val = fp32_to_bf16(0.0f);
            const uint8_t* tok = sTokPtrs[token_in_tile];
            if (tok != nullptr)
                val = reinterpret_cast<const bf16_t*>(tok + SPARSE_ROPE_OFFSET)[rope_d];

            sKV_T[dim * SKV_STRIDE + token_in_tile] = val;
        }
        __syncthreads();

        // ── Process each head group using the same KV tile ──
        #pragma unroll
        for (int g = 0; g < HEADS_PER_BLOCK; g++) {
            MFMA_BF16::vtype_c s_acc;
            opus::clear(s_acc);

            for (int k = 0; k < MFMA_QK_ITERS; k++) {
                MFMA_BF16::vtype_a a_val;
                a_val[0] = q_reg[g][k * 4 + 0];
                a_val[1] = q_reg[g][k * 4 + 1];
                a_val[2] = q_reg[g][k * 4 + 2];
                a_val[3] = q_reg[g][k * 4 + 3];

                MFMA_BF16::vtype_b b_val;
                int b_base = (k * MFMA_K + mfma_col_base) * SKV_STRIDE + mfma_row;
                b_val[0] = sKV_T[b_base + 0 * SKV_STRIDE];
                b_val[1] = sKV_T[b_base + 1 * SKV_STRIDE];
                b_val[2] = sKV_T[b_base + 2 * SKV_STRIDE];
                b_val[3] = sKV_T[b_base + 3 * SKV_STRIDE];

                s_acc = mma(b_val, a_val, s_acc);
            }

            s_acc[0] *= sm_scale;
            s_acc[1] *= sm_scale;
            s_acc[2] *= sm_scale;
            s_acc[3] *= sm_scale;

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int tok = mfma_col_base + i;
                if (tok >= tile_count || sTokPtrs[tok] == nullptr)
                    s_acc[i] = -INFINITY;
            }

            float local_max = fmaxf(fmaxf(s_acc[0], s_acc[1]),
                                    fmaxf(s_acc[2], s_acc[3]));
            local_max = fmaxf(local_max, __shfl_xor(local_max, 16, MFMA_WARP));
            local_max = fmaxf(local_max, __shfl_xor(local_max, 32, MFMA_WARP));

            float m_new = fmaxf(m_val[g], local_max);
            float correction = (m_val[g] == -INFINITY) ? 0.0f : __expf(m_val[g] - m_new);

            #pragma unroll
            for (int c = 0; c < MFMA_PV_CHUNKS; c++) {
                o_acc[g][c][0] *= correction;
                o_acc[g][c][1] *= correction;
                o_acc[g][c][2] *= correction;
                o_acc[g][c][3] *= correction;
            }

            float ev[4], local_sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                ev[i] = (m_new == -INFINITY) ? 0.0f : __expf(s_acc[i] - m_new);
                local_sum += ev[i];
            }
            local_sum += __shfl_xor(local_sum, 16, MFMA_WARP);
            local_sum += __shfl_xor(local_sum, 32, MFMA_WARP);

            l_val[g] = l_val[g] * correction + local_sum;
            m_val[g] = m_new;

            MFMA_BF16::vtype_a p_reg;
            p_reg[0] = fp32_to_bf16(ev[0]);
            p_reg[1] = fp32_to_bf16(ev[1]);
            p_reg[2] = fp32_to_bf16(ev[2]);
            p_reg[3] = fp32_to_bf16(ev[3]);

            #pragma unroll
            for (int c = 0; c < MFMA_PV_CHUNKS; c++) {
                MFMA_BF16::vtype_b b_pv;
                int vt_off = (c * MFMA_N + mfma_row) * SKV_STRIDE + mfma_col_base;
                b_pv[0] = sKV_T[vt_off + 0];
                b_pv[1] = sKV_T[vt_off + 1];
                b_pv[2] = sKV_T[vt_off + 2];
                b_pv[3] = sKV_T[vt_off + 3];

                o_acc[g][c] = mma(b_pv, p_reg, o_acc[g][c]);
            }
        }

        __syncthreads();
    }

    // ── Write output for each head group ──
    const int64_t o_bsq = (static_cast<int64_t>(batch_idx) * s_q + sq_idx)
                           * num_heads * MFMA_HD;

    #pragma unroll
    for (int g = 0; g < HEADS_PER_BLOCK; g++) {
        int g_head_start = head_start + g * MFMA_M;
        if (g_head_start >= num_heads) break;
        int g_valid = min(MFMA_M, num_heads - g_head_start);

        float inv_l = (l_val[g] > 0.0f) ? (1.0f / l_val[g]) : 0.0f;
        bf16_t* o_ptr = O + o_bsq + static_cast<int64_t>(g_head_start) * MFMA_HD;

        #pragma unroll
        for (int c = 0; c < MFMA_PV_CHUNKS; c++) {
            int head_local = mfma_row;
            int dim = c * MFMA_N + mfma_col_base;

            if (head_local < g_valid) {
                bf16_t* out = o_ptr + head_local * MFMA_HD + dim;
                out[0] = fp32_to_bf16(o_acc[g][c][0] * inv_l);
                out[1] = fp32_to_bf16(o_acc[g][c][1] * inv_l);
                out[2] = fp32_to_bf16(o_acc[g][c][2] * inv_l);
                out[3] = fp32_to_bf16(o_acc[g][c][3] * inv_l);
            }
        }

        if (mfma_col_base == 0 && mfma_row < g_valid) {
            float lse = (l_val[g] > 0.0f) ? (m_val[g] + __logf(l_val[g])) : -INFINITY;
            int64_t lse_idx = (static_cast<int64_t>(batch_idx) * s_q + sq_idx)
                              * num_heads + g_head_start + mfma_row;
            lse_out[lse_idx] = lse;
        }
    }
}


/***************************************************************************************************
 * flash_mla_sparse_decode_splitk_kernel — Split-K variant for parallelism
 *
 * Splits topk range across num_splits groups for higher GPU utilization.
 * Writes unnormalized partial results for the combine kernel to merge.
 *
 * Grid:  (ceil(h_q/16) * num_splits, b * s_q)
 * Block: 256 threads (4 wavefronts)
 **************************************************************************************************/
constexpr int SPLITK_BLOCK = 256;
constexpr int NUM_WAVES    = SPLITK_BLOCK / MFMA_WARP;  // 4

__global__ __launch_bounds__(SPLITK_BLOCK)
void flash_mla_sparse_decode_splitk_kernel(
    const bf16_t*  __restrict__ Q,
    const uint8_t* __restrict__ KV_packed,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ topk_length_ptr,
    float*         __restrict__ O_partial,
    float*         __restrict__ lse_partial,
    int num_heads,
    int topk,
    int s_q,
    float sm_scale,
    int num_splits)
{
    const int head_groups  = (num_heads + MFMA_M - 1) / MFMA_M;
    const int linear_x     = blockIdx.x;
    const int split_id     = linear_x / head_groups;
    const int head_block   = linear_x % head_groups;
    const int head_start   = head_block * MFMA_M;

    const int batch_sq_idx = blockIdx.y;
    const int batch_idx    = batch_sq_idx / s_q;
    const int sq_idx       = batch_sq_idx % s_q;
    const int tid          = threadIdx.x;
    const int lid          = tid & 63;
    const int wave_id      = tid >> 6;

    if (head_start >= num_heads) return;

    const int valid_heads  = min(MFMA_M, num_heads - head_start);
    const int mfma_row     = lid & 15;
    const int mfma_col_base= (lid >> 4) << 2;

    __shared__ bf16_t sQ   [MFMA_M  * SQ_STRIDE];
    __shared__ bf16_t sKV_T[MFMA_HD * SKV_STRIDE];
    __shared__ const uint8_t* sTokPtrs[MFMA_KV_TILE];

    const int64_t q_bsq = (static_cast<int64_t>(batch_idx) * s_q + sq_idx)
                           * num_heads * MFMA_HD;
    const bf16_t* q_ptr = Q + q_bsq + static_cast<int64_t>(head_start) * MFMA_HD;

    for (int i = tid; i < MFMA_M * MFMA_HD; i += SPLITK_BLOCK) {
        int h = i / MFMA_HD;
        int d = i % MFMA_HD;
        sQ[h * SQ_STRIDE + d] = (h < valid_heads) ? q_ptr[i] : fp32_to_bf16(0.0f);
    }
    __syncthreads();

    int eff_topk = topk;
    if (topk_length_ptr != nullptr) {
        eff_topk = min(topk, static_cast<int>(topk_length_ptr[batch_idx]));
        if (eff_topk < 0) eff_topk = 0;
    }

    int tokens_per_split = (eff_topk + num_splits - 1) / num_splits;
    int split_start = split_id * tokens_per_split;
    int split_end   = min(split_start + tokens_per_split, eff_topk);
    if (split_start >= eff_topk) {
        split_start = eff_topk;
        split_end   = eff_topk;
    }

    const int32_t* idx_base = indices
        + (static_cast<int64_t>(batch_idx) * s_q + sq_idx) * topk;

    MFMA_BF16 mma;

    MFMA_BF16::vtype_c o_acc[MFMA_PV_CHUNKS];
    #pragma unroll
    for (int c = 0; c < MFMA_PV_CHUNKS; c++)
        opus::clear(o_acc[c]);

    float m_val = -INFINITY;
    float l_val = 0.0f;

    for (int kv_tile = split_start; kv_tile < split_end; kv_tile += MFMA_KV_TILE) {
        const int tile_count = min(MFMA_KV_TILE, split_end - kv_tile);

        if (tid < MFMA_KV_TILE) {
            const uint8_t* ptr = nullptr;
            if (tid < tile_count) {
                int32_t flat_idx = idx_base[kv_tile + tid];
                if (flat_idx >= 0) {
                    ptr = KV_packed + static_cast<int64_t>(flat_idx) * SPARSE_BYTES_PER_TOKEN;
                }
            }
            sTokPtrs[tid] = ptr;
        }
        __syncthreads();

        for (int idx = tid; idx < MFMA_KV_TILE * I4_NOPE; idx += SPLITK_BLOCK) {
            int token_in_tile = idx / I4_NOPE;
            int i4            = idx % I4_NOPE;
            int dim           = i4 * 4;
            int scale_idx     = dim / TILE_SIZE;

            int32_t packed    = 0;
            float tile_scale  = 0.0f;

            const uint8_t* tok = sTokPtrs[token_in_tile];
            if (tok != nullptr) {
                packed = reinterpret_cast<const int32_t*>(tok + SPARSE_NOPE_OFFSET)[i4];
                tile_scale = e8m0_to_float(tok[SPARSE_SCALE_OFFSET + scale_idx]);
            }

            auto f01 = __builtin_amdgcn_cvt_pk_f32_fp8(packed, 0);
            auto f23 = __builtin_amdgcn_cvt_pk_f32_fp8(packed, 1);

            sKV_T[(dim + 0) * SKV_STRIDE + token_in_tile] = fp32_to_bf16(f01[0] * tile_scale);
            sKV_T[(dim + 1) * SKV_STRIDE + token_in_tile] = fp32_to_bf16(f01[1] * tile_scale);
            sKV_T[(dim + 2) * SKV_STRIDE + token_in_tile] = fp32_to_bf16(f23[0] * tile_scale);
            sKV_T[(dim + 3) * SKV_STRIDE + token_in_tile] = fp32_to_bf16(f23[1] * tile_scale);
        }

        for (int idx = tid; idx < MFMA_KV_TILE * D_ROPE; idx += SPLITK_BLOCK) {
            int token_in_tile = idx / D_ROPE;
            int rope_d        = idx % D_ROPE;
            int dim           = D_NOPE + rope_d;

            bf16_t val = fp32_to_bf16(0.0f);
            const uint8_t* tok = sTokPtrs[token_in_tile];
            if (tok != nullptr) {
                val = reinterpret_cast<const bf16_t*>(tok + SPARSE_ROPE_OFFSET)[rope_d];
            }
            sKV_T[dim * SKV_STRIDE + token_in_tile] = val;
        }
        __syncthreads();

        MFMA_BF16::vtype_c s_acc;
        opus::clear(s_acc);

        for (int k = 0; k < MFMA_QK_ITERS; k++) {
            MFMA_BF16::vtype_a a_val;
            int a_off = mfma_row * SQ_STRIDE + k * MFMA_K + mfma_col_base;
            a_val[0] = sQ[a_off + 0];
            a_val[1] = sQ[a_off + 1];
            a_val[2] = sQ[a_off + 2];
            a_val[3] = sQ[a_off + 3];

            MFMA_BF16::vtype_b b_val;
            int b_base = (k * MFMA_K + mfma_col_base) * SKV_STRIDE + mfma_row;
            b_val[0] = sKV_T[b_base + 0 * SKV_STRIDE];
            b_val[1] = sKV_T[b_base + 1 * SKV_STRIDE];
            b_val[2] = sKV_T[b_base + 2 * SKV_STRIDE];
            b_val[3] = sKV_T[b_base + 3 * SKV_STRIDE];

            s_acc = mma(b_val, a_val, s_acc);
        }

        s_acc[0] *= sm_scale;
        s_acc[1] *= sm_scale;
        s_acc[2] *= sm_scale;
        s_acc[3] *= sm_scale;

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int tok = mfma_col_base + i;
            if (tok >= tile_count || sTokPtrs[tok] == nullptr)
                s_acc[i] = -INFINITY;
        }

        float local_max = fmaxf(fmaxf(s_acc[0], s_acc[1]),
                                fmaxf(s_acc[2], s_acc[3]));
        local_max = fmaxf(local_max, __shfl_xor(local_max, 16, MFMA_WARP));
        local_max = fmaxf(local_max, __shfl_xor(local_max, 32, MFMA_WARP));

        float m_new = fmaxf(m_val, local_max);
        float correction = (m_val == -INFINITY) ? 0.0f : __expf(m_val - m_new);

        #pragma unroll
        for (int c = 0; c < MFMA_PV_CHUNKS; c++) {
            o_acc[c][0] *= correction;
            o_acc[c][1] *= correction;
            o_acc[c][2] *= correction;
            o_acc[c][3] *= correction;
        }

        float ev[4], local_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            ev[i] = (m_new == -INFINITY) ? 0.0f : __expf(s_acc[i] - m_new);
            local_sum += ev[i];
        }
        local_sum += __shfl_xor(local_sum, 16, MFMA_WARP);
        local_sum += __shfl_xor(local_sum, 32, MFMA_WARP);

        l_val = l_val * correction + local_sum;
        m_val = m_new;

        MFMA_BF16::vtype_a p_reg;
        p_reg[0] = fp32_to_bf16(ev[0]);
        p_reg[1] = fp32_to_bf16(ev[1]);
        p_reg[2] = fp32_to_bf16(ev[2]);
        p_reg[3] = fp32_to_bf16(ev[3]);

        #pragma unroll
        for (int c = 0; c < MFMA_PV_CHUNKS; c++) {
            MFMA_BF16::vtype_b b_pv;
            int vt_off = (c * MFMA_N + mfma_row) * SKV_STRIDE + mfma_col_base;
            b_pv[0] = sKV_T[vt_off + 0];
            b_pv[1] = sKV_T[vt_off + 1];
            b_pv[2] = sKV_T[vt_off + 2];
            b_pv[3] = sKV_T[vt_off + 3];

            o_acc[c] = mma(b_pv, p_reg, o_acc[c]);
        }

        __syncthreads();
    }

    if (wave_id == 0) {
        float inv_l = (l_val > 0.0f) ? (1.0f / l_val) : 0.0f;
        float lse = (l_val > 0.0f) ? (m_val + __logf(l_val)) : -INFINITY;

        const int64_t bsq_heads = static_cast<int64_t>(batch_sq_idx) * num_heads;
        float* o_part_base = O_partial
            + static_cast<int64_t>(split_id) * (gridDim.y * num_heads) * MFMA_HD
            + (bsq_heads + head_start) * MFMA_HD;

        #pragma unroll
        for (int c = 0; c < MFMA_PV_CHUNKS; c++) {
            int head_local = mfma_row;
            int dim = c * MFMA_N + mfma_col_base;

            if (head_local < valid_heads) {
                float* out = o_part_base + head_local * MFMA_HD + dim;
                out[0] = o_acc[c][0] * inv_l;
                out[1] = o_acc[c][1] * inv_l;
                out[2] = o_acc[c][2] * inv_l;
                out[3] = o_acc[c][3] * inv_l;
            }
        }

        if (mfma_col_base == 0 && mfma_row < valid_heads) {
            float* lse_part = lse_partial
                + static_cast<int64_t>(split_id) * (gridDim.y * num_heads)
                + bsq_heads + head_start + mfma_row;
            *lse_part = lse;
        }
    }
}


/***************************************************************************************************
 * flash_mla_sparse_combine_kernel — merge Split-K partial results
 **************************************************************************************************/
constexpr int COMBINE_ELEMS = MFMA_HD / MFMA_WARP;  // 8

__global__ __launch_bounds__(64)
void flash_mla_sparse_combine_kernel(
    const float*   __restrict__ O_partial,
    const float*   __restrict__ lse_partial,
    bf16_t*        __restrict__ O_final,
    float*         __restrict__ lse_final,
    int num_heads,
    int num_splits,
    int bsq_total)
{
    const int head_idx     = blockIdx.x;
    const int batch_sq_idx = blockIdx.y;
    const int lid          = threadIdx.x;

    if (head_idx >= num_heads) return;

    const int64_t head_stride_o   = static_cast<int64_t>(bsq_total) * num_heads * MFMA_HD;
    const int64_t head_stride_lse = static_cast<int64_t>(bsq_total) * num_heads;
    const int64_t bsq_head_o     = static_cast<int64_t>(batch_sq_idx) * num_heads * MFMA_HD
                                   + static_cast<int64_t>(head_idx) * MFMA_HD;
    const int64_t bsq_head_lse   = static_cast<int64_t>(batch_sq_idx) * num_heads + head_idx;

    float lse_max = -INFINITY;
    for (int s = 0; s < num_splits; s++) {
        float lse_s = lse_partial[static_cast<int64_t>(s) * head_stride_lse + bsq_head_lse];
        lse_max = fmaxf(lse_max, lse_s);
    }

    float o_acc[COMBINE_ELEMS];
    #pragma unroll
    for (int i = 0; i < COMBINE_ELEMS; i++) o_acc[i] = 0.0f;
    float l_total = 0.0f;

    int dim_base = lid * COMBINE_ELEMS;

    for (int s = 0; s < num_splits; s++) {
        float lse_s = lse_partial[static_cast<int64_t>(s) * head_stride_lse + bsq_head_lse];
        float w = __expf(lse_s - lse_max);
        l_total += w;

        const float* o_s = O_partial + static_cast<int64_t>(s) * head_stride_o
                           + bsq_head_o + dim_base;
        #pragma unroll
        for (int i = 0; i < COMBINE_ELEMS; i++) {
            o_acc[i] += w * o_s[i];
        }
    }

    float inv_l = (l_total > 0.0f) ? (1.0f / l_total) : 0.0f;

    bf16_t* out = O_final + bsq_head_o + dim_base;
    #pragma unroll
    for (int i = 0; i < COMBINE_ELEMS; i++) {
        out[i] = fp32_to_bf16(o_acc[i] * inv_l);
    }

    if (lid == 0) {
        float final_lse = (l_total > 0.0f) ? (lse_max + __logf(l_total)) : -INFINITY;
        lse_final[bsq_head_lse] = final_lse;
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
// Host launchers
// ═══════════════════════════════════════════════════════════════════════════════

void flash_mla_decode(
    const void* Q,
    const void* KV_nope,
    const void* KV_rope,
    const void* KV_scales,
    void* O,
    const void* kv_indptr,
    int batch_size,
    int num_heads,
    float sm_scale,
    void* stream)
{
    dim3 grid((num_heads + MFMA_M - 1) / MFMA_M, batch_size);
    dim3 block(MFMA_WARP);

    flash_mla_decode_kernel<<<grid, block, 0, static_cast<hipStream_t>(stream)>>>(
        reinterpret_cast<const bf16_t*>(Q),
        reinterpret_cast<const int32_t*>(KV_nope),
        reinterpret_cast<const bf16_t*>(KV_rope),
        reinterpret_cast<const float*>(KV_scales),
        reinterpret_cast<bf16_t*>(O),
        reinterpret_cast<const int*>(kv_indptr),
        num_heads,
        sm_scale);
}

void flash_mla_sparse_decode(
    const void* Q,
    const void* KV_packed,
    const void* indices,
    const void* topk_length,
    void* O,
    void* lse,
    int b,
    int s_q,
    int num_heads,
    int topk,
    float sm_scale,
    void* stream)
{
    dim3 block(SPARSE_BLOCK);
    int bsq = b * s_q;

    int hg2 = (num_heads + 2 * MFMA_M - 1) / (2 * MFMA_M);
    int blocks2 = hg2 * bsq;

    if (blocks2 >= MI308_NUM_CUS && num_heads >= 2 * MFMA_M) {
        dim3 grid(hg2, bsq);
        flash_mla_sparse_decode_kernel<2><<<grid, block, 0,
            static_cast<hipStream_t>(stream)>>>(
            reinterpret_cast<const bf16_t*>(Q),
            reinterpret_cast<const uint8_t*>(KV_packed),
            reinterpret_cast<const int32_t*>(indices),
            reinterpret_cast<const int32_t*>(topk_length),
            reinterpret_cast<bf16_t*>(O),
            reinterpret_cast<float*>(lse),
            num_heads, topk, s_q, sm_scale);
    } else {
        int hg1 = (num_heads + MFMA_M - 1) / MFMA_M;
        dim3 grid(hg1, bsq);
        flash_mla_sparse_decode_kernel<1><<<grid, block, 0,
            static_cast<hipStream_t>(stream)>>>(
            reinterpret_cast<const bf16_t*>(Q),
            reinterpret_cast<const uint8_t*>(KV_packed),
            reinterpret_cast<const int32_t*>(indices),
            reinterpret_cast<const int32_t*>(topk_length),
            reinterpret_cast<bf16_t*>(O),
            reinterpret_cast<float*>(lse),
            num_heads, topk, s_q, sm_scale);
    }
}

void flash_mla_sparse_decode_splitk(
    const void* Q,
    const void* KV_packed,
    const void* indices,
    const void* topk_length,
    void* O_partial,
    void* lse_partial,
    void* O_final,
    void* lse_final,
    int b,
    int s_q,
    int num_heads,
    int topk,
    int num_splits,
    float sm_scale,
    void* stream)
{
    auto hip_stream = static_cast<hipStream_t>(stream);
    int bsq = b * s_q;
    int head_groups = (num_heads + MFMA_M - 1) / MFMA_M;

    {
        dim3 grid(head_groups * num_splits, bsq);
        dim3 block(SPLITK_BLOCK);

        flash_mla_sparse_decode_splitk_kernel<<<grid, block, 0, hip_stream>>>(
            reinterpret_cast<const bf16_t*>(Q),
            reinterpret_cast<const uint8_t*>(KV_packed),
            reinterpret_cast<const int32_t*>(indices),
            reinterpret_cast<const int32_t*>(topk_length),
            reinterpret_cast<float*>(O_partial),
            reinterpret_cast<float*>(lse_partial),
            num_heads, topk, s_q, sm_scale, num_splits);
    }

    {
        dim3 grid(num_heads, bsq);
        dim3 block(MFMA_WARP);

        flash_mla_sparse_combine_kernel<<<grid, block, 0, hip_stream>>>(
            reinterpret_cast<const float*>(O_partial),
            reinterpret_cast<const float*>(lse_partial),
            reinterpret_cast<bf16_t*>(O_final),
            reinterpret_cast<float*>(lse_final),
            num_heads, num_splits, bsq);
    }
}
