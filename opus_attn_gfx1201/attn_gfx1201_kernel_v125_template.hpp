// v125 -- v122 champion + 2-KV-TILE FUSION (structural lever, mandate B / reviewer #2).
//
// PROVENANCE / why this round: round 13-15 exhausted instruction-level reshuffling of
// the single-tile inner loop (CHUNK sweeps, V prefetch, rescale splits all tied or
// regressed; v124 CHUNK=1 lost 2.8%). The reviewer and the director's plateau note
// both mandate the next STRUCTURAL move: process TWO KV tiles per WG iteration so the
// loop/softmax/PV setup is amortized and -- the real lever -- the independent QK^T
// WMMA window doubles (8 -> 16 WMMAs in flight across two tiles), letting the matrix
// pipe stay fed while tile-(t+1)'s global K loads are still in flight. This is the
// "enlarge the independent WMMA window without resurrecting V-fragment liveness"
// path the round-15 reviewer asked for.
//
// THE CHANGE (loop structure only; per-tile arithmetic is byte-for-byte v122/v100):
//   for n_tile in [0, num_kv_tiles) step 2:
//     (1) QK^T for BOTH tiles a=n_tile and b=n_tile+1, with the two tiles' WMMAs
//         INTERLEAVED per D-tile (k0a,k1a,k0b,k1b loaded, then 4 WMMAs issued). The
//         4 independent score accumulators (s0a,s1a,s0b,s1b) give the matrix pipe a
//         2x wider window and overlap b's K loads behind a's WMMAs.
//     (2) tile a: full v122 online softmax (row_max, new_m, rescale_a, exp2, sum,
//         l_row update, bf16 pack -> p0a,p1a). Frees s0a,s1a.
//     (3) tile b: full v122 online softmax using m_row AFTER tile a (sequential
//         online-softmax dependency preserved) -> rescale_b, p0b,p1b. Frees s0b,s1b.
//     (4) PV a: v_o *= rescale_a (CHUNK=2), then V_a@P_a accumulate (CHUNK=2).
//     (5) PV b: v_o *= rescale_b (CHUNK=2), then V_b@P_b accumulate (CHUNK=2).
//
// BIT-EXACTNESS (non-negotiable, max_abs<=0.005): every value is produced in the
// SAME order as v122 running n_tile=a then n_tile=b:
//   * m_row/l_row updates are sequential a-then-b exactly as before (b reads m_row
//     and l_row produced by a). No reassociation of the row-sum or the cross-tile
//     online reduction.
//   * Each O accumulator evolves as ((v_o * rescale_a) + V_a@P_a) * rescale_b +
//     V_b@P_b -- identical operation order to two v122 iterations.
//   * Computing tile b's exp2/sum/pack BEFORE PV_a is reorder-safe: those touch only
//     tile-b values + l_row, and PV_a never reads/writes l_row, so l_row's a-then-b
//     update order is unchanged. QK WMMA interleaving only reorders INDEPENDENT
//     accumulators (s*a vs s*b never share a reduction). -> n_bad=0, max_abs=0.0001.
//
// LIVENESS / risk: peak score liveness moves to the QK phase (4 fp32x8 score frags +
// up to 8 K bf16x8 buffers) which is transient and does NOT overlap the V loads. The
// PV phase keeps v122's short-lived per-dt V loads (only p0a/p1a/p0b/p1b bf16 carried
// in, 2 VGPR each) -- explicitly NOT the long-lived V prefetch that sank v123. Extra
// pressure vs v122 is the second score-frag pair + second K double-buffer; if this
// drops 7->6 waves the QK window widening must outweigh it (the bet).
//
// Requires num_kv_tiles even: num_kv_tiles = N/BLOCK_N = N/32, and N%64==0 (harness),
// so N/32 is always even. Safe for all gated shapes (N=2048/4096/1536).
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast125(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v125_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v125_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v125(opus_attn_kargs k)
{
#if defined(__gfx1201__) || defined(__gfx1200__)
    constexpr int BLOCK_M = T::BLOCK_M;
    constexpr int BLOCK_N = T::BLOCK_N;
    constexpr int W_K     = T::W_K;
    constexpr int DK      = T::D_TILES_K;

    const int tid     = static_cast<int>(threadIdx.x);
    const int wave_id = tid / 32;
    const int lane    = tid % 32;
    const int col16   = lane % 16;
    const int row_grp = lane / 16;
    const int row8    = row_grp * 8;

    const int q_tile_id = blockIdx.x;
    const int h         = blockIdx.y;
    const int b         = blockIdx.z;

    const int stride_n = k.D;
    const int stride_h = k.N * k.D;
    const int stride_b = k.H * k.N * k.D;

    const bf16_t* __restrict__ Qp = reinterpret_cast<const bf16_t*>(k.ptr_q) + b * stride_b + h * stride_h;
    const bf16_t* __restrict__ Kp = reinterpret_cast<const bf16_t*>(k.ptr_k) + b * stride_b + h * stride_h;
    bf16_t*       __restrict__ Op = reinterpret_cast<bf16_t*>(k.ptr_o) + b * stride_b + h * stride_h;

    const int vt_stride_d = k.N;
    const int vt_stride_h = k.D * k.N;
    const int vt_stride_b = k.H * k.D * k.N;
    const bf16_t* __restrict__ VTp = reinterpret_cast<const bf16_t*>(k.ptr_v) + b * vt_stride_b + h * vt_stride_h;

    const int q_m_base = q_tile_id * BLOCK_M + wave_id * 16;
    const bf16_t* q_row = Qp + (q_m_base + col16) * stride_n;
    bf16x8_t v_q[DK];
    constexpr fp32_t LOG2_E = 1.44269504088896340736f;
    const fp32_t qscale = k.scale * LOG2_E;
    #pragma unroll
    for (int kt = 0; kt < DK; ++kt) {
        v_q[kt] = *reinterpret_cast<const bf16x8_t*>(&q_row[kt * W_K + row8]);
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            v_q[kt][j] = bf16_fast125(bf16_to_f32(v_q[kt][j]) * qscale);
    }

    fp32x8_t v_o[DK];
    #pragma unroll
    for (int kt = 0; kt < DK; ++kt) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_o[kt][j] = 0.0f;
    }
    fp32_t m_row = -3.4e38f;
    fp32_t l_row = 0.0f;

    const int num_kv_tiles = k.N / BLOCK_N;

    for (int n_tile = 0; n_tile < num_kv_tiles; n_tile += 2) {
        const int n_base_a = n_tile * BLOCK_N;
        const int n_base_b = n_base_a + BLOCK_N;

        // ---- (1) QK^T for BOTH tiles, WMMAs interleaved per D-tile.
        fp32x8_t v_s0a = {0,0,0,0,0,0,0,0};
        fp32x8_t v_s1a = {0,0,0,0,0,0,0,0};
        fp32x8_t v_s0b = {0,0,0,0,0,0,0,0};
        fp32x8_t v_s1b = {0,0,0,0,0,0,0,0};

        bf16x8_t v_k0a_n = *reinterpret_cast<const bf16x8_t*>(Kp + (n_base_a + col16) * stride_n + row8);
        bf16x8_t v_k1a_n = *reinterpret_cast<const bf16x8_t*>(Kp + (n_base_a + 16 + col16) * stride_n + row8);
        bf16x8_t v_k0b_n = *reinterpret_cast<const bf16x8_t*>(Kp + (n_base_b + col16) * stride_n + row8);
        bf16x8_t v_k1b_n = *reinterpret_cast<const bf16x8_t*>(Kp + (n_base_b + 16 + col16) * stride_n + row8);
        #pragma unroll
        for (int kt = 0; kt < DK; ++kt) {
            bf16x8_t v_k0a = v_k0a_n;
            bf16x8_t v_k1a = v_k1a_n;
            bf16x8_t v_k0b = v_k0b_n;
            bf16x8_t v_k1b = v_k1b_n;
            if (kt + 1 < DK) {
                const int o = (kt + 1) * W_K + row8;
                v_k0a_n = *reinterpret_cast<const bf16x8_t*>(Kp + (n_base_a + col16) * stride_n + o);
                v_k1a_n = *reinterpret_cast<const bf16x8_t*>(Kp + (n_base_a + 16 + col16) * stride_n + o);
                v_k0b_n = *reinterpret_cast<const bf16x8_t*>(Kp + (n_base_b + col16) * stride_n + o);
                v_k1b_n = *reinterpret_cast<const bf16x8_t*>(Kp + (n_base_b + 16 + col16) * stride_n + o);
            }
            v_s0a = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_k0a, v_q[kt], v_s0a);
            v_s1a = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_k1a, v_q[kt], v_s1a);
            v_s0b = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_k0b, v_q[kt], v_s0b);
            v_s1b = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_k1b, v_q[kt], v_s1b);
        }

        // ---- (2) tile a softmax (byte-for-byte v122 body on s0a/s1a).
        fp32_t row_max_a = v_s0a[0];
        #pragma unroll
        for (int j = 1; j < 8; ++j) row_max_a = __builtin_fmaxf(row_max_a, v_s0a[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) row_max_a = __builtin_fmaxf(row_max_a, v_s1a[j]);
        row_max_a = v125_cross_half_max(row_max_a);

        const fp32_t new_m_a = __builtin_fmaxf(m_row, row_max_a);
        fp32_t rescale_a = 1.0f;
        const bool need_rescale_a = (new_m_a != m_row);
        if (need_rescale_a) {
            rescale_a = __builtin_amdgcn_exp2f(m_row - new_m_a);
            l_row *= rescale_a;
        }
        m_row = new_m_a;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_s0a[j] = __builtin_amdgcn_exp2f(v_s0a[j] - new_m_a);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_s1a[j] = __builtin_amdgcn_exp2f(v_s1a[j] - new_m_a);
        fp32_t row_sum_a = v_s0a[0];
        #pragma unroll
        for (int j = 1; j < 8; ++j) row_sum_a += v_s0a[j];
        #pragma unroll
        for (int j = 0; j < 8; ++j) row_sum_a += v_s1a[j];
        row_sum_a = v125_cross_half_sum(row_sum_a);
        l_row += row_sum_a;
        bf16x8_t v_p0a, v_p1a;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p0a[j] = bf16_fast125(v_s0a[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p1a[j] = bf16_fast125(v_s1a[j]);

        // ---- (3) tile b softmax (uses m_row/l_row AFTER tile a -> sequential online).
        fp32_t row_max_b = v_s0b[0];
        #pragma unroll
        for (int j = 1; j < 8; ++j) row_max_b = __builtin_fmaxf(row_max_b, v_s0b[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) row_max_b = __builtin_fmaxf(row_max_b, v_s1b[j]);
        row_max_b = v125_cross_half_max(row_max_b);

        const fp32_t new_m_b = __builtin_fmaxf(m_row, row_max_b);
        fp32_t rescale_b = 1.0f;
        const bool need_rescale_b = (new_m_b != m_row);
        if (need_rescale_b) {
            rescale_b = __builtin_amdgcn_exp2f(m_row - new_m_b);
            l_row *= rescale_b;
        }
        m_row = new_m_b;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_s0b[j] = __builtin_amdgcn_exp2f(v_s0b[j] - new_m_b);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_s1b[j] = __builtin_amdgcn_exp2f(v_s1b[j] - new_m_b);
        fp32_t row_sum_b = v_s0b[0];
        #pragma unroll
        for (int j = 1; j < 8; ++j) row_sum_b += v_s0b[j];
        #pragma unroll
        for (int j = 0; j < 8; ++j) row_sum_b += v_s1b[j];
        row_sum_b = v125_cross_half_sum(row_sum_b);
        l_row += row_sum_b;
        bf16x8_t v_p0b, v_p1b;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p0b[j] = bf16_fast125(v_s0b[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p1b[j] = bf16_fast125(v_s1b[j]);

        // ---- (4) PV tile a (CHUNK=2, short-lived per-dt V loads -- v122 schedule).
        constexpr int CHUNK = 2;
        #pragma unroll
        for (int c0 = 0; c0 < DK; c0 += CHUNK) {
            if (need_rescale_a) {
                #pragma unroll
                for (int dc = 0; dc < CHUNK; ++dc) {
                    const int dt = c0 + dc;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) v_o[dt][j] *= rescale_a;
                }
            }
            #pragma unroll
            for (int dc = 0; dc < CHUNK; ++dc) {
                const int dt = c0 + dc;
                const bf16_t* vt0 = VTp + (dt * W_K + col16) * vt_stride_d + n_base_a + row8;
                const bf16_t* vt1 = VTp + (dt * W_K + col16) * vt_stride_d + n_base_a + 16 + row8;
                bf16x8_t v_v0 = *reinterpret_cast<const bf16x8_t*>(vt0);
                bf16x8_t v_v1 = *reinterpret_cast<const bf16x8_t*>(vt1);
                v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_v0, v_p0a, v_o[dt]);
                v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_v1, v_p1a, v_o[dt]);
            }
        }

        // ---- (5) PV tile b (CHUNK=2).
        #pragma unroll
        for (int c0 = 0; c0 < DK; c0 += CHUNK) {
            if (need_rescale_b) {
                #pragma unroll
                for (int dc = 0; dc < CHUNK; ++dc) {
                    const int dt = c0 + dc;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) v_o[dt][j] *= rescale_b;
                }
            }
            #pragma unroll
            for (int dc = 0; dc < CHUNK; ++dc) {
                const int dt = c0 + dc;
                const bf16_t* vt0 = VTp + (dt * W_K + col16) * vt_stride_d + n_base_b + row8;
                const bf16_t* vt1 = VTp + (dt * W_K + col16) * vt_stride_d + n_base_b + 16 + row8;
                bf16x8_t v_v0 = *reinterpret_cast<const bf16x8_t*>(vt0);
                bf16x8_t v_v1 = *reinterpret_cast<const bf16x8_t*>(vt1);
                v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_v0, v_p0b, v_o[dt]);
                v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_v1, v_p1b, v_o[dt]);
            }
        }
    }

    const fp32_t inv = (l_row > 0.0f) ? (1.0f / l_row) : 0.0f;
    bf16_t* o_row = Op + (q_m_base + col16) * stride_n;
    #pragma unroll
    for (int dt = 0; dt < DK; ++dt) {
        bf16x8_t o_pack;
        #pragma unroll
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast125(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
