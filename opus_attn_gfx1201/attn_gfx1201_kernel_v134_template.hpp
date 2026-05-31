// v134 -- STRUCTURAL change (director mandate B): process 2 KV-tiles per WG
// iteration with paired QK^T issued UP FRONT, to overlap tile-B's QK^T WMMAs
// (matrix pipe) with tile-A's softmax (VALU/exp2/cross-lane pipe).
//
// PROVENANCE / why this round: the plateau intervention note says the
// single-tile PV instruction-shuffle lever (v122..v133, deeper K rings, PV
// groupings, rolling V prefetch) is EXHAUSTED -- ~20 rounds tied or regressed,
// and the last few (v132 91.x, v133 91.2) all sit just under the 92.04 bar.
// The note mandates a STRUCTURAL change: "process 2 KV-tiles per WG iteration
// with a shared Q register file ... changes the loop STRUCTURE, not just
// instruction order -- the only remaining path to the 93-95 ceiling."
//
// THE BOTTLENECK this targets: in the v100/v122 single-tile loop, each KV-tile
// is QK^T (matrix) -> softmax (VALU + exp2 + 2 ds_bpermute cross-half folds, a
// long serial transcendental/shuffle chain) -> PV (matrix). The softmax sits on
// the critical path BETWEEN the two matrix phases with a hard RAW dependency, so
// the matrix pipe idles through it. The PV-only reshuffles never addressed this
// gap because they only move work that is already AFTER the softmax. With two
// KV-tiles unrolled and BOTH QK^Ts emitted before either softmax, the compiler
// is free to issue tile-B's QK^T WMMAs (which depend on nothing from tile A)
// during tile-A's softmax VALU window -> the matrix pipe stays fed across the
// softmax bubble. This is cross-pipe overlap at the LOOP-STRUCTURE granularity,
// distinct from every prior round's PV micro-scheduling.
//
// BIT-EXACTNESS (non-negotiable, max_abs<=0.005): the online-softmax state
// (m_row, l_row) and the O accumulators are updated in EXACTLY the v100 order:
// tile A's softmax (m_row/l_row update) precedes tile B's softmax, and tile A's
// PV (v_o update) precedes tile B's PV. QK^T is a pure function of K and Q with
// NO dependence on m_row/l_row/v_o, so hoisting tile B's QK^T ahead of tile A's
// softmax changes only the issue order of independent matrix ops -- it is
// arithmetically identical. Within each tile the reductions (row_max fmax chain,
// row_sum add chain, ds_bpermute(lane^16) folds, rescale fold, exp2) are
// byte-for-byte v100/v122. Independent O accumulators never share an fp32
// reduction. -> n_bad=0, max_abs=0.0001 expected (same as v100/v122).
//
// LOOP BOUND: num_kv_tiles = N / BLOCK_N. The harness gate guarantees N%64==0,
// and BLOCK_N=32, so num_kv_tiles = N/32 is always EVEN -> the pair loop never
// has a remainder. (A scalar tail would be dead code under the gate; omitted to
// keep one clean basic block. If N%64!=0 the host already rejects the run.)
//
// VGPR: holding two score-pair sets live (s0_A,s1_A,s0_B,s1_B = 4 fp32x8 = 32
// fp32/lane) until their respective softmaxes adds ~16 VGPR over v122's single
// pair (s_A consumed before s_B exists there). Estimated ~221 VGPR < 256 ceiling
// -> occupancy stays 7 waves/SIMD (no spill, no LDS, no barriers). PV uses the
// proven v122 single-chain per-accumulator order (lowest pressure, no PV-shuffle
// risk) -- the novelty is purely the KV-unroll-by-2 with paired QK^T.
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast134(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v134_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v134_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v134(opus_attn_kargs k)
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
            v_q[kt][j] = bf16_fast134(bf16_to_f32(v_q[kt][j]) * qscale);
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

    // ---- Process TWO KV-tiles per iteration. num_kv_tiles is even (N%64==0).
    for (int n_pair = 0; n_pair < num_kv_tiles; n_pair += 2) {
        const int n_base_A = (n_pair)     * BLOCK_N;
        const int n_base_B = (n_pair + 1) * BLOCK_N;

        // ===== QK^T for BOTH tiles, issued UP FRONT (independent matrix work).
        // Tile B's WMMAs have no dependence on tile A's softmax/PV, so the
        // compiler can slot them into tile A's softmax VALU window.
        fp32x8_t v_s0_A = {0,0,0,0,0,0,0,0};
        fp32x8_t v_s1_A = {0,0,0,0,0,0,0,0};
        fp32x8_t v_s0_B = {0,0,0,0,0,0,0,0};
        fp32x8_t v_s1_B = {0,0,0,0,0,0,0,0};

        bf16x8_t kA0n = *reinterpret_cast<const bf16x8_t*>(Kp + (n_base_A + col16) * stride_n + row8);
        bf16x8_t kA1n = *reinterpret_cast<const bf16x8_t*>(Kp + (n_base_A + 16 + col16) * stride_n + row8);
        bf16x8_t kB0n = *reinterpret_cast<const bf16x8_t*>(Kp + (n_base_B + col16) * stride_n + row8);
        bf16x8_t kB1n = *reinterpret_cast<const bf16x8_t*>(Kp + (n_base_B + 16 + col16) * stride_n + row8);
        #pragma unroll
        for (int kt = 0; kt < DK; ++kt) {
            bf16x8_t kA0 = kA0n, kA1 = kA1n, kB0 = kB0n, kB1 = kB1n;
            if (kt + 1 < DK) {
                kA0n = *reinterpret_cast<const bf16x8_t*>(Kp + (n_base_A + col16) * stride_n + (kt+1) * W_K + row8);
                kA1n = *reinterpret_cast<const bf16x8_t*>(Kp + (n_base_A + 16 + col16) * stride_n + (kt+1) * W_K + row8);
                kB0n = *reinterpret_cast<const bf16x8_t*>(Kp + (n_base_B + col16) * stride_n + (kt+1) * W_K + row8);
                kB1n = *reinterpret_cast<const bf16x8_t*>(Kp + (n_base_B + 16 + col16) * stride_n + (kt+1) * W_K + row8);
            }
            v_s0_A = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(kA0, v_q[kt], v_s0_A);
            v_s1_A = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(kA1, v_q[kt], v_s1_A);
            v_s0_B = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(kB0, v_q[kt], v_s0_B);
            v_s1_B = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(kB1, v_q[kt], v_s1_B);
        }

        // ===== TILE A: softmax (v100 order) -> PV (v122 single-chain order).
        {
            fp32_t row_max = v_s0_A[0];
            #pragma unroll
            for (int j = 1; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s0_A[j]);
            #pragma unroll
            for (int j = 0; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s1_A[j]);
            row_max = v134_cross_half_max(row_max);

            const fp32_t new_m = __builtin_fmaxf(m_row, row_max);
            fp32_t rescale = 1.0f;
            const bool need_rescale = (new_m != m_row);
            if (need_rescale) {
                rescale = __builtin_amdgcn_exp2f(m_row - new_m);
                l_row *= rescale;
            }
            m_row = new_m;

            #pragma unroll
            for (int j = 0; j < 8; ++j) v_s0_A[j] = __builtin_amdgcn_exp2f(v_s0_A[j] - new_m);
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_s1_A[j] = __builtin_amdgcn_exp2f(v_s1_A[j] - new_m);

            fp32_t row_sum = v_s0_A[0];
            #pragma unroll
            for (int j = 1; j < 8; ++j) row_sum += v_s0_A[j];
            #pragma unroll
            for (int j = 0; j < 8; ++j) row_sum += v_s1_A[j];
            row_sum = v134_cross_half_sum(row_sum);
            l_row += row_sum;

            bf16x8_t v_p0, v_p1;
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_p0[j] = bf16_fast134(v_s0_A[j]);
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_p1[j] = bf16_fast134(v_s1_A[j]);

            if (need_rescale) {
                #pragma unroll
                for (int dt = 0; dt < DK; ++dt) {
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) v_o[dt][j] *= rescale;
                }
            }
            #pragma unroll
            for (int dt = 0; dt < DK; ++dt) {
                const bf16_t* vt_addr0 = VTp + (dt * W_K + col16) * vt_stride_d + n_base_A + row8;
                const bf16_t* vt_addr1 = VTp + (dt * W_K + col16) * vt_stride_d + n_base_A + 16 + row8;
                bf16x8_t v_v0 = *reinterpret_cast<const bf16x8_t*>(vt_addr0);
                bf16x8_t v_v1 = *reinterpret_cast<const bf16x8_t*>(vt_addr1);
                v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_v0, v_p0, v_o[dt]);
                v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_v1, v_p1, v_o[dt]);
            }
        }

        // ===== TILE B: softmax (v100 order, post-A state) -> PV. Bit-exact: B's
        // m_row/l_row/v_o updates strictly follow A's, exactly as in v100.
        {
            fp32_t row_max = v_s0_B[0];
            #pragma unroll
            for (int j = 1; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s0_B[j]);
            #pragma unroll
            for (int j = 0; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s1_B[j]);
            row_max = v134_cross_half_max(row_max);

            const fp32_t new_m = __builtin_fmaxf(m_row, row_max);
            fp32_t rescale = 1.0f;
            const bool need_rescale = (new_m != m_row);
            if (need_rescale) {
                rescale = __builtin_amdgcn_exp2f(m_row - new_m);
                l_row *= rescale;
            }
            m_row = new_m;

            #pragma unroll
            for (int j = 0; j < 8; ++j) v_s0_B[j] = __builtin_amdgcn_exp2f(v_s0_B[j] - new_m);
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_s1_B[j] = __builtin_amdgcn_exp2f(v_s1_B[j] - new_m);

            fp32_t row_sum = v_s0_B[0];
            #pragma unroll
            for (int j = 1; j < 8; ++j) row_sum += v_s0_B[j];
            #pragma unroll
            for (int j = 0; j < 8; ++j) row_sum += v_s1_B[j];
            row_sum = v134_cross_half_sum(row_sum);
            l_row += row_sum;

            bf16x8_t v_p0, v_p1;
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_p0[j] = bf16_fast134(v_s0_B[j]);
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_p1[j] = bf16_fast134(v_s1_B[j]);

            if (need_rescale) {
                #pragma unroll
                for (int dt = 0; dt < DK; ++dt) {
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) v_o[dt][j] *= rescale;
                }
            }
            #pragma unroll
            for (int dt = 0; dt < DK; ++dt) {
                const bf16_t* vt_addr0 = VTp + (dt * W_K + col16) * vt_stride_d + n_base_B + row8;
                const bf16_t* vt_addr1 = VTp + (dt * W_K + col16) * vt_stride_d + n_base_B + 16 + row8;
                bf16x8_t v_v0 = *reinterpret_cast<const bf16x8_t*>(vt_addr0);
                bf16x8_t v_v1 = *reinterpret_cast<const bf16x8_t*>(vt_addr1);
                v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_v0, v_p0, v_o[dt]);
                v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_v1, v_p1, v_o[dt]);
            }
        }
    }

    const fp32_t inv = (l_row > 0.0f) ? (1.0f / l_row) : 0.0f;
    bf16_t* o_row = Op + (q_m_base + col16) * stride_n;
    #pragma unroll
    for (int dt = 0; dt < DK; ++dt) {
        bf16x8_t o_pack;
        #pragma unroll
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast134(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
