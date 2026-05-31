// v116 -- v106 (MEASURED champion, 90.20 TFLOPS) + CROSS-KV-TILE K PREFETCH (n_pipe=2 at
//          KV-block granularity, bounded to the first D-tile pair so occupancy is preserved).
//
// Base choice: v106 (running best, measured 90.20 @ b4h32n4096). v106 = v103's two-ahead
// intra-tile QK^T K-prefetch ring + 2-D-tile grouped PV (4 live V fragments, two
// independent accumulator chains per pair) + one-pair-ahead software-pipelined rescale.
// PV restructuring is SATURATED on this box (v104/v105 4-chain regressed, v107/v108
// rescale-placement regressed, v109/v110 V-prefetch rings dipped, v102 all-8 rescale
// starved the pipe) and pure softmax reorders (v112/v113/v114/v115) only tie. So v116
// leaves the entire QK^T D-tile ring + PV WMMA schedule + softmax of v106 BYTE-FOR-BYTE
// untouched, and attacks the ONE structural lever never applied to the v100+ family:
// the EXPOSED COLD K LOAD AT EACH KV-TILE HEAD.
//
// THE BOTTLENECK. v106's intra-tile K ring hides the kt>=2 D-tile loads of QK^T under the
// kt-1 WMMAs, but the FIRST D-tile pair (kt=0) of every KV tile is a cold global load
// issued at the top of the QK^T loop, and the very first two QK WMMAs of the tile stall on
// its full VMEM latency. Across N/BLOCK_N = 64 (N=2048) .. 128 (N=4096) KV tiles that cold
// head-of-tile latency is paid once per tile, on the critical path, with no matrix work to
// cover it (PV of the previous tile has already retired by then).
//
// THE CHANGE (the ONLY change vs v106). Software-pipeline the KV loop by ONE tile, but
// carry ONLY the next tile's kt=0 K pair (two bf16x8 fragments = 8 VGPR). The next tile's
// kt=0 loads are ISSUED right before this tile's PV WMMA block, so their VMEM latency hides
// in the shadow of the PV matrix pipe (playbook lever 4 "n_pipe=2 at the KV-block level",
// proven in v63; playbook lever 1 "separate memory-issuing from VALU/matrix work"). At the
// next tile's QK^T head the kt=0 fragments are already resident -> the first two QK WMMAs
// issue with NO exposed cold-load latency. kt=1 is still cold-loaded at the head as in v106
// (it overlaps the kt=0 WMMA), and kt>=2 keep v106's two-ahead ring verbatim.
//
// Why only the kt=0 pair (not the whole tile): carrying all 8 D-tiles would add 64 VGPR and
// collapse occupancy (the v104 cliff). Carrying 4 fragments (kt=0,1) adds 16 VGPR -> 213 ->
// drops 7->6 waves. Carrying the kt=0 pair adds exactly 8 VGPR: 197 -> 205, still <=208, so
// 7 waves/SIMD are preserved (the same footprint as v100/v101). This buys the head-of-tile
// latency hide WITHOUT the occupancy regression that killed every "carry more state" attempt.
//
// BIT-EXACTNESS. The kt=0 K fragments loaded for tile n are read from the SAME global
// addresses whether cold-loaded at the head (v106) or prefetched at the end of tile n-1
// (v116) -> byte-identical K operands. Every other load, the QK^T D-tile ring, the WMMA
// operand order, the entire softmax (row_max tree position, exp2 args, serial row_sum, RNE
// pack) and the entire PV schedule + rescale are v106 verbatim -> byte-identical row_max /
// new_m / rescale / probs / l_row / O. The boundary prefetch is guarded n_tile+1 <
// num_kv_tiles (no OOB; the last tile leaves the carry stale and unread).
//
// RISK: low-med. If the compiler spills past 205 (e.g. fails to free a transient) occupancy
// could drop to 6 waves and regress; mitigated by carrying the minimum 8 VGPR. If the HW
// scoreboard already covered the head-of-tile load via the previous tile's tail, v116 ties
// v106. No correctness risk (pure address-equivalent reorder of one load).
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast116(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v116_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v116_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v116(opus_attn_kargs k)
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
            v_q[kt][j] = bf16_fast116(bf16_to_f32(v_q[kt][j]) * qscale);
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

    // ---- Cross-tile K prefetch carry: the kt=0 K pair of the CURRENT tile. Primed here
    // for tile 0 (cold, unavoidable on the first tile), then refilled at the end of every
    // tile with the NEXT tile's kt=0 pair so its VMEM latency hides under PV WMMAs.
    bf16x8_t nk0, nk1;
    {
        const bf16_t* kb0 = Kp + (0 * BLOCK_N + col16) * stride_n + row8;
        const bf16_t* kb1 = Kp + (0 * BLOCK_N + 16 + col16) * stride_n + row8;
        nk0 = *reinterpret_cast<const bf16x8_t*>(kb0);
        nk1 = *reinterpret_cast<const bf16x8_t*>(kb1);
    }

    for (int n_tile = 0; n_tile < num_kv_tiles; ++n_tile) {
        const int n_base = n_tile * BLOCK_N;

        // ---- QK^T: BLOCK_N=32 -> two 16x16x16 WMMAs per D-tile.
        // TWO-AHEAD software-pipelined K loads (v101/v103/v106). The kt=0 pair is supplied
        // from the cross-tile carry (already resident -> first two QK WMMAs do not stall on
        // a cold load); kt=1 is cold-loaded here (overlaps the kt=0 WMMA); kt>=2 use the
        // v106 two-ahead ring verbatim.
        fp32x8_t v_s0 = {0,0,0,0,0,0,0,0};
        fp32x8_t v_s1 = {0,0,0,0,0,0,0,0};

        const bf16_t* kbase0 = Kp + (n_base + col16) * stride_n + row8;
        const bf16_t* kbase1 = Kp + (n_base + 16 + col16) * stride_n + row8;
        bf16x8_t kr0[2];
        bf16x8_t kr1[2];
        kr0[0] = nk0;   // carried kt=0 pair (== *(kbase0), same address)
        kr1[0] = nk1;   // carried kt=0 pair (== *(kbase1), same address)
        if (DK > 1) {
            kr0[1] = *reinterpret_cast<const bf16x8_t*>(kbase0 + W_K);
            kr1[1] = *reinterpret_cast<const bf16x8_t*>(kbase1 + W_K);
        }
        #pragma unroll
        for (int kt = 0; kt < DK; ++kt) {
            const int cur = kt & 1;
            bf16x8_t v_k0 = kr0[cur];
            bf16x8_t v_k1 = kr1[cur];
            if (kt + 2 < DK) {
                kr0[cur] = *reinterpret_cast<const bf16x8_t*>(kbase0 + (kt+2) * W_K);
                kr1[cur] = *reinterpret_cast<const bf16x8_t*>(kbase1 + (kt+2) * W_K);
            }
            v_s0 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_k0, v_q[kt], v_s0);
            v_s1 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_k1, v_q[kt], v_s1);
        }

        // Softmax over 16 values (8 from s0, 8 from s1)
        fp32_t row_max = v_s0[0];
        #pragma unroll
        for (int j = 1; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s1[j]);
        row_max = v116_cross_half_max(row_max);

        const fp32_t new_m = __builtin_fmaxf(m_row, row_max);
        fp32_t rescale = 1.0f;
        const bool need_rescale = (new_m != m_row);
        if (need_rescale) {
            rescale = __builtin_amdgcn_exp2f(m_row - new_m);
            l_row *= rescale;
        }
        m_row = new_m;

        #pragma unroll
        for (int j = 0; j < 8; ++j) v_s0[j] = __builtin_amdgcn_exp2f(v_s0[j] - new_m);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_s1[j] = __builtin_amdgcn_exp2f(v_s1[j] - new_m);

        fp32_t row_sum = v_s0[0];
        #pragma unroll
        for (int j = 1; j < 8; ++j) row_sum += v_s0[j];
        #pragma unroll
        for (int j = 0; j < 8; ++j) row_sum += v_s1[j];
        row_sum = v116_cross_half_sum(row_sum);
        l_row += row_sum;

        bf16x8_t v_p0, v_p1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p0[j] = bf16_fast116(v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p1[j] = bf16_fast116(v_s1[j]);

        // ---- Cross-tile K prefetch: issue the NEXT tile's kt=0 K pair NOW, so its global
        // VMEM latency hides under the PV WMMA block below. At the next iteration's QK^T head
        // these fragments are already resident. Guarded against OOB on the last tile.
        if (n_tile + 1 < num_kv_tiles) {
            const int nn_base = n_base + BLOCK_N;
            const bf16_t* nkb0 = Kp + (nn_base + col16) * stride_n + row8;
            const bf16_t* nkb1 = Kp + (nn_base + 16 + col16) * stride_n + row8;
            nk0 = *reinterpret_cast<const bf16x8_t*>(nkb0);
            nk1 = *reinterpret_cast<const bf16x8_t*>(nkb1);
        }

        // ---- PV: v106's 2-D-tile grouped PV (4 live V fragments, two independent
        // accumulator chains per pair) with the online rescale SOFTWARE-PIPELINED BY ONE
        // PAIR. Byte-for-byte identical to v106. DK=8 even -> exact pairing, no remainder.
        if (need_rescale) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_o[0][j] *= rescale;
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_o[1][j] *= rescale;
        }
        #pragma unroll
        for (int dt = 0; dt < DK; dt += 2) {
            const int dt1 = dt + 1;
            const bf16_t* a0 = VTp + (dt  * W_K + col16) * vt_stride_d + n_base + row8;
            const bf16_t* a1 = VTp + (dt  * W_K + col16) * vt_stride_d + n_base + 16 + row8;
            const bf16_t* b0 = VTp + (dt1 * W_K + col16) * vt_stride_d + n_base + row8;
            const bf16_t* b1 = VTp + (dt1 * W_K + col16) * vt_stride_d + n_base + 16 + row8;
            bf16x8_t va0 = *reinterpret_cast<const bf16x8_t*>(a0);
            bf16x8_t va1 = *reinterpret_cast<const bf16x8_t*>(a1);
            bf16x8_t vb0 = *reinterpret_cast<const bf16x8_t*>(b0);
            bf16x8_t vb1 = *reinterpret_cast<const bf16x8_t*>(b1);
            // One-pair-ahead rescale: prepare the NEXT pair's accumulators while the
            // current pair's V loads are in flight. Independent of the WMMAs below.
            if (dt + 2 < DK && need_rescale) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) v_o[dt + 2][j] *= rescale;
                #pragma unroll
                for (int j = 0; j < 8; ++j) v_o[dt + 3][j] *= rescale;
            }
            v_o[dt]  = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(va0, v_p0, v_o[dt]);
            v_o[dt1] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(vb0, v_p0, v_o[dt1]);
            v_o[dt]  = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(va1, v_p1, v_o[dt]);
            v_o[dt1] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(vb1, v_p1, v_o[dt1]);
        }
    }

    const fp32_t inv = (l_row > 0.0f) ? (1.0f / l_row) : 0.0f;
    bf16_t* o_row = Op + (q_m_base + col16) * stride_n;
    #pragma unroll
    for (int dt = 0; dt < DK; ++dt) {
        bf16x8_t o_pack;
        #pragma unroll
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast116(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
