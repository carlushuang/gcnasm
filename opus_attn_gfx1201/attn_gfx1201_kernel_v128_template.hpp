// v128 -- v122 champion + V-HALF-SPLIT PV (widen the independent-WMMA window with
//          ZERO added fragment liveness). Bit-exact vs v122/v100.
//
// PROVENANCE / why this round (round 19): v122 = 91.68 TFLOPS is the champion. The
// director's plateau note says the single-tile K-prefetch lever is EXHAUSTED and
// ~20 rounds of intra-loop reshuffling have tied or regressed. The ledger is blunt:
//   * EVERY cross-boundary structural scheme fell off the ~205->256 VGPR / 7-wave
//     occupancy cliff -- v116=83.8, v117=67.3, v118=73.1, v119=75.0 (hoist QK(t+1),
//     carries score frags), v125=67.0 (2-KV fusion, 4 score frags), v126=82.0
//     (cross-tile K carry). Adding ANY fragment across the KV-tile boundary loses.
//   * EVERY pure permutation of v122's inner loop tied/regressed -- v123=77.9 (V
//     prefetch), v124=89.1, v127=90.2 (QK reorder). Instruction shuffling is spent.
// The round-18 reviewer's NEXT FOCUS: keep v122's QK^T verbatim (both next-K loads
// issued early) and test a NO-NEW-FRAGMENT placement change in PV.
//
// THE INSIGHT (a real scheduling lever, not a permutation). v122's PV walks D-tiles
// and per tile issues a 2-DEEP DEPENDENT WMMA CHAIN on the SAME accumulator:
//     v_o[dt] = wmma(v_v0, v_p0, v_o[dt]);   // writes v_o[dt]
//     v_o[dt] = wmma(v_v1, v_p1, v_o[dt]);   // STALLS on the previous write of v_o[dt]
// The second WMMA of each tile cannot issue until the first has retired (RAW on
// v_o[dt]). With CHUNK=2 the compiler interleaves two such chains, but the structure
// still pairs each accumulator's two WMMAs back-to-back -> the matrix pipe repeatedly
// eats the WMMA->WMMA accumulate latency.
//
// THE CHANGE (PV phase only; QK^T / softmax / exp2 / sum / pack / normalize are all
// byte-for-byte v122): split PV into two passes BY V-HALF across ALL 8 accumulators.
//   Pass A: for every dt -- rescale (CHUNK=2, exactly v122) then wmma(v_v0,v_p0,v_o[dt]).
//           The 8 v0 WMMAs write 8 DISTINCT accumulators -> fully independent, an
//           8-wide WMMA window with no intra-accumulator RAW stall.
//   Pass B: for every dt -- wmma(v_v1,v_p1,v_o[dt]). Each depends on Pass A's same-dt
//           result, but that WMMA retired ~7 issues earlier, so the accumulate latency
//           is fully hidden behind the other tiles' Pass-B WMMAs.
// Net: the same 16 WMMAs, but reordered so dependent pairs are maximally separated ->
// the matrix pipe stops paying the per-tile WMMA->WMMA accumulate stall.
//
// LIVENESS / VGPR (why this is NOT v103/v104/v105 and NOT v125). The failed pairing
// schemes (v103 "two independent chains per pair", v104/v105 "4-chain") kept FOUR V
// fragments (v0a,v1a,v0b,v1b) live at once, and v125 kept four SCORE fragments live --
// both blew past the occupancy cliff. v128 keeps EXACTLY ONE V fragment live at a time
// (v_v0 in Pass A is dead after its WMMA; v_v1 loaded fresh in Pass B). That is FEWER
// live V regs than v122's CHUNK=2 group (which holds v_v0 AND v_v1). So VGPR <= v122
// (~205, 7 waves/SIMD) -- no new buffers, no LDS, no barriers, no score doubling.
//
// BIT-EXACTNESS (non-negotiable, max_abs<=0.005). For each accumulator v_o[dt] the
// arithmetic is the IDENTICAL ordered sequence as v122/v100:
//     v_o[dt] *= rescale   (only when need_rescale, same rescale)
//     v_o[dt]  = wmma(v_v0, v_p0, v_o[dt])
//     v_o[dt]  = wmma(v_v1, v_p1, v_o[dt])
// Splitting the loop changes only the INTERLEAVING of operations belonging to
// DIFFERENT (independent) accumulators -- v_o[0] and v_o[3] never share an fp32
// reduction, so reordering their WMMAs is associativity-neutral. The intermediate
// value of v_o[dt] handed from Pass A to Pass B is byte-identical to v122's value
// between its two WMMAs. row_max (serial fmax), new_m, rescale, exp2(s0)/exp2(s1),
// row_sum (serial add s0 then s1, then cross-half fold), l_row, the bf16 packs, and
// the final normalize are ALL byte-for-byte v122. -> n_bad=0, max_abs=0.0001 expected.
//
// WHY this is distinct from v127 (90.24, QK reorder): v127 touched the QK^T half-tile
// load schedule and shortened K lead time. v128 leaves QK^T verbatim (both next-K
// loads early, as the reviewer mandated) and attacks the DIFFERENT bottleneck of the
// PV dependent-WMMA-chain latency. The two are orthogonal.
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast128(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v128_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v128_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v128(opus_attn_kargs k)
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
            v_q[kt][j] = bf16_fast128(bf16_to_f32(v_q[kt][j]) * qscale);
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

    for (int n_tile = 0; n_tile < num_kv_tiles; ++n_tile) {
        const int n_base = n_tile * BLOCK_N;

        // ---- QK^T: BLOCK_N=32 -> two 16x16x16 WMMAs per D-tile. BYTE-FOR-BYTE v122:
        // both next-D-tile K loads (v_k0_next, v_k1_next) issued early so global VMEM
        // overlaps the WMMA issue window.
        fp32x8_t v_s0 = {0,0,0,0,0,0,0,0};
        fp32x8_t v_s1 = {0,0,0,0,0,0,0,0};

        bf16x8_t v_k0_next = *reinterpret_cast<const bf16x8_t*>(Kp + (n_base + col16) * stride_n + row8);
        bf16x8_t v_k1_next = *reinterpret_cast<const bf16x8_t*>(Kp + (n_base + 16 + col16) * stride_n + row8);
        #pragma unroll
        for (int kt = 0; kt < DK; ++kt) {
            bf16x8_t v_k0 = v_k0_next;
            bf16x8_t v_k1 = v_k1_next;
            if (kt + 1 < DK) {
                v_k0_next = *reinterpret_cast<const bf16x8_t*>(Kp + (n_base + col16) * stride_n + (kt+1) * W_K + row8);
                v_k1_next = *reinterpret_cast<const bf16x8_t*>(Kp + (n_base + 16 + col16) * stride_n + (kt+1) * W_K + row8);
            }
            v_s0 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_k0, v_q[kt], v_s0);
            v_s1 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_k1, v_q[kt], v_s1);
        }

        // ---- Softmax over 16 values (8 from s0, 8 from s1). BYTE-FOR-BYTE v122.
        fp32_t row_max = v_s0[0];
        #pragma unroll
        for (int j = 1; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s1[j]);
        row_max = v128_cross_half_max(row_max);

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
        row_sum = v128_cross_half_sum(row_sum);
        l_row += row_sum;

        bf16x8_t v_p0, v_p1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p0[j] = bf16_fast128(v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p1[j] = bf16_fast128(v_s1[j]);

        // ---- PV: V-HALF SPLIT. Pass A issues all 8 v0-half WMMAs (independent across
        // accumulators -> 8-wide window), Pass B issues all 8 v1-half WMMAs (the same-dt
        // dependency is hidden behind the other tiles' WMMAs). Per accumulator the order
        // is exactly v122: `*= rescale` (if need_rescale), wmma(v_v0,v_p0), wmma(v_v1,v_p1).
        // At most ONE V fragment is live at a time -> liveness <= v122's CHUNK=2 group.

        // Pass A: rescale (CHUNK=2, exactly v122) + the v0-half WMMA for every D-tile.
        constexpr int CHUNK = 2;
        #pragma unroll
        for (int c0 = 0; c0 < DK; c0 += CHUNK) {
            if (need_rescale) {
                #pragma unroll
                for (int dc = 0; dc < CHUNK; ++dc) {
                    const int dt = c0 + dc;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) v_o[dt][j] *= rescale;
                }
            }
            #pragma unroll
            for (int dc = 0; dc < CHUNK; ++dc) {
                const int dt = c0 + dc;
                const bf16_t* vt_addr0 = VTp + (dt * W_K + col16) * vt_stride_d + n_base + row8;
                bf16x8_t v_v0 = *reinterpret_cast<const bf16x8_t*>(vt_addr0);
                v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_v0, v_p0, v_o[dt]);
            }
        }

        // Pass B: the v1-half WMMA for every D-tile. Each depends on Pass A's same-dt
        // accumulator write, but that retired ~7 WMMAs earlier -> latency fully hidden.
        #pragma unroll
        for (int dt = 0; dt < DK; ++dt) {
            const bf16_t* vt_addr1 = VTp + (dt * W_K + col16) * vt_stride_d + n_base + 16 + row8;
            bf16x8_t v_v1 = *reinterpret_cast<const bf16x8_t*>(vt_addr1);
            v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_v1, v_p1, v_o[dt]);
        }
    }

    const fp32_t inv = (l_row > 0.0f) ? (1.0f / l_row) : 0.0f;
    bf16_t* o_row = Op + (q_m_base + col16) * stride_n;
    #pragma unroll
    for (int dt = 0; dt < DK; ++dt) {
        bf16x8_t o_pack;
        #pragma unroll
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast128(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
