// v115 -- v106 champion (90.20 TFLOPS) + 3-WIDE PV GROUPING (three independent
// accumulator chains, <=6 live V fragments) with the rescale software-pipelined ONE
// GROUP AHEAD.
//
// Base: v106 (the MEASURED running best, 90.20 TFLOPS). QK^T is BYTE-FOR-BYTE identical
// to v106 (two-ahead K prefetch ring). The softmax/exp2/pack path is identical. The
// ONLY change is the PV-phase grouping width.
//
// Why 3-wide (reviewer's explicit NEXT FOCUS for this round):
//   v106 uses 2-wide PV groups -> TWO independent accumulator chains (v_o[dt], v_o[dt+1]).
//   Within a pair, the two dependent WMMAs on the SAME accumulator (p0 then p1) are
//   separated by only ONE independent WMMA (the other chain's p0). On RDNA4 the WMMA
//   result latency wants ~2 independent issues of cover, so v106 still leaves a small
//   bubble on each accumulator's second WMMA.
//   v104 (4-wide, 8 live V frags) and v105 (4-wide two-wave JIT) BOTH REGRESSED hard
//   (~79 TFLOPS) -- 8 live V fragments blew the VGPR/scheduling budget (the reviewer's
//   diagnosed "VGPR cliff"). 3-wide is the sweet spot the reviewer asked for:
//     * THREE independent chains -> the second WMMA on each accumulator is now separated
//       by TWO independent WMMAs (the other two chains' p0) -> full result-latency cover.
//     * exactly 6 live V fragments (3x p0-col + 3x p1-col) -- the reviewer's stated <=6
//       ceiling, ~+8 VGPR over v106's 4 (197 -> ~205, still the 7-wave band v39 ran at),
//       NOT v104's 8-fragment cliff.
//
// Grouping: DK=8 -> groups {0,1,2},{3,4,5},{6,7} (last group is a 2-wide remainder,
// exactly v106's pair). Rescale is pipelined ONE GROUP ahead exactly as v106 pipelined
// one PAIR ahead: a prologue rescales group 0; iteration over group g issues that
// group's V loads, then interleaves the NEXT group's rescale (DIFFERENT, independent
// accumulator regs) before issuing the current group's WMMAs (all p0 then all p1).
//
// Arithmetic: each v_o[i] is rescaled EXACTLY ONCE before its PV accumulation, with the
// same rescale scalar, and each accumulator's own WMMA order is unchanged (its p0
// contribution then its p1 contribution -- identical to v106). Only the interleaving
// ACROSS independent accumulators changes, which does not alter any single accumulator's
// fp32 sum -> bit-exact / n_bad==0, same as v106.
// RISK: low-med. The +8 live-V VGPR may tip occupancy on a bad allocation; if so this
// ties or slightly trails v106. No correctness risk.
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast115(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v115_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v115_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v115(opus_attn_kargs k)
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
            v_q[kt][j] = bf16_fast115(bf16_to_f32(v_q[kt][j]) * qscale);
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

        // ---- QK^T: BLOCK_N=32 -> two 16x16x16 WMMAs per D-tile.
        // TWO-AHEAD software-pipelined K loads (v101/v103, byte-for-byte identical).
        fp32x8_t v_s0 = {0,0,0,0,0,0,0,0};
        fp32x8_t v_s1 = {0,0,0,0,0,0,0,0};

        const bf16_t* kbase0 = Kp + (n_base + col16) * stride_n + row8;
        const bf16_t* kbase1 = Kp + (n_base + 16 + col16) * stride_n + row8;
        bf16x8_t kr0[2];
        bf16x8_t kr1[2];
        kr0[0] = *reinterpret_cast<const bf16x8_t*>(kbase0);
        kr1[0] = *reinterpret_cast<const bf16x8_t*>(kbase1);
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
        row_max = v115_cross_half_max(row_max);

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
        row_sum = v115_cross_half_sum(row_sum);
        l_row += row_sum;

        bf16x8_t v_p0, v_p1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p0[j] = bf16_fast115(v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p1[j] = bf16_fast115(v_s1[j]);

        // ---- PV: 3-WIDE grouped PV (THREE independent accumulator chains, <=6 live V
        // fragments) with the online rescale SOFTWARE-PIPELINED BY ONE GROUP.
        // Groups of GW=3 D-tiles: {0,1,2},{3,4,5},{6,7} for DK=8. Within a group we
        // load all V fragments (p0-col + p1-col = up to 6 frags), interleave the NEXT
        // group's rescale (independent accumulator regs), then issue all p0 WMMAs
        // followed by all p1 WMMAs. Each accumulator's own WMMA order (its p0 then its
        // p1 contribution) is identical to v106 -> bit-exact. Three chains separate the
        // two dependent WMMAs on each accumulator by TWO independent WMMAs -> full
        // result-latency cover (v106's pairs gave only one).
        constexpr int GW = 3;
        if (need_rescale) {
            #pragma unroll
            for (int i = 0; i < (GW < DK ? GW : DK); ++i) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) v_o[i][j] *= rescale;
            }
        }
        #pragma unroll
        for (int g0 = 0; g0 < DK; g0 += GW) {
            const int gsz = (DK - g0 < GW) ? (DK - g0) : GW;
            bf16x8_t va[GW];  // p0-column V fragments (paired with v_p0)
            bf16x8_t vb[GW];  // p1-column V fragments (paired with v_p1)
            #pragma unroll
            for (int i = 0; i < GW; ++i) {
                if (i < gsz) {
                    const int dt = g0 + i;
                    const bf16_t* pa = VTp + (dt * W_K + col16) * vt_stride_d + n_base + row8;
                    const bf16_t* pb = VTp + (dt * W_K + col16) * vt_stride_d + n_base + 16 + row8;
                    va[i] = *reinterpret_cast<const bf16x8_t*>(pa);
                    vb[i] = *reinterpret_cast<const bf16x8_t*>(pb);
                }
            }
            // One-group-ahead rescale: prepare the NEXT group's accumulators (different,
            // independent regs) while the current group's V loads are in flight.
            if (need_rescale) {
                #pragma unroll
                for (int i = 0; i < GW; ++i) {
                    const int nd = g0 + GW + i;
                    if (nd < DK) {
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) v_o[nd][j] *= rescale;
                    }
                }
            }
            #pragma unroll
            for (int i = 0; i < GW; ++i) {
                if (i < gsz) {
                    const int dt = g0 + i;
                    v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(va[i], v_p0, v_o[dt]);
                }
            }
            #pragma unroll
            for (int i = 0; i < GW; ++i) {
                if (i < gsz) {
                    const int dt = g0 + i;
                    v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(vb[i], v_p1, v_o[dt]);
                }
            }
        }
    }

    const fp32_t inv = (l_row > 0.0f) ? (1.0f / l_row) : 0.0f;
    bf16_t* o_row = Op + (q_m_base + col16) * stride_n;
    #pragma unroll
    for (int dt = 0; dt < DK; ++dt) {
        bf16x8_t o_pack;
        #pragma unroll
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast115(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
