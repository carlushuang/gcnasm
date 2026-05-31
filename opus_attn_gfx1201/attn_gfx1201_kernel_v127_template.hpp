// v127 -- v100 champion + CHUNKED (CHUNK=2) PV online-rescale schedule.
//
// PROVENANCE / why this round: the director's plateau note (round 13, mandate A)
// asks to recover the speed of the "fast v111" (measured 91.6-92.3 TFLOPS in the
// other ablation lane, the fastest kernel found) while keeping v100's EXACT fp32
// accumulation order. Investigation of the ledgers shows the fast v111 is the
// CHUNK=2 PV-rescale schedule, and that lane's reported max_abs=0.0360 is a
// lane-LOCAL verify artifact: in that lane EVERY version from v103 onward reports
// the identical 0.0360 regardless of the change (even an fp-contract(off) build),
// while THIS lane reports 0.0001 for the byte-identical sources. So the 0.0360 is
// not produced by the chunked schedule; the schedule itself is the real +1.5 TFLOPS
// lever and it has NEVER been measured in this (clean-verify) lane.
//
// THE CHANGE (PV phase only; QK^T / softmax / exp2 / sum / pack / normalize are all
// byte-for-byte v100): v100 applies the online rescale to all 8 O accumulators in
// one monolithic VALU pass, THEN runs the 8-D-tile PV WMMA loop. v127 instead walks
// the D-tiles in CHUNK=2 groups; for each group it (a) rescales just that group's 2
// O fragments, then (b) issues that group's V loads + PV WMMAs. The next group's
// rescale FMAs (VALU pipe) then overlap the current group's PV WMMAs (matrix pipe),
// and no single VALU region keeps more than 2 fp32x8 O fragments hot -> shorter live
// ranges and more scheduler freedom around the accumulator-pressure peak.
//
// BIT-EXACTNESS (non-negotiable, max_abs<=0.005): the arithmetic touching each
// accumulator v_o[dt] is UNCHANGED from v100 -- for every dt it is still exactly
// `v_o[dt] *= rescale` (only when need_rescale, same `rescale`), then the same two
// WMMAs in the same order: wmma(v_v0,v_p0) then wmma(v_v1,v_p1). Chunking only
// changes the INTERLEAVING of independent accumulators (v_o[0]'s ops vs v_o[2]'s
// ops never share an fp32 reduction), which is associativity-neutral: the result
// of each accumulator is rounding-identical to v100. row_max/row_sum/exp2/l_row are
// untouched. -> n_bad=0, max_abs=0.0001 expected (same as v100 in this lane).
//
// WHY this differs from this lane's earlier v103 (89.68, no win): v103 *paired* the
// V loads into two INDEPENDENT WMMA chains (v_o[dt] and v_o[dt+1] interleaved at the
// WMMA level), which doubled live V fragments per group (4 V regs) and reordered the
// WMMA issue. v127 keeps v100's single-chain per-dt WMMA order and only groups the
// RESCALE -- minimal liveness delta, the lever the fast lane actually used.
//
// VGPR: ~v100 (~205, 7 waves/SIMD). No new V buffers, no LDS, no barriers.
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast127(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v127_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v127_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v127(opus_attn_kargs k)
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
            v_q[kt][j] = bf16_fast127(bf16_to_f32(v_q[kt][j]) * qscale);
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
        // v127 narrow intra-tile QK scheduling change (vs v122): the two next-D-tile
        // K refills are SPLIT so each is issued immediately before the WMMA that does
        // NOT consume it. v122 issued BOTH next-K loads back-to-back before BOTH
        // WMMAs, so at the WMMA-issue point all 4 K fragments (v_k0,v_k1,v_k0_next,
        // v_k1_next = 16 VGPR) were simultaneously live. v127 issues v_k0_next, then
        // s0's WMMA (which overlaps that load; only v_k0/v_k1/v_k0_next live = 3
        // frags), then v_k1_next, then s1's WMMA. Peak K-fragment liveness drops 4->3
        // inside the QK loop and each VMEM load overlaps an independent WMMA's issue
        // shadow. WMMA order (s0 then s1) and ALL arithmetic are byte-for-byte
        // v122/v100 -> bit-exact (max_abs<=0.0001). Nothing kept live across softmax/PV.
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
            }
            v_s0 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_k0, v_q[kt], v_s0);
            if (kt + 1 < DK) {
                v_k1_next = *reinterpret_cast<const bf16x8_t*>(Kp + (n_base + 16 + col16) * stride_n + (kt+1) * W_K + row8);
            }
            v_s1 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_k1, v_q[kt], v_s1);
        }

        // Softmax over 16 values (8 from s0, 8 from s1)
        fp32_t row_max = v_s0[0];
        #pragma unroll
        for (int j = 1; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s1[j]);
        row_max = v127_cross_half_max(row_max);

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
        row_sum = v127_cross_half_sum(row_sum);
        l_row += row_sum;

        bf16x8_t v_p0, v_p1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p0[j] = bf16_fast127(v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p1[j] = bf16_fast127(v_s1[j]);

        // ---- PV: CHUNK=2 grouped online-rescale interleaved with the PV WMMAs.
        // Per group of CHUNK D-tiles: rescale just this group's O fragments (a short
        // VALU region, only 2 fp32x8 accumulators hot), then issue the group's V
        // loads and PV WMMAs. The next group's rescale FMAs overlap this group's
        // WMMAs in the matrix-pipe shadow. Each accumulator v_o[dt]'s arithmetic is
        // byte-for-byte v100: `*= rescale` (only if need_rescale), then wmma(v0,p0),
        // then wmma(v1,p1). Independent accumulators never share a reduction, so the
        // regrouping is associativity-neutral -> bit-exact vs v100.
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
                const bf16_t* vt_addr1 = VTp + (dt * W_K + col16) * vt_stride_d + n_base + 16 + row8;
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
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast127(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
