// v129 -- v122 champion (CHUNK=2 PV rescale) + 1-deep DEFERRED-B PV software pipeline.
//
// PROVENANCE: v122 (91.68 TFLOPS, this lane's champion) issues PV as, per D-tile dt:
//   rescale(o[dt]); A_dt = wmma(v0,p0,o[dt]); B_dt = wmma(v1,p1,o[dt]);
// The two WMMAs for the SAME accumulator are back-to-back, so B_dt has a RAW
// dependency on A_dt's matrix-pipe result with NO independent WMMA between them ->
// the WMMA accumulate latency is exposed once per D-tile.
//
// Round 19's v128 tried to fix this by full V-half splitting (Pass A = all 8 A_dt,
// Pass B = all 8 B_dt). That maximised WMMA separation but DESTROYED V-load lead
// time: Pass B re-issued 8 fresh v1 loads with immediate load->WMMA use, exposing
// global VMEM latency -> regressed to 73.5 TFLOPS. The reviewer's diagnosis: keep
// V-load lead time, but separate each accumulator's two WMMAs by only 1-2 other
// (independent) WMMAs -- a small rotating schedule, not a full pass split.
//
// THE CHANGE (PV phase only; QK^T/softmax/exp2/sum/pack/normalize byte-for-byte
// v122): a 1-deep software pipeline that DEFERS the v1-half WMMA (B) by one D-tile.
// Per dt we still load BOTH v0 and v1 together (lead time preserved, exactly as
// v122), issue A_dt immediately, but issue the PREVIOUS tile's B_{dt-1} instead of
// B_dt. So the instruction stream becomes:
//   ... A_{dt-1}; B_{dt-2}; A_dt; B_{dt-1}; A_{dt+1}; B_dt; ...
// Now between A_{dt} (write to o[dt]) and B_{dt} (read o[dt]) sits A_{dt+1} -- one
// independent WMMA on a different accumulator -- so the accumulate latency of A_dt
// is hidden behind A_{dt+1} instead of stalling B_dt. V loads still lead their use
// by a full tile (both halves prefetched at the top of each dt step), so no VMEM
// bubble is introduced. Liveness adds exactly ONE bf16x8 carry (v1 + its index)
// across one loop step -> ~+1 VGPR, well below the occupancy cliff.
//
// BIT-EXACTNESS (non-negotiable, max_abs<=0.005): every accumulator v_o[dt] sees the
// IDENTICAL ops in the IDENTICAL order as v122: `*= rescale` (only if need_rescale,
// same rescale), then wmma(v_v0,v_p0,o[dt]), then wmma(v_v1,v_p1,o[dt]). The pipeline
// only changes WHEN B_dt is issued relative to OTHER accumulators' WMMAs -- never the
// order of the two WMMAs touching the same o[dt], and independent accumulators never
// share an fp32 reduction. Associativity-neutral -> rounding-identical to v122/v100.
// The rescale of o[dt] still happens (CHUNK=2 group) strictly before A_dt, and B_dt
// is the last write to o[dt] before normalize, so the deferral cannot race rescale of
// a later tile (rescale of o[dt] for the NEXT n_tile only runs after this tile's B_dt
// has retired the accumulator). -> n_bad=0, max_abs=0.0001 expected.
//
// VGPR: ~v122 (+1 carry reg, ~205, 7 waves/SIMD). No LDS, no barriers, no extra loads.
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast129(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v129_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v129_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v129(opus_attn_kargs k)
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
            v_q[kt][j] = bf16_fast129(bf16_to_f32(v_q[kt][j]) * qscale);
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
        // Software-pipelined K loads: preload D-tile 0, then issue dt+1's K loads
        // before the WMMAs on dt so global VMEM overlaps the WMMA issue window.
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

        // Softmax over 16 values (8 from s0, 8 from s1)
        fp32_t row_max = v_s0[0];
        #pragma unroll
        for (int j = 1; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s1[j]);
        row_max = v129_cross_half_max(row_max);

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
        row_sum = v129_cross_half_sum(row_sum);
        l_row += row_sum;

        bf16x8_t v_p0, v_p1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p0[j] = bf16_fast129(v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p1[j] = bf16_fast129(v_s1[j]);

        // ---- PV: online rescale (CHUNK=2 liveness) then a 1-deep DEFERRED-B pipeline.
        //
        // Step 1: rescale all O accumulators. Walked in CHUNK=2 groups so no VALU
        // region keeps more than 2 fp32x8 accumulators hot (identical to v122's
        // rescale-group liveness). For each o[dt] this is exactly `*= rescale` (only
        // if need_rescale) and runs strictly before any WMMA touching o[dt] -> the
        // per-accumulator op order is byte-for-byte v100/v122.
        constexpr int CHUNK = 2;
        if (need_rescale) {
            #pragma unroll
            for (int c0 = 0; c0 < DK; c0 += CHUNK) {
                #pragma unroll
                for (int dc = 0; dc < CHUNK; ++dc) {
                    const int dt = c0 + dc;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) v_o[dt][j] *= rescale;
                }
            }
        }

        // Step 2: deferred-B PV. Per D-tile we load BOTH V halves together (full
        // tile of V-load lead time, exactly as v122 -- no Pass-B load-use bubble),
        // issue A_dt = wmma(v0,p0,o[dt]) immediately, but DEFER the v1-half WMMA to
        // the NEXT iteration: we issue the PREVIOUS tile's B_{dt-1} after A_dt. The
        // emitted WMMA stream is  A0 | A1 B0 | A2 B1 | ... | A7 B6 | B7, so between
        // A_dt (writes o[dt]) and B_dt (reads o[dt]) sits A_{dt+1} -- one independent
        // WMMA on a different accumulator -- hiding A_dt's accumulate latency instead
        // of stalling B_dt back-to-back as v122 did. Per accumulator the two WMMAs
        // are still in order A_dt then B_dt, and independent accumulators never share
        // an fp32 reduction -> bit-exact vs v122. Carries one bf16x8 (v_v1_prev).
        bf16x8_t v_v1_prev;
        #pragma unroll
        for (int dt = 0; dt < DK; ++dt) {
            const bf16_t* vt_addr0 = VTp + (dt * W_K + col16) * vt_stride_d + n_base + row8;
            const bf16_t* vt_addr1 = VTp + (dt * W_K + col16) * vt_stride_d + n_base + 16 + row8;
            bf16x8_t v_v0 = *reinterpret_cast<const bf16x8_t*>(vt_addr0);
            bf16x8_t v_v1 = *reinterpret_cast<const bf16x8_t*>(vt_addr1);
            v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_v0, v_p0, v_o[dt]);
            if (dt > 0)
                v_o[dt-1] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_v1_prev, v_p1, v_o[dt-1]);
            v_v1_prev = v_v1;
        }
        v_o[DK-1] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_v1_prev, v_p1, v_o[DK-1]);
    }

    const fp32_t inv = (l_row > 0.0f) ? (1.0f / l_row) : 0.0f;
    bf16_t* o_row = Op + (q_m_base + col16) * stride_n;
    #pragma unroll
    for (int dt = 0; dt < DK; ++dt) {
        bf16x8_t o_pack;
        #pragma unroll
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast129(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
