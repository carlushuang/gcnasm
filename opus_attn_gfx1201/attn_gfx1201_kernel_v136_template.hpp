// v136 -- v135 (s_setprio matrix-window shaping) with a TIGHTENED PV priority
// window. ZERO VGPR cost, ZERO arithmetic change, bit-exact vs v135/v122/v100.
//
// PROVENANCE: v135 (92.42 TFLOPS, round-13 PASS) added s_setprio(1) around the
// QK^T and PV matrix regions so a wave currently in its WMMA window wins the SIMD
// issue arbiter over a sibling mid-softmax. The round-13 reviewer PASSED it but
// flagged ONE imprecision (its explicit "NEXT"): v135's PV high-priority window
// ALSO wraps the per-CHUNK `need_rescale` online-rescale VALU (up to CHUNK*8 = 16
// fp32 mults/lane per group, 64/lane per KV tile). Those FMAs are NOT matrix work,
// yet at the 7-wave occupancy knee a wave running them at high priority steals an
// issue slot from a SIBLING wave that is actually mid-WMMA -- the exact wave the
// hint is supposed to let win the matrix pipe. So v135's PV window is too wide and
// partially fights itself.
//
// THE CHANGE (PV s_setprio placement ONLY; entire v135/v122 dataflow byte-for-byte
// identical -- same QK^T + K double-buffer, same softmax/exp2/pack, same CHUNK=2
// PV arithmetic, same normalize/store; QK^T priority window unchanged):
//   * the per-CHUNK rescale FMAs now run at priority 0 (yield the issue slot, since
//     they go on the VALU pipe regardless and cannot use the matrix pipe), and
//   * priority is raised to 1 ONLY around each group's V loads + the two PV WMMAs,
//     then dropped to 0 again before the next group's rescale.
// Net effect: the PV matrix pipe is now fed ONLY by whichever wave is truly inside
// a WMMA window, never by a wave doing rescale VALU. This sharpens exactly the
// cross-wave arbitration win that gave v135 its +1.2 TFLOPS over v122.
//
// s_setprio is a SCALAR instruction: NO VGPR / NO SGPR live range. Moving / adding
// the hint changes only hardware issue PRIORITY, so v136 stays byte-for-byte at
// v122/v135's register footprint (~197-205 VGPR, 7 waves/SIMD) -- it does NOT touch
// the occupancy knee that killed every cross-iteration ILP attempt (v119/v125/v134).
//
// BIT-EXACTNESS (non-negotiable, max_abs<=0.005): s_setprio cannot change any
// computed value, any instruction, or the intra-wave ordering of dependent ops
// (priority affects only inter-wave arbitration, never intra-wave program order).
// Every fp32 reduction, exp2 arg, rescale fold, and WMMA accumulate is byte-for-byte
// v135 / v122 / v100. -> n_bad=0, max_abs=0.0001 expected (identical to v135).
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast136(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v136_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v136_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v136(opus_attn_kargs k)
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
            v_q[kt][j] = bf16_fast136(bf16_to_f32(v_q[kt][j]) * qscale);
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
        // Raise issue priority for the QK^T matrix region so this wave wins the
        // SIMD issue slot over any sibling currently stalled in its softmax VALU
        // window. Scalar op: no VGPR/SGPR liveness, no arithmetic effect.
        __builtin_amdgcn_s_setprio(1);
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
        // Drop priority for the softmax VALU/exp2/cross-lane stretch: yield issue
        // bandwidth (which this wave cannot use for the matrix pipe anyway) to a
        // sibling wave that is in its high-prio QK^T / PV WMMA region.
        __builtin_amdgcn_s_setprio(0);

        // Softmax over 16 values (8 from s0, 8 from s1)
        fp32_t row_max = v_s0[0];
        #pragma unroll
        for (int j = 1; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) row_max = __builtin_fmaxf(row_max, v_s1[j]);
        row_max = v136_cross_half_max(row_max);

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
        row_sum = v136_cross_half_sum(row_sum);
        l_row += row_sum;

        bf16x8_t v_p0, v_p1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p0[j] = bf16_fast136(v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p1[j] = bf16_fast136(v_s1[j]);

        // ---- PV: CHUNK=2 grouped online-rescale interleaved with the PV WMMAs.
        // Per group of CHUNK D-tiles: rescale just this group's O fragments (a short
        // VALU region, only 2 fp32x8 accumulators hot), then issue the group's V
        // loads and PV WMMAs. The next group's rescale FMAs overlap this group's
        // WMMAs in the matrix-pipe shadow. Each accumulator v_o[dt]'s arithmetic is
        // byte-for-byte v100: `*= rescale` (only if need_rescale), then wmma(v0,p0),
        // then wmma(v1,p1). Independent accumulators never share a reduction, so the
        // regrouping is associativity-neutral -> bit-exact vs v100.
        constexpr int CHUNK = 2;
        // v136 TIGHTENING (round-13 reviewer NEXT): unlike v135, the per-CHUNK
        // online-rescale FMAs run at priority 0, and priority is raised to 1 ONLY
        // around each group's V-load + PV-WMMA matrix region (then dropped again).
        // Rationale: v135's PV high-prio window also wrapped the need_rescale VALU
        // (up to CHUNK*8 = 16 fp32 mults/lane per group, 64/lane per tile). At the
        // 7-wave knee, a wave running rescale at high prio steals an issue slot from
        // a SIBLING wave that is actually mid-WMMA -- the exact wave we want to win
        // the matrix pipe. Keeping rescale at prio 0 lets the rescaling wave yield
        // (its FMAs go on the VALU pipe regardless), so the matrix pipe is fed only
        // by whoever is truly in a WMMA window. Pure scheduling: no VGPR, no
        // arithmetic change -> byte-for-byte bit-exact vs v135/v122/v100.
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
            // Raise priority only for this group's matrix region (V loads + WMMAs).
            __builtin_amdgcn_s_setprio(1);
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
            // Drop priority for the next group's rescale VALU (yield to matrix-phase siblings).
            __builtin_amdgcn_s_setprio(0);
        }
    }

    const fp32_t inv = (l_row > 0.0f) ? (1.0f / l_row) : 0.0f;
    bf16_t* o_row = Op + (q_m_base + col16) * stride_n;
    #pragma unroll
    for (int dt = 0; dt < DK; ++dt) {
        bf16x8_t o_pack;
        #pragma unroll
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast136(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
