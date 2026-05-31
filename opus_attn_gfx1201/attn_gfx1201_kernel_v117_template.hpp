// v117 -- v106 champion (90.20 TFLOPS) + UNCONDITIONAL (BRANCHLESS) PV RESCALE FORM.
//
// Base choice: v106 (the MEASURED running best, 90.20 TFLOPS primary @ b4h32n4096).
// QK^T, softmax math, V layout, the 2-D-tile grouped PV (4 live V fragments, two
// independent accumulator chains per pair) and the one-pair-ahead rescale SCHEDULE are
// all kept structurally identical to v106. v107-v116 already proved that restructuring
// PV (mid-load rescale v107=85.3, split rescale v108=88.7, V-prefetch rings
// v109=85.1/v110) or reordering softmax (v112-v115, tied) does not beat v106: every
// register-adding change hits the ~207-VGPR / 7-wave occupancy cliff, and every
// reorder ties. The ONE sanctioned lever never implemented is the round-7 reviewer's
// "Next Lever 3": A/B the CONDITIONAL `if (need_rescale)` rescale against an
// UNCONDITIONAL multiply.
//
// What changed vs v106 (the ONLY change): the three `if (need_rescale)` guards around
// the O-accumulator and l_row rescales are REMOVED. rescale = exp2f(m_row - new_m) is
// computed unconditionally and l_row, v_o[0..1] (prologue), and v_o[dt+2..dt+3] (the
// one-pair-ahead step) are multiplied unconditionally.
//
// WHY THIS SHOULD HELP (primary, n=4096): online-softmax record maxima grow ~ln(N), so
// `need_rescale` is TRUE on only ~a handful of the 64 KV tiles -- the rescale FMAs are
// already cheap. Their real cost in v106 is NOT the arithmetic but the data-dependent
// `s_cbranch` that fences the compiler's instruction scheduler at three points inside
// PV: the scheduler cannot freely hoist/interleave the rescale FMAs with the V loads
// and PV WMMAs across a branch boundary, which is exactly why v106's placement was so
// fragile (v107/v108 moved it a little and regressed). The rescale FMAs are pure VALU
// that co-issue with the WMMAs on a SEPARATE pipe (RDNA4 dual-issue), so running them
// unconditionally costs ~0 matrix-pipe cycles while removing the three scheduler fences
// -- letting the compiler pack the FMAs into the WMMA issue shadow with no barrier.
// Zero added VGPR (no new live state), so occupancy is unchanged at 7 waves/SIMD.
//
// BIT-EXACTNESS (vs v106, provable):
//   * When new_m == m_row: rescale = exp2f(0) = EXACTLY 1.0 (hw v_exp2_f32(0)=1.0).
//     v106 skipped the multiply; v117 does `x * 1.0f`, which is the IEEE-754 identity
//     (exact for every finite/inf/zero x; no rounding). l_row *= 1.0f likewise exact.
//   * When new_m != m_row: v117 takes the same arithmetic path v106's branch took --
//     same rescale value, same multiplies, same order.
//   * First tile (m_row = -3.4e38): new_m = row_max != m_row in BOTH versions ->
//     rescale = exp2f(-huge) = 0.0; v_o (already 0) *= 0 and l_row (0) *= 0 -> identical.
// Every v_o[dt] is still rescaled EXACTLY ONCE before its own PV WMMA, same WMMA
// operand order -> byte-identical row_max / new_m / rescale / probs / l_row / O.
//
// RISK: low. Correctness is an IEEE identity (verified by the n_bad==0 gate). Perf:
// neutral-to-positive; if the compiler had already if-converted v106's branch this
// simply ties v106 (no regression -- same instruction count, same VGPR).
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast117(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v117_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v117_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v117(opus_attn_kargs k)
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
            v_q[kt][j] = bf16_fast117(bf16_to_f32(v_q[kt][j]) * qscale);
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
        row_max = v117_cross_half_max(row_max);

        const fp32_t new_m = __builtin_fmaxf(m_row, row_max);
        // UNCONDITIONAL (branchless) rescale: exp2f(0)=1.0 and x*1.0f is exact, so the
        // no-rescale case is bit-identical to v106's skipped branch -- but with no
        // data-dependent s_cbranch fencing the PV instruction scheduler.
        const fp32_t rescale = __builtin_amdgcn_exp2f(m_row - new_m);
        l_row *= rescale;
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
        row_sum = v117_cross_half_sum(row_sum);
        l_row += row_sum;

        bf16x8_t v_p0, v_p1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p0[j] = bf16_fast117(v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p1[j] = bf16_fast117(v_s1[j]);

        // ---- PV: v106's 2-D-tile grouped PV (4 live V fragments, two independent
        // accumulator chains per pair) with the one-pair-ahead rescale SCHEDULE kept
        // byte-for-byte. The ONLY change: the rescale multiplies are now UNCONDITIONAL
        // (no `if (need_rescale)` guard), removing the scheduler-fencing branches so the
        // rescale FMAs pack freely into the V-load + WMMA issue shadow. Each v_o[dt] is
        // still rescaled exactly once before its PV WMMA -> bit-exact. DK=8 is even.
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_o[0][j] *= rescale;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_o[1][j] *= rescale;
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
            // One-pair-ahead rescale (UNCONDITIONAL): prepare the NEXT pair's
            // accumulators while the current pair's V loads are in flight. Independent
            // of the WMMAs below. exp2f(0)=1.0 -> *=1.0 is the IEEE identity when no
            // rescale is needed, matching v106 bit-for-bit.
            if (dt + 2 < DK) {
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
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast117(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
