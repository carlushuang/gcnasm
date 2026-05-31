// v114 -- v106 (MEASURED champion, 90.20 TFLOPS) + CROSS-PIPE exp2/pack INTERLEAVE of
//         subtile 1 into the matrix-pipe shadow of pair-0's first two PV WMMAs (v45 lever).
//
// Base choice: v106, the measured best @ b4h32n4096. QK^T (two-ahead K prefetch ring) and
// the ENTIRE PV WMMA cadence + accumulator chaining + one-pair-ahead rescale are kept
// BYTE-FOR-BYTE. PV restructuring is saturated on this box (v104/v105 4-chain regressed,
// v107/v108 rescale-placement regressed, v109/v110 V-prefetch rings dipped below v106), so
// v114 does NOT touch the PV schedule. Instead it attacks the SAME unsolved softmax wedge
// the reviewer named, but with the one playbook lever never yet implemented: lever 2 / v45
// CROSS-PIPE INTERLEAVE -- hide the transcendental exp2 in the WMMA issue shadow.
//
// THE ONLY CHANGE (cross-pipe interleave of subtile 1):
//   In v106 the softmax does, in program order: exp2(s0)+exp2(s1) -> row_sum -> pack
//   v_p0(from s0)+pack v_p1(from s1) -> prologue rescale -> PV loop. The first PV WMMA
//   thus waits for BOTH subtiles' 16 exp2 transcendentals to retire even though pair-0's
//   first two WMMAs consume ONLY v_p0 (= exp2(s0)). The exp2 on subtile 1 (8 transcendentals
//   on the exp pipe) sits needlessly on the path to the matrix pipe starting work.
//
//   v114 peels pair 0 out of the PV loop and reorders so subtile-1's exp2+pack run in the
//   shadow of pair-0's first two PV WMMAs:
//     1. exp2(s0); pack v_p0.                              (path to first WMMA)
//     2. prologue rescale pair 0.
//     3. load pair-0 V (va0/va1/vb0/vb1); rescale pair 2 ahead (v106's dt=0 body verbatim).
//     4. WMMA(va0,v_p0,o0); WMMA(vb0,v_p0,o1).             <- matrix pipe runs on v_p0 only
//     5. exp2(s1); pack v_p1.                              <- exp pipe runs UNDER step 4
//     6. WMMA(va1,v_p1,o0); WMMA(vb1,v_p1,o1).             <- pair-0 completes (v106 order)
//     7. row_sum (off the QK->PV critical path; l_row consumed only after the KV loop).
//     8. PV loop for pairs dt=2,4,6 -- v106's body VERBATIM.
//   The 8 exp2(s1) transcendentals now co-execute with two bf16 WMMAs (~12.6-cyc issue
//   window each) instead of blocking the matrix pipe's start. Playbook: exp2 partially
//   co-executes with wmma; ~0.4 cyc exp hides per bf16 WMMA.
//
// BIT-EXACT: exp2 args (new_m), the bf16 RNE pack, the serial row_sum order, the rescale
// value, and the per-accumulator WMMA operand order (o0: va0*p0 then va1*p1; o1: vb0*p0 then
// vb1*p1) are ALL identical to v106 -> byte-identical row_max/new_m/probs/l_row/O.
// VGPR: no new accumulators or V fragments; v_s1 stays live across two extra WMMAs and
// v_p1 is produced later -> net live state ~unchanged -> same 4-live-V-fragment footprint
// and occupancy as v106. RISK: low. If the compiler already hoisted v106's WMMAs above
// exp2(s1), v114 simply ties v106.
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast114(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v114_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v114_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v114(opus_attn_kargs k)
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
            v_q[kt][j] = bf16_fast114(bf16_to_f32(v_q[kt][j]) * qscale);
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
        row_max = v114_cross_half_max(row_max);

        const fp32_t new_m = __builtin_fmaxf(m_row, row_max);
        fp32_t rescale = 1.0f;
        const bool need_rescale = (new_m != m_row);
        if (need_rescale) {
            rescale = __builtin_amdgcn_exp2f(m_row - new_m);
            l_row *= rescale;
        }
        m_row = new_m;

        // exp2 of SUBTILE 0 ONLY (the path to pair-0's first two PV WMMAs, which consume
        // v_p0 only). Subtile-1 exp2 + pack + row_sum are DEFERRED below into the
        // matrix-pipe shadow of those two WMMAs (cross-pipe interleave, playbook lever 2).
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_s0[j] = __builtin_amdgcn_exp2f(v_s0[j] - new_m);

        bf16x8_t v_p0, v_p1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p0[j] = bf16_fast114(v_s0[j]);

        // ---- PV: v106's 2-D-tile grouped PV, one-pair-ahead pipelined rescale, with
        // PAIR 0 PEELED so subtile-1's exp2/pack/row_sum run UNDER pair-0's first two
        // WMMAs. Prologue rescales pair 0 (v106 verbatim).
        if (need_rescale) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_o[0][j] *= rescale;
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_o[1][j] *= rescale;
        }

        // ---- PEELED PAIR 0 (cross-pipe interleave) ----
        {
            const bf16_t* a0 = VTp + (0 * W_K + col16) * vt_stride_d + n_base + row8;
            const bf16_t* a1 = VTp + (0 * W_K + col16) * vt_stride_d + n_base + 16 + row8;
            const bf16_t* b0 = VTp + (1 * W_K + col16) * vt_stride_d + n_base + row8;
            const bf16_t* b1 = VTp + (1 * W_K + col16) * vt_stride_d + n_base + 16 + row8;
            bf16x8_t va0 = *reinterpret_cast<const bf16x8_t*>(a0);
            bf16x8_t va1 = *reinterpret_cast<const bf16x8_t*>(a1);
            bf16x8_t vb0 = *reinterpret_cast<const bf16x8_t*>(b0);
            bf16x8_t vb1 = *reinterpret_cast<const bf16x8_t*>(b1);
            // One-pair-ahead rescale of pair 2 (v106's dt=0 body, identical guard/order).
            if (DK > 2 && need_rescale) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) v_o[2][j] *= rescale;
                #pragma unroll
                for (int j = 0; j < 8; ++j) v_o[3][j] *= rescale;
            }
            // First two WMMAs consume ONLY v_p0 -> matrix pipe starts BEFORE exp2(s1).
            v_o[0] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(va0, v_p0, v_o[0]);
            v_o[1] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(vb0, v_p0, v_o[1]);

            // CROSS-PIPE: subtile-1 exp2 (8 transcendentals) + pack now execute on the
            // exp/VALU pipe in the shadow of the two WMMAs above. row_sum (off the
            // QK->PV critical path -- l_row consumed only after the KV loop) follows.
            // Bit-exact: same exp2 args, same RNE pack, same serial s0-then-s1 sum order.
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_s1[j] = __builtin_amdgcn_exp2f(v_s1[j] - new_m);
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_p1[j] = bf16_fast114(v_s1[j]);

            fp32_t row_sum = v_s0[0];
            #pragma unroll
            for (int j = 1; j < 8; ++j) row_sum += v_s0[j];
            #pragma unroll
            for (int j = 0; j < 8; ++j) row_sum += v_s1[j];
            row_sum = v114_cross_half_sum(row_sum);
            l_row += row_sum;

            // pair-0 completes with v_p1 (same per-accumulator WMMA order as v106).
            v_o[0] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(va1, v_p1, v_o[0]);
            v_o[1] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(vb1, v_p1, v_o[1]);
        }

        // ---- Remaining pairs dt=2,4,6: v106 PV body VERBATIM ----
        #pragma unroll
        for (int dt = 2; dt < DK; dt += 2) {
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
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast114(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
