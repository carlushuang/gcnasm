// v126 -- v122 champion (CHUNK=2 PV rescale) + CROSS-TILE next-K prefetch.
//
// PROVENANCE / why this round (round 17): v122 = 91.68 TFLOPS is the champion. The
// round-16 reviewer's explicit NEXT FOCUS after v125's 2-KV-fusion failed on the VGPR
// /occupancy cliff (67 TFLOPS): "Revert to v122 and test a 'next-K prefetch only'
// variant that overlaps tile-(t+1)'s K loads without keeping tile-b score
// accumulators live." v125 died because it kept FOUR fp32x8 score fragments
// (s0a,s1a,s0b,s1b = 32 VGPR) live across both tiles' softmax -- doubling the
// accumulator-pressure peak that already sits at the 205->256 ceiling. The fix is to
// overlap ONLY the next tile's K *loads* (2 bf16x8 = 8 VGPR), never its scores.
//
// THE CHANGE (K-load TIMING only; all arithmetic is byte-for-byte v122/v100):
// v122's QK^T D-tile loop already runs a 2-deep K double-buffer (v_k0_next/v_k1_next),
// but it PRIMES that buffer fresh at the TOP of every KV iteration -- so the first
// WMMA of each tile waits on a COLD global K load (a guaranteed VMEM stall at the
// start of every QK phase, num_kv_tiles times). v126 makes the prefetch ring wrap
// ACROSS the KV-tile boundary: the K buffers are hoisted above the KV loop and primed
// ONCE before it; inside each iteration the last D-tile (kt == DK-1) prefetches the
// NEXT KV tile's dt=0 K columns (n_base + BLOCK_N) instead of going idle. Those loads
// are then in flight through the entire softmax + CHUNK=2 PV phase (a long VALU/WMMA
// region with no K dependency), so when the next iteration's first QK WMMA fires its
// K operand is already resident -> the per-tile cold-load stall is eliminated.
//
// LIVENESS / VGPR (the lesson from v125 and v123/v124): the only state carried across
// the iteration boundary is the SAME two K buffers v122 already allocates -- their
// live range is extended from "within the QK loop" to "across softmax+PV", costing at
// most one occupancy tier of headroom (8 VGPR), NOT the 32 VGPR of score fragments
// that broke v125. No new V buffers (v123/v124 falsified PV-side prefetch), no LDS, no
// score-fragment doubling. Expected VGPR ~205-213, still 7 waves/SIMD.
//
// BIT-EXACTNESS (non-negotiable, max_abs<=0.005): NOTHING in the arithmetic changes.
// The QK WMMAs consume identical K/Q operands in identical order (kt=0..DK-1, s0 then
// s1); only WHEN the K bytes are fetched from DRAM moves earlier. Softmax, exp2,
// row_max/row_sum reductions, the CHUNK=2 PV rescale+WMMA order, and normalize are all
// byte-for-byte v122. Prefetching K for a tile that exists (guarded by n_tile+1 <
// num_kv_tiles) reads only valid memory; the final tile does not prefetch. Each v_o[dt]
// result is rounding-identical to v122 -> n_bad=0, max_abs=0.0001 expected.
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast126(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v126_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v126_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v126(opus_attn_kargs k)
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
            v_q[kt][j] = bf16_fast126(bf16_to_f32(v_q[kt][j]) * qscale);
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

    // ---- Cross-tile K prefetch ring (v126): the 2-deep K double-buffer is hoisted
    // ABOVE the KV loop and primed ONCE with tile 0's dt=0 K. Inside each iteration the
    // last D-tile prefetches the NEXT KV tile's dt=0 K, so the buffer is never re-primed
    // from a cold load at the top of a QK phase -- the next tile's K is fetched during
    // this tile's softmax+PV (a long region with no K dependency).
    bf16x8_t v_k0_next = *reinterpret_cast<const bf16x8_t*>(Kp + (0 + col16) * stride_n + row8);
    bf16x8_t v_k1_next = *reinterpret_cast<const bf16x8_t*>(Kp + (0 + 16 + col16) * stride_n + row8);

    for (int n_tile = 0; n_tile < num_kv_tiles; ++n_tile) {
        const int n_base = n_tile * BLOCK_N;
        const int n_base_next = n_base + BLOCK_N;  // next KV tile's column base
        const bool have_next_tile = (n_tile + 1 < num_kv_tiles);

        // ---- QK^T: BLOCK_N=32 -> two 16x16x16 WMMAs per D-tile.
        // K loads are pipelined 2-deep AND wrap across the KV-tile boundary: for the
        // last D-tile (kt==DK-1) we prefetch the NEXT tile's dt=0 K (guarded), so its
        // VMEM latency hides behind this tile's softmax + PV WMMAs.
        fp32x8_t v_s0 = {0,0,0,0,0,0,0,0};
        fp32x8_t v_s1 = {0,0,0,0,0,0,0,0};

        #pragma unroll
        for (int kt = 0; kt < DK; ++kt) {
            bf16x8_t v_k0 = v_k0_next;
            bf16x8_t v_k1 = v_k1_next;
            if (kt + 1 < DK) {
                // prefetch next D-tile of the CURRENT KV tile
                v_k0_next = *reinterpret_cast<const bf16x8_t*>(Kp + (n_base + col16) * stride_n + (kt+1) * W_K + row8);
                v_k1_next = *reinterpret_cast<const bf16x8_t*>(Kp + (n_base + 16 + col16) * stride_n + (kt+1) * W_K + row8);
            } else if (have_next_tile) {
                // last D-tile: prefetch dt=0 of the NEXT KV tile (cross-tile wrap)
                v_k0_next = *reinterpret_cast<const bf16x8_t*>(Kp + (n_base_next + col16) * stride_n + row8);
                v_k1_next = *reinterpret_cast<const bf16x8_t*>(Kp + (n_base_next + 16 + col16) * stride_n + row8);
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
        row_max = v126_cross_half_max(row_max);

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
        row_sum = v126_cross_half_sum(row_sum);
        l_row += row_sum;

        bf16x8_t v_p0, v_p1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p0[j] = bf16_fast126(v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p1[j] = bf16_fast126(v_s1[j]);

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
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast126(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
