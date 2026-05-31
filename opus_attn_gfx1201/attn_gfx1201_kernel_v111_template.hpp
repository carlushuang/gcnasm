// v111 -- v100 + bit-exact BALANCED-TREE softmax max-reduction (zero-register change).
//
// Base: v100 (the measured champion @ ~89.4-90.45 TFLOPS). Every later attempt
// (v101-v110) failed to clear the gate. The decisive lesson from v109/v110 is that
// ANY change adding register liveness across the PV phase (where the fp32x8 O
// accumulator pressure peaks at ~205 VGPR / 7 waves) tips an occupancy/allocation
// cliff and REGRESSES (v109 -5, v110 ~tie/-). So the only safe lever left is one with
// ZERO added register cost.
//
// THE EXPOSED CHAIN: between the QK^T WMMAs (produce v_s0/v_s1) and the PV WMMAs, the
// kernel must compute row_max, because the PV operand v_p = bf16(exp2(s - new_m))
// depends on new_m = max(m_row, row_max). v100 computes row_max with a STRICTLY SERIAL
// fmax chain over 16 values:
//     row_max = v_s0[0]; for j: row_max = fmax(row_max, v_s0[j]); for j: fmax(.., v_s1[j]);
// That is a ~15-deep dependency chain. The compiler is NOT allowed to reassociate
// fmax of floats without -ffast-math (the build uses plain -O3), so this serial chain
// survives to the ISA and sits exposed on the QK->softmax->PV critical path, stalling
// the matrix pipe between the two WMMA phases.
//
// THE CHANGE (only this block differs from v100): replace the serial chain with a
// balanced pairwise tree of depth 4:
//   t[j] = fmax(v_s0[j], v_s1[j])   (8 independent fmax, depth 1)
//   then reduce the 8 lane-local partials 8->4->2->1 (depths 2-4).
// This shortens the exposed reduction latency from ~15 to ~4 dependent fmax ops with
// ZERO extra live registers (t[] is consumed immediately; v_s0/v_s1 were already live).
//
// BIT-EXACTNESS: fmax over a fixed set of finite values is fully associative and
// commutative -- it merely SELECTS the maximum element, it never rounds. So the tree
// yields the byte-identical row_max as v100's serial chain for every input. The
// cross-half fold (v111_cross_half_max) and everything downstream (new_m, rescale,
// exp2, the serial row_sum, bf16 pack, both WMMA phases, K-prefetch ring) are
// byte-for-byte v100. The row_sum reduction is left SERIAL and identical to v100
// (it is off the PV critical path -- l_row is consumed only after the KV loop -- so
// there is no reason to perturb its arithmetic). Result: n_bad(>0.05)=0, bit-exact.
//
// VGPR: identical to v100 (~205, 7 waves/SIMD). No LDS, no barriers, no new buffers.
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast111(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v111_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v111_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v111(opus_attn_kargs k)
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
            v_q[kt][j] = bf16_fast111(bf16_to_f32(v_q[kt][j]) * qscale);
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

        // ---- Softmax row max over 16 values (8 from s0, 8 from s1).
        // BALANCED TREE (depth 4) instead of v100's serial ~15-deep fmax chain.
        // fmax is associative+commutative for finite values -> selects the same
        // maximum element -> byte-identical row_max as v100, but on a far shorter
        // dependency chain that no longer stalls the QK->PV transition. Zero extra
        // live registers (t[] is consumed immediately).
        fp32_t t[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) t[j] = __builtin_fmaxf(v_s0[j], v_s1[j]); // depth 1
        #pragma unroll
        for (int j = 0; j < 4; ++j) t[j] = __builtin_fmaxf(t[j], t[j + 4]);   // depth 2
        t[0] = __builtin_fmaxf(t[0], t[2]);                                    // depth 3
        t[1] = __builtin_fmaxf(t[1], t[3]);
        fp32_t row_max = __builtin_fmaxf(t[0], t[1]);                          // depth 4
        row_max = v111_cross_half_max(row_max);

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

        // row_sum: kept SERIAL and byte-for-byte identical to v100 (off the PV
        // critical path -- l_row is consumed only after the KV loop).
        fp32_t row_sum = v_s0[0];
        #pragma unroll
        for (int j = 1; j < 8; ++j) row_sum += v_s0[j];
        #pragma unroll
        for (int j = 0; j < 8; ++j) row_sum += v_s1[j];
        row_sum = v111_cross_half_sum(row_sum);
        l_row += row_sum;

        bf16x8_t v_p0, v_p1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p0[j] = bf16_fast111(v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p1[j] = bf16_fast111(v_s1[j]);

        // ---- PV: two WMMAs per D-tile, rescale fused per D-tile (v39/v100 structure).
        #pragma unroll
        for (int dt = 0; dt < DK; ++dt) {
            if (need_rescale) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) v_o[dt][j] *= rescale;
            }
            const bf16_t* vt_addr0 = VTp + (dt * W_K + col16) * vt_stride_d + n_base + row8;
            const bf16_t* vt_addr1 = VTp + (dt * W_K + col16) * vt_stride_d + n_base + 16 + row8;
            bf16x8_t v_v0 = *reinterpret_cast<const bf16x8_t*>(vt_addr0);
            bf16x8_t v_v1 = *reinterpret_cast<const bf16x8_t*>(vt_addr1);
            v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_v0, v_p0, v_o[dt]);
            v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(v_v1, v_p1, v_o[dt]);
        }
    }

    const fp32_t inv = (l_row > 0.0f) ? (1.0f / l_row) : 0.0f;
    bf16_t* o_row = Op + (q_m_base + col16) * stride_n;
    #pragma unroll
    for (int dt = 0; dt < DK; ++dt) {
        bf16x8_t o_pack;
        #pragma unroll
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast111(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
