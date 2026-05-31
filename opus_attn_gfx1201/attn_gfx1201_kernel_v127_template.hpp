// v127 -- v126's PROVEN-FAST chunked-rescale PV schedule (91.52 primary, bit-exact)
//          + a targeted CHUNK=2 double-buffered V software pipeline inside the PV
//          loop. This is reviewer NEXT-lever #2: "v126 plus a targeted V-load
//          prefetch/double-buffer inside the PV chunk." The schedule is now
//          correct (bit-exact) and the next plausible bottleneck is V-load latency
//          feeding back-to-back PV WMMAs.
//
// ================= WHY THIS IS NEW (and why earlier attempts didn't count) =====
//   * v126 = chunked rescale + V loaded inline per chunk -> every chunk's V loads
//     block on s_waitcnt before that chunk's first PV WMMA can issue: a per-chunk
//     head-of-group VMEM bubble in the matrix pipe.
//   * v124 = double-buffered V prefetch, BUT it paired it with the SLOWER
//     monolithic (all-8-fragments-at-once) v100 rescale, AND it was never added to
//     the host dVT allow-list, so it silently read the non-transposed dV buffer
//     (the round-19 root cause) -> it showed max_abs=0.0360 and was never honestly
//     measured. The double-buffer lever has therefore never been tested ON TOP OF
//     the fast chunked-rescale schedule with correct V wiring.
//   * v127 = v126 chunked rescale (fast, proven bit-exact once wired to dVT)
//     + v124's cross-chunk V prefetch (hide the head-of-chunk VMEM bubble),
//     and IS registered in the dVT allow-list this round.
//
// ================= THE SCHEDULE ==============================================
// Preload chunk 0's CHUNK*2 V fragments. Then for each CHUNK-sized D-tile group:
//   1. snapshot the resident V buffer for THIS chunk (cv0/cv1),
//   2. issue the NEXT chunk's CHUNK*2 V loads (memory-level-parallel burst that
//      retires under this chunk's PV WMMAs),
//   3. rescale ONLY this chunk's O fragments (VALU, overlaps the in-flight loads),
//   4. issue this chunk's PV WMMAs from the resident snapshot.
// Step 2's loads overlap step 4's WMMAs across the chunk boundary, eliminating the
// per-chunk VMEM stall v126 paid, while step 3 keeps the rescale VALU region small
// (only CHUNK O fragments hot at a time -> tight live ranges, v111/v126 lever).
//
// ================= BIT-EXACTNESS (non-negotiable) ============================
// Identical to v126's arithmetic:
//   * each v_o[dt][j] receives `*= rescale` EXACTLY ONCE (same scalar rescale),
//     before that fragment's two PV WMMAs -- same as v126;
//   * each v_o[dt] still accumulates v0-WMMA then v1-WMMA, same operands/order;
//   * distinct dt are independent accumulators (relative order irrelevant);
//   * row_max / row_sum reductions + cross-half ds_bpermute folds byte-identical.
// The ONLY change vs v126 is WHEN independent global V loads retire, which cannot
// alter any rounding. (Lesson banked: relocating loads is bit-exact; relocating
// the rescale past exp2/sum is NOT -- we keep the rescale per-chunk, pre-WMMA.)
//
// EXPECTED: primary (b4 h32 n4096) ~91.5-92.5 TFLOPS (v126 + hidden head-of-chunk
// VMEM bubble), max_abs ~0.0001, n_bad=0.
// RISK: low-med. Extra VGPR = one double-buffer of CHUNK*2 bf16x8 next-regs
// (~16 VGPR) live only during PV; v_o (64 fp32/lane) still dominates pressure.
// Worst case the scheduler collapses the prefetch back to v126's schedule -> ties
// it, still bit-exact.
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

        fp32x8_t v_s0 = {0,0,0,0,0,0,0,0};
        fp32x8_t v_s1 = {0,0,0,0,0,0,0,0};

        // ---- QKT with software-pipelined K loads (v63) ----
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
            // v_o rescale deferred into per-chunk PV loop (same as v126).
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

        // ---- PV: chunked rescale (v126) + CHUNK=2 double-buffered V SW-pipeline (v124) ----
        // Preload chunk 0's V. For each chunk: snapshot resident V, prefetch NEXT
        // chunk's V (overlaps this chunk's WMMAs), rescale ONLY this chunk's O
        // fragments (VALU under the in-flight loads), then this chunk's PV WMMAs.
        constexpr int CHUNK = 2;
        bf16x8_t vv0[CHUNK];
        bf16x8_t vv1[CHUNK];
        #pragma unroll
        for (int dc = 0; dc < CHUNK; ++dc) {
            const int dt = dc; // chunk 0 preload
            vv0[dc] = *reinterpret_cast<const bf16x8_t*>(VTp + (dt * W_K + col16) * vt_stride_d + n_base + row8);
            vv1[dc] = *reinterpret_cast<const bf16x8_t*>(VTp + (dt * W_K + col16) * vt_stride_d + n_base + 16 + row8);
        }
        #pragma unroll
        for (int c0 = 0; c0 < DK; c0 += CHUNK) {
            // Snapshot the resident buffer for THIS chunk before overwriting it.
            bf16x8_t cv0[CHUNK];
            bf16x8_t cv1[CHUNK];
            #pragma unroll
            for (int dc = 0; dc < CHUNK; ++dc) { cv0[dc] = vv0[dc]; cv1[dc] = vv1[dc]; }

            // Prefetch the NEXT chunk's V loads so they retire under this chunk's
            // PV WMMAs (hide the head-of-chunk VMEM bubble v126 paid).
            const int n0 = c0 + CHUNK;
            if (n0 < DK) {
                #pragma unroll
                for (int dc = 0; dc < CHUNK; ++dc) {
                    const int dt = n0 + dc;
                    vv0[dc] = *reinterpret_cast<const bf16x8_t*>(VTp + (dt * W_K + col16) * vt_stride_d + n_base + row8);
                    vv1[dc] = *reinterpret_cast<const bf16x8_t*>(VTp + (dt * W_K + col16) * vt_stride_d + n_base + 16 + row8);
                }
            }

            // Chunked rescale: only THIS chunk's O fragments hot (tight live ranges,
            // overlaps the in-flight prefetch loads). Same scalar rescale, applied
            // exactly once per fragment before its PV WMMAs -> bit-exact vs v126.
            if (need_rescale) {
                #pragma unroll
                for (int dc = 0; dc < CHUNK; ++dc) {
                    const int dt = c0 + dc;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) v_o[dt][j] *= rescale;
                }
            }

            // This chunk's PV WMMAs from the resident snapshot (v0 then v1 per dt).
            #pragma unroll
            for (int dc = 0; dc < CHUNK; ++dc) {
                const int dt = c0 + dc;
                v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(cv0[dc], v_p0, v_o[dt]);
                v_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(cv1[dc], v_p1, v_o[dt]);
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
