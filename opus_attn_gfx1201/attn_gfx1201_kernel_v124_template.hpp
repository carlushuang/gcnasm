// v124 -- DIRECTOR DIRECTION (A): recover v111's measured fast PV schedule while
//          staying BIT-EXACT vs v100 (max_abs ~1e-4, n_bad=0). New, coherent change:
//          combine the two proven-bit-exact PV memory-level-parallelism levers that
//          were each tried alone but NEVER combined:
//            * v123: grouped loads-then-WMMAs within a CHUNK=2 D-tile group
//                    (4 V loads issued back-to-back -> memory-level parallelism,
//                     only one chunk's V regs live at a time).
//            * v121: software-pipeline the V loads across the iteration boundary
//                    (prefetch the next group's V while the current group's PV
//                     WMMAs run -> hides global V-load latency in the matrix-pipe
//                     issue shadow).
//
// v123 grouped the loads but still WAITED on each chunk's V loads before that
// chunk's WMMAs -> the HEAD of every chunk has an exposed VMEM bubble (the matrix
// pipe stalls on s_waitcnt before the first WMMA of the chunk can issue). v121
// pipelined single D-tiles but issued loads one-fragment-at-a-time, giving the
// memory subsystem less in-flight parallelism per request burst. v124 does BOTH:
// double-buffer at CHUNK=2 granularity. Preload chunk 0's 4 V fragments, then for
// each chunk: (1) issue the NEXT chunk's 4 V loads (memory-level parallel burst),
// (2) issue THIS chunk's 4 PV WMMAs from the already-resident buffer. The next
// chunk's loads overlap this chunk's WMMAs => the per-chunk head-of-group VMEM
// bubble is hidden, and we keep the 4-loads-in-flight MLP burst.
//
// BIT-EXACTNESS (the non-negotiable gate): this changes ONLY the order/timing of
// independent global V *loads*. Every fp32 accumulation is untouched:
//   * the online-softmax rescale is the SAME monolithic all-8-fragment pass v100
//     does, in v100's EXACT location (inside need_rescale, BEFORE exp2/sum/pack);
//   * each v_o[dt] still receives v0-WMMA then v1-WMMA, same operands, same order;
//   * distinct dt are independent accumulators (their relative order is irrelevant);
//   * row_max / row_sum reductions and the cross-half ds_bpermute folds are byte
//     identical to v100.
// => arithmetically byte-identical to v100. The only thing the compiler sees
// differently is WHEN the V VMEM loads retire, which cannot change any rounding.
// (Lesson from v111/v122: relocating the *rescale* past exp2/sum broke
// bit-exactness; relocating *loads* does not. v121 already confirmed load
// pipelining stays bit-exact. v124 only reorders loads.)
//
// EXPECTED: primary (b4 h32 n4096) ~90.5-91.5 TFLOPS (v123 grouping + the missing
// cross-chunk latency hiding from v121), max_abs ~0.0001, n_bad=0.
// RISK: low. Extra VGPR = one double-buffer of CHUNK*2 bf16x8 next-regs (16 VGPR),
// live only during PV; v_o already dominates pressure. Worst case the scheduler
// collapses to v123's schedule -> ties it, still bit-exact.
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast124(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v124_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v124_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v124(opus_attn_kargs k)
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
            v_q[kt][j] = bf16_fast124(bf16_to_f32(v_q[kt][j]) * qscale);
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
        row_max = v124_cross_half_max(row_max);

        const fp32_t new_m = __builtin_fmaxf(m_row, row_max);
        fp32_t rescale = 1.0f;
        const bool need_rescale = (new_m != m_row);
        if (need_rescale) {
            rescale = __builtin_amdgcn_exp2f(m_row - new_m);
            l_row *= rescale;
            // ---- Batched pre-PV rescale (v43/v100), EXACT v100 placement ----
            // Monolithic single pass over all 8 O accumulators, BEFORE exp2/sum/
            // pack. Byte-identical to v100 -> bit-exact. (v111/v122 lost
            // bit-exactness precisely by deferring this past exp2/sum into PV.)
            #pragma unroll
            for (int dt = 0; dt < DK; ++dt)
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                    v_o[dt][j] *= rescale;
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
        row_sum = v124_cross_half_sum(row_sum);
        l_row += row_sum;

        bf16x8_t v_p0, v_p1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p0[j] = bf16_fast124(v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p1[j] = bf16_fast124(v_s1[j]);

        // ---- PV: branch-free, CHUNK=2 double-buffered V software pipeline (v124) ----
        // Rescale is ALREADY done (monolithic, above) -> this loop is purely PV and
        // bit-exact. Combine v123's grouped 4-loads-in-flight MLP burst with v121's
        // cross-iteration V prefetch: preload chunk 0's V, then for each chunk issue
        // the NEXT chunk's 4 V loads (so they retire under the current chunk's WMMAs)
        // and run THIS chunk's 4 PV WMMAs from the resident double-buffer. Hides the
        // head-of-chunk VMEM bubble v123 still paid. Pure load reordering -> the
        // accumulation order into each v_o[dt] (v0 then v1) is unchanged.
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

            // Prefetch the NEXT chunk's V loads (memory-level-parallel burst) so they
            // overlap this chunk's PV WMMAs. Guarded to never read out of range.
            const int n0 = c0 + CHUNK;
            if (n0 < DK) {
                #pragma unroll
                for (int dc = 0; dc < CHUNK; ++dc) {
                    const int dt = n0 + dc;
                    vv0[dc] = *reinterpret_cast<const bf16x8_t*>(VTp + (dt * W_K + col16) * vt_stride_d + n_base + row8);
                    vv1[dc] = *reinterpret_cast<const bf16x8_t*>(VTp + (dt * W_K + col16) * vt_stride_d + n_base + 16 + row8);
                }
            }

            // This chunk's PV WMMAs (same per-fragment order as v100 -> bit-exact).
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
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast124(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
