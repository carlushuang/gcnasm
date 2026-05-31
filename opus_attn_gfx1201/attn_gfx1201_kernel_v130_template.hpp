// v130 -- PERSISTENT-WG KV-STREAMING, GRID-STRIDE (L2-locality fix of v129).
//
// ================= WHY v129 DID NOT WIN ====================================
// v129 was the first persistent-WG kernel: a machine-sized pool of WGs, each
// owning a CONTIGUOUS run of work items wi = g*chunk .. g*chunk+chunk. The
// intent ("a WG stays inside one head, K/V hot in L2") is right for the END of
// the kernel, but WRONG for the FRONTIER that is actually co-resident on the GPU
// at any instant. At t=0 every WG g=0..nwg-1 starts at wi=g*chunk; with
// total_tiles=4096 and nwg~=192, chunk~=21, so the live wi set is
// {0,21,42,...,4011} -- which touches ALL ~126 distinct heads at the SAME time.
// 126 heads x 2 MB K/V each massively exceeds L2 -> every head's K/V is streamed
// cold from VRAM anyway. The contiguous-chunk partition optimizes intra-WG reuse
// but destroys INTER-WG (co-resident) reuse, which is what L2 actually caches.
//
// ================= HYPOTHESIS (v130) =======================================
// Keep the persistent machine-sized grid, but distribute work with a GRID-STRIDE
// loop over the head-major global index:
//       wi = (b*H + h)*num_m_tiles + q_tile_id        (m-tile fastest in a head)
//       for (wi = g; wi < total_tiles; wi += nwg)
// Now the co-resident WG frontier at step t is the CONTIGUOUS window
//       [t*nwg, (t+1)*nwg)
// which spans only ceil(nwg / num_m_tiles) distinct heads. At the primary shape
// (num_m_tiles = 4096/128 = 32, nwg ~= 192) that is ~6 heads live at once, each
// shared by ~32 WGs that all read the SAME 2 MB K/V out of L2. The grid advances
// head-by-head in lock-step, so a head's K/V is fetched from VRAM ~once and
// reused L2-resident by every m-tile of that head -- the bandwidth win v129 was
// supposed to deliver but didn't. The persistent grid still removes v126's
// ragged tail wave (4096 tiles / 192 WGs = 21.3 grid-waves spread evenly over a
// fixed, fully-packed pool) and pays one launch's WG setup, not 4096.
//
// ================= BIT-EXACTNESS ===========================================
// The PER-WORK-ITEM body is the v126 device code VERBATIM (same K double-buffer
// QK^T, same v130 cross-half max/sum reduction order, same CHUNK=2 chunked
// pre-PV rescale, same fp32 online-softmax accumulation order, same dVT
// pre-transposed V buffer). v130 changes ONLY which (b,h,m) each persistent WG
// visits and in what order -- a pure work permutation. Each output (b,h,m) tile
// is still computed by exactly one WG with the identical instruction stream, so
// no fp32 add is reordered -> max_abs must stay 0.0001 (bit-exact, == v126/v129).
//
// RISK: low. Same launcher / occupancy story as v129 (grid sized from the
// kernel's measured max-active-blocks). If grid-stride still does not beat v126
// then L2 reuse is not the limiter and the persistent-WG lever is exhausted.
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast130(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v130_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v130_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v130(opus_attn_kargs k)
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

    const int stride_n = k.D;
    const int stride_h = k.N * k.D;
    const int stride_b = k.H * k.N * k.D;

    const int vt_stride_h = k.D * k.N;
    const int vt_stride_b = k.H * k.D * k.N;
    const int vt_stride_d = k.N;

    const int num_m_tiles = k.N / BLOCK_M;
    const int num_kv_tiles = k.N / BLOCK_N;
    const int total_tiles = k.B * k.H * num_m_tiles;

    // ---- persistent GRID-STRIDE work partition (head-major wi) --------------
    // wi = (b*H + h)*num_m_tiles + q_tile_id  -> m-tile fastest WITHIN a head.
    // A grid-stride loop (wi = g, g+nwg, g+2*nwg, ...) makes the CO-RESIDENT WG
    // frontier at any instant the contiguous window [t*nwg, (t+1)*nwg), which
    // spans only ceil(nwg / num_m_tiles) heads -- e.g. nwg~=192, num_m_tiles=32
    // -> ~6 heads live at once, each shared by ~32 WGs reading its K/V out of L2.
    // (v129's contiguous-chunk partition started every WG at g*chunk, so its
    //  frontier spanned ALL ~126 heads simultaneously -> L2 thrash. Same body;
    //  this only reorders which (b,h,m) each persistent WG visits, so it is
    //  bit-exact with v129/v126 by construction.)
    const int g     = static_cast<int>(blockIdx.x);
    const int nwg   = static_cast<int>(gridDim.x);

    constexpr fp32_t LOG2_E = 1.44269504088896340736f;
    const fp32_t qscale = k.scale * LOG2_E;

    for (int wi = g; wi < total_tiles; wi += nwg) {
        const int q_tile_id = wi % num_m_tiles;
        const int bh        = wi / num_m_tiles;
        const int h         = bh % k.H;
        const int b         = bh / k.H;

        const bf16_t* __restrict__ Qp = reinterpret_cast<const bf16_t*>(k.ptr_q) + b * stride_b + h * stride_h;
        const bf16_t* __restrict__ Kp = reinterpret_cast<const bf16_t*>(k.ptr_k) + b * stride_b + h * stride_h;
        bf16_t*       __restrict__ Op = reinterpret_cast<bf16_t*>(k.ptr_o) + b * stride_b + h * stride_h;
        const bf16_t* __restrict__ VTp = reinterpret_cast<const bf16_t*>(k.ptr_v) + b * vt_stride_b + h * vt_stride_h;

        const int q_m_base = q_tile_id * BLOCK_M + wave_id * 16;
        const bf16_t* q_row = Qp + (q_m_base + col16) * stride_n;
        bf16x8_t v_q[DK];
        #pragma unroll
        for (int kt = 0; kt < DK; ++kt) {
            v_q[kt] = *reinterpret_cast<const bf16x8_t*>(&q_row[kt * W_K + row8]);
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                v_q[kt][j] = bf16_fast130(bf16_to_f32(v_q[kt][j]) * qscale);
        }

        fp32x8_t v_o[DK];
        #pragma unroll
        for (int kt = 0; kt < DK; ++kt) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_o[kt][j] = 0.0f;
        }
        fp32_t m_row = -3.4e38f;
        fp32_t l_row = 0.0f;

        for (int n_tile = 0; n_tile < num_kv_tiles; ++n_tile) {
            const int n_base = n_tile * BLOCK_N;

            fp32x8_t v_s0 = {0,0,0,0,0,0,0,0};
            fp32x8_t v_s1 = {0,0,0,0,0,0,0,0};

            // ---- QKT with software-pipelined K loads (v63/v126) ----
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
            row_max = v130_cross_half_max(row_max);

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
            row_sum = v130_cross_half_sum(row_sum);
            l_row += row_sum;

            bf16x8_t v_p0, v_p1;
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_p0[j] = bf16_fast130(v_s0[j]);
            #pragma unroll
            for (int j = 0; j < 8; ++j) v_p1[j] = bf16_fast130(v_s1[j]);

            // ---- PV: chunked rescale interleaved with WMMAs (v126) ----
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
            for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast130(v_o[dt][j] * inv);
            *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
        }
    }
#else
    (void)k;
#endif
}
