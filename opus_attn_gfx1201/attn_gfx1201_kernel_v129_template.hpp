// v129 -- PERSISTENT-WG KV-STREAMING (lane_B structural lever, round 11).
//
// ================= HYPOTHESIS ===============================================
// v126 already streams the full N (all KV blocks) for ONE (b,h,m-tile) in a
// single kernel instance, but the launch still spawns B*H*(N/BLOCK_M)
// INDEPENDENT workgroups (4096 WGs at the b4h32n4096 primary shape). Each WG
// cold-reads its head's entire K/V from VRAM. Although the HW dispatcher tends
// to co-schedule nearby blockIdx WGs, that ordering is best-effort and the tail
// of the grid wave (a partial CU fill) plus per-WG launch/retire bookkeeping
// leaves a few % on the table.
//
// v129 launches a FIXED pool of persistent workgroups sized to exactly fill the
// GPU (num_CU * hipOccupancyMaxActiveBlocksPerMultiprocessor, computed on the
// host), and gives each WG a CONTIGUOUS range of work items ordered so the
// m-tile index varies fastest within a head:
//       wi = ((b*H + h) * num_m_tiles) + m_tile
// A contiguous chunk of wi therefore stays inside ONE head, so that head's K/V
// (1 MB each at N=4096) is read from VRAM once and reused out of L2 across all
// the m-tiles the WG owns -- guaranteed, not best-effort. The persistent grid
// is exactly machine-sized, so every CU is fully packed for the entire kernel
// (no ragged tail wave) and there is one launch's worth of WG setup instead of
// thousands.
//
// ================= BIT-EXACTNESS ===========================================
// The PER-WORK-ITEM body is the v126 device code VERBATIM (same K double-buffer
// QK^T, same v126 cross-half max/sum reduction order, same CHUNK=2 chunked
// pre-PV rescale, same fp32 online-softmax accumulation order, same dVT
// pre-transposed V buffer). Only the OUTER driver changed: a host-sized
// persistent loop that re-runs that identical body for each (b,h,m) the WG
// owns. No fp32 add is reordered -> max_abs must stay 0.0001 (bit-exact).
//
// RISK: if the persistent grid under-fills a CU relative to the default launch's
// natural packing, occupancy could drop and regress. Mitigated by sizing the
// grid from the kernel's own measured max-active-blocks (full occupancy by
// construction). Falls back to >= default behavior when total tiles < grid.
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

    const int stride_n = k.D;
    const int stride_h = k.N * k.D;
    const int stride_b = k.H * k.N * k.D;

    const int vt_stride_h = k.D * k.N;
    const int vt_stride_b = k.H * k.D * k.N;
    const int vt_stride_d = k.N;

    const int num_m_tiles = k.N / BLOCK_M;
    const int num_kv_tiles = k.N / BLOCK_N;
    const int total_tiles = k.B * k.H * num_m_tiles;

    // ---- persistent contiguous-chunk work partition (m-tile fastest) --------
    const int g     = static_cast<int>(blockIdx.x);
    const int nwg   = static_cast<int>(gridDim.x);
    const int chunk = total_tiles / nwg;
    const int rem   = total_tiles - chunk * nwg;
    const int wi_start = g * chunk + (g < rem ? g : rem);
    const int wi_count = chunk + (g < rem ? 1 : 0);

    constexpr fp32_t LOG2_E = 1.44269504088896340736f;
    const fp32_t qscale = k.scale * LOG2_E;

    for (int wi = wi_start; wi < wi_start + wi_count; ++wi) {
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
            for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast129(v_o[dt][j] * inv);
            *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
        }
    }
#else
    (void)k;
#endif
}
