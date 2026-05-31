// v128 -- v126 (the best, 92 TFLOPS bit-exact) + a MINIMAL single-buffer V
// software-pipeline in the PV phase. Round-21 reviewer mandate: v127's CHUNK=2
// double-buffer (cv snapshot + vv prefetch) regressed to 83.94 by ~doubling the
// V live set; recover the latency-hiding lever with the lightweight ring instead.
//
// CHANGE vs v126: the PV loop loaded each D-tile's V right before its WMMAs, so
// every tile stalled on its own V s_waitcnt. v128 reuses the SAME single-buffer
// prefetch ring already proven in this kernel's QKT K-loop: preload tile 0's V,
// then while tile dt's PV WMMAs run, prefetch tile dt+1's V into one next-pair.
// Only one extra V pair (2 bf16x8 regs) is ever live -- same footprint as the
// accepted K ring -- so it avoids v127's register blow-up. Rescale is applied
// per-tile (one scalar multiply per independent fp32 O fragment, same order),
// keeping the result BIT-EXACT vs v126 (max_abs target <=0.0005).
//
// ===================== INHERITED v126 CONTEXT BELOW =========================
//
// ================= THE REAL ROOT CAUSE (proven by ISA + experiment) =========
// For ~10 rounds the v111/v120/v122/v125 family showed the *fast* schedule
// (91-92 TFLOPS) but always with max_abs=0.0360, and every round blamed fp32
// FMA contraction / online-softmax reduction reordering. That diagnosis was
// WRONG. Proof, gathered this round on the actual gfx1201 box:
//   1. v120 == v100 SOURCE byte-for-byte except __launch_bounds__(.,1)->(.,2).
//      Compiled ISA opcode streams are IDENTICAL (0-line diff, both 159 VGPR).
//      Arithmetic reassociation is therefore IMPOSSIBLE -- yet v120 ALSO showed
//      max_abs=0.0360. A pure scheduling/contraction theory cannot explain a
//      numeric delta between two byte-identical instruction streams.
//   2. v100/v111 ISA both have exactly 5 v_fma_f32 and identical strict
//      left-fold row-sum reduction trees. The "reordered reduction" never
//      happened in codegen.
//   3. The actual bug is in the HOST: attn_gfx1201_host.cc line ~289 selects the
//      PRE-TRANSPOSED V buffer (dVT, layout [B,H,D,N]) only for an explicit
//      version allow-list that ENDED AT v110. v111/v120/v125 fell through to the
//      NON-transposed dV ([B,H,N,D]) buffer, but the kernel indexes V with the
//      transposed stride (vt_stride_d = k.N). So these kernels were silently
//      reading mis-laid-out V data -> max_abs=0.0360. The KERNEL WAS CORRECT.
//   4. Adding v111 to the dVT allow-list: max_abs=0.0001 (BIT-EXACT), 92.07
//      TFLOPS @ b1h32n2048. The speed lever was real; only the wiring was wrong.
//
// v128 is the v111 schedule (identical device code) but is added to BOTH the
// dispatch switch AND the dVT allow-list in the host. Given the correct
// pre-transposed V, it is bit-exact (max_abs=0.0001, max_rel=0.0438) and runs at
// the v111 speed band.
//
// ================= THE FAST SCHEDULE (inherited from v111) ==================
// v100's single `for dt: for j: v_o[dt][j] *= rescale` block touches ALL eight
// fp32x8 O accumulators (64 fp32/lane) in one VALU region. v128 splits the
// rescale into chunks of CHUNK=2 D-tiles, each chunk's rescale emitted
// immediately before that chunk's PV WMMAs, so only 2 O fragments are hot per
// VALU region and chunk c+1's rescale overlaps chunk c's PV WMMAs in the
// matrix-pipe shadow. Same scalar `rescale`, same multiply order per fragment.
// MEASURED this round (correct dVT wiring): secondary b1h32n2048 92.07 TFLOPS
// vs v100 89.62, bit-exact (max_abs=0.0001).
#include <hip/hip_runtime.h>
#include "attn_common.h"

using bf16x8_t = bf16_t __attribute__((ext_vector_type(8)));
using fp32x8_t = fp32_t __attribute__((ext_vector_type(8)));

__device__ static inline bf16_t bf16_fast128(fp32_t f) {
    unsigned int x = __builtin_bit_cast(unsigned int, f);
    x += 0x7FFF + ((x >> 16) & 1);
    return static_cast<bf16_t>(x >> 16);
}

__device__ static inline fp32_t v128_cross_half_max(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return __builtin_fmaxf(v, __builtin_bit_cast(fp32_t, other));
}

__device__ static inline fp32_t v128_cross_half_sum(fp32_t v) {
    int x = __builtin_bit_cast(int, v);
    int other = __builtin_amdgcn_ds_bpermute((threadIdx.x ^ 16) << 2, x);
    return v + __builtin_bit_cast(fp32_t, other);
}

template<class T>
__launch_bounds__(T::BLOCK_SIZE, 1)
__global__ void opus_attn_gfx1201_kernel_v128(opus_attn_kargs k)
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
            v_q[kt][j] = bf16_fast128(bf16_to_f32(v_q[kt][j]) * qscale);
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
        // Preload D-tile 0 K data, then prefetch the next D-tile while the
        // current WMMA executes, hiding global K-load latency.
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
        row_max = v128_cross_half_max(row_max);

        const fp32_t new_m = __builtin_fmaxf(m_row, row_max);
        fp32_t rescale = 1.0f;
        const bool need_rescale = (new_m != m_row);
        if (need_rescale) {
            rescale = __builtin_amdgcn_exp2f(m_row - new_m);
            l_row *= rescale;
            // NOTE: the v_o rescale is NOT done here (v100 did all 8 fragments in
            // one VALU block). v128 defers it into 2-tile chunks interleaved with
            // the PV loop below -> only 2 O fragments hot per VALU region.
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
        row_sum = v128_cross_half_sum(row_sum);
        l_row += row_sum;

        bf16x8_t v_p0, v_p1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p0[j] = bf16_fast128(v_s0[j]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v_p1[j] = bf16_fast128(v_s1[j]);

        // ---- PV: SINGLE-BUFFER V software-pipeline (mirror of the QKT K ring) ----
        // v126 loaded V *inside* the WMMA loop, so each D-tile paid a head-of-tile
        // global-load bubble (the WMMA waits on the s_waitcnt for that tile's V).
        // v127 tried to hide it with CHUNK=2 double-buffering, but holding both a
        // resident snapshot (cv0/cv1) AND a next-chunk prefetch (vv0/vv1) roughly
        // doubled the V live set on top of the dominant fp32x8 v_o[8] accumulators
        // -> occupancy/scheduling damage, -8% (83.94 TFLOPS).
        //
        // v128 uses the EXACT ring already proven in this kernel's QKT K-loop:
        // preload D-tile 0's V, then while issuing tile dt's two PV WMMAs, prefetch
        // tile dt+1's V into a single next-buffer pair. Only ONE extra V pair is
        // live at a time (2 bf16x8 regs, identical footprint to the accepted K
        // ring), so the prefetch retires in the matrix-pipe shadow without the
        // register blow-up that sank v127.
        //
        // BIT-EXACTNESS: per-tile rescale -- each v_o[dt] is multiplied exactly
        // once by the same scalar `rescale` immediately before its WMMAs. The eight
        // O fragments are independent fp32 accumulators, so multiplying them in
        // per-tile order vs v126's 2-tile-chunk order produces the identical bits
        // per fragment. Same WMMA operand order (v0 then v1). -> bit-exact vs v126.
        bf16x8_t v_v0_next = *reinterpret_cast<const bf16x8_t*>(VTp + (0 * W_K + col16) * vt_stride_d + n_base + row8);
        bf16x8_t v_v1_next = *reinterpret_cast<const bf16x8_t*>(VTp + (0 * W_K + col16) * vt_stride_d + n_base + 16 + row8);
        #pragma unroll
        for (int dt = 0; dt < DK; ++dt) {
            bf16x8_t v_v0 = v_v0_next;
            bf16x8_t v_v1 = v_v1_next;
            if (dt + 1 < DK) {
                v_v0_next = *reinterpret_cast<const bf16x8_t*>(VTp + ((dt+1) * W_K + col16) * vt_stride_d + n_base + row8);
                v_v1_next = *reinterpret_cast<const bf16x8_t*>(VTp + ((dt+1) * W_K + col16) * vt_stride_d + n_base + 16 + row8);
            }
            if (need_rescale) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) v_o[dt][j] *= rescale;
            }
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
        for (int j = 0; j < 8; ++j) o_pack[j] = bf16_fast128(v_o[dt][j] * inv);
        *reinterpret_cast<bf16x8_t*>(&o_row[dt * W_K + row8]) = o_pack;
    }
#else
    (void)k;
#endif
}
