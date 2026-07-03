// Per-arch opus tiled-MMA matrix-core kernels with the register plan applied via
// the amdgpu_pin_{vgpr,agpr} attributes (no inline asm). One __global__ per
// target, each built on opus::make_tiled_mma (the same high-level path as the
// op_tests/opus/device tiled tests):
//
//   matrix_core_gfx942 / matrix_core_gfx950  (CDNA3, MFMA, wave64):
//       make_tiled_mma<..>(.., mfma_adaptor_swap_ab)  + pin_agpr A/B
//       => buffer_load a[..] ; v_mfma_f32_16x16x16_f16 v[C], a[A], a[B]
//   matrix_core_gfx1201 (RDNA4, WMMA, wave32, no AGPR file):
//       make_tiled_mma(make_wmma<..>(.., wmma_adaptor_swap_ab), ..) + pin_vgpr A/B
//       => buffer_load v[..] ; v_wmma_f32_16x16x16_f16 v[C], v[A], v[B]
//
// 16x16x16 f16, one wave, BLOCK == WAVE (single tile per thread) so the A/B
// fragment is one load and the pin folds directly into the requested registers.
// Each kernel is compiled/dumped only for its own arch (guarded by the arch
// predefine).
#include <hip/hip_runtime.h>
#include "opus/opus.hpp"

#if defined(__gfx942__) || defined(__gfx950__)
template <int WM, int WN, int WK>
__device__ void mfma_tile(const opus::fp16_t *ptr_a, const opus::fp16_t *ptr_b,
                          opus::fp32_t *ptr_c, int k, int stride_a,
                          int stride_b, int stride_c) {
  using opus::operator""_I;
  constexpr int E_M = 1, E_N = 1, E_K = 1, T_M = 1, T_N = 1, T_K = 1;
  constexpr int ELEM_A = WM * WK / 64, ELEM_B = WN * WK / 64;
  using d_a = opus::fp16_t; using d_b = opus::fp16_t; using d_c = opus::fp32_t;
  int lane_id = __builtin_amdgcn_workitem_id_x() % opus::get_warp_size();
  int wave_id = __builtin_amdgcn_workitem_id_x() / opus::get_warp_size();
  int g_im = __builtin_amdgcn_workgroup_id_x() * WM;
  int g_in = __builtin_amdgcn_workgroup_id_y() * WN;
  auto mma = opus::make_tiled_mma<d_a, d_b, d_c>(
      opus::seq<E_M, E_N, E_K>{}, opus::seq<T_M, T_N, T_K>{},
      opus::seq<WM, WN, WK>{}, opus::mfma_adaptor_swap_ab{});
  auto u_a = opus::partition_layout_a<ELEM_A>(
      mma, opus::make_tuple(stride_a, 1_I),
      opus::make_tuple(wave_id / 2, lane_id % mma.grpm_a, 0_I, lane_id / mma.grpm_a));
  auto u_b = opus::partition_layout_b<ELEM_B>(
      mma, opus::make_tuple(stride_b, 1_I),
      opus::make_tuple(wave_id % 2, lane_id % mma.grpn_b, 0_I, lane_id / mma.grpn_b));
  auto u_c = opus::partition_layout_c(
      mma, opus::make_tuple(stride_c, 1_I),
      opus::make_tuple(wave_id / 2, lane_id % mma.grpn_c, wave_id % 2, lane_id / mma.grpn_c));
  auto g_a = opus::make_gmem(ptr_a + g_im * stride_a);
  auto g_b = opus::make_gmem(ptr_b + g_in * stride_b);
  auto g_c = opus::make_gmem(ptr_c + g_im * stride_c + g_in);
  int loops = (k + WK - 1) / WK;
  typename decltype(mma)::vtype_c v_c;             // accumulator -> VGPR
  opus::clear(v_c);
  for (int i = 0; i < loops; i++) {
    __attribute__((amdgpu_pin_agpr(0)))  auto v_a = g_a.template load<ELEM_A>(u_a);  u_a += WK;
    __attribute__((amdgpu_pin_agpr(8)))  auto v_b = g_b.template load<ELEM_B>(u_b);  u_b += WK;
    v_c = mma(v_a, v_b, v_c);
  }
  g_c.template store<4>(v_c, u_c);
}
#endif

#if defined(__gfx942__)
__global__ void matrix_core_gfx942(const opus::fp16_t *a, const opus::fp16_t *b,
                                   opus::fp32_t *c, int k, int sa, int sb, int sc) {
  mfma_tile<16, 16, 16>(a, b, c, k, sa, sb, sc);
}
#endif
#if defined(__gfx950__)
__global__ void matrix_core_gfx950(const opus::fp16_t *a, const opus::fp16_t *b,
                                   opus::fp32_t *c, int k, int sa, int sb, int sc) {
  mfma_tile<16, 16, 16>(a, b, c, k, sa, sb, sc);
}
#endif

#if defined(__gfx1201__) || defined(__gfx1200__)
__global__ void matrix_core_gfx1201(const opus::fp16_t *ptr_a, const opus::fp16_t *ptr_b,
                                    opus::fp32_t *ptr_c, int k, int stride_a,
                                    int stride_b, int stride_c) {
  using opus::operator""_I;
  constexpr int WM = 16, WN = 16, WK = 16;
  constexpr int E_M = 1, E_N = 1, E_K = 1, T_M = 1, T_N = 1, T_K = 1;
  constexpr int PACK_A = WM * WK / 32, PACK_B = WN * WK / 32; // 8 f16 = 128-bit
  constexpr int PACK_C = 4;                                   // fp32 store: 16 bytes
  using d_a = opus::fp16_t; using d_b = opus::fp16_t; using d_c = opus::fp32_t;
  int lane_id = __builtin_amdgcn_workitem_id_x() % opus::get_warp_size();
  int g_im = __builtin_amdgcn_workgroup_id_x() * WM;
  int g_in = __builtin_amdgcn_workgroup_id_y() * WN;
  auto mma = opus::make_tiled_mma(
      opus::make_wmma<d_a, d_b, d_c>(opus::seq<WM, WN, WK>{}, opus::wmma_adaptor_swap_ab{}),
      opus::seq<E_M, E_N, E_K>{}, opus::seq<T_M, T_N, T_K>{});
  auto u_a = opus::partition_layout_a<PACK_A>(mma, opus::make_tuple(stride_a, 1_I),
      opus::make_tuple(0_I, lane_id % mma.grpm_a, 0_I, lane_id / mma.grpm_a));
  auto u_b = opus::partition_layout_b<PACK_B>(mma, opus::make_tuple(stride_b, 1_I),
      opus::make_tuple(0_I, lane_id % mma.grpn_b, 0_I, lane_id / mma.grpn_b));
  auto u_c = opus::partition_layout_c(mma, opus::make_tuple(stride_c, 1_I),
      opus::make_tuple(0_I, lane_id % mma.grpn_c, 0_I, lane_id / mma.grpn_c));
  auto g_a = opus::make_gmem(ptr_a + g_im * stride_a);
  auto g_b = opus::make_gmem(ptr_b + g_in * stride_b);
  auto g_c = opus::make_gmem(ptr_c + g_im * stride_c + g_in);
  int loops = (k + WK - 1) / WK;
  typename decltype(mma)::vtype_c v_c;             // accumulator -> VGPR
  opus::clear(v_c);
  for (int i = 0; i < loops; i++) {
    __attribute__((amdgpu_pin_vgpr(8)))  auto v_a = g_a.template load<PACK_A>(u_a);  u_a += WK;
    __attribute__((amdgpu_pin_vgpr(12))) auto v_b = g_b.template load<PACK_B>(u_b);  u_b += WK;
    v_c = mma(v_a, v_b, v_c);
  }
  g_c.template store<PACK_C>(v_c, u_c);
}
#endif
