// Per-arch matrix-core tile kernels with the opus register plan expressed via the
// amdgpu_pin_{vgpr,agpr} attributes (no inline asm). One __global__ per target:
//
//   matrix_core_gfx942 / matrix_core_gfx950  (CDNA3, MFMA):
//       A/B -> AGPR, C -> VGPR  =>  v_mfma_f32_16x16x16_f16 v[C], a[A], a[B]
//   matrix_core_gfx1201 (RDNA4, WMMA, no AGPR file):
//       A/B -> VGPR, C -> VGPR  =>  v_wmma_f32_16x16x16_f16 v[C], v[A], v[B]
//
// Each kernel is compiled and dumped only for its own arch (guarded by the arch
// predefine). A single 16x16x16 f16 matrix-core op is one GEMM tile step; the
// opus tiled block kernel (matrix_core.cc) reuses the same pinned registers
// across the K loop.
//
// NOTE on the full opus block_v2 kernel: its A/B fragments are wide (>=256-bit,
// assembled from several buffer_load_dwordx2), and that wide multi-load AGPR
// fold does not reproduce in this build environment (it stays in VGPR). These
// single-tile kernels use the foldable form so the AGPR/VGPR pins take effect
// deterministically, which is what these ISA dumps demonstrate.
#include <hip/hip_runtime.h>

using f16x4 = _Float16 __attribute__((ext_vector_type(4)));  // MFMA A/B: 2 dwords
using f16x8 = _Float16 __attribute__((ext_vector_type(8)));  // WMMA A/B: 4 dwords
using f32x4 = float    __attribute__((ext_vector_type(4)));  // MFMA C:   4 dwords
using f32x8 = float    __attribute__((ext_vector_type(8)));  // WMMA C:   8 dwords

#if defined(__gfx942__)
__global__ void matrix_core_gfx942(const _Float16 *pa, const _Float16 *pb, float *pc) {
  int lane = __builtin_amdgcn_workitem_id_x();
  __attribute__((amdgpu_pin_agpr(0)))  f16x4 a = *reinterpret_cast<const f16x4 *>(pa + lane * 4);
  __attribute__((amdgpu_pin_agpr(2)))  f16x4 b = *reinterpret_cast<const f16x4 *>(pb + lane * 4);
  f32x4 c = {};
  __attribute__((amdgpu_pin_vgpr(0))) f32x4 d = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);
  *reinterpret_cast<f32x4 *>(pc + lane * 4) = d;
}
#endif

#if defined(__gfx950__)
__global__ void matrix_core_gfx950(const _Float16 *pa, const _Float16 *pb, float *pc) {
  int lane = __builtin_amdgcn_workitem_id_x();
  __attribute__((amdgpu_pin_agpr(0)))  f16x4 a = *reinterpret_cast<const f16x4 *>(pa + lane * 4);
  __attribute__((amdgpu_pin_agpr(2)))  f16x4 b = *reinterpret_cast<const f16x4 *>(pb + lane * 4);
  f32x4 c = {};
  __attribute__((amdgpu_pin_vgpr(0))) f32x4 d = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);
  *reinterpret_cast<f32x4 *>(pc + lane * 4) = d;
}
#endif

#if defined(__gfx1201__)
__global__ void matrix_core_gfx1201(const _Float16 *pa, const _Float16 *pb, float *pc) {
  int lane = __builtin_amdgcn_workitem_id_x();
  __attribute__((amdgpu_pin_vgpr(8)))  f16x8 a = *reinterpret_cast<const f16x8 *>(pa + lane * 8);
  __attribute__((amdgpu_pin_vgpr(12))) f16x8 b = *reinterpret_cast<const f16x8 *>(pb + lane * 8);
  f32x8 c = {};
  __attribute__((amdgpu_pin_vgpr(20))) f32x8 d = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
  *reinterpret_cast<f32x8 *>(pc + lane * 8) = d;
}
#endif
